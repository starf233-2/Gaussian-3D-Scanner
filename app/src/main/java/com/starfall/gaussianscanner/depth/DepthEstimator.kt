package com.starfall.gaussianscanner.depth

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.starfall.gaussianscanner.model.DepthMap
import com.starfall.gaussianscanner.qnn.QnnEngine
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeoutException
import kotlin.math.abs

/**
 * Depth Anything 3 Small inference.
 *
 * The model runs through the Qualcomm QNN TFLite delegate when possible so the
 * workload is offloaded from the CPU to the Adreno GPU (all Snapdragon) or the
 * Hexagon DSP/HTP (NPU). Fallbacks, in order: QNN GPU -> TFLite GPU delegate ->
 * CPU (XNNPACK).
 *
 * Primary model: da3_small_gpu_fp32.tflite — the official Qualcomm float export
 * (NHWC [1,518,518,3] input, raw [0,1] pixels, ImageNet stats handled inside the
 * graph). A legacy NCHW fp16/INT8 export can be dropped in for compatibility;
 * layout and normalization are detected from the input tensor shape.
 *
 * Every hardware attempt runs under a hard timeout so we never freeze the app.
 */
class DepthEstimator(
    context: Context,
    enginePref: String,
    private val onStatus: (String) -> Unit = {},
    debug: Boolean = false,
    saveDiag: Boolean = false
) {
    private val ctx = context.applicationContext

    companion object {
        private const val TAG = "DepthEstimator"
        const val MODEL_DEFAULT = "da3_small_gpu_fp32.tflite"   // 高通官方 small (NHWC, raw [0,1], 内嵌标准化)
        const val MODEL_FP16 = "da3_small_gpu_fp16.tflite"       // 旧版 NCHW 小模型（兼容）
        const val MODEL_INT8 = "da3_small_int8.tflite"           // 可选 INT8 小模型（兼容）

        /** Overall budget for one QNN backend (delegate + graph prepare). */
        private const val QNN_GPU_TIMEOUT_MS = 600_000L
        /** Budget for a retry with skip-ops. */
        private const val QNN_RETRY_TIMEOUT_MS = 180_000L
        /** Budget for DSP/HTP (NPU) attempts: on-device HTP graph prepare for a
         *  large fp32 model can take minutes on the first run; the cache is
         *  only written after a successful prepare, so a short timeout would
         *  make HTP fail forever. */
        private const val QNN_NPU_TIMEOUT_MS = 300_000L
        /** Budget for the classic TFLite GPU delegate. */
        private const val GPU_DELEGATE_TIMEOUT_MS = 120_000L
    }

    private class Engine(
        interpreter: Interpreter,
        val delegate: org.tensorflow.lite.Delegate?,
        model: ByteBuffer,
        val w: Int, val h: Int, val nhwc: Boolean
    ) : AutoCloseable {
        private var live: Interpreter? = interpreter
        private var modelBuffer: ByteBuffer? = model
        val interpreter: Interpreter get() = checkNotNull(live) { "模型已释放" }
        @Volatile var closed = false
        var input: ByteBuffer? = null
        var output: ByteBuffer? = null
        override fun close() {
            if (closed) return
            closed = true
            // Interpreter must release graph references BEFORE its delegate.
            try { live?.close() } finally {
                live = null; modelBuffer = null; input = null; output = null
                NativeRunner.forgetResident(this)
                delegate?.close()
            }
        }
    }
    @Volatile private var engine: Engine? = null
    // Accessed only on the native lane, so close() cannot invalidate a running shot.
    private var runningEngine: Engine? = null
    @Volatile private var builtReady = false
    val ready: Boolean get() = builtReady && engine?.closed == false

    /** Human readable label of the backend that actually runs the model. */
    @Volatile var backendLabel = "CPU (XNNPACK)"; private set
    @Volatile var inferenceReport = "单图推理"; private set

    /** Input geometry of the active model (filled after the interpreter loads). */
    @Volatile var inputW = 518
        private set
    @Volatile var inputH = 518
        private set
    /** true when the active model's input is NHWC ([1,H,W,3]). */
    @Volatile var inputNhwc = true
        private set
    /** true when the active model expects externally ImageNet-normalized NCHW data. */
    private var legacyNormalize = false

    /** Details of every backend attempt (diagnostics / debug log). */
    val attemptLog = ArrayList<String>()

    val deviceInfo = QnnEngine.detectDevice()
    val enginePreference = QnnEngine.preferenceFromString(enginePref)

    init {
        QnnEngine.verboseLogging = debug
        try {
            builtReady = buildEngine(ctx, pickModelForPreference())
            if (ready) {
                persistActualEngine()
                if (saveDiag) dumpDiagnostics()
            }
        } catch (t: TimeoutException) {
            attemptLog.add("原生任务超时: ${t.message}")
            status("驱动任务尚未退出，暂不启动并行回退；请稍后重试")
        } catch (t: InterruptedException) {
            Thread.currentThread().interrupt()
            throw t // Closing the page must not trigger a fresh CPU build.
        } catch (t: IllegalStateException) {
            if (NativeRunner.busy) {
                attemptLog.add(t.message ?: "原生任务繁忙")
                status(t.message ?: "原生任务繁忙")
            } else {
                status("模型加载失败，尝试 CPU …")
                buildCpu(MODEL_DEFAULT, null)
                builtReady = engine != null
            }
        } catch (t: Throwable) {
            attemptLog.add("模型加载失败: ${t.message}")
            status("模型加载失败，尝试 CPU …")
            buildCpu(MODEL_DEFAULT, null)
            builtReady = engine != null
            if (ready) persistActualEngine()
            if (saveDiag) dumpDiagnostics()
        }
    }

    /**
     * When the chosen NPU preference cannot run on this device (e.g. no HTP
     * capability), persist the actually-used backend as the engine preference
     * so the Settings page reflects reality and later starts don't re-attempt
     * a backend that can never work. The NPU option stays available for manual
     * re-selection.
     */
    private fun persistActualEngine() {
        if (enginePreference != QnnEngine.Preference.NPU) return
        if (backendLabel.contains("NPU")) return
        val actual = if (backendLabel.contains("GPU")) "gpu" else "cpu"
        try {
            ctx.getSharedPreferences("gaussianscanner", Context.MODE_PRIVATE)
                .edit().putString("engine", actual).apply()
            status("NPU在此设备不可用，已自动切换为 $actual")
        } catch (_: Throwable) {}
    }

    private fun assetExists(name: String): Boolean =
        runCatching { ctx.assets.open(name).close(); true }.getOrDefault(false)

    private fun pickModelForPreference(): String {
        if (enginePreference == QnnEngine.Preference.NPU && assetExists(MODEL_INT8)) return MODEL_INT8
        if (assetExists(MODEL_DEFAULT)) return MODEL_DEFAULT
        if (assetExists(MODEL_FP16)) return MODEL_FP16
        return MODEL_DEFAULT
    }

    private fun buildEngine(ctx: Context, modelName: String): Boolean {
        legacyNormalize = modelName == MODEL_FP16 || modelName == MODEL_INT8
        val b = loadModel(modelName)

        // NPU backends that are known to fail on this device (remembered in
        // SharedPreferences) are skipped in AUTO mode so a rejected/hanging DSP
        // no longer costs time on every app start. Choosing "NPU" explicitly
        // retries them and clears the flag on success.
        val prefs = ctx.getSharedPreferences("gaussianscanner", Context.MODE_PRIVATE)
        val gpuQuality = QnnEngine.GpuQuality.fromPref(prefs.getString("gpu_quality", "balanced"))
        val gpuMode = prefs.getString("gpu_mode", "auto") ?: "auto"
        val gpuCompat = prefs.getBoolean("gpu_compat", false)
        // low-memory protection: delegate graph compilation can peak at GBs and
        // freeze low-RAM devices — CPU only there
        val lowRam = QnnEngine.isLowRamDevice(ctx)
        if (lowRam) attemptLog.add("低内存设备，启用保护模式（仅CPU）")
        var attempts = QnnEngine.planBackends(enginePreference, deviceInfo, lowRam)
        if (enginePreference != QnnEngine.Preference.NPU) {
            attempts = attempts.filter { a ->
                when (a) {
                    QnnEngine.BackendAttempt.HTP -> !prefs.getBoolean("qnn_htp_bad", false)
                    QnnEngine.BackendAttempt.DSP -> !prefs.getBoolean("qnn_dsp_bad", false)
                    else -> true // GPU is always worth trying
                }
            }
        }

        if (attempts.isNotEmpty()) {
            for (a in attempts) {
                // per-attempt lazy preload: load only THIS backend's runtime
                // libs (the 50 MB libQnnHtpPrepare is only pinned when NPU is
                // actually attempted)
                val hint = if (a == QnnEngine.BackendAttempt.GPU)
                    "（${gpuQuality.label}，首次编译GPU内核可能需要几分钟）" else ""
                status("正在尝试 ${a.displayName} $hint …")
                val ok = try { buildWithQnn(ctx, modelName, b, a, gpuQuality, gpuMode, gpuCompat) }
                catch (t: TimeoutException) {
                    if (a != QnnEngine.BackendAttempt.GPU) {
                        prefs.edit().putBoolean(if (a == QnnEngine.BackendAttempt.HTP) "qnn_htp_bad" else "qnn_dsp_bad", true).apply()
                    }
                    throw t
                }
                if (a != QnnEngine.BackendAttempt.GPU) {
                    val key = when (a) {
                        QnnEngine.BackendAttempt.HTP -> "qnn_htp_bad"
                        else -> "qnn_dsp_bad"
                    }
                    prefs.edit().putBoolean(key, !ok).apply()
                }
                if (ok) return true
            }
        }

        if (enginePreference != QnnEngine.Preference.CPU && !lowRam) {
            // If the preference only requested NPU and it failed, still fall back
            // to QNN GPU so the user gets acceleration instead of raw CPU.
            // (Skipped entirely on low-RAM devices — the protection must hold.)
            if (attempts.none { it == QnnEngine.BackendAttempt.GPU }) {
                val reason = attemptLog.lastOrNull { it.startsWith("尝试 QNN NPU") } ?: ""
                status("NPU不可用${if (reason.isNotBlank()) "（${reason.substringAfter("— ", "")}）" else ""}，尝试 ${QnnEngine.BackendAttempt.GPU.displayName} …")
                if (buildWithQnn(ctx, modelName, b, QnnEngine.BackendAttempt.GPU, gpuQuality, gpuMode, gpuCompat)) {
                    return true
                }
            }
            status("QNN不可用，尝试 TFLite GPU …")
            if (buildWithTfliteGpu(modelName, b)) {
                return true
            }
        }

        status("GPU不可用，回退 CPU (XNNPACK) …")
        buildCpu(modelName, b)
        return engine != null
    }

    /** Always write a short diagnostic file so failures are visible without Debug Mode. */
    private fun dumpDiagnostics() {
        val contents = buildString {
            appendLine("=== GaussianScanner QNN 诊断 ===")
            appendLine("时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())}")
            appendLine("设备: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
            appendLine("SoC: ${deviceInfo.socModel} / ${deviceInfo.hardware}")
            appendLine("平台: ${deviceInfo.acceleratorSummary}")
            appendLine("引擎偏好: ${enginePreference}")
            appendLine("模型输入: ${inputW}x${inputH} ${if (inputNhwc) "NHWC" else "NCHW"}")
            appendLine("最终后端: $backendLabel")
            attemptLog.forEach { appendLine("  $it") }
        }
        val candidates = ArrayList<java.io.File>()
        try {
            // 1) public Documents (works when "all files access" is granted)
            val base = if (android.os.Build.VERSION.SDK_INT >= 30) {
                if (android.os.Environment.isExternalStorageManager())
                    android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS)
                else java.io.File(ctx.getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS) ?: ctx.filesDir, "documents")
            } else {
                android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS)
            }
            candidates.add(java.io.File(base, "GaussianScanner"))   // /Documents/GaussianScanner (public) or app dir
            candidates.add(java.io.File(ctx.filesDir, "qnn_diag")) // guaranteed-writable fallback
        } catch (_: Throwable) {
            candidates.add(java.io.File(ctx.filesDir, "qnn_diag"))
        }

        val written = ArrayList<String>()
        for (dir in candidates.distinct()) {
            try {
                if (dir.exists() || dir.mkdirs()) {
                    val f = java.io.File(dir, "qnn_diag_${System.currentTimeMillis()}.txt")
                    f.writeText(contents)
                    written.add(f.absolutePath)
                }
            } catch (_: Throwable) {}
        }
        if (written.isNotEmpty()) {
            attemptLog.add("诊断文件: ${written.joinToString("; ")}")
            status("诊断日志: ${written.first()}")
            Log.i(TAG, "diag saved to ${written.first()} (${written.size} copies)")
        } else {
            attemptLog.add("诊断写入失败")
            status("诊断日志写入失败")
        }
    }

    /** Run delegate creation + Interpreter build for one QNN backend, under timeout.
     *
     * For the GPU backend we retry with progressively larger skip lists: the
     * QNN delegate converts the WHOLE graph at once and one offending op makes
     * the entire delegation fail ("Failed to apply delegate"). Skipping a few
     * suspicious ops (they then run on CPU) lets the rest of the graph use the
     * GPU. TFLite builtin codes: GELU=150, BATCH_MATMUL=126, EXP=47, RSQRT=76,
     * SQUARED_DIFFERENCE=99, MEAN=40, RESIZE_NEAREST_NEIGHBOR=97, SLICE=65.
     */
    private fun buildWithQnn(ctx: Context, modelName: String, b: ByteBuffer, attempt: QnnEngine.BackendAttempt,
                             gpuQuality: QnnEngine.GpuQuality = QnnEngine.GpuQuality.BALANCED,
                             gpuMode: String = "auto",
                             gpuCompat: Boolean = false): Boolean {
        // DEFAULT: full-graph delegation only. The skip-ops fallback chain was
        // the cause of an intermittent "skip 3 op types" state that degraded the
        // depth (whole-graph prepare occasionally ran slower than the timeout and
        // fell through to a broad-skip level, whose many CPU/GPU partition
        // boundaries produce bad depth). It is now OFF unless the user explicitly
        // enables "GPU兼容模式" in Settings.
        val skipLevels: Array<IntArray?> = when {
            attempt != QnnEngine.BackendAttempt.GPU -> arrayOf(null)
            gpuMode == "quality" -> arrayOf(
                intArrayOf(150, 126, 25),                  // GELU + BatchMatMul + Softmax -> CPU
                intArrayOf(150, 126, 25, 47, 76, 99, 40, 97, 65)
            )
            gpuCompat -> arrayOf(
                null,                       // 0: whole graph
                intArrayOf(150),            // 1: +GELU
                intArrayOf(150, 126),       // 2: +BatchMatMul
                intArrayOf(150, 126, 47, 76, 99, 40, 97, 65) // 3: broad
            )
            else -> arrayOf(null)           // full graph only
        }

        for ((level, skips) in skipLevels.withIndex()) {
            val timeout = when {
                attempt != QnnEngine.BackendAttempt.GPU -> QNN_NPU_TIMEOUT_MS
                level == 0 -> QNN_GPU_TIMEOUT_MS
                else -> QNN_RETRY_TIMEOUT_MS
            }
            val desc = skips?.let { "跳过算子 ${it.joinToString(" ")}" } ?: "全图委托"
            status("正在尝试 ${attempt.displayName}（$desc）…")
            if (tryBuildQnn(ctx, modelName, b, attempt, skips, timeout, gpuQuality)) return true
        }
        return false
    }

    private fun tryBuildQnn(
        ctx: Context, modelName: String, b: ByteBuffer, attempt: QnnEngine.BackendAttempt,
        skipOps: IntArray?, timeout: Long, gpuQuality: QnnEngine.GpuQuality = QnnEngine.GpuQuality.BALANCED
    ): Boolean {
        // Cache token MUST encode backend + precision + skip level, otherwise the
        // QNN delegate reuses the first prepared/cached context binary for every
        // setting (which is why switching GPU precision changed nothing).
        val installStamp = ctx.packageManager.getPackageInfo(ctx.packageName, 0).lastUpdateTime
        val token = "$modelName|$installStamp|${attempt.name}|${gpuQuality.name}|${skipOps?.sorted()?.joinToString(".") ?: "full"}"
        return try {
            val built = NativeRunner.call(timeout, discard = { value: Engine -> value.close() }) {
                NativeRunner.checkActive()
                val failures = QnnEngine.preloadRuntimes(attempt)
                check(failures.isEmpty()) { failures.joinToString("; ") }
                NativeRunner.checkActive()
                val res = QnnEngine.createQnnDelegate(ctx, attempt, modelToken = token, skipOps = skipOps,
                    gpuQuality = gpuQuality, quantized = modelName == MODEL_INT8)
                val delegate = res.delegate ?: error(res.detail)
                createEngine(b, delegate)
            }
            adopt(built, modelName)
            backendLabel = attempt.displayName + if (skipOps == null || skipOps.isEmpty()) "" else "（跳过${skipOps.size}类算子）"
            attemptLog.add("尝试 ${attempt.displayName}: 成功")
            status("已使用 $backendLabel")
            true
        } catch (t: TimeoutException) {
            throw t // JNI is still running; allocating a fallback now is unsafe.
        } catch (t: Throwable) {
            if (NativeRunner.busy || t is InterruptedException) throw t
            attemptLog.add("尝试 ${attempt.displayName}: 失败 — ${t.message}")
            status("${attempt.displayName} 不可用: ${t.message}")
            false
        }
    }

    /** Run the classic TFLite GPU delegate path under a timeout. */
    private fun buildWithTfliteGpu(modelName: String, b: ByteBuffer): Boolean {
        return try {
            val built = NativeRunner.call(GPU_DELEGATE_TIMEOUT_MS, discard = { value: Engine -> value.close() }) {
                check(CompatibilityList().isDelegateSupportedOnThisDevice) { "设备不支持 TFLite GPU" }
                createEngine(b, GpuDelegate())
            }
            adopt(built, modelName)
            backendLabel = "TFLite GPU"
            status("已使用 $backendLabel")
            true
        } catch (t: TimeoutException) {
            throw t
        } catch (t: Throwable) {
            if (NativeRunner.busy || t is InterruptedException) throw t
            attemptLog.add("尝试 TFLite GPU: 异常 — ${t.message}")
            false
        }
    }

    private fun buildCpu(modelName: String, b: ByteBuffer?) {
        try {
            val buf = b ?: loadModel(modelName)
            val built = NativeRunner.call(QNN_NPU_TIMEOUT_MS, discard = { value: Engine -> value.close() }) {
                createEngine(buf, null)
            }
            adopt(built, modelName)
            backendLabel = "CPU (XNNPACK)"
            attemptLog.add("回退: CPU (XNNPACK)")
        } catch (t: Throwable) {
            if (t is TimeoutException || t is InterruptedException || NativeRunner.busy) throw t
            attemptLog.add("CPU回退失败: ${t.message}")
        }
    }

    private fun loadModel(name: String): ByteBuffer = ctx.assets.openFd(name).use { descriptor ->
        descriptor.createInputStream().use { stream ->
            stream.channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
        }
    }

    private fun createEngine(buf: ByteBuffer, delegate: org.tensorflow.lite.Delegate?): Engine {
        var interpreter: Interpreter? = null
        try {
            NativeRunner.checkActive()
            NativeRunner.clearResident()
            val options = Interpreter.Options().apply {
                setNumThreads(if (QnnEngine.isLowRamDevice(ctx)) 2 else 4)
                setUseXNNPACK(delegate == null)
                if (delegate != null) addDelegate(delegate)
            }
            interpreter = Interpreter(buf, options)
            val shape = interpreter.getInputTensor(0).shape()
            require(shape.size == 4 && shape[0] == 1 && shape.all { it > 0 }) { "不支持的模型输入形状" }
            val nhwc = shape[3] == 3
            require(nhwc || shape[1] == 3) { "模型输入必须为三通道" }
            val w = if (nhwc) shape[2] else shape[3]
            val h = if (nhwc) shape[1] else shape[2]
            require(interpreter.getOutputTensor(0).numElements().toLong() == w.toLong() * h) { "模型输出不是同尺寸单通道深度图" }
            NativeRunner.checkActive()
            return Engine(interpreter, delegate, buf, w, h, nhwc).also { NativeRunner.keepResident(it) }
        } catch (t: Throwable) {
            runCatching { interpreter?.close() }
            runCatching { delegate?.close() }
            throw t
        }
    }

    private fun adopt(value: Engine, modelName: String) {
        engine = value
        inputW = value.w; inputH = value.h; inputNhwc = value.nhwc
        attemptLog.add("模型输入: ${inputW}x${inputH} ${if (inputNhwc) "NHWC" else "NCHW"} ($modelName)")
    }

    fun estimate(bmp: Bitmap): DepthMap? {
        val active = engine ?: return null
        val prefs = ctx.getSharedPreferences("gaussianscanner", Context.MODE_PRIVATE)
        return try {
            val tiled = prefs.getBoolean("tiled_inference", false)
            // The tiled operation can run ten model invocations, sequentially.
            NativeRunner.call(if (tiled) QNN_GPU_TIMEOUT_MS * 3 else QNN_GPU_TIMEOUT_MS,
                onAbandoned = { if (!bmp.isRecycled) bmp.recycle() }) {
                check(!active.closed) { "模型已释放" }
                runningEngine = active
                try {
                    if (tiled) estimateTiled(bmp) else estimateSingle(bmp)
                } finally { runningEngine = null }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "inference failed", e)
            if (e is TimeoutException || e is InterruptedException) { close(); throw e }
            null
        }
    }

    /** Run the model on a bitmap already at model size; returns the RAW depth. */
    private fun runModel(px: IntArray): FloatArray? {
        NativeRunner.checkActive()
        val active = runningEngine ?: return null
        val it = active.interpreter
        val w = inputW; val h = inputH
        val inTensor = it.getInputTensor(0)
        val inType = inTensor.dataType()
        val outTensor = it.getOutputTensor(0)
        val supported = setOf(org.tensorflow.lite.DataType.FLOAT32, org.tensorflow.lite.DataType.INT8, org.tensorflow.lite.DataType.UINT8)
        require(inType in supported && outTensor.dataType() in supported) { "不支持的张量数据类型" }
        val inp = active.input ?: ByteBuffer.allocateDirect(inTensor.numBytes()).order(ByteOrder.nativeOrder()).also { active.input = it }
        val out = active.output ?: ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder()).also { active.output = it }
        inp.clear(); out.clear()

        val quantization = inTensor.quantizationParams()
        TensorInput.write(inp, px, inputNhwc, legacyNormalize && !inputNhwc,
            inType == org.tensorflow.lite.DataType.FLOAT32, quantization.scale, quantization.zeroPoint,
            inType == org.tensorflow.lite.DataType.INT8)
        it.run(inp, out); out.rewind()
        NativeRunner.checkActive()
        val d = FloatArray(h * w)
        val oT = it.getOutputTensor(0)
        val oq = oT.quantizationParams()
        val outSigned = oT.dataType() == org.tensorflow.lite.DataType.INT8
        if (oT.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            for (j in d.indices) d[j] = out.float
        } else {
            for (j in d.indices) {
                val raw = if (outSigned) out.get().toInt() else (out.get().toInt() and 0xFF)
                d[j] = (raw - oq.zeroPoint) * oq.scale
            }
        }
        require(d.any { it.isFinite() }) { "模型输出没有有效深度值" }
        DepthMath.sanitize(d)
        return d
    }

    private fun smoothOn(): Boolean =
        ctx.getSharedPreferences("gaussianscanner", Context.MODE_PRIVATE)
            .getBoolean("depth_smooth", true)

    private fun estimateSingle(bmp: Bitmap): DepthMap? {
        inferenceReport = "单图推理"
        val w = inputW; val h = inputH
        val rs = Bitmap.createScaledBitmap(bmp, w, h, true)
        val px = IntArray(w * h)
        try { rs.getPixels(px, 0, w, 0, 0, w, h) } finally { if (rs !== bmp) rs.recycle() }
        val raw = runModel(px) ?: return null
        val finalD = if (smoothOn()) postProcessDepth(raw, w, h, px) else raw
        return normalizeDepth(finalD, w, h)
    }

    /** Equal-size native-resolution tiles with a globally anchored depth scale. */
    private fun estimateTiled(bmp: Bitmap): DepthMap? {
        val w = inputW; val h = inputH
        val plan = TiledDepthFusion.plan(w, h, bmp.width, bmp.height)
        if (plan.tiles.isEmpty()) {
            val single = estimateSingle(bmp)
            inferenceReport = "原图像素不足，使用单图深度 ${w}×${h}"
            status(inferenceReport)
            return single
        }
        status("高分辨率 1/${plan.tiles.size + 1}：全图参考")
        val refSmall = Bitmap.createScaledBitmap(bmp, w, h, true)
        val refPx = IntArray(w * h)
        try { refSmall.getPixels(refPx, 0, w, 0, 0, w, h) } finally { if (refSmall !== bmp) refSmall.recycle() }
        val ref = runModel(refPx) ?: return null
        val cleanRef = if (smoothOn()) postProcessDepth(ref, w, h, refPx) else ref
        val reference = TiledDepthFusion.upsample(cleanRef, w, h, plan.w, plan.h)
        val src = Bitmap.createScaledBitmap(bmp, plan.w, plan.h, true)
        try {
            val pixels = IntArray(w * h)
            val fused = TiledDepthFusion.fuse(reference, plan) { tile, index ->
                NativeRunner.checkActive()
                status("高分辨率 ${index + 2}/${plan.tiles.size + 1}：瓦片 ${index + 1}/${plan.tiles.size}")
                // Read an exact input-sized rectangle. No crop Bitmap and no
                // second resize, so every tile really sees higher source detail.
                src.getPixels(pixels, 0, w, tile.x, tile.y, w, h)
                try {
                    runModel(pixels)
                } catch (t: Exception) {
                    if (t is java.util.concurrent.CancellationException) throw t
                    Log.w(TAG, "tile ${index + 1} failed", t)
                    null
                }
            }
            if (fused.accepted == 0 || fused.changedFraction < .005f) {
                inferenceReport = "高分辨率未得到有效增量（瓦片 ${fused.accepted}/${plan.tiles.size}），回退单图 ${w}×${h}"
                status(inferenceReport)
                return normalizeDepth(cleanRef, w, h)
            }
            // Mild edge-aware filtering only; the old full post-processing at
            // the end erased precisely the new tile edges we wanted to retain.
            val out = if (smoothOn()) {
                val px = IntArray(plan.w * plan.h)
                src.getPixels(px, 0, plan.w, 0, 0, plan.w, plan.h)
                val luma = FloatArray(px.size) { j ->
                    val p = px[j]
                    (.299f * ((p shr 16) and 255) + .587f * ((p shr 8) and 255) + .114f * (p and 255)) / 255f
                }
                DepthMath.guidedFilter(luma, fused.data, plan.w, plan.h, 1, .001f)
            } else fused.data
            inferenceReport = "深度 ${plan.w}×${plan.h} · 瓦片 ${fused.accepted}/${plan.tiles.size} · 深度变化区域 ${"%.1f".format(fused.changedFraction * 100)}%"
            status(inferenceReport)
            return normalizeDepth(out, plan.w, plan.h)
        } finally { if (src !== bmp) src.recycle() }
    }
    /** Robust percentile normalization: maps [p1,p99] to [0,1] (深度一致性校准). */
    private fun normalizeDepth(d: FloatArray, w: Int, h: Int): DepthMap {
        return DepthMath.normalize(d, w, h)
    }

    /**
     * Depth post-processing: (1) remove isolated outliers, (2) handle sky /
     * specular regions, (3) luma-guided edge-preserving filter. Applied on the
     * RAW depth before percentile normalization.
     */
    private fun postProcessDepth(src: FloatArray, w: Int, h: Int, px: IntArray): FloatArray {
        if (w < 3 || h < 3) return src
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
        for (v in src) { if (v < mn) mn = v; if (v > mx) mx = v }
        val range = (mx - mn).coerceAtLeast(1e-6f)

        // robust noise scale: mean absolute deviation from the median
        val sortedArr = src.copyOf()
        sortedArr.sort()
        val centre = sortedArr[sortedArr.size / 2]
        var madSum = 0.0
        for (v in sortedArr) madSum += abs(v - centre)
        val mad = (madSum / sortedArr.size).toFloat().let { maxOf(it, 1e-4f) }

        // --- outlier removal: samples far from their 8-neighbour mean ---
        val out = sortedArr
        System.arraycopy(src, 0, out, 0, src.size)
        val thresh = maxOf(6f * mad, range * 0.05f)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                var sum = 0f
                sum += src[idx - 1]; sum += src[idx + 1]
                sum += src[idx - w]; sum += src[idx + w]
                sum += src[idx - w - 1]; sum += src[idx - w + 1]
                sum += src[idx + w - 1]; sum += src[idx + w + 1]
                val mean = sum / 8f
                if (abs(src[idx] - mean) > thresh) out[idx] = mean
            }
        }

        // --- specular highlight handling (sky handling removed: unreliable) ---
        handleSpecular(out, w, h, px)

        // --- luma-guided edge-preserving filter ---
        val luma = FloatArray(w * h)
        for (j in luma.indices) {
            val p = px[j]
            luma[j] = (0.299f * ((p shr 16) and 0xFF) +
                0.587f * ((p shr 8) and 0xFF) +
                0.114f * (p and 0xFF)) / 255f
        }
        return guidedFilter(luma, out, w, h, radius = 2, eps = 0.0025f)
    }

    /**
     * Specular highlight handling: near-clipped bright spots produce noisy
     * depth; they are replaced by their 8-neighbour median.
     * (Sky detection was removed — the heuristic misclassified too many
     *  real scenes.)
     */
    private fun handleSpecular(d: FloatArray, w: Int, h: Int, px: IntArray) {
        val tmp = FloatArray(8)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                val p = px[idx]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                if (maxOf(r, g, b) > 0.95f) {   // specular highlight
                    tmp[0] = d[idx - 1]; tmp[1] = d[idx + 1]
                    tmp[2] = d[idx - w]; tmp[3] = d[idx + w]
                    tmp[4] = d[idx - w - 1]; tmp[5] = d[idx - w + 1]
                    tmp[6] = d[idx + w - 1]; tmp[7] = d[idx + w + 1]
                    tmp.sort()
                    d[idx] = (tmp[3] + tmp[4]) / 2f
                }
            }
        }
    }

    /**
     * Fast guided filter (box-filter via integral images), guided by luma so
     * depth edges follow image edges. radius=2 keeps memory bounded.
     */
    private fun guidedFilter(guide: FloatArray, src: FloatArray, w: Int, h: Int, radius: Int, eps: Float): FloatArray {
        return DepthMath.guidedFilter(guide, src, w, h, radius, eps)
    }

    private fun status(msg: String) {
        Log.i(TAG, msg)
        onStatus(msg)
    }

    fun close() {
        builtReady = false
        val active = engine
        engine = null
        if (active != null) NativeRunner.release { active.close() }
    }
}
