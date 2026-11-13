package com.starfall.gaussianscanner.qnn

import android.content.Context
import android.os.Build
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import java.io.File

/**
 * Wraps the Qualcomm QNN (AI Engine Direct) TFLite delegate.
 *
 * The QNN delegate lets a TFLite model be partially or fully executed on
 * Qualcomm silicon instead of the CPU:
 *
 *  - GPU  : Adreno GPU via OpenCL   (libQnnGpu.so)            — works on ALL Adreno
 *  - HTP  : Hexagon Tensor Processor (libQnnHtp.so + Vxx)      — Snapdragon 888 and newer
 *  - DSP  : Hexagon DSP w/ HVX v66   (libQnnDsp.so + V66)      — Snapdragon 865 / 870
 *
 * The delegate is obtained from com.qualcomm.qti.QnnDelegate (qtld.jar), and the
 * runtime .so files live in jniLibs/arm64-v8a. DSP/HTP skel libraries live in
 * assets/qnn/skels and are copied to a private dir at runtime so the fastRPC
 * driver can send them to the DSP/HTP.
 */
object QnnEngine {

    private const val TAG = "QnnEngine"

    enum class Preference { AUTO, GPU, NPU, CPU }

    /** When true the QNN delegate/backends log at INFO (for debugging). */
    @Volatile var verboseLogging = false

    /** Platform capability summary used for decision making and the UI. */
    data class DeviceInfo(
        val socModel: String,
        val hardware: String,
        val chipset: String,
        val isQualcomm: Boolean,
        val hasHtp: Boolean,
        val hasDspV66: Boolean,
        val hasAdrenoGpu: Boolean
    ) {
        val acceleratorSummary: String
            get() = buildString {
                append(chipset.ifBlank { socModel.ifBlank { hardware } })
                if (isQualcomm) {
                    when {
                        hasHtp -> append(" (NPU: HTP)")
                        hasDspV66 -> append(" (NPU: DSP v66)")
                        else -> append(" (无可用NPU)")
                    }
                    append(" / GPU: Adreno")
                } else append(" / 非骁龙平台")
            }
    }

    data class DelegateResult(val delegate: QnnDelegate?, val label: String, val detail: String)

    // Board name -> friendly chipset
    private val BOARD_NAMES = mapOf(
        "alchemy" to "Snapdragon 855",
        "kona" to "Snapdragon 865/870",
        "lahaina" to "Snapdragon 888",
        "taro" to "Snapdragon 8 Gen 1",
        "kalama" to "Snapdragon 8 Gen 2",
        "pineapple" to "Snapdragon 8 Gen 3",
        "sun" to "Snapdragon 8 Elite"
    )

    // soc_model (Build.SOC_MODEL) -> friendly chipset
    // Note: Snapdragon 870 is SM8250-AC, but many 870 devices (e.g. Xiaomi)
    // report plain "SM8250" in Build.SOC_MODEL — same die, same DSP/GPU.
    private val SOC_NAMES = mapOf(
        "SM8150" to "Snapdragon 855",
        "SM8250" to "Snapdragon 865/870",
        "SM8250-AC" to "Snapdragon 870",
        "SM8250AB" to "Snapdragon 865",
        "SM8350" to "Snapdragon 888",
        "SM8450" to "Snapdragon 8 Gen 1",
        "SM8475" to "Snapdragon 8+ Gen 1",
        "SM8550" to "Snapdragon 8 Gen 2",
        "SM8650" to "Snapdragon 8 Gen 3",
        "SM8635" to "Snapdragon 8s Gen 3",
        "SM8750" to "Snapdragon 8 Elite"
    )

    // HTP-capable => Snapdragon 888 (SM8350) and newer.
    private val HTP_SOC = setOf(
        "SM8350", "SM8450", "SM8475", "SM8550", "SM8650", "SM8635", "SM8750"
    )
    private val HTP_BOARDS = setOf(
        "lahaina", // SM8350
        "taro", // SM8450 / SM8475
        "kalama", // SM8550
        "pineapple", // SM8650
        "sun", // SM8750
        "cliffs", // SM8635
        "garnet", "parrot", "pit", "zuk" // Snapdragon 7/8 Gen 2+ variants
    )

    // DSP v66 (HVX) => Snapdragon 865 / 870 (SM8250 family, board "kona").
    private val DSP66_SOC = setOf(
        "SM8250", "SM8250-AC", "SM8250AB", "SM8250M", "SM8250-CD"
    )
    private val DSP66_BOARDS = setOf("kona")

    // Known numeric SoC ids from /sys/devices/soc0/soc_id.
    private val SOC_ID = mapOf(
        356L to "SM8250", 415L to "SM8350", 441L to "SM8450",
        457L to "SM8475", 453L to "SM8550", 617L to "SM8650"
    )

    /** Detect the device platform without loading any QNN library. */
    fun detectDevice(): DeviceInfo {
        val socModel = readSocModel()
        val board = (Build.BOARD + " " + Build.HARDWARE).trim().lowercase()
        val manufacturer = Build.MANUFACTURER.lowercase()
        val socId = readSocId()

        val isQualcomm = manufacturer.contains("qualcomm") ||
            board.contains("qcom") ||
            BOARD_NAMES.keys.any { board.contains(it) } ||
            socModel.startsWith("SM", ignoreCase = true) ||
            socId != null && SOC_ID.containsKey(socId)

        val upper = socModel.uppercase()
        val hasHtp = HTP_SOC.any { upper.startsWith(it) } ||
            HTP_BOARDS.any { board.contains(it) } ||
            (socId != null && SOC_ID[socId]?.let { HTP_SOC.any { s -> it == s } } == true)

        val hasDsp66 = DSP66_SOC.any { upper.startsWith(it) } ||
            DSP66_BOARDS.any { board.contains(it) } ||
            (socId != null && SOC_ID[socId] == "SM8250")

        val chipset = SOC_NAMES.entries.sortedByDescending { it.key.length }.firstOrNull { upper.startsWith(it.key) }?.value
            ?: BOARD_NAMES.entries.firstOrNull { board.contains(it.key) }?.value
            ?: (socId?.let { SOC_ID[it] }?.let { SOC_NAMES[it] })
            ?: ""

        return DeviceInfo(
            socModel = socModel,
            hardware = Build.HARDWARE,
            chipset = chipset,
            isQualcomm = isQualcomm,
            hasHtp = hasHtp,
            hasDspV66 = hasDsp66,
            hasAdrenoGpu = isQualcomm
        )
    }

    /** Map a saved preference string to an engine. */
    fun preferenceFromString(value: String?): Preference {
        return when (value?.lowercase()) {
            "gpu" -> Preference.GPU
            "npu" -> Preference.NPU
            "cpu" -> Preference.CPU
            else -> Preference.AUTO
        }
    }

    /**
     * Low-memory protection: building the QNN delegate compiles the whole
     * model graph, whose peak can reach GBs. On low-RAM devices that freezes
     * or kills the whole system, so QNN is disabled there (CPU only).
     */
    fun isLowRamDevice(context: Context): Boolean {
        return try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            if (am.isLowRamDevice) return true
            val mi = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            mi.totalMem < 6L * 1024 * 1024 * 1024 || mi.lowMemory
        } catch (_: Throwable) {
            false
        }
    }

    /**
     * Return the ordered list of backends to attempt for this preference.
     * Each element is a [BackendAttempt] describing a QNN backend to try.
     * Returns an empty list on non-Qualcomm platforms (no QNN runtime).
     *
     * Ordering notes:
     *  - Snapdragon 865/870 have no HTP; their "NPU" is the Hexagon DSP v66.
     *    On a locked retail phone the DSP will almost always refuse our unsigned
     *    skel and the fastRPC open can HANG for a long time, so in AUTO mode we
     *    go straight to the reliable Adreno GPU instead.
     *  - Snapdragon 888+ try HTP first, then GPU.
     */
    fun planBackends(pref: Preference, info: DeviceInfo, lowRam: Boolean = false): List<BackendAttempt> {
        if (!info.isQualcomm || lowRam) return emptyList()
        val ordered = ArrayList<BackendAttempt>(4)
        when (pref) {
            Preference.CPU -> {} // no QNN
            Preference.GPU -> ordered.add(BackendAttempt.GPU)
            Preference.NPU -> {
                if (info.hasHtp) ordered.add(BackendAttempt.HTP)
                else if (info.hasDspV66) ordered.add(BackendAttempt.DSP)
                else ordered.add(BackendAttempt.GPU)
            }
            Preference.AUTO -> {
                // NPU (HTP on 888+, DSP v66 on 865/870), then the reliable GPU.
                if (info.hasHtp) ordered.add(BackendAttempt.HTP)
                else if (info.hasDspV66) ordered.add(BackendAttempt.DSP)
                ordered.add(BackendAttempt.GPU)
            }
        }
        return ordered
    }

    /** Remember which runtime groups were already loaded (loading is irreversible). */
    private val loadedLibraries = HashSet<String>()

    /**
     * Preload ONLY the runtime libraries needed by the given backend, in
     * dependency order. Loading everything up front pins ~75 MB of native libs
     * (incl. the 50 MB libQnnHtpPrepare) even when NPU is never used, which
     * can push low-RAM devices over the edge. Once loaded, groups stay mapped
     * (dlopen by name still resolves them for the delegate).
     * Returns a list of load failures (missing/incompatible lib).
     */
    @Synchronized fun preloadRuntimes(attempt: BackendAttempt): List<String> {
        val order = when (attempt) {
            BackendAttempt.GPU -> listOf("QnnSystem", "QnnIr", "QnnGpu", "QnnCpu")
            BackendAttempt.DSP -> listOf("QnnSystem", "QnnDsp", "QnnDspV66Stub")
            // The backend dlopens the appropriate extracted Vxx stub itself.
            // Do not map every architecture or the GPU on an NPU attempt.
            BackendAttempt.HTP -> listOf("QnnSystem", "QnnHtp")
        }
        val failures = ArrayList<String>()
        for (name in order) {
            if (name in loadedLibraries) continue
            try {
                System.loadLibrary(name)
                loadedLibraries.add(name) // Never mark a failed load as successful.
            } catch (t: Throwable) {
                failures.add("$name: ${t.message ?: t.javaClass.simpleName}")
            }
        }
        return failures
    }

    /**
     * Attempt to build a QnnDelegate for the given backend.
     * The native construction runs under [ATTEMPT_TIMEOUT_MS] so a hanging
     * backend (e.g. blocked fastRPC/DSP connect) cannot stall the app forever.
     */
/** Preview GPU accuracy/performance trade-off. */
    enum class GpuQuality(val precision: QnnDelegate.Options.GpuPrecision,
                          val perf: QnnDelegate.Options.GpuPerformanceMode,
                          val label: String) {
        FAST(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16,
            QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH,
            "GPU快速(fp16)"),
        BALANCED(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_HYBRID,
            QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_NORMAL,
            "GPU均衡(默认)"),
        PRECISE(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP32,
            QnnDelegate.Options.GpuPerformanceMode.GPU_PERFORMANCE_HIGH,
            "GPU高精度(fp32)");

        companion object {
            fun fromPref(v: String?): GpuQuality = when (v) {
                "fast" -> FAST
                "precise" -> PRECISE
                else -> BALANCED
            }
        }
    }

    fun createQnnDelegate(context: Context, attempt: BackendAttempt,
                          modelToken: String = "da3",
                          skipOps: IntArray? = null,
                          gpuQuality: GpuQuality = GpuQuality.BALANCED,
                          quantized: Boolean = false): DelegateResult {
        // This method runs on NativeRunner, the SAME thread as graph build/run.
        val skelDir = if (attempt == BackendAttempt.GPU) null else ensureSkels(context).absolutePath
        if (attempt == BackendAttempt.HTP) {
            val capability = if (quantized) QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED else QnnDelegate.Capability.HTP_RUNTIME_FP16
            if (!QnnDelegate.checkCapability(capability)) {
                return DelegateResult(null, attempt.displayName, "HTP ${if (quantized) "量化" else "FP16"} 运行时能力不可用")
            }
            // Only map the 50MB graph preparer after the requested capability passes.
            System.loadLibrary("QnnHtpPrepare")
        }

        val opts = QnnDelegate.Options().apply {
            try {
                // Backend libs are preloaded via preloadRuntimes() (RTLD_GLOBAL)
                // AND extracted to real files on disk (legacy packaging), so the
                // delegate's dlopen-by-name resolves them reliably. This matches
                // the official Java example, which sets neither library_path
                // nor any path tricks.
                try {
                    setLogLevel(
                        if (verboseLogging) QnnDelegate.Options.LogLevel.LOG_LEVEL_INFO
                        else QnnDelegate.Options.LogLevel.LOG_LEVEL_ERROR
                    )
                } catch (_: Throwable) {}
                // Skip listed TFLite builtin op codes: those ops then run on the
                // CPU while the rest of the graph still goes to the QNN backend.
                if (skipOps != null && skipOps.isNotEmpty()) {
                    try {
                        setSkipOps(skipOps)
                    } catch (t: Throwable) {
                        Log.w(TAG, "setSkipOps failed: ${t.message}")
                    }
                }
                // Cache the prepared/converted graph so later app starts (and
                // pre-built context binaries dropped into the cache dir) are fast.
                val cacheDir = File(context.filesDir, "qnn_cache").apply { mkdirs() }
                if (cacheDir.exists()) {
                    setCacheDir(cacheDir.absolutePath)
                    setModelToken(modelToken)
                }
                if (attempt != BackendAttempt.GPU && skelDir != null) {
                    setSkelLibraryDir(skelDir)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "option setup failed: ${t.message}")
            }
            when (attempt) {
                BackendAttempt.GPU -> {
                    setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND)
                    try {
                        setGpuOptions(gpuQuality.perf, gpuQuality.precision)
                    } catch (t: Throwable) {
                        Log.w(TAG, "gpu option failed: ${t.message}")
                    }
                }
                BackendAttempt.HTP -> {
                    setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                    try {
                        setHtpOptions(
                            QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST,
                            if (quantized) QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED
                            else QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16,
                            QnnDelegate.Options.HtpPdSession.HTP_PD_SESSION_UNSIGNED,
                            QnnDelegate.Options.HtpOptimizationStrategy.HTP_OPTIMIZE_FOR_INFERENCE
                        )
                        // SDK defaults to MANUAL, but this app does not issue
                        // manual performanceVote/performanceRelease calls.
                        setHtpPerfCtrlStrategy(QnnDelegate.Options.HtpPerfCtrlStrategy.HTP_PERF_CTRL_AUTO)
                    } catch (t: Throwable) {
                        Log.w(TAG, "htp option failed: ${t.message}")
                    }
                }
                BackendAttempt.DSP -> {
                    setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND)
                    try {
                        setDspOptions(
                            QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_HIGH_PERFORMANCE,
                            QnnDelegate.Options.DspPdSession.DSP_PD_SESSION_UNSIGNED
                        )
                        setDspPerfCtrlStrategy(QnnDelegate.Options.DspPerfCtrlStrategy.DSP_PERF_CTRL_AUTO)
                    } catch (t: Throwable) {
                        Log.w(TAG, "dsp option failed: ${t.message}")
                    }
                }
            }
        }

        return try {
            val delegate = QnnDelegate(opts)
            if (!delegate.isAvailable) {
                delegate.close()
                DelegateResult(null, attempt.displayName, "平台能力不足")
            } else {
                DelegateResult(delegate, attempt.displayName, "OK")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "QnnDelegate(${attempt}) failed: ${t.message}")
            DelegateResult(null, attempt.displayName, t.message ?: t.javaClass.simpleName)
        }
    }

    /**
     * Copy the DSP/HTP skel libraries from assets/qnn/skels into a private
     * directory. Returns that directory. The QNN stub uses it (ADSP_LIBRARY_PATH)
     * to upload the skel to the DSP/HTP over fastRPC.
     */
    @Synchronized fun ensureSkels(context: Context): File {
        val dir = File(context.filesDir, "qnn_skels")
        try {
            val assets = context.assets.list("qnn/skels") ?: emptyArray()
            if (assets.isNotEmpty()) {
                dir.mkdirs()
                val marker = File(dir, ".install")
                val install = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
                val refresh = !marker.isFile || marker.readText() != install
                for (name in assets) {
                    val out = File(dir, name)
                    context.assets.open("qnn/skels/$name").use { input ->
                        // Recover partial/zero-length files from interrupted copies.
                        val expected = input.available().toLong()
                        if (refresh || !out.isFile || out.length() != expected) {
                            val temp = File(dir, "$name.part")
                            temp.outputStream().use { output -> input.copyTo(output) }
                            check(temp.renameTo(out)) { "无法更新 $name" }
                        }
                    }
                }
                marker.writeText(install)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "skel copy failed: ${t.message}")
        }
        return dir
    }

    private fun readSocModel(): String {
        // Build.SOC_MODEL only exists on Android 12 (API 31)+; accessing it on
        // older versions throws NoSuchFieldError, so it must be guarded.
        if (Build.VERSION.SDK_INT >= 31) {
            try {
                val direct = Build.SOC_MODEL
                if (!direct.isNullOrBlank()) return direct.trim()
            } catch (_: Throwable) {
            }
        }
        // fallback: /sys/devices/soc0/soc_id (available on older devices too)
        return runCatching {
            File("/sys/devices/soc0/soc_id").readText().trim().let {
                SOC_ID[it.toLong()]
            }
        }.getOrNull() ?: ""
    }

    private fun readSocId(): Long? = runCatching {
        File("/sys/devices/soc0/soc_id").readText().trim().toLong()
    }.getOrNull()

    enum class BackendAttempt(val displayName: String) {
        HTP("QNN NPU (HTP)"),
        DSP("QNN NPU (DSP)"),
        GPU("QNN GPU (Adreno)");
    }
}
