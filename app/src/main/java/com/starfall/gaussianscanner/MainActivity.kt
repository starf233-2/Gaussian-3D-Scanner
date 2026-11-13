package com.starfall.gaussianscanner

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.View
import android.widget.EditText
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.starfall.gaussianscanner.camera.CameraCapture
import com.starfall.gaussianscanner.databinding.ActivityMainBinding
import com.starfall.gaussianscanner.depth.DepthEstimator
import com.starfall.gaussianscanner.model.PointCloud
import com.starfall.gaussianscanner.rendering.PointCloudBuilder
import com.starfall.gaussianscanner.rendering.CloudSampling
import com.starfall.gaussianscanner.rendering.PointCloudRenderer
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var b: ActivityMainBinding
    private lateinit var r: PointCloudRenderer
    @Volatile private var da3: DepthEstimator? = null
    private var cam: CameraCapture? = null
    private var view3D = false
    private var processing = false
    private var capturing = false
    @Volatile private var destroyed = false
    private var resumed = false
    private var glActive = false
    private var exporting = false
    private var depthResolution: String? = null
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var axisLock = 0 // 0=free, 1=horizontal only, 2=vertical only
    private var touchRx = 0f
    private var touchRy = 0f
    private lateinit var prefs: SharedPreferences
    private val exec = Executors.newSingleThreadExecutor()
    private val h = Handler(Looper.getMainLooper())
    private var rt: Runnable? = null
    private var pd = 0f
    private var lastTap = 0L
    private var tapX = 0f
    private var tapY = 0f
    private var currentZoom = 5
    private val zoomLevels = floatArrayOf(0.3f, 0.6f, 1f, 1.5f, 2.2f, 3f, 5f, 8f, 12f, 18f)

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { processGalleryImage(it) } }

    private val importPlyLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { importPly(it) } }

    /** The point cloud currently shown in the 3D view (for export). */
    private var currentCloud = PointCloud.EMPTY

    private fun postUi(action: () -> Unit) { h.post { if (!destroyed) action() } }

    private fun setBusyUi(busy: Boolean) {
        b.btnCapture.isEnabled = !busy
        b.btnGallery.isEnabled = !busy
        b.btnImport.isEnabled = !busy
        b.btnSettings.isEnabled = !busy
    }

    private fun failOperation(message: String, restartCamera: Boolean = true) {
        processing = false; capturing = false
        setBusyUi(false)
        b.previewView.visibility = View.VISIBLE
        b.tvStatus.visibility = View.VISIBLE
        if (restartCamera) startCam()
        b.tvStatus.text = message
        scheduleIdleRelease()
        if (!resumed) releaseEngine()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        r = PointCloudRenderer { message -> postUi {
            b.tvStatus.visibility = View.VISIBLE
            b.tvStatus.text = message
        } }
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)
        setupGL()
        b.btnCapture.setOnClickListener {
            if (!processing) {
                processing = true; capturing = true; setBusyUi(true)
                if (cam?.capture() != true) failOperation("相机尚未就绪")
            }
        }
        b.btnRetake.setOnClickListener { exit3D() }
        b.btnGallery.setOnClickListener { pickFromGallery() }
        b.btnImport.setOnClickListener { importPlyLauncher.launch("*/*") }
        b.btnExport.setOnClickListener { exportPly() }
        b.btnSettings.setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
        b.btnAxisLock.setOnClickListener { cycleAxisLock() }
        setupSeekBar()
        setupExposureSeekBar()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)

        checkStoragePermission()

        // QNN delegate construction may convert/prepare the whole model graph
        // on first use, which can take a while — never block the UI.
        b.tvStatus.text = "点击拍摄 · 模型按需加载"
    }

    /** The engine preference that the current DepthEstimator was built with. */
    @Volatile private var builtEngine: String? = null
    /** Only the first failure message per engine build pops a Toast. */
    private var failureToasted = false

    /** All settings that change delegate construction belong in the build key. */
    private fun engineKey(): String = listOf(
        SettingsActivity.getEngine(prefs), prefs.getString("gpu_quality", "balanced"),
        prefs.getString("gpu_mode", "auto"), prefs.getBoolean("gpu_compat", false),
        prefs.getBoolean("debug_mode", false), prefs.getBoolean("debug_log", false)
    ).joinToString("|")

    /** Called on the serial processing worker, not the UI. Build only when a
     * photo needs inference; release the old graph before allocating the next. */
    private fun ensureEngine(): DepthEstimator {
        check(!destroyed) { "页面已关闭" }
        val key = engineKey()
        da3?.let { if (it.ready && builtEngine == key) return it }
        releaseEngine()
        val pref = prefs.getString("engine", "auto") ?: "auto"
        failureToasted = false
        val value = DepthEstimator(applicationContext, pref,
            debug = SettingsActivity.getDebugPanelOn(prefs), saveDiag = prefs.getBoolean("debug_log", false),
            onStatus = { message -> postUi {
                if (!view3D) b.tvStatus.text = message
                if (!failureToasted && (message.contains("不可用") || message.contains("失败"))) {
                    failureToasted = true
                    android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_LONG).show()
                }
            } })
        synchronized(this) {
            if (!value.ready || destroyed) {
                value.close()
                error(value.attemptLog.lastOrNull() ?: "模型加载失败")
            }
            da3 = value
            // NPU fallback may have updated the saved preference during construction.
            builtEngine = engineKey()
        }
        return value
    }

    @Synchronized private fun releaseEngine() {
        val old = da3; da3 = null; builtEngine = null
        old?.close()
    }

    private val idleRelease = Runnable { if (!processing) releaseEngine() }
    private fun scheduleIdleRelease() {
        h.removeCallbacks(idleRelease)
        if (!destroyed) h.postDelayed(idleRelease, 60_000)
    }

    private fun checkStoragePermission() {
        if (StorageUtil.canWritePublic(this)) return
        if (android.os.Build.VERSION.SDK_INT >= 30) {
            StorageUtil.requestAllFilesAccess(this)
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                101
            )
        }
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(rc, p, g)
        if (rc == 100 && g.isNotEmpty() && g.all { it == PackageManager.PERMISSION_GRANTED }) startCam()
    }

    private fun startCam() {
        if (!resumed || destroyed || camStarted() || processing || view3D ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val d = da3
        b.tvStatus.text = if (d != null && d.ready)
            "点击拍摄 · ${d.backendLabel}" else "点击拍摄 · 模型按需加载"
        cam = CameraCapture(this, this, b.previewView,
            highResolution = prefs.getBoolean("tiled_inference", false), onError = {
            if (it.startsWith("相机启动失败")) { stopCam(); failOperation(it, restartCamera = false) }
            else failOperation(it)
        }) { bmp, focal ->
            capturing = false
            stopCam()
            processBitmap(bmp, focal)
        }
        cam?.start()
        setupTapToFocus()
    }

    private fun setupTapToFocus() {
        var focusX = 0f; var focusY = 0f
        b.previewView.setOnTouchListener { view, event ->
            if (event.actionMasked == MotionEvent.ACTION_UP) {
                focusX = event.x; focusY = event.y; view.performClick()
            }
            true
        }
        b.previewView.setOnClickListener {
            if (view3D || processing) return@setOnClickListener
            val ok = cam?.focusAt(focusX, focusY) ?: false
            if (ok) {
                b.tvStatus.text = "已请求对焦"
                b.sbExposure.translationX = 80f * resources.displayMetrics.density
                b.sbExposure.visibility = View.VISIBLE
            }
        }
    }

    private fun setupExposureSeekBar() {
        b.sbExposure.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                val ev = (p - 240) / 10f  // -24 .. +24
                cam?.setExposure(ev)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
    }

    private fun stopCam() {
        // shutdown() (not just stop()): a CameraCapture instance is never
        // reused — startCam() builds a fresh one — so its single-thread
        // executor must exit, otherwise each capture leaks one live thread.
        cam?.shutdown()
        cam = null
    }

    private fun setupSeekBar() {
        b.sbPtSize.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                r.ptSize = 1.0f + p * 0.1f
                b.tvPtSize.text = "%.1f".format(r.ptSize)
                if (view3D) b.glSurfaceView.requestRender()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
    }

    private fun pickFromGallery() {
        galleryLauncher.launch("image/*")
    }

    /** Export the current point cloud to a PLY file in the output folder. */
    private fun exportPly() {
        if (exporting) return
        if (currentCloud.size == 0) {
            android.widget.Toast.makeText(this, "当前没有可导出的模型", android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        exporting = true
        b.btnExport.isEnabled = false
        val snapshot = currentCloud
        exec.execute {
            try {
                val dir = StorageUtil.documentsDir(this)
                if (!dir.exists()) dir.mkdirs()
                val f = java.io.File(dir, "scan_${System.currentTimeMillis()}.ply")
                val ok = com.starfall.gaussianscanner.model.PlyIO.export(snapshot, f)
                if (!ok) f.delete() // This operation's incomplete output only.
                postUi { android.widget.Toast.makeText(this, if (ok) "已导出: ${f.absolutePath}" else "导出失败", android.widget.Toast.LENGTH_LONG).show() }
            } catch (e: Throwable) {
                postUi { android.widget.Toast.makeText(this, "导出失败: ${e.message}", android.widget.Toast.LENGTH_LONG).show() }
            } finally { postUi { exporting = false; b.btnExport.isEnabled = true } }
        }
    }

    /** Import a PLY file and show it in the 3D view (parsed off the main thread). */
    private fun importPly(uri: Uri) {
        if (processing || destroyed) return
        processing = true
        setBusyUi(true)
        stopCam()
        b.previewView.visibility = View.GONE
        b.tvStatus.visibility = View.VISIBLE
        b.tvStatus.text = "正在导入PLY…"
        exec.execute {
            try {
                val pts = contentResolver.openInputStream(uri)?.use {
                    com.starfall.gaussianscanner.model.PlyIO.parse(it)
                } ?: PointCloud.EMPTY
                if (pts.size == 0) {
                    postUi { failOperation("PLY中没有有效点") }
                    return@execute
                }
                postUi {
                    processing = false
                    setBusyUi(false)
                    depthResolution = null
                    currentCloud = pts
                    r.setCloud(pts)
                    enter3D("已导入 ${pts.size} 点（最多保留150万点）")
                }
            } catch (e: Throwable) {
                postUi { failOperation("导入失败: ${e.message}") }
            }
        }
    }

    private fun processGalleryImage(uri: Uri) {
        if (processing || destroyed) return
        if (SettingsActivity.getManualFocal(prefs)) {
            showFocalInputDialog { proceedGallery(uri) }
        } else {
            proceedGallery(uri)
        }
    }

    /** Prompt for focal length (mm, 35mm-equivalent); blank -> 45 default. */
    private fun showFocalInputDialog(onDone: () -> Unit) {
        val input = EditText(this).apply {
            hint = "焦距(mm)，默认45"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(String.format(java.util.Locale.US, "%.0f", SettingsActivity.getManualFocalMm(prefs)))
            setSelectAllOnFocus(true)
            setPadding(48, 24, 48, 24)
        }
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("输入焦距")
            .setMessage("导入图片的等效35mm焦距（单位mm），不填则默认 45mm")
            .setView(input)
            .setPositiveButton("确定") { _, _ ->
                val t = input.text?.toString()?.trim()
                val mm = t?.toFloatOrNull()?.takeIf { it.isFinite() } ?: 45f
                prefs.edit().putFloat("manual_focal_mm", mm.coerceIn(1f, 200f)).apply()
                onDone()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun proceedGallery(uri: Uri) {
        if (processing || destroyed) return
        processing = true
        setBusyUi(true)
        stopCam()
        postUi {
            b.previewView.visibility = View.GONE
            b.tvStatus.text = "正在思考…"
        }
        exec.execute {
            try {
                val bmp = decodeSampledBitmap(uri, if (prefs.getBoolean("tiled_inference", false)) 3072 else 2048)
                if (bmp != null) processBitmapWorker(bmp, 0f)
                else postUi { failOperation("图片加载失败") }
            } catch (e: Throwable) {
                postUi { failOperation("图片加载失败: ${e.message}") }
            }
        }
    }

    private fun decodeSampledBitmap(uri: Uri, maxDim: Int): android.graphics.Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / sample > maxDim) {
                sample *= 2
            }
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            val decoded = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
            try {
                val orientation = contentResolver.openInputStream(uri)?.use {
                    androidx.exifinterface.media.ExifInterface(it).getAttributeInt(
                        androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                        androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL)
                } ?: 1
                val matrix = android.graphics.Matrix()
                when (orientation) {
                    2 -> matrix.setScale(-1f, 1f)
                    3 -> matrix.setRotate(180f)
                    4 -> matrix.setScale(1f, -1f)
                    5 -> { matrix.setRotate(90f); matrix.postScale(-1f, 1f) }
                    6 -> matrix.setRotate(90f)
                    7 -> { matrix.setRotate(270f); matrix.postScale(-1f, 1f) }
                    8 -> matrix.setRotate(270f)
                }
                if (matrix.isIdentity) decoded else android.graphics.Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
                    .also { if (it !== decoded) decoded.recycle() }
            } catch (_: Exception) { decoded }
        } catch (_: Exception) {
            null
        }
    }

    private fun processBitmap(bmp: android.graphics.Bitmap, focalPx: Float = 0f) {
        if (destroyed) {
            bmp.recycle()
            return
        }
        processing = true
        setBusyUi(true)
        postUi {
            b.previewView.visibility = View.GONE
            b.tvStatus.visibility = View.VISIBLE
            b.tvStatus.text = "正在思考…"
        }
        exec.execute { processBitmapWorker(bmp, focalPx) }
    }

    private fun processBitmapWorker(bmp: android.graphics.Bitmap, focalPx: Float) {
        var nativeOwnsBitmap = false
        try {
            val d = ensureEngine()
            val debug = SettingsActivity.getDebugMode(prefs)
            val t0 = android.os.SystemClock.elapsedRealtime()
            val depth = try { d.estimate(bmp) } catch (e: Exception) {
                if (e is java.util.concurrent.TimeoutException || e is InterruptedException) nativeOwnsBitmap = true
                throw e
            }
            var cloud = PointCloud.EMPTY
            val maxPts = SettingsActivity.getMaxPoints(prefs)
            var targetW = SettingsActivity.computeCustomWidth(prefs, bmp.width, bmp.height)
            if (depth != null) {
                if (depth.w > d.inputW || depth.h > d.inputH) {
                    targetW = CloudSampling.highResolutionWidth(bmp.width, bmp.height, targetW, depth.w, depth.h)
                }
                val (edgeOn, edgeStrength) = SettingsActivity.getEdgeTrim(prefs)
                val manualFoc = SettingsActivity.manualFocalPx(prefs, bmp.width)
                val effectiveFocal = when {
                    manualFoc > 0f -> manualFoc
                    SettingsActivity.getUseFocal(prefs) -> focalPx
                    else -> 0f
                }
                cloud = PointCloudBuilder.build(bmp, depth, targetW,
                    edgeTrim = if (edgeOn) edgeStrength else 0f,
                    focalPx = effectiveFocal,
                    interp = SettingsActivity.getDepthInterp(prefs),
                    colorEnhance = prefs.getBoolean("color_enhance", true),
                    lighting = prefs.getBoolean("lighting", true))
                if (SettingsActivity.getDepthRay(prefs)) saveDepthRayMaps(depth)
            }
            val elapsed = android.os.SystemClock.elapsedRealtime() - t0
            logInferenceTime(elapsed, d.backendLabel)
            if (debug) {
                val log = buildString {
                    appendLine("=== 深度推理日志 ===")
                    appendLine("时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())}")
                    appendLine("设备: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                    appendLine("Android: ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT})")
                    appendLine("SoC: ${d.deviceInfo.socModel} / ${d.deviceInfo.hardware}")
                    appendLine("平台: ${d.deviceInfo.acceleratorSummary}")
                    appendLine("推理引擎: ${d.backendLabel}")
                    appendLine("引擎偏好: ${d.enginePreference}")
                    d.attemptLog.forEach { appendLine("后端尝试: $it") }
                    appendLine("内存: 总 ${Runtime.getRuntime().totalMemory()/(1024*1024)}MB / 最大 ${Runtime.getRuntime().maxMemory()/(1024*1024)}MB")
                    appendLine("推理分辨率: ${d.inputW} × ${d.inputH} ${if (d.inputNhwc) "NHWC" else "NCHW"}")
                    appendLine("真实焦距(px): ${if (focalPx > 0f) "%.0f".format(focalPx) else "未获取(用估算)"}")
                    appendLine("手动焦距: ${if (SettingsActivity.getManualFocal(prefs)) "${SettingsActivity.getManualFocalMm(prefs)}mm" else "关"}")
                    appendLine("输入图片: ${bmp.width} × ${bmp.height}")
                    appendLine("推理耗时: ${elapsed}ms")
                    appendLine("深度图: ${depth?.w} × ${depth?.h}  结果=${if (depth != null) "成功" else "失败"}")
                    appendLine("点云: ${cloud.size} 点 / 目标 $maxPts")
                    appendLine("边缘修剪: ${SettingsActivity.getEdgeTrim(prefs).first} (${SettingsActivity.getEdgeTrim(prefs).second})")
                    appendLine("点云毛边: ${prefs.getBoolean("fuzzy", false)}")
                    appendLine("深度平滑: ${prefs.getBoolean("depth_smooth", true)} · 颜色增强: ${prefs.getBoolean("color_enhance", true)} · 光照: ${prefs.getBoolean("lighting", true)}")
                    appendLine("推理结果: ${d.inferenceReport}")
                    appendLine("质量预设: ${prefs.getString("quality", "medium")}")
                }
                writeDebugLog(log)
            }
            postUi {
                processing = false
                setBusyUi(false)
                if (cloud.size > 0) {
                    depthResolution = depth?.let { "${it.w}×${it.h}" }
                    currentCloud = cloud; r.setCloud(cloud)
                    enter3D(if (prefs.getBoolean("tiled_inference", false)) d.inferenceReport else "")
                } else failOperation("深度估计失败，请重试")
            }
        } catch (e: Throwable) {
            android.util.Log.e("MainActivity", "processBitmap failed", e)
            postUi { failOperation("处理失败: ${e.message ?: e.javaClass.simpleName}") }
        } finally {
            // An abandoned JNI task owns/recycles this bitmap after it exits.
            if (!nativeOwnsBitmap && !bmp.isRecycled) bmp.recycle()
        }
    }

    /** Append one inference-time line to the app-private log (never hangs). */
    private fun logInferenceTime(elapsed: Long, backend: String) {
        try {
            val dir = java.io.File(getExternalFilesDir(null) ?: filesDir, "GaussianScanner")
            if (dir.exists() || dir.mkdirs()) {
                val f = java.io.File(dir, "inference_times.log")
                val line = "${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())} 后端=$backend 推理耗时=${elapsed}ms\n"
                f.appendText(line)
            }
        } catch (_: Exception) {}
    }

    private fun writeDebugLog(content: String) {
        // Write to the app-external dir first (always writable, never hangs),
        // and to public Documents only when all-files access is granted.
        val candidates = ArrayList<java.io.File>()
        try {
            val appDir = java.io.File(getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS) ?: filesDir, "GaussianScanner")
            candidates.add(appDir)
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                if (android.os.Environment.isExternalStorageManager()) {
                    candidates.add(java.io.File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS), "GaussianScanner"))
                }
            } else {
                candidates.add(java.io.File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS), "GaussianScanner"))
            }
        } catch (_: Exception) {}
        for (dir in candidates.distinct()) {
            try {
                if (dir.exists() || dir.mkdirs()) {
                    val file = java.io.File(dir, "debug_${System.currentTimeMillis()}.log")
                    file.writeText(content)
                }
            } catch (_: Exception) {}
        }
    }

    private fun saveDepthRayMaps(depth: com.starfall.gaussianscanner.model.DepthMap) {
        try {
            val dir = StorageUtil.picturesDir(this)

            val w = depth.w
            val h = depth.h
            val ts = System.currentTimeMillis()

            // Reuse one bitmap/pixel array for both images.
            val depthBmp = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888)
            try {
            val depthPx = IntArray(w * h)
            for (i in depth.data.indices) {
                val v = (depth.data[i] * 255f).toInt().coerceIn(0, 255)
                depthPx[i] = 0xFF000000.toInt() or (v shl 16) or (v shl 8) or v
            }
            depthBmp.setPixels(depthPx, 0, w, 0, 0, w, h)
            java.io.FileOutputStream(java.io.File(dir, "depth_$ts.png")).use { out ->
                depthBmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            }
            for (i in depth.data.indices) {
                val t = depth.data[i].coerceIn(0f, 1f)
                depthPx[i] = turboColor(t)
            }
            depthBmp.setPixels(depthPx, 0, w, 0, 0, w, h)
            java.io.FileOutputStream(java.io.File(dir, "ray_$ts.png")).use { out ->
                depthBmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            }
            } finally { depthBmp.recycle() }
        } catch (_: Exception) {}
    }

    /** approximate turbo colormap, returns [r,g,b] in 0..255. */
    private fun turboColor(t: Float): Int {
        val r = (34.61f + t * (1172.33f - t * (10793.56f - t * (33300.12f - t * (38394.49f - t * 14825.05f)))))
        val g = (23.31f + t * (557.33f + t * (1225.33f - t * (3574.96f - t * (1073.77f + t * 707.56f)))))
        val b = (27.2f + t * (3211.1f - t * (15327.97f - t * (27814.0f - t * (22569.18f - t * 6838.66f)))))
        return 0xFF000000.toInt() or (r.coerceIn(0f, 255f).toInt() shl 16) or
            (g.coerceIn(0f, 255f).toInt() shl 8) or b.coerceIn(0f, 255f).toInt()
    }

    private fun setupGL() {
        b.glSurfaceView.setEGLContextClientVersion(3)
        b.glSurfaceView.setRenderer(r)
        b.glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        b.glSurfaceView.setOnTouchListener { _, e ->
            if (!view3D) return@setOnTouchListener false
            when (e.actionMasked) {
                MotionEvent.ACTION_SCROLL -> {
                    val delta = e.getAxisValue(MotionEvent.AXIS_VSCROLL)
                    r.zoom -= delta * 0.25f
                    r.zoom = r.zoom.coerceIn(zoomLevels.first(), zoomLevels.last())
                    updateZoomLabel()
                    renderCloud()
                    true
                }
                MotionEvent.ACTION_DOWN -> {
                    pd = 0f
                    lastTouchX = e.x; lastTouchY = e.y
                    val now = android.os.SystemClock.uptimeMillis()
                    if (now - lastTap < 300 && Math.abs(e.x - tapX) < 50 && Math.abs(e.y - tapY) < 50) {
                        val idx = zoomLevels.indexOfFirst { it > r.zoom }
                        currentZoom = if (idx >= 0) idx else 0
                        r.zoom = zoomLevels[currentZoom]
                        updateZoomLabel()
                        renderCloud()
                        lastTap = 0L
                    } else { lastTap = now; tapX = e.x; tapY = e.y }
                    true
                }
                MotionEvent.ACTION_MOVE -> when (e.pointerCount) {
                    2 -> {
                        val d = Math.hypot((e.getX(0)-e.getX(1)).toDouble(),(e.getY(0)-e.getY(1)).toDouble()).toFloat()
                        if(pd>0f){r.zoom-=(d-pd)*0.02f;r.zoom=r.zoom.coerceIn(zoomLevels.first(),zoomLevels.last())}
                        pd=d; lastTouchX=e.x; lastTouchY=e.y; updateZoomLabel(); renderCloud(); true
                    }
                    1 -> {
                        val dx=e.x-lastTouchX
                        val dy=e.y-lastTouchY
                        lastTouchX=e.x; lastTouchY=e.y
                        when (axisLock) {
                            1 -> touchRy += dx * 0.5f
                            2 -> touchRx += dy * 0.5f
                            else -> { touchRy += dx * 0.5f; touchRx += dy * 0.5f }
                        }
                        r.rx = touchRx; r.ry = touchRy
                        renderCloud();true
                    }
                    else -> false
                }
                MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP, MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    pd = 0f
                    val index = if (e.actionMasked == MotionEvent.ACTION_POINTER_UP && e.actionIndex == 0 && e.pointerCount > 1) 1 else 0
                    lastTouchX = e.getX(index); lastTouchY = e.getY(index)
                    true
                }
                else -> false
            }
        }
    }

    private fun updateZoomLabel() {
        val pct = (r.zoom / 3f * 100f).toInt().coerceIn(10, 600)
        b.tvZoom.text = "$pct%"
    }

    private fun renderCloud() {
        r.autoPointSize = SettingsActivity.getAutoPtSize(prefs)
        b.glSurfaceView.requestRender()
    }

    /** Show/hide the current model's point count and depth resolution. */
    private fun updatePtCountDisplay() {
        if (prefs.getBoolean("show_pt_count", false)) {
            b.tvPtCount.text = "点数: ${currentCloud.size}" + (depthResolution?.let { " · 深度 $it" } ?: " · PLY")
            b.tvPtCount.visibility = View.VISIBLE
        } else {
            b.tvPtCount.visibility = View.GONE
        }
    }

    private fun cycleAxisLock() {
        axisLock = (axisLock + 1) % 3
        when (axisLock) {
            0 -> { b.btnAxisLock.text = "FREE"; b.btnAxisLock.setTextColor(0xFFCCCCCC.toInt()) }
            1 -> { b.btnAxisLock.text = "H-LK"; b.btnAxisLock.setTextColor(0xFF00D4AA.toInt()) }
            2 -> { b.btnAxisLock.text = "V-LK"; b.btnAxisLock.setTextColor(0xFF00D4AA.toInt()) }
        }
    }

    private fun enter3D(message: String = "") {
        view3D = true
        if (resumed && !glActive) { b.glSurfaceView.onResume(); glActive = true }
        b.previewView.visibility = View.GONE
        b.glSurfaceView.visibility = View.VISIBLE
        b.btnCapture.visibility = View.GONE
        b.btnRetake.visibility = View.VISIBLE
        b.btnGallery.visibility = View.GONE
        b.btnImport.visibility = View.GONE
        b.btnExport.visibility = View.VISIBLE
        b.tvZoom.visibility = View.VISIBLE
        b.ptSizePanel.visibility = if (SettingsActivity.getAutoPtSize(prefs)) View.GONE else View.VISIBLE
        b.btnSettings.visibility = View.GONE
        b.btnAxisLock.visibility = View.VISIBLE
        axisLock = 0
        b.btnAxisLock.text = "FREE"
        b.btnAxisLock.setTextColor(0xFFCCCCCC.toInt())
        b.sbExposure.visibility = View.GONE
        touchRx = 0f; touchRy = 0f
        b.tvFps.visibility = View.VISIBLE
        b.tvFps.text = "按需渲染"
        updatePtCountDisplay()
        if (message.isNotEmpty()) { b.tvStatus.visibility = View.VISIBLE; b.tvStatus.text = message }
        else b.tvStatus.visibility = View.GONE
        r.zoom = 3f
        currentZoom = 5
        updateZoomLabel()
        r.fuzzy = prefs.getBoolean("fuzzy", false)
        r.fuzzyRadius = SettingsActivity.getFuzzyRadius(prefs)
        r.rx = 0f; r.ry = 0f
        renderCloud()
        startFrameMonitor()
        scheduleIdleRelease()
        if (!resumed) releaseEngine()
    }

    private fun exit3D() {
        view3D = false
        stopFrameMonitor()
        currentCloud = PointCloud.EMPTY
        depthResolution = null
        r.setCloud(PointCloud.EMPTY)
        // Destroy the idle EGL context as well: hiding a SurfaceView alone
        // can keep its old VBO allocated after all Java cloud references clear.
        if (glActive) { b.glSurfaceView.onPause(); glActive = false }
        b.previewView.visibility = View.VISIBLE
        b.glSurfaceView.visibility = View.GONE
        b.btnCapture.visibility = View.VISIBLE
        b.btnRetake.visibility = View.GONE
        b.tvStatus.visibility = View.VISIBLE
        b.tvZoom.visibility = View.GONE
        b.btnGallery.visibility = View.VISIBLE
        b.btnImport.visibility = View.VISIBLE
        b.btnExport.visibility = View.GONE
        b.ptSizePanel.visibility = View.GONE
        b.btnSettings.visibility = View.VISIBLE
        b.btnAxisLock.visibility = View.GONE
        b.tvFps.visibility = View.GONE
        b.tvPtCount.visibility = View.GONE
        b.sbExposure.visibility = View.GONE
        touchRx = 0f; touchRy = 0f
        r.rx=0f;r.ry=0f;r.zoom=3f
        scheduleIdleRelease()
        if (!camStarted()) startCam()
    }

    private fun camStarted() = cam != null

    private fun stopFrameMonitor() {
        rt?.let { h.removeCallbacks(it) }
        rt = null
    }

    private fun startFrameMonitor() {
        stopFrameMonitor()
        if (!resumed || !view3D) return
        var frames = r.frameCount
        var time = android.os.SystemClock.elapsedRealtime()
        rt = object : Runnable {
            override fun run() {
                if (!resumed || !view3D || destroyed) return
                val next = r.frameCount
                val now = android.os.SystemClock.elapsedRealtime()
                b.tvFps.text = "${(next - frames) * 1000 / (now - time).coerceAtLeast(1)} FPS · 按需渲染"
                frames = next
                time = now
                h.postDelayed(this, 500)
            }
        }
        h.postDelayed(rt!!, 500)
    }

    override fun onDestroy() {
        destroyed = true
        stopFrameMonitor()
        h.removeCallbacksAndMessages(null)
        stopCam()
        releaseEngine()
        currentCloud = PointCloud.EMPTY
        r.setCloud(PointCloud.EMPTY)
        exec.shutdownNow()
        super.onDestroy()
    }

    override fun onPause() {
        resumed = false
        stopFrameMonitor()
        if (capturing) { capturing = false; processing = false; setBusyUi(false) }
        stopCam()
        if (glActive) { b.glSurfaceView.onPause(); glActive = false }
        scheduleIdleRelease()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        if (view3D && !glActive) { b.glSurfaceView.onResume(); glActive = true }
        // Invalidate changed delegates, but don't rebuild until a photo needs one.
        if (!processing && da3 != null && builtEngine != engineKey()) releaseEngine()
        if (view3D) {
            // reflect any settings change (e.g. auto point size) from Settings
            b.ptSizePanel.visibility = if (SettingsActivity.getAutoPtSize(prefs)) View.GONE else View.VISIBLE
            updatePtCountDisplay()
            renderCloud()
            startFrameMonitor()
        } else if (!processing && !camStarted()) startCam()
    }

    override fun onStop() {
        super.onStop()
        if (!processing) releaseEngine()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (!processing && level >= TRIM_MEMORY_RUNNING_LOW) releaseEngine()
    }
}
