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
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.starfall.gaussianscanner.camera.CameraCapture
import com.starfall.gaussianscanner.databinding.ActivityMainBinding
import com.starfall.gaussianscanner.depth.DepthEstimator
import com.starfall.gaussianscanner.rendering.PointCloudBuilder
import com.starfall.gaussianscanner.rendering.PointCloudRenderer
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var b: ActivityMainBinding
    private lateinit var r: PointCloudRenderer
    private lateinit var da3: DepthEstimator
    private var cam: CameraCapture? = null
    private var view3D = false
    private var processing = false
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
    private var currentZoom = 100
    private val zoomLevels = floatArrayOf(0.3f, 0.6f, 1f, 1.5f, 2.2f, 3f, 5f, 8f, 12f, 18f)

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { processGalleryImage(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        r = PointCloudRenderer()
        da3 = DepthEstimator(this)
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)
        setupGL()
        b.btnCapture.setOnClickListener { cam?.capture() }
        b.btnRetake.setOnClickListener { exit3D() }
        b.btnGallery.setOnClickListener { pickFromGallery() }
        b.btnSettings.setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
        b.btnAxisLock.setOnClickListener { cycleAxisLock() }
        setupSeekBar()
        setupExposureSeekBar()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCam() else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(rc, p, g)
        if (rc == 100 && g.all { it == PackageManager.PERMISSION_GRANTED }) startCam()
    }

    private fun startCam() {
        b.tvStatus.text = if (da3.ready) "点击拍摄" else "正在加载DA3模型…"
        cam = CameraCapture(this, this, b.previewView) { bmp, focal ->
            stopCam()
            processBitmap(bmp, focal)
        }
        cam?.start()
        setupTapToFocus()
    }

    private fun setupTapToFocus() {
        b.previewView.setOnClickListener {
            if (view3D || processing) return@setOnClickListener
            val x = it.x
            val y = it.y
            val w = it.width
            val h = it.height
            if (w == 0 || h == 0) return@setOnClickListener

            val ok = cam?.focusAt(x / w, y / h) ?: false
            if (ok) {
                it.postDelayed({
                    b.tvStatus.text = "已对焦"
                    b.sbExposure.translationX = 80f * resources.displayMetrics.density
                    b.sbExposure.visibility = View.VISIBLE
                }, 50)
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
        cam?.stop()
        cam = null
    }

    private fun setupSeekBar() {
        b.sbPtSize.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                r.ptSize = 2.1f + p * 0.1f
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

    private fun processGalleryImage(uri: Uri) {
        processing = true
        stopCam()
        h.post {
            b.previewView.visibility = View.GONE
            b.tvStatus.text = "正在思考…"
        }
        exec.execute {
            try {
                val bmp = decodeSampledBitmap(uri, 1600)
                if (bmp != null) processBitmap(bmp, 0f)
                else h.post {
                    processing = false
                    b.previewView.visibility = View.VISIBLE
                    b.tvStatus.text = "图片加载失败"
                    startCam()
                }
            } catch (_: Exception) {
                h.post {
                    processing = false
                    b.previewView.visibility = View.VISIBLE
                    b.tvStatus.text = "图片加载失败"
                    startCam()
                }
            }
        }
    }

    private fun decodeSampledBitmap(uri: Uri, maxDim: Int): android.graphics.Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

            var sample = 1
            while (bounds.outWidth / (sample * 2) >= maxDim || bounds.outHeight / (sample * 2) >= maxDim) {
                sample *= 2
            }
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        } catch (_: Exception) {
            null
        }
    }

    private fun processBitmap(bmp: android.graphics.Bitmap, focalPx: Float = 0f) {
        h.post {
            b.previewView.visibility = View.GONE
            b.tvStatus.text = "正在思考…"
        }
        exec.execute {
            try {
                val debug = SettingsActivity.getDebugMode(prefs)
                val t0 = System.currentTimeMillis()
                val depth = da3.estimate(bmp)
                var cloud = emptyList<com.starfall.gaussianscanner.model.ColoredPt>()
                val maxPts = SettingsActivity.getMaxPoints(prefs)
                val targetW = SettingsActivity.computeCustomWidth(prefs, bmp.width, bmp.height)
                if (depth != null) {
                    val (edgeOn, edgeStrength) = SettingsActivity.getEdgeTrim(prefs)
                    cloud = PointCloudBuilder.build(bmp, depth, targetW, maxPts,
                        edgeTrim = if (edgeOn) edgeStrength else 0f,
                        focalPx = if (SettingsActivity.getUseFocal(prefs)) focalPx else 0f,
                        interp = SettingsActivity.getDepthInterp(prefs))
                }
                val elapsed = System.currentTimeMillis() - t0
                if (debug) {
                    val log = buildString {
                        appendLine("=== 深度推理日志 ===")
                        appendLine("时间: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())}")
                        appendLine("设备: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                        appendLine("Android: ${android.os.Build.VERSION.RELEASE} (API ${android.os.Build.VERSION.SDK_INT})")
                        appendLine("内存: 总 ${Runtime.getRuntime().totalMemory()/(1024*1024)}MB / 最大 ${Runtime.getRuntime().maxMemory()/(1024*1024)}MB")
                        appendLine("推理分辨率: ${com.starfall.gaussianscanner.depth.DepthEstimator.W} × ${com.starfall.gaussianscanner.depth.DepthEstimator.H}")
                        appendLine("真实焦距(px): ${if (focalPx > 0f) "%.0f".format(focalPx) else "未获取(用估算)"}")
                        appendLine("输入图片: ${bmp.width} × ${bmp.height}")
                        appendLine("推理耗时: ${elapsed}ms")
                        appendLine("深度图: ${depth?.w} × ${depth?.h}  结果=${if (depth != null) "成功" else "失败"}")
                        appendLine("点云: ${cloud.size} 点 / 目标 ${maxPts}")
                        appendLine("边缘修剪: ${SettingsActivity.getEdgeTrim(prefs).first} (${SettingsActivity.getEdgeTrim(prefs).second})")
                        appendLine("点云毛边: ${prefs.getBoolean("fuzzy", false)}")
                        appendLine("质量预设: ${prefs.getString("quality", "medium")}")
                    }
                    writeDebugLog(log)
                }
                bmp.recycle()
                System.gc()
                h.post {
                    processing = false
                    if (cloud.isNotEmpty()) { r.setCloud(cloud); enter3D() }
                    else {
                        b.previewView.visibility = View.VISIBLE
                        b.tvStatus.text = "深度估计失败，请重试"
                        startCam()
                    }
                }
            } catch (e: Throwable) {
                android.util.Log.e("MainActivity", "processBitmap failed", e)
                h.post {
                    processing = false
                    b.previewView.visibility = View.VISIBLE
                    b.tvStatus.text = "处理失败: ${e.message ?: e.javaClass.simpleName}"
                    startCam()
                }
            }
        }
    }

    private fun writeDebugLog(content: String) {
        try {
            val dir: java.io.File = if (android.os.Build.VERSION.SDK_INT >= 29) {
                java.io.File(getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS), "GaussianScanner")
            } else {
                java.io.File(
                    android.os.Environment.getExternalStoragePublicDirectory(
                        android.os.Environment.DIRECTORY_DOCUMENTS
                    ), "GaussianScanner"
                )
            }
            if (!dir.exists()) dir.mkdirs()
            val file = java.io.File(dir, "debug_${System.currentTimeMillis()}.log")
            file.writeText(content)
        } catch (_: Exception) {}
    }

    private fun setupGL() {
        b.glSurfaceView.setEGLContextClientVersion(3)
        b.glSurfaceView.setRenderer(r)
        b.glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        b.glSurfaceView.setOnTouchListener { _, e ->
            if (!view3D) return@setOnTouchListener false
            when (e.actionMasked) {
                MotionEvent.ACTION_SCROLL -> {
                    val delta = e.getAxisValue(MotionEvent.AXIS_VSCROLL)
                    r.zoom -= delta * 0.25f
                    r.zoom = r.zoom.coerceIn(zoomLevels.first(), zoomLevels.last())
                    updateZoomLabel()
                    b.glSurfaceView.requestRender()
                    true
                }
                MotionEvent.ACTION_DOWN -> {
                    pd = 0f
                    val now = System.currentTimeMillis()
                    if (now - lastTap < 300 && Math.abs(e.x - tapX) < 50 && Math.abs(e.y - tapY) < 50) {
                        val idx = zoomLevels.indexOfFirst { it > r.zoom }
                        currentZoom = if (idx >= 0 && idx < zoomLevels.size - 1) idx + 1 else zoomLevels.size - 2
                        r.zoom = zoomLevels[currentZoom]
                        updateZoomLabel()
                        b.glSurfaceView.requestRender()
                        lastTap = 0L
                    } else { lastTap = now; tapX = e.x; tapY = e.y }
                    true
                }
                MotionEvent.ACTION_MOVE -> when (e.pointerCount) {
                    2 -> {
                        val d = Math.hypot((e.getX(0)-e.getX(1)).toDouble(),(e.getY(0)-e.getY(1)).toDouble()).toFloat()
                        if(pd>0f){r.zoom-=(d-pd)*0.02f;r.zoom=r.zoom.coerceIn(zoomLevels.first(),zoomLevels.last())}
                        pd=d; updateZoomLabel(); b.glSurfaceView.requestRender(); true
                    }
                    1 -> {
                        val dx=e.x-(if(e.historySize>0)e.getHistoricalX(0,0)else e.x)
                        val dy=e.y-(if(e.historySize>0)e.getHistoricalY(0,0)else e.y)
                        when (axisLock) {
                            1 -> touchRy += dx * 0.5f
                            2 -> touchRx += dy * 0.5f
                            else -> { touchRy += dx * 0.5f; touchRx += dy * 0.5f }
                        }
                        r.rx = touchRx; r.ry = touchRy
                        b.glSurfaceView.requestRender();true
                    }
                    else -> false
                }
                else -> false
            }
        }
    }

    private fun updateZoomLabel() {
        val pct = (r.zoom / 3f * 100f).toInt().coerceIn(10, 500)
        b.tvZoom.text = "$pct%"
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
        b.previewView.visibility = View.GONE
        b.glSurfaceView.visibility = View.VISIBLE
        b.btnCapture.visibility = View.GONE
        b.btnRetake.visibility = View.VISIBLE
        b.btnGallery.visibility = View.GONE
        b.tvZoom.visibility = View.VISIBLE
        b.ptSizePanel.visibility = View.VISIBLE
        b.btnSettings.visibility = View.GONE
        b.btnAxisLock.visibility = View.VISIBLE
        axisLock = 0
        b.btnAxisLock.text = "FREE"
        b.btnAxisLock.setTextColor(0xFFCCCCCC.toInt())
        b.sbExposure.visibility = View.GONE
        touchRx = 0f; touchRy = 0f
        b.tvFps.visibility = View.VISIBLE
        if (message.isNotEmpty()) { b.tvStatus.visibility = View.VISIBLE; b.tvStatus.text = message }
        else b.tvStatus.visibility = View.GONE
        r.zoom = 3f
        currentZoom = 5
        updateZoomLabel()
        r.fuzzy = prefs.getBoolean("fuzzy", false)
        r.fuzzyRadius = SettingsActivity.getFuzzyRadius(prefs)
        rt = object : Runnable {
            override fun run() {
                if(!view3D)return
                r.rx = touchRx; r.ry = touchRy
                b.glSurfaceView.requestRender()
                b.tvFps.text = "${r.fps} FPS"
                h.postDelayed(this, 32)
            }
        }
        h.post(rt!!)
    }

    private fun exit3D() {
        view3D = false
        rt = null
        b.previewView.visibility = View.VISIBLE
        b.glSurfaceView.visibility = View.GONE
        b.btnCapture.visibility = View.VISIBLE
        b.btnRetake.visibility = View.GONE
        b.tvStatus.visibility = View.VISIBLE
        b.tvZoom.visibility = View.GONE
        b.btnGallery.visibility = View.VISIBLE
        b.ptSizePanel.visibility = View.GONE
        b.btnSettings.visibility = View.VISIBLE
        b.btnAxisLock.visibility = View.GONE
        b.tvFps.visibility = View.GONE
        b.sbExposure.visibility = View.GONE
        touchRx = 0f; touchRy = 0f
        r.rx=0f;r.ry=0f;r.zoom=3f
        System.gc()
        if (!camStarted()) startCam()
    }

    private fun camStarted() = cam != null

    override fun onDestroy() {
        super.onDestroy()
        rt = null
        cam?.shutdown()
        da3.close()
        exec.shutdown()
    }

    override fun onPause() {
        super.onPause()
        rt = null
        if (!view3D) stopCam()
    }

    override fun onResume() {
        super.onResume()
        if (view3D) {
            rt = object : Runnable {
                override fun run() {
                    if(!view3D)return
                    r.rx = touchRx; r.ry = touchRy
                    b.glSurfaceView.requestRender()
                    h.postDelayed(this, 32)
                }
            }
            h.post(rt!!)
        } else if (!processing && !camStarted()) startCam()
    }
}
