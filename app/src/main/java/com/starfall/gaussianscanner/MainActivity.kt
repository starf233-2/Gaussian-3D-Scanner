package com.starfall.gaussianscanner

import android.Manifest; import android.content.Intent; import android.content.SharedPreferences
import android.content.pm.PackageManager; import android.graphics.BitmapFactory
import android.net.Uri; import android.opengl.GLSurfaceView; import android.os.*; import android.view.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat; import androidx.core.content.ContextCompat
import com.starfall.gaussianscanner.camera.CameraCapture
import com.starfall.gaussianscanner.databinding.ActivityMainBinding
import com.starfall.gaussianscanner.depth.DepthEstimator
import com.starfall.gaussianscanner.rendering.PointCloudBuilder
import com.starfall.gaussianscanner.rendering.PointCloudRenderer
import android.widget.SeekBar
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var b: ActivityMainBinding
    private lateinit var r: PointCloudRenderer
    private lateinit var da3: DepthEstimator
    private var cam: CameraCapture? = null
    private var view3D = false
    private lateinit var prefs: SharedPreferences
    private val exec = Executors.newSingleThreadExecutor(); private val h = Handler(Looper.getMainLooper())
    private var rt: Runnable? = null; private var pd = 0f
    private var lastTap = 0L; private var tapX = 0f; private var tapY = 0f
    private var currentZoom = 100; private val zoomLevels = floatArrayOf(0.3f, 0.6f, 1f, 1.5f, 2.2f, 3f, 5f, 8f, 12f, 18f)

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { processGalleryImage(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater); setContentView(b.root)
        r = PointCloudRenderer(); da3 = DepthEstimator(this)
        prefs = getSharedPreferences("gaussianscanner", MODE_PRIVATE)
        setupGL(); b.btnCapture.setOnClickListener { cam?.capture() }
        b.btnRetake.setOnClickListener { exit3D() }; b.btnGallery.setOnClickListener { pickFromGallery() }
        b.btnSettings.setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
        setupSeekBar()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCam() else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(rc, p, g)
        if (rc == 100 && g.all { it == PackageManager.PERMISSION_GRANTED }) startCam()
    }

    private fun startCam() {
        b.tvStatus.text = if (da3.ready) "Tap to capture" else "Loading DA3…"
        cam = CameraCapture(this, this, b.previewView) { bmp ->
            stopCam()
            processBitmap(bmp)
        }
        cam?.start()
    }

    private fun stopCam() {
        cam?.stop(); cam = null
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

    private fun pickFromGallery() { galleryLauncher.launch("image/*") }

    private fun processGalleryImage(uri: Uri) {
        stopCam()
        h.post { b.previewView.visibility = View.GONE; b.tvStatus.text = "Thinking…" }
        exec.execute {
            try {
                contentResolver.openInputStream(uri)?.use { stream ->
                    val bmp = BitmapFactory.decodeStream(stream)
                    if (bmp != null) processBitmap(bmp)
                    else h.post { b.previewView.visibility = View.VISIBLE; b.tvStatus.text = "Failed to load image" }
                }
            } catch (_: Exception) {
                h.post { b.previewView.visibility = View.VISIBLE; b.tvStatus.text = "Failed to load image" }
            }
        }
    }

    private fun processBitmap(bmp: android.graphics.Bitmap) {
        h.post { b.previewView.visibility = View.GONE; b.tvStatus.text = "Thinking…" }
        exec.execute {
            val depth = da3.estimate(bmp)
            var cloud = emptyList<com.starfall.gaussianscanner.model.ColoredPt>()
            val targetW = SettingsActivity.getTargetWidth(prefs)
            if (depth != null) cloud = PointCloudBuilder.build(bmp, depth, targetW)
            bmp.recycle(); System.gc()
            h.post {
                if (cloud.isNotEmpty()) { r.setCloud(cloud); enter3D() }
                else { b.previewView.visibility = View.VISIBLE; b.tvStatus.text = "Depth failed. Try again." }
            }
        }
    }

    private fun setupGL() {
        b.glSurfaceView.setEGLContextClientVersion(3); b.glSurfaceView.setRenderer(r)
        b.glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        b.glSurfaceView.setOnTouchListener { _, e ->
            if (!view3D) return@setOnTouchListener false
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    pd = 0f; val now = System.currentTimeMillis()
                    if (now - lastTap < 300 && Math.abs(e.x - tapX) < 50 && Math.abs(e.y - tapY) < 50) {
                        val idx = zoomLevels.indexOfFirst { it > r.zoom }
                        currentZoom = if (idx >= 0 && idx < zoomLevels.size - 1) idx + 1 else zoomLevels.size - 2
                        r.zoom = zoomLevels[currentZoom]
                        updateZoomLabel(); b.glSurfaceView.requestRender()
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
                        r.ry+=dx*0.5f;r.rx+=dy*0.5f;b.glSurfaceView.requestRender();true
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

    private fun enter3D() {
        view3D = true; b.previewView.visibility = View.GONE; b.glSurfaceView.visibility = View.VISIBLE
        b.btnCapture.visibility = View.GONE; b.btnRetake.visibility = View.VISIBLE; b.tvStatus.visibility = View.GONE
        b.btnGallery.visibility = View.GONE; b.tvZoom.visibility = View.VISIBLE
        b.ptSizePanel.visibility = View.VISIBLE; b.btnSettings.visibility = View.GONE
        r.zoom = 3f; currentZoom = 5; updateZoomLabel()
        rt = object : Runnable { override fun run() { if(!view3D)return;b.glSurfaceView.requestRender();h.postDelayed(this,32) } }
        h.post(rt!!)
    }

    private fun exit3D() {
        view3D = false; rt = null; b.previewView.visibility = View.VISIBLE
        b.glSurfaceView.visibility = View.GONE; b.btnCapture.visibility = View.VISIBLE
        b.btnRetake.visibility = View.GONE; b.tvStatus.visibility = View.VISIBLE; b.tvZoom.visibility = View.GONE
        b.btnGallery.visibility = View.VISIBLE; b.ptSizePanel.visibility = View.GONE
        b.btnSettings.visibility = View.VISIBLE
        r.rx=0f;r.ry=0f;r.zoom=3f; System.gc()
        if (!camStarted()) startCam()
    }

    private fun camStarted() = cam != null

    override fun onDestroy() { super.onDestroy();rt=null;cam?.shutdown();da3.close();exec.shutdown() }

    override fun onPause() { super.onPause(); rt = null; if (!view3D) stopCam() }

    override fun onResume() {
        super.onResume()
        if (view3D) {
            rt = object : Runnable { override fun run() { if(!view3D)return;b.glSurfaceView.requestRender();h.postDelayed(this,32) } }
            h.post(rt!!)
        } else if (!camStarted()) startCam()
    }
}
