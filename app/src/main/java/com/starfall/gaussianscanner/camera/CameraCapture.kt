package com.starfall.gaussianscanner.camera

import android.content.Context; import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import androidx.camera.core.*; import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView; import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner; import java.util.concurrent.Executors

class CameraCapture(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    val onPhoto: (Bitmap, Float) -> Unit
) {
    private var cap: ImageCapture? = null
    private var provider: ProcessCameraProvider? = null
    private val exec = Executors.newSingleThreadExecutor()
    private var started = false

    private var focalPx = 0f

    @Volatile var camera: Camera? = null; private set
    val cameraControl: CameraControl? get() = camera?.cameraControl

    fun start() {
        if (started) return
        started = true
        readIntrinsics()
        ProcessCameraProvider.getInstance(context).addListener({
            provider = ProcessCameraProvider.getInstance(context).get()
            cap = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(90)
                .setTargetRotation(previewView.display?.rotation ?: 0)
                .build()
            val prv = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            try {
                provider?.unbindAll()
                camera = provider?.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, prv, cap)
            } catch (_: Exception) {}
        }, ContextCompat.getMainExecutor(context))
    }

    /** trigger tap-to-focus at view coordinates [0..1] on the preview view. */
    fun focusAt(nx: Float, ny: Float): Boolean {
        val control = cameraControl ?: return false
        return try {
            val point = previewView.meteringPointFactory.createPoint(nx, ny)
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(2, java.util.concurrent.TimeUnit.SECONDS)
                .build()
            control.startFocusAndMetering(action)
            true
        } catch (_: Exception) { false }
    }

    /** set exposure compensation in EV (range depends on device, typically -12..+12 or wider). */
    fun setExposure(ev: Float): Boolean {
        val control = cameraControl ?: return false
        return try {
            control.setExposureCompensationIndex(ev.toInt())
            true
        } catch (_: Exception) { false }
    }

    private fun readIntrinsics() {
        try {
            val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val id = cm.cameraIdList.firstOrNull { cid ->
                cm.getCameraCharacteristics(cid)
                    .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: cm.cameraIdList.firstOrNull() ?: return
            val ch = cm.getCameraCharacteristics(id)
            val calib = ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val aa = ch.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            if (calib != null && aa != null && calib.size >= 4 && aa.width() > 0) {
                focalPx = calib[0]
            }
        } catch (_: Exception) { focalPx = 0f }
    }

    fun stop() {
        started = false
        try { provider?.unbindAll() } catch (_: Exception) {}
        cap = null; camera = null
    }

    fun capture() {
        cap?.takePicture(exec, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(proxy: ImageProxy) {
                try {
                    val buf = proxy.planes[0].buffer
                    val bytes = ByteArray(buf.remaining()); buf.get(bytes)
                    var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    val deg = proxy.imageInfo.rotationDegrees
                    if (bmp != null && deg != 0) {
                        val m = Matrix(); m.postRotate(deg.toFloat())
                        val r = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                        if (r != bmp) bmp.recycle(); bmp = r
                    }
                    proxy.close()
                    if (bmp != null) onPhoto(bmp, focalPx)
                } catch (_: Exception) { proxy.close() }
            }
            override fun onError(e: ImageCaptureException) { e.printStackTrace() }
        })
    }

    fun shutdown() { stop(); exec.shutdown() }
}
