package com.starfall.gaussianscanner.camera

import android.content.Context; import android.graphics.*
import androidx.camera.core.*; import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView; import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner; import java.util.concurrent.Executors

class CameraCapture(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    val onPhoto: (Bitmap) -> Unit
) {
    private var cap: ImageCapture? = null
    private var provider: ProcessCameraProvider? = null
    private val exec = Executors.newSingleThreadExecutor()
    private var started = false

    fun start() {
        if (started) return
        started = true
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
                provider?.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, prv, cap)
            } catch (_: Exception) {}
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        started = false
        try { provider?.unbindAll() } catch (_: Exception) {}
        cap = null
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
                    bmp?.let(onPhoto)
                } catch (_: Exception) { proxy.close() }
            }
            override fun onError(e: ImageCaptureException) { e.printStackTrace() }
        })
    }

    fun shutdown() { stop(); exec.shutdown() }
}
