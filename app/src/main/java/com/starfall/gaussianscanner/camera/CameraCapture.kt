package com.starfall.gaussianscanner.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class CameraCapture(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    private val highResolution: Boolean = false,
    private val onError: (String) -> Unit = {},
    private val onPhoto: (Bitmap, Float) -> Unit
) {
    private var capture: ImageCapture? = null
    private var preview: Preview? = null
    private var provider: ProcessCameraProvider? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val main = ContextCompat.getMainExecutor(context)
    @Volatile private var started = false
    private var generation = 0
    private var shutDown = false
    private var capturing = false
    private var focalPx = 0f
    private var activeW = 0
    private var activeH = 0
    @Volatile var camera: Camera? = null; private set

    @Suppress("DEPRECATION")
    @androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
    fun start() {
        if (started || shutDown) return
        started = true
        val gen = ++generation
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (!started || gen != generation) return@addListener
            try {
                provider = future.get()
                capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .setJpegQuality(90)
                    // Reconstruction uses at most ~1MP of color. Avoid decoding
                    // a 12/48MP full-sensor shot just to scale it down again.
                    .setTargetResolution(if (highResolution) Size(2048, 1536) else Size(1600, 1200))
                    .setTargetRotation(previewView.display?.rotation ?: 0)
                    .build()
                preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                camera = provider?.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                readIntrinsics(Camera2CameraInfo.from(camera!!.cameraInfo).cameraId)
            } catch (e: Exception) {
                stop()
                onError("相机启动失败: ${e.message}")
            }
        }, main)
    }

    /** PreviewView's metering factory takes VIEW PIXELS, not normalized coords. */
    fun focusAt(x: Float, y: Float): Boolean { return try {
        val active = camera ?: return false
        val point = previewView.meteringPointFactory.createPoint(x, y)
        active.cameraControl.startFocusAndMetering(
            FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(2, java.util.concurrent.TimeUnit.SECONDS).build()
        )
        true
    } catch (_: Exception) { false } }

    fun setExposure(ev: Float): Boolean { return try {
        val active = camera ?: return false
        val state = active.cameraInfo.exposureState
        val step = state.exposureCompensationStep.toFloat()
        if (!state.isExposureCompensationSupported || step <= 0f) return false
        val index = (ev / step).roundToInt().coerceIn(state.exposureCompensationRange.lower, state.exposureCompensationRange.upper)
        active.cameraControl.setExposureCompensationIndex(index)
        true
    } catch (_: Exception) { false } }

    private fun readIntrinsics(id: String) {
        focalPx = 0f; activeW = 0; activeH = 0
        try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val chars = manager.getCameraCharacteristics(id)
            val calibration = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val array = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            if (calibration != null && calibration.size >= 4 && array != null && calibration[0].isFinite() && calibration[0] > 0f) {
                focalPx = calibration[0]; activeW = array.width(); activeH = array.height()
            }
        } catch (_: Exception) {}
    }

    fun stop() {
        started = false; generation++
        val cases = listOfNotNull(preview, capture).toTypedArray()
        if (cases.isNotEmpty()) runCatching { provider?.unbind(*cases) }
        capture = null; preview = null; camera = null
    }

    fun capture(): Boolean {
        val active = capture ?: return false
        if (capturing || !started || shutDown) return false
        capturing = true
        active.targetRotation = previewView.display?.rotation ?: 0
        active.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(proxy: ImageProxy) {
                var bitmap: Bitmap? = null
                try {
                    require(proxy.format == ImageFormat.JPEG) { "不支持的相机输出格式" }
                    val buffer = proxy.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                    var sample = 1
                    val decodeLimit = if (highResolution) 3072 else 2048
                    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > decodeLimit) sample *= 2
                    bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
                        ?: error("照片解码失败")
                    val degrees = proxy.imageInfo.rotationDegrees
                    if (degrees != 0) {
                        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height,
                            Matrix().apply { postRotate(degrees.toFloat()) }, true)
                        if (rotated !== bitmap) bitmap.recycle()
                        bitmap = rotated
                    }
                    val result = bitmap
                    val focal = if (activeW > 0 && activeH > 0) focalPx * maxOf(result.width, result.height) / maxOf(activeW, activeH) else 0f
                    main.execute {
                        capturing = false
                        if (started) {
                            try { onPhoto(result, focal) } catch (e: Exception) {
                                if (!result.isRecycled) result.recycle()
                                onError("照片处理失败: ${e.message}")
                            }
                        } else result.recycle()
                    }
                    bitmap = null // ownership transferred to the main-thread callback
                } catch (e: Throwable) {
                    bitmap?.recycle()
                    main.execute { capturing = false; if (started) onError("拍照失败: ${e.message}") }
                } finally { proxy.close() }
            }
            override fun onError(exception: ImageCaptureException) {
                main.execute { capturing = false; if (started) onError("拍照失败: ${exception.message}") }
            }
        })
        return true
    }

    fun shutdown() {
        if (shutDown) return
        shutDown = true
        stop()
        executor.shutdown()
    }
}
