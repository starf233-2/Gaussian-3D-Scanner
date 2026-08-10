package com.starfall.gaussianscanner.depth

import android.content.Context; import android.graphics.Bitmap
import com.starfall.gaussianscanner.model.DepthMap
import org.tensorflow.lite.Interpreter; import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate; import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer; import java.nio.ByteOrder

class DepthEstimator(ctx: Context) {

    companion object {
        private val M = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val S = floatArrayOf(0.229f, 0.224f, 0.225f)
        // DA3 model input is FIXED [1,3,896,504] (W=504, H=896) by the conversion.
        const val W = 504
        const val H = 896
    }

    private var i: Interpreter? = null
    private var g: GpuDelegate? = null
    @Volatile var ready = false; private set

    init {
        try {
            val b = FileUtil.loadMappedFile(ctx, "da3_small_gpu_fp16.tflite")
            val o = Interpreter.Options().apply {
                setNumThreads(4); setUseXNNPACK(true)
                try { if (CompatibilityList().isDelegateSupportedOnThisDevice) { g = GpuDelegate(); addDelegate(g) } } catch (_: Throwable) {}
            }
            i = Interpreter(b, o); ready = true
        } catch (_: Throwable) {}
    }

    fun estimate(bmp: Bitmap): DepthMap? {
        val it = i ?: return null
        return try {
            val rs = Bitmap.createScaledBitmap(bmp, W, H, true)
            val px = IntArray(W * H); rs.getPixels(px, 0, W, 0, 0, W, H)
            val inp = ByteBuffer.allocateDirect(3 * H * W * 4).apply { order(ByteOrder.nativeOrder()) }
            for (c in 0..2) for (y in 0 until H) for (x in 0 until W) {
                val v = ((px[y * W + x] shr (16 - 8 * c)) and 0xFF)
                inp.putFloat(((v / 255f) - M[c]) / S[c])
            }
            val out = ByteBuffer.allocateDirect(H * W * 4).apply { order(ByteOrder.nativeOrder()) }
            it.run(inp, out); out.rewind()
            val d = FloatArray(H * W)
            var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
            for (j in d.indices) { d[j] = out.float; if (d[j] < mn) mn = d[j]; if (d[j] > mx) mx = d[j] }
            val rng = mx - mn
            if (rng > 0f) for (j in d.indices) d[j] = (d[j] - mn) / rng
            DepthMap(d, W, H)
        } catch (e: Throwable) {
            android.util.Log.e("DepthEstimator", "inference failed", e)
            null
        }
    }

    fun close() {
        try { i?.close() } catch (_: Throwable) {}
        try { g?.close() } catch (_: Throwable) {}
        i = null; g = null
    }
}
