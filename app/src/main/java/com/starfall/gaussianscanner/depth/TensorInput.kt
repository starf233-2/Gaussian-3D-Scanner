package com.starfall.gaussianscanner.depth

import java.nio.ByteBuffer
import kotlin.math.roundToInt

internal object TensorInput {
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun write(buffer: ByteBuffer, pixels: IntArray, nhwc: Boolean, normalize: Boolean,
              floatInput: Boolean, scale: Float = 1f, zeroPoint: Int = 0, signed: Boolean = true) {
        buffer.clear()
        if (!floatInput) require(scale.isFinite() && scale > 0f) { "量化比例无效" }
        fun put(pixel: Int, channel: Int) {
            val raw = ((pixel shr (16 - channel * 8)) and 255) / 255f
            val value = if (normalize) (raw - mean[channel]) / std[channel] else raw
            if (floatInput) buffer.putFloat(value) else {
                val quantized = (value / scale + zeroPoint).roundToInt()
                buffer.put(quantized.coerceIn(if (signed) -128 else 0, if (signed) 127 else 255).toByte())
            }
        }
        if (nhwc) for (pixel in pixels) for (channel in 0..2) put(pixel, channel)
        else for (channel in 0..2) for (pixel in pixels) put(pixel, channel)
        buffer.rewind()
    }
}
