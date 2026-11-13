package com.starfall.gaussianscanner.depth

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TensorInputTest {
    private val pixels = intArrayOf(0xFF0000, 0x00FF00)
    @Test fun nhwcFloatMatchesOriginalInputExactly() {
        val buffer = ByteBuffer.allocateDirect(24).order(ByteOrder.nativeOrder())
        TensorInput.write(buffer, pixels, true, false, true)
        val actual = FloatArray(6); buffer.asFloatBuffer().get(actual)
        assertArrayEquals(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f), actual, 0f)
        assertEquals(0, buffer.position())
    }
    @Test fun nchwQuantizedUsesPlanesNotInterleavedRgb() {
        val buffer = ByteBuffer.allocate(6)
        TensorInput.write(buffer, pixels, false, false, false, 1f / 255f, 0, false)
        assertArrayEquals(byteArrayOf(-1, 0, 0, -1, 0, 0), buffer.array())
    }
    @Test fun quantizationRoundsAndClampsAndBuffersCanBeReused() {
        val buffer = ByteBuffer.allocate(6)
        TensorInput.write(buffer, pixels, true, false, false, 0.4f, -1, true)
        assertArrayEquals(byteArrayOf(2, -1, -1, -1, 2, -1), buffer.array())
        TensorInput.write(buffer, pixels, true, false, false, 0.0001f, 0, true)
        assertArrayEquals(byteArrayOf(127, 0, 0, 0, 127, 0), buffer.array())
    }
    @Test fun legacyImageNetNormalizationIsPreserved() {
        val buffer = ByteBuffer.allocate(24).order(ByteOrder.nativeOrder())
        TensorInput.write(buffer, pixels, false, true, true)
        val values = FloatArray(6); buffer.asFloatBuffer().get(values)
        assertArrayEquals(floatArrayOf((1f - .485f) / .229f, -0.485f / .229f,
            -0.456f / .224f, (1f - .456f) / .224f, -.406f / .225f, -.406f / .225f), values, 0f)
    }
}
