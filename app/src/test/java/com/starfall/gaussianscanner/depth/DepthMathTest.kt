package com.starfall.gaussianscanner.depth

import org.junit.Assert.*
import org.junit.Test
import kotlin.random.Random

class DepthMathTest {
    @Test fun finiteNormalizationPreservesOriginalFormula() {
        val values = FloatArray(10000) { Random(10 + it).nextFloat() * 8f }
        val expected = values.copyOf()
        val sorted = values.sortedArray()
        val lo = sorted[100]; val hi = sorted[9900]
        for (j in expected.indices) expected[j] = ((expected[j] - lo) / (hi - lo)).coerceIn(0f, 1f)
        assertArrayEquals(expected, DepthMath.normalize(values, 100, 100).data, 0f)
    }

    @Test fun majorityNaNUsesFiniteMedianNotNaN() {
        val values = floatArrayOf(Float.NaN, Float.NaN, Float.NaN, 1f, 3f, Float.POSITIVE_INFINITY)
        DepthMath.sanitize(values)
        assertArrayEquals(floatArrayOf(3f, 3f, 3f, 1f, 3f, 3f), values, 0f)
        assertTrue(DepthMath.normalize(values, 3, 2).data.all { it.isFinite() && it in 0f..1f })
    }

    @Test fun allInvalidOrConstantBecomeNeutral() {
        for (data in arrayOf(FloatArray(4) { Float.NaN }, FloatArray(4) { Float.NEGATIVE_INFINITY }, FloatArray(4) { 3f })) {
            assertArrayEquals(FloatArray(4) { 0.5f }, DepthMath.normalize(data, 2, 2).data, 0f)
        }
    }

    @Test fun hugeFiniteRangeDoesNotOverflow() {
        val result = DepthMath.normalize(floatArrayOf(-Float.MAX_VALUE, 0f, Float.MAX_VALUE), 3, 1).data
        assertArrayEquals(floatArrayOf(0f, 0.5f, 1f), result, 0f)
    }

    @Test fun invalidDimensionsRejected() {
        assertThrows(IllegalArgumentException::class.java) { DepthMath.normalize(FloatArray(0), 0, 0) }
        assertThrows(IllegalArgumentException::class.java) { DepthMath.normalize(FloatArray(3), 2, 2) }
    }

    @Test fun rollingFilterMatchesIndependentDoubleReferenceIncludingBorders() {
        val random = Random(4)
        for ((w, h) in listOf(1 to 1, 1 to 8, 9 to 1, 3 to 2, 31 to 29)) {
            val guide = FloatArray(w * h) { random.nextFloat() }
            val src = FloatArray(w * h) { random.nextFloat() * 12f }
            for (r in listOf(0, 1, 2, 7)) {
                val expected = naive(guide, src, w, h, r, 0.0025f)
                assertArrayEquals("$w x $h radius=$r", expected, DepthMath.guidedFilter(guide, src, w, h, r, 0.0025f), 0.000002f)
            }
        }
    }

    @Test fun fullResolutionConstantSurfaceStaysFlatAndFinite() {
        val w = 1036; val h = 1036
        val guide = FloatArray(w * h) { 0.98f }
        val src = FloatArray(w * h) { 7.31f }
        val result = DepthMath.guidedFilter(guide, src, w, h, 2, 0.0025f)
        assertTrue(result.all { it.isFinite() && kotlin.math.abs(it - 7.31f) < 0.000001f })
    }

    private fun naive(guide: FloatArray, src: FloatArray, w: Int, h: Int, radius: Int, eps: Float): FloatArray {
        return FloatArray(w * h) { index ->
            val x = index % w; val y = index / w
            var si = 0.0; var sp = 0.0; var sii = 0.0; var sip = 0.0; var n = 0
            for (yy in maxOf(0, y - radius)..minOf(h - 1, y + radius)) {
                for (xx in maxOf(0, x - radius)..minOf(w - 1, x + radius)) {
                    val i = guide[yy * w + xx].toDouble(); val p = src[yy * w + xx].toDouble()
                    si += i; sp += p; sii += i * i; sip += i * p; n++
                }
            }
            val mi = si / n; val mp = sp / n
            val a = (sip / n - mi * mp) / (maxOf(0.0, sii / n - mi * mi) + eps)
            (a * (guide[index] - mi) + mp).toFloat()
        }
    }
}
