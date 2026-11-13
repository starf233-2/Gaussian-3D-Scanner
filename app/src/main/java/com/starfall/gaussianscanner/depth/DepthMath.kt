package com.starfall.gaussianscanner.depth

import com.starfall.gaussianscanner.model.DepthMap
import java.util.Arrays

internal object DepthMath {
    /** Clean BEFORE any neighbourhood/filter operation; one NaN otherwise
     * contaminates a whole integral image. Preserve every finite input value. */
    fun sanitize(values: FloatArray) {
        if (values.all { it.isFinite() }) return
        val finite = FloatArray(values.size)
        var n = 0
        for (v in values) if (v.isFinite()) finite[n++] = v
        Arrays.sort(finite, 0, n)
        val median = if (n == 0) 0.5f else finite[n / 2]
        for (j in values.indices) if (!values[j].isFinite()) values[j] = median
    }

    fun normalize(values: FloatArray, w: Int, h: Int): DepthMap {
        require(w > 0 && h > 0 && w.toLong() * h == values.size.toLong())
        sanitize(values)
        val sorted = values.copyOf().apply { sort() }
        val n = sorted.size
        val lo = sorted[(n * 0.01f).toInt().coerceIn(0, n - 1)]
        val hi = sorted[(n * 0.99f).toInt().coerceIn(0, n - 1)]
        val range = hi.toDouble() - lo.toDouble()
        // Keep the original float arithmetic for ordinary model values.
        if (range > 0 && (hi - lo).isFinite()) {
            for (j in values.indices) values[j] = ((values[j] - lo) / (hi - lo)).coerceIn(0f, 1f)
        } else if (range > 0) {
            for (j in values.indices) values[j] = ((values[j].toDouble() - lo) / range).toFloat().coerceIn(0f, 1f)
        } else values.fill(0.5f)
        return DepthMap(values, w, h)
    }

    /** Same local guided-filter formula and clipped windows as before, using
     * a rolling row window instead of four full-resolution integral images.
     * Double sums avoid catastrophic cancellation/negative variance at 1036².
     * Scratch memory: O(width * radius), not O(width * height).
     */
    fun guidedFilter(guide: FloatArray, src: FloatArray, w: Int, h: Int, radius: Int, eps: Float): FloatArray {
        require(w > 0 && h > 0 && guide.size == w * h && src.size == w * h && radius >= 0 && eps > 0)
        val window = minOf(h, radius * 2 + 1)
        val rows = Array(window) { DoubleArray(w * 4) }
        val sums = DoubleArray(w * 4)
        val result = FloatArray(w * h)
        var bottom = -1
        var top = 0
        for (y in 0 until h) {
            val nextTop = maxOf(0, y - radius)
            while (top < nextTop) {
                val old = rows[top % window]
                for (k in sums.indices) sums[k] -= old[k]
                top++
            }
            val nextBottom = minOf(h - 1, y + radius)
            while (bottom < nextBottom) {
                bottom++
                val row = rows[bottom % window]
                var si = 0.0; var sp = 0.0; var sii = 0.0; var sip = 0.0
                val base = bottom * w
                fun add(x: Int, sign: Double) {
                    val iv = guide[base + x].toDouble(); val pv = src[base + x].toDouble()
                    si += iv * sign; sp += pv * sign
                    sii += iv * iv * sign; sip += iv * pv * sign
                }
                for (x in 0..minOf(w - 1, radius)) add(x, 1.0)
                for (x in 0 until w) {
                    if (x > 0) {
                        if (x - radius - 1 >= 0) add(x - radius - 1, -1.0)
                        if (x + radius < w) add(x + radius, 1.0)
                    }
                    val k = x * 4
                    row[k] = si; row[k + 1] = sp; row[k + 2] = sii; row[k + 3] = sip
                }
                for (k in sums.indices) sums[k] += row[k]
            }
            for (x in 0 until w) {
                val area = (bottom - top + 1) * (minOf(w - 1, x + radius) - maxOf(0, x - radius) + 1)
                val k = x * 4
                val mi = sums[k] / area; val mp = sums[k + 1] / area
                val variance = maxOf(0.0, sums[k + 2] / area - mi * mi)
                val covariance = sums[k + 3] / area - mi * mp
                val a = covariance / (variance + eps)
                result[y * w + x] = (a * (guide[y * w + x] - mi) + mp).toFloat()
            }
        }
        return result
    }
}
