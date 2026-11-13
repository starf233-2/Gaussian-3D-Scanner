package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.PointCloud
import kotlin.math.*

internal object PointFootprint {
    const val SLOPE_RANGE = 4f
    const val COVERAGE = 1.12f

    /** Use a valid one-sided slope at silhouettes, never span a foreground /
     * background jump and create an enormous splat that bridges that gap. */
    fun slope(z: Float, before: Float, after: Float, spacing: Float): Short {
        if (spacing <= 0f || !spacing.isFinite()) return 0
        val limit = maxOf(spacing * SLOPE_RANGE, z * .03f)
        val left = z - before; val right = after - z
        val goodLeft = before.isFinite() && abs(left) <= limit
        val goodRight = after.isFinite() && abs(right) <= limit
        val delta = when {
            goodLeft && goodRight -> (left + right) * .5f
            goodLeft -> left
            goodRight -> right
            else -> 0f
        }
        return (delta / spacing / SLOPE_RANGE).coerceIn(-1f, 1f).times(32767f).roundToInt().toShort()
    }

    /** Irregular PLY has no camera grid: bounded sampled nearest-neighbour
     * estimate, independent of file order. Surface-density correction accounts
     * for sampling. It remains an approximation for mixed-density imports. */
    fun estimateImportedSpacing(cloud: PointCloud): Float {
        val count = minOf(1024, cloud.size)
        if (count < 2) return .01f
        val indices = IntArray(count) { a ->
            val start = (a.toLong() * cloud.size / count).toInt()
            val end = ((a + 1L) * cloud.size / count).toInt()
            var hash = a + 0x9e3779b9.toInt()
            hash = (hash xor (hash ushr 16)) * 0x85ebca6b.toInt()
            hash = (hash xor (hash ushr 13)) * 0xc2b2ae35.toInt()
            hash = hash xor (hash ushr 16)
            // Jitter within strata avoids selecting one column of an ordered
            // 1024-wide PLY grid, which would underestimate spacing by 32x.
            (start + (hash and Int.MAX_VALUE) % (end - start)) * PointCloud.STRIDE
        }
        val nearest = FloatArray(count) { Float.POSITIVE_INFINITY }
        for (a in 0 until count) {
            val ia = indices[a]
            for (b in a + 1 until count) {
                val ib = indices[b]
                val x = cloud.data[ia].toDouble() - cloud.data[ib]
                val y = cloud.data[ia + 1].toDouble() - cloud.data[ib + 1]
                val z = cloud.data[ia + 2].toDouble() - cloud.data[ib + 2]
                val d = (x * x + y * y + z * z).toFloat()
                if (d > 0f && d.isFinite()) {
                    nearest[a] = minOf(nearest[a], d); nearest[b] = minOf(nearest[b], d)
                }
            }
        }
        val finite = nearest.filter { it.isFinite() }.sorted()
        if (finite.isEmpty()) return .01f
        // Median nearest distance of a sparse random 2D sample is ~0.47 times
        // sqrt(area/count). Correct that sampling bias, not only sample count.
        val correction = if (cloud.size > count * 4) 2.1f else 1f
        return (sqrt(finite[finite.size / 2]) * sqrt(count.toFloat() / cloud.size) * correction).coerceAtLeast(1e-7f)
    }

    /** CPU reference for the GLSL footprint equation (column-major MVP).
     * Axis-aligned bounding square of the projected camera-grid cell. */
    fun projectedSize(position: FloatArray, center: FloatArray, mvp: FloatArray,
                      viewportW: Int, viewportH: Int, stepOverFocal: Float,
                      slopeX: Short = 0, slopeY: Short = 0, maximum: Float = 64f): Float {
        val z = -position[2]
        val spacing = z * stepOverFocal
        val dx = slopeX / 32767f * SLOPE_RANGE * spacing
        val dy = slopeY / 32767f * SLOPE_RANGE * spacing
        val tx = floatArrayOf(spacing + position[0] / z * dx, position[1] / z * dx, -dx)
        val ty = floatArrayOf(position[0] / z * dy, spacing + position[1] / z * dy, -dy)
        fun transform(p: FloatArray, w: Float): FloatArray = FloatArray(4) { row ->
            mvp[row] * p[0] + mvp[4 + row] * p[1] + mvp[8 + row] * p[2] + mvp[12 + row] * w
        }
        val clip = transform(FloatArray(3) { position[it] - center[it] }, 1f)
        if (clip[3] <= .1f || z <= 0f) return 1f
        fun projected(t: FloatArray): FloatArray {
            val v = transform(t, 0f)
            return floatArrayOf((v[0] * clip[3] - clip[0] * v[3]) / (clip[3] * clip[3]) * viewportW * .5f,
                (v[1] * clip[3] - clip[1] * v[3]) / (clip[3] * clip[3]) * viewportH * .5f)
        }
        val x = projected(tx); val y = projected(ty)
        return (maxOf(abs(x[0]) + abs(y[0]), abs(x[1]) + abs(y[1])) * COVERAGE).coerceIn(1f, maximum)
    }
}
