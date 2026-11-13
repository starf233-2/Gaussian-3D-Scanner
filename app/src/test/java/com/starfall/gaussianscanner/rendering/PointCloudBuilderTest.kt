package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.DepthMap
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/** Exercises pure geometry without calling Android's Bitmap stub on the JVM. */
class PointCloudBuilderTest {
    @Test fun allocationFreeLightingMatchesOriginalVectorMath() {
        val random = Random(15)
        val depth = DepthMap(FloatArray(33 * 29) { .45f + random.nextFloat() * .02f }, 33, 29)
        val method = PointCloudBuilder::class.java.declaredMethods.single { it.name == "lambertShade" }.apply { isAccessible = true }
        for (interp in listOf(false, true)) {
            for (step in listOf(1, 2, 3)) {
                for (v in 0 until 29) for (u in 0 until 33) {
                    val expected = original(u, v, step, depth, interp)
                    val actual = method.invoke(PointCloudBuilder, u, v, step, 33, 29, 33f, 29f,
                        36.3f, 16.5f, 14.5f, depth, interp, .45f, .55f, .7f, sqrt(.45f*.45f + .55f*.55f + .7f*.7f)) as Float
                    assertEquals(expected, actual, .000001f)
                }
            }
        }
    }

    private fun original(u: Int, v: Int, step: Int, depth: DepthMap, interp: Boolean): Float {
        fun sample(x: Int, y: Int): Float? {
            if (x !in 0 until 33 || y !in 0 until 29) return null
            val fx = (x.toFloat() / 33 * depth.w).coerceIn(0f, 32f)
            val fy = (y.toFloat() / 29 * depth.h).coerceIn(0f, 28f)
            val x0 = fx.toInt(); val y0 = fy.toInt()
            if (!interp) return depth.data[y0 * 33 + x0]
            val x1 = minOf(x0 + 1, 32); val y1 = minOf(y0 + 1, 28)
            val ax = fx - x0; val ay = fy - y0
            val top = depth.data[y0 * 33 + x0] * (1 - ax) + depth.data[y0 * 33 + x1] * ax
            val bottom = depth.data[y1 * 33 + x0] * (1 - ax) + depth.data[y1 * 33 + x1] * ax
            return top * (1 - ay) + bottom * ay
        }
        val coords = arrayOf(u - step to v, u + step to v, u to v - step, u to v + step)
        val d = coords.map { sample(it.first, it.second) ?: return 1f }
        if (abs(d[0] - d[1]) > .3f || abs(d[2] - d[3]) > .3f) return 1f
        val p = coords.mapIndexed { i, c ->
            val z = d[i] * 3f + .3f
            floatArrayOf((c.first - 16.5f) * z / 36.3f, (14.5f - c.second) * z / 36.3f, -z)
        }
        val tx = p[1][0] - p[0][0]; val ty = p[1][1] - p[0][1]; val tz = p[1][2] - p[0][2]
        val bx = p[3][0] - p[2][0]; val by = p[3][1] - p[2][1]; val bz = p[3][2] - p[2][2]
        var nx = ty * bz - tz * by; var ny = tz * bx - tx * bz; var nz = tx * by - ty * bx
        val len = sqrt(nx * nx + ny * ny + nz * nz)
        if (len < .000001f) return 1f
        nx /= len; ny /= len; nz /= len
        if (nz < 0) { nx = -nx; ny = -ny; nz = -nz }
        return .62f + .38f * maxOf((nx * .45f + ny * .55f + nz * .7f) / sqrt(.45f*.45f + .55f*.55f + .7f*.7f), 0f)
    }
}
