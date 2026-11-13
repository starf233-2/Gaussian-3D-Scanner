package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.PointCloud
import com.starfall.gaussianscanner.model.DepthMap
import com.starfall.gaussianscanner.model.SurfaceGrid
import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.sqrt

object PointCloudBuilder {

    /**
     * @param focalPx real camera focal length in pixels at the ORIGINAL image resolution (0 = estimate).
     * @param interp true = bilinear depth sampling, false = nearest neighbor.
     * @param colorEnhance boost saturation/contrast of sampled colors.
     * @param lighting bake a fixed-direction Lambertian shade from the local
     *        surface normal (computed from depth) into each point color, which
     *        gives the point cloud a much stronger sense of depth/volume.
     */
    fun build(
        bmp: Bitmap, depth: DepthMap,
        targetW: Int = 500,
        edgeTrim: Float = 0f,
        focalPx: Float = 0f,
        interp: Boolean = true,
        colorEnhance: Boolean = true,
        lighting: Boolean = true
    ): PointCloud {
        require(targetW > 0 && depth.w > 0 && depth.h > 0 && depth.w.toLong() * depth.h == depth.data.size.toLong())
        val bw = bmp.width; val bh = bmp.height
        val (tw, th) = CloudSampling.dimensions(bw, bh, targetW)

        val scaled = Bitmap.createScaledBitmap(bmp, tw, th, true)
        try {
        val px = IntArray(tw * th); scaled.getPixels(px, 0, tw, 0, 0, tw, th)

        val gridStep = 1 // Dimensions are capped smoothly before allocating pixels.

        // depth gradient mask for edge trimming
        val trimEnabled = edgeTrim > 0.001f
        val edgeMask = if (trimEnabled) FloatArray(depth.data.size) else null
        if (trimEnabled) {
            for (v in 1 until depth.h - 1) {
                for (u in 1 until depth.w - 1) {
                    val idx = v * depth.w + u
                    val gx = abs(depth.data[idx + 1] - depth.data[idx - 1])
                    val gy = abs(depth.data[idx + depth.w] - depth.data[idx - depth.w])
                    edgeMask!![idx] = maxOf(gx, gy)
                }
            }
        }
        val trimThresh = 1.0f - (1.0f - 0.04f) * edgeTrim

        // focal: use real intrinsic if provided, else estimate.
        val focal = if (focalPx > 0f) focalPx * (tw.toFloat() / bw) else tw.coerceAtLeast(th).toFloat() * 1.1f
        val cx = tw / 2f; val cy = th / 2f
        val dw = depth.w.toFloat(); val dh = depth.h.toFloat()
        // constant light direction (upper-right, slightly toward viewer)
        val lx = 0.45f; val ly = 0.55f; val lz = 0.7f
        val llen = sqrt(lx * lx + ly * ly + lz * lz)

        var count = ((tw + gridStep - 1) / gridStep) * ((th + gridStep - 1) / gridStep)
        if (trimEnabled) {
            count = 0
            for (v in 0 until th step gridStep) for (u in 0 until tw step gridStep) {
                val du = (u.toFloat() / tw * dw).coerceIn(0f, (depth.w - 1).toFloat())
                val dv = (v.toFloat() / th * dh).coerceIn(0f, (depth.h - 1).toFloat())
                if (edgeMask!![dv.toInt() * depth.w + du.toInt()] > trimThresh) continue
                count++
            }
        }
        val pts = FloatArray(count * PointCloud.STRIDE)
        val slopes = ShortArray(count * 2)
        var written = 0

        for (v in 0 until th step gridStep) {
            for (u in 0 until tw step gridStep) {
                val du = (u.toFloat() / tw * dw).coerceIn(0f, (depth.w - 1).toFloat())
                val dv = (v.toFloat() / th * dh).coerceIn(0f, (depth.h - 1).toFloat())
                if (trimEnabled) {
                    val di = dv.toInt() * depth.w + du.toInt()
                    if (edgeMask!![di] > trimThresh) continue
                }
                val d = if (interp) sampleDepth(depth, du, dv)
                        else depth.data[dv.toInt() * depth.w + du.toInt()]

                val pixel = px[v * tw + u]
                var r = ((pixel shr 16) and 0xFF) / 255f
                var g = ((pixel shr 8) and 0xFF) / 255f
                var b = (pixel and 0xFF) / 255f

                if (colorEnhance) {
                    val luma = 0.299f * r + 0.587f * g + 0.114f * b
                    // saturation boost + slight contrast
                    r = (luma + (r - luma) * 1.18f - 0.5f) * 1.06f + 0.5f
                    g = (luma + (g - luma) * 1.18f - 0.5f) * 1.06f + 0.5f
                    b = (luma + (b - luma) * 1.18f - 0.5f) * 1.06f + 0.5f
                    r = r.coerceIn(0f, 1f); g = g.coerceIn(0f, 1f); b = b.coerceIn(0f, 1f)
                }

                val z = d * 3f + 0.3f
                val wx = (u - cx) * z / focal
                val wy = (cy - v) * z / focal
                val wz = -z

                if (lighting) {
                    val shade = lambertShade(u, v, gridStep, tw, th, dw, dh, focal, cx, cy, depth, interp, lx, ly, lz, llen)
                    r *= shade; g *= shade; b *= shade
                }

                val spacing = z * gridStep / focal
                val leftZ = depthAt(u - gridStep, v, tw, th, dw, dh, depth, interp) * 3f + .3f
                val rightZ = depthAt(u + gridStep, v, tw, th, dw, dh, depth, interp) * 3f + .3f
                val topZ = depthAt(u, v - gridStep, tw, th, dw, dh, depth, interp) * 3f + .3f
                val bottomZ = depthAt(u, v + gridStep, tw, th, dw, dh, depth, interp) * 3f + .3f
                slopes[written * 2] = PointFootprint.slope(z, leftZ, rightZ, spacing)
                // Source y points upward; image rows point downward.
                slopes[written * 2 + 1] = PointFootprint.slope(z, bottomZ, topZ, spacing)
                val offset = written++ * PointCloud.STRIDE
                pts[offset] = wx; pts[offset + 1] = wy; pts[offset + 2] = wz
                pts[offset + 3] = r.coerceIn(0f, 1f); pts[offset + 4] = g.coerceIn(0f, 1f); pts[offset + 5] = b.coerceIn(0f, 1f)
            }
        }

        return PointCloud(pts, written, SurfaceGrid(gridStep / focal, slopes))
        } finally { if (scaled !== bmp) scaled.recycle() }
    }

    /** Fixed-direction Lambert shade from the surface normal at (u,v). */
    private fun lambertShade(
        u: Int, v: Int, step: Int, tw: Int, th: Int,
        dw: Float, dh: Float, focal: Float, cx: Float, cy: Float,
        depth: DepthMap, interp: Boolean,
        lx: Float, ly: Float, lz: Float, llen: Float
    ): Float {
        // depth-edge guard: across an occlusion boundary the neighbour depths
        // jump wildly and the normal becomes garbage (visible shading spikes
        // at object tops). Fall back to neutral shading there.
        val d0 = depthAt(u - step, v, tw, th, dw, dh, depth, interp)
        val d1 = depthAt(u + step, v, tw, th, dw, dh, depth, interp)
        val d2 = depthAt(u, v - step, tw, th, dw, dh, depth, interp)
        val d3 = depthAt(u, v + step, tw, th, dw, dh, depth, interp)
        if (!d0.isFinite() || !d1.isFinite() || !d2.isFinite() || !d3.isFinite()) return 1f
        if (abs(d0 - d1) > 0.3f || abs(d2 - d3) > 0.3f) return 1f

        // Reuse the four depths already sampled above. No FloatArray world
        // vectors (previously four arrays + duplicate depth samples PER point).
        val z0 = d0 * 3f + 0.3f; val z1 = d1 * 3f + 0.3f
        val z2 = d2 * 3f + 0.3f; val z3 = d3 * 3f + 0.3f
        val tx = (u + step - cx) * z1 / focal - (u - step - cx) * z0 / focal
        val ty = (cy - v) * z1 / focal - (cy - v) * z0 / focal
        val tz = -z1 - (-z0)
        val bx = (u - cx) * z3 / focal - (u - cx) * z2 / focal
        val by = (cy - v - step) * z3 / focal - (cy - v + step) * z2 / focal
        val bz = -z3 - (-z2)
        // normal = T x B
        var nx = ty * bz - tz * by
        var ny = tz * bx - tx * bz
        var nz = tx * by - ty * bx
        val nlen = sqrt(nx * nx + ny * ny + nz * nz)
        if (nlen < 1e-6f) return 1f
        nx /= nlen; ny /= nlen; nz /= nlen
        // orient toward viewer (+z of camera looks along -z in this setup means
        // surfaces facing the camera have normal ~ +z_camera). Flip if needed.
        if (nz < 0f) { nx = -nx; ny = -ny; nz = -nz }
        val ndotl = (nx * lx + ny * ly + nz * lz) / llen
        return 0.62f + 0.38f * maxOf(ndotl, 0f)
    }

    /** Normalized depth sample at grid pixel (u,v), or null when out of range. */
    private fun depthAt(
        u: Int, v: Int, tw: Int, th: Int,
        dw: Float, dh: Float, depth: DepthMap, interp: Boolean
    ): Float {
        if (u < 0 || u >= tw || v < 0 || v >= th) return Float.NaN
        val du = (u.toFloat() / tw * dw).coerceIn(0f, (depth.w - 1).toFloat())
        val dv = (v.toFloat() / th * dh).coerceIn(0f, (depth.h - 1).toFloat())
        return if (interp) sampleDepth(depth, du, dv) else depth.data[dv.toInt() * depth.w + du.toInt()]
    }

    /** bilinear sample of the depth map for smooth, non-blocky depth. */
    private fun sampleDepth(depth: DepthMap, x: Float, y: Float): Float {
        val x0 = x.toInt(); val y0 = y.toInt()
        val x1 = minOf(x0 + 1, depth.w - 1); val y1 = minOf(y0 + 1, depth.h - 1)
        val fx = x - x0; val fy = y - y0
        val d00 = depth.data[y0 * depth.w + x0]
        val d10 = depth.data[y0 * depth.w + x1]
        val d01 = depth.data[y1 * depth.w + x0]
        val d11 = depth.data[y1 * depth.w + x1]
        val top = d00 * (1 - fx) + d10 * fx
        val bot = d01 * (1 - fx) + d11 * fx
        return top * (1 - fy) + bot * fy
    }
}
