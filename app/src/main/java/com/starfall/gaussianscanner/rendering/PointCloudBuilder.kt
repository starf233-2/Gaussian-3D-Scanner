package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.ColoredPt
import com.starfall.gaussianscanner.model.DepthMap
import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

object PointCloudBuilder {

    /**
     * @param focalPx real camera focal length in pixels at the ORIGINAL image resolution (0 = estimate).
     * @param interp true = bilinear depth sampling, false = nearest neighbor.
     */
    fun build(
        bmp: Bitmap, depth: DepthMap,
        targetW: Int = 500,
        maxPoints: Int = Int.MAX_VALUE,
        edgeTrim: Float = 0f,
        focalPx: Float = 0f,
        interp: Boolean = true
    ): List<ColoredPt> {
        val bw = bmp.width; val bh = bmp.height
        val scale = targetW.toFloat() / bw
        val tw = targetW
        val th = (bh * scale).toInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(bmp, tw, th, true)
        val px = IntArray(tw * th); scaled.getPixels(px, 0, tw, 0, 0, tw, th)

        // uniform grid sampling: every gridStep-th row and column
        var gridStep = 1
        val total = tw.toLong() * th
        if (maxPoints < total) {
            gridStep = ceil(sqrt(total.toDouble() / maxPoints)).toInt().coerceAtLeast(1)
        }

        // depth gradient mask for edge trimming
        val edgeMask = FloatArray(depth.w * depth.h)
        val trimEnabled = edgeTrim > 0.001f
        if (trimEnabled) {
            for (v in 1 until depth.h - 1) {
                for (u in 1 until depth.w - 1) {
                    val idx = v * depth.w + u
                    val gx = abs(depth.data[idx + 1] - depth.data[idx - 1])
                    val gy = abs(depth.data[idx + depth.w] - depth.data[idx - depth.w])
                    edgeMask[idx] = maxOf(gx, gy)
                }
            }
        }
        val trimThresh = 1.0f - (1.0f - 0.04f) * edgeTrim

        // focal: use real intrinsic if provided, else estimate.
        // scale from original image resolution down to the working resolution.
        val focal = if (focalPx > 0f) focalPx * (tw.toFloat() / bw) else tw.coerceAtLeast(th).toFloat() * 1.1f
        val cx = tw / 2f; val cy = th / 2f
        val dw = depth.w.toFloat(); val dh = depth.h.toFloat()

        val pts = ArrayList<ColoredPt>((tw / gridStep) * (th / gridStep))

        for (v in 0 until th step gridStep) {
            for (u in 0 until tw step gridStep) {
                val du = (u.toFloat() / tw * dw).coerceIn(0f, (depth.w - 1).toFloat())
                val dv = (v.toFloat() / th * dh).coerceIn(0f, (depth.h - 1).toFloat())
                val d = if (interp) sampleDepth(depth, du, dv)
                        else depth.data[dv.toInt() * depth.w + du.toInt()]

                if (trimEnabled) {
                    val di = dv.toInt() * depth.w + du.toInt()
                    if (edgeMask[di] > trimThresh) continue
                }

                val pixel = px[v * tw + u]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val z = d * 3f + 0.3f

                val wx = (u - cx) * z / focal
                val wy = (cy - v) * z / focal
                val wz = -z

                pts.add(ColoredPt(wx, wy, wz, r, g, b))
            }
        }

        return pts
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
