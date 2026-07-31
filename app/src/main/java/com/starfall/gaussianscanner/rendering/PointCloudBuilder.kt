package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.ColoredPt
import com.starfall.gaussianscanner.model.DepthMap
import android.graphics.Bitmap

object PointCloudBuilder {

    fun build(bmp: Bitmap, depth: DepthMap, targetW: Int = 500): List<ColoredPt> {
        val bw = bmp.width; val bh = bmp.height
        val scale = targetW.toFloat() / bw
        val tw = targetW
        val th = (bh * scale).toInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(bmp, tw, th, true)
        val px = IntArray(tw * th); scaled.getPixels(px, 0, tw, 0, 0, tw, th)

        val focal = tw.coerceAtLeast(th).toFloat() * 1.1f
        val cx = tw / 2f; val cy = th / 2f
        val dw = depth.w.toFloat(); val dh = depth.h.toFloat()

        val pts = ArrayList<ColoredPt>(tw * th)

        for (v in 0 until th) {
            for (u in 0 until tw) {
                val pixel = px[v * tw + u]
                val r = ((pixel shr 16) and 0xFF) / 255f
                val g = ((pixel shr 8) and 0xFF) / 255f
                val b = (pixel and 0xFF) / 255f

                val du = (u.toFloat() / tw * dw).toInt().coerceIn(0, depth.w - 1)
                val dv = (v.toFloat() / th * dh).toInt().coerceIn(0, depth.h - 1)
                val d = depth.data[dv * depth.w + du] * 3f + 0.3f

                val wx = (u - cx) * d / focal
                val wy = (cy - v) * d / focal
                val wz = d

                pts.add(ColoredPt(wx, wy, wz, r, g, b))
            }
        }

        return pts
    }
}
