package com.starfall.gaussianscanner.depth

import kotlin.math.*

/** Model-independent, streaming fusion. Tiles have EXACTLY the model input
 * dimensions: no differently sized crops stretched back to the same tensor.
 * Fits scale/offset against pooled global depth, corrects a coarse bias field,
 * then blends complete aligned depths. No edge suppression/high-pass gain.
 */
internal object TiledDepthFusion {
    data class Tile(val x: Int, val y: Int, val w: Int, val h: Int)
    data class Plan(val w: Int, val h: Int, val tiles: List<Tile>)
    data class Result(val data: FloatArray, val accepted: Int, val rejected: Int,
                      val changedFraction: Float, val meanChange: Float)

    fun plan(modelW: Int, modelH: Int, sourceW: Int, sourceH: Int): Plan {
        require(modelW > 1 && modelH > 1 && sourceW > 0 && sourceH > 0)
        val factor = minOf(2.0, sourceW.toDouble() / modelW, sourceH.toDouble() / modelH).coerceAtLeast(1.0)
        val w = (modelW * factor).toInt(); val h = (modelH * factor).toInt()
        if (w == modelW && h == modelH) return Plan(w, h, emptyList())
        fun starts(size: Int, tile: Int): IntArray {
            if (size == tile) return intArrayOf(0)
            val count = ceil((size - tile) / (tile * 0.75)).toInt() + 1
            return IntArray(count) { j -> (j.toDouble() * (size - tile) / (count - 1)).roundToInt() }
        }
        val xs = starts(w, modelW); val ys = starts(h, modelH)
        return Plan(w, h, ys.flatMap { y -> xs.map { x -> Tile(x, y, modelW, modelH) } })
    }

    fun upsample(src: FloatArray, w: Int, h: Int, outW: Int, outH: Int): FloatArray {
        require(src.size == w * h && w > 0 && h > 0 && outW > 0 && outH > 0)
        return FloatArray(outW * outH) { i ->
            val x = ((i % outW + .5f) * w / outW - .5f).coerceIn(0f, (w - 1).toFloat())
            val y = ((i / outW + .5f) * h / outH - .5f).coerceIn(0f, (h - 1).toFloat())
            sample(src, w, h, x, y)
        }
    }

    /** [infer] supplies one raw model result at a time; null/invalid tiles only
     * lose their own contribution. Rejected areas retain the global reference.
     */
    fun fuse(reference: FloatArray, plan: Plan, infer: (Tile, Int) -> FloatArray?): Result {
        require(reference.size == plan.w * plan.h && reference.all { it.isFinite() })
        val sorted = reference.sortedArray()
        val span = maxOf(1e-6f, sorted[(sorted.size * .99f).toInt().coerceAtMost(sorted.lastIndex)] - sorted[(sorted.size * .01f).toInt()])
        val sum = FloatArray(reference.size)
        val weights = FloatArray(reference.size)
        val squares = FloatArray(reference.size)
        var accepted = 0
        for ((index, tile) in plan.tiles.withIndex()) {
            val raw = infer(tile, index) ?: continue
            if (raw.size != tile.w * tile.h || raw.any { !it.isFinite() }) continue
            val target = FloatArray(raw.size) { j -> reference[(tile.y + j / tile.w) * plan.w + tile.x + j % tile.w] }
            val fit = align(raw, target, tile.w, tile.h, span) ?: continue
            val aligned = FloatArray(raw.size) { j ->
                var value = raw[j]
                val x = j % tile.w; val y = j / tile.w
                if (x in 1 until tile.w - 1 && y in 1 until tile.h - 1) {
                    val low = minOf(minOf(raw[j-1], raw[j+1]), minOf(raw[j-tile.w], raw[j+tile.w]))
                    val high = maxOf(maxOf(raw[j-1], raw[j+1]), maxOf(raw[j-tile.w], raw[j+tile.w]))
                    val allowance = maxOf(4f * (high - low), span / fit.first * .03f)
                    if (value < low - allowance || value > high + allowance) value = (low + high) * .5f
                }
                value * fit.first + fit.second
            }
            // Low frequency bias anchors scene shape; preserve full tile edges
            // and medium/fine detail instead of injecting only a tiny high-pass.
            val bias = coarseBias(aligned, target, tile.w, tile.h)
            val featherX = maxOf(1, tile.w / 4)
            val featherY = maxOf(1, tile.h / 4)
            for (y in 0 until tile.h) for (x in 0 until tile.w) {
                val j = y * tile.w + x
                val global = (tile.y + y) * plan.w + tile.x + x
                val residual = aligned[j] - bias[j] - target[j]
                // Gross disagreement is not useful detail. This is a wide
                // reference-relative gate, not the previous ±4% detail clamp.
                if (!residual.isFinite() || abs(residual) > span * .35f) continue
                val wx = edgeWeight(x, tile.w, featherX, tile.x == 0, tile.x + tile.w == plan.w)
                val wy = edgeWeight(y, tile.h, featherY, tile.y == 0, tile.y + tile.h == plan.h)
                val weight = wx * wy
                sum[global] += residual * weight
                squares[global] += residual * residual * weight
                weights[global] += weight
            }
            accepted++
        }
        var change = 0.0; var changed = 0
        val out = FloatArray(reference.size) { j ->
            val weight = weights[j]
            var residual = if (weight > 1e-8f) sum[j] / weight else 0f
            if (weight > 1e-8f) {
                val variance = maxOf(0f, squares[j] / weight - residual * residual)
                val disagreement = variance / (span * span * .06f * .06f)
                residual *= 1f / (1f + disagreement)
                // With a missing neighbour, feather into the global reference
                // instead of dividing a tiny border weight back to full strength.
                residual *= minOf(1f, weight)
            }
            val delta = abs(residual) / span
            change += delta
            if (delta > .002f) changed++
            reference[j] + residual
        }
        return Result(out, accepted, plan.tiles.size - accepted, changed.toFloat() / out.size, (change / out.size).toFloat())
    }

    private fun edgeWeight(pos: Int, size: Int, feather: Int, first: Boolean, last: Boolean): Float {
        val distance = minOf(if (first) 1f else pos.toFloat() / feather,
            if (last) 1f else (size - 1 - pos).toFloat() / feather).coerceIn(0f, 1f)
        return distance * distance * (3f - 2f * distance)
    }

    /** Robust affine calibration from block MEANS, not high frequency pixels.
     * Positive scale only; inverted/unrelated/collapsed predictions rejected. */
    private fun align(raw: FloatArray, ref: FloatArray, w: Int, h: Int, span: Float): Pair<Float, Float>? {
        val cell = maxOf(2, minOf(w, h) / 16)
        val a = ArrayList<Double>(); val b = ArrayList<Double>()
        for (y in 0 until h step cell) for (x in 0 until w step cell) {
            var sa = 0.0; var sb = 0.0; var n = 0
            for (yy in y until minOf(h, y + cell)) for (xx in x until minOf(w, x + cell)) {
                sa += raw[yy * w + xx]; sb += ref[yy * w + xx]; n++
            }
            a.add(sa / n); b.add(sb / n)
        }
        var scale = 1.0; var offset = 0.0
        val weights = DoubleArray(a.size) { 1.0 }
        repeat(3) {
            var sw = 0.0; var sx = 0.0; var sy = 0.0
            for (j in a.indices) { sw += weights[j]; sx += a[j] * weights[j]; sy += b[j] * weights[j] }
            val mx = sx / sw; val my = sy / sw
            var vx = 0.0; var vy = 0.0; var covariance = 0.0
            for (j in a.indices) {
                val dx = a[j] - mx; val dy = b[j] - my
                vx += weights[j] * dx * dx; vy += weights[j] * dy * dy; covariance += weights[j] * dx * dy
            }
            if (vy / sw < span * span * 1e-6) {
                // A textureless global patch cannot determine a scale. Keep
                // unit scale only if tile structure itself remains modest.
                if (vx / sw > span * span * .02) return null
                scale = 1.0
            } else {
                if (vx <= 1e-18 || covariance <= 0 || covariance / sqrt(vx * vy) < .45) return null
                scale = covariance / vx
                if (!scale.isFinite() || scale !in .02..50.0) return null
            }
            offset = my - scale * mx
            val errors = DoubleArray(a.size) { abs(scale * a[it] + offset - b[it]) }
            val median = errors.sortedArray()[errors.size / 2]
            val threshold = maxOf(span * .01, 2.5 * median)
            for (j in weights.indices) weights[j] = minOf(1.0, threshold / maxOf(1e-12, errors[j]))
        }
        if (!offset.isFinite()) return null
        return scale.toFloat() to offset.toFloat()
    }

    /** Small bias lattice, bilinearly interpolated: no megapixel integral
     * images and no smoothing of the actual high-resolution depth edges. */
    private fun coarseBias(aligned: FloatArray, ref: FloatArray, w: Int, h: Int): FloatArray {
        val step = maxOf(4, minOf(w, h) / 8)
        val cols = (w - 1 + step - 1) / step + 1
        val rows = (h - 1 + step - 1) / step + 1
        val bias = FloatArray(cols * rows)
        for (gy in 0 until rows) for (gx in 0 until cols) {
            val cx = gx * step; val cy = gy * step
            var sum = 0.0; var n = 0
            for (y in maxOf(0, cy - step) until minOf(h, cy + step + 1)) {
                for (x in maxOf(0, cx - step) until minOf(w, cx + step + 1)) {
                    val j = y * w + x
                    sum += aligned[j] - ref[j]; n++
                }
            }
            bias[gy * cols + gx] = if (n > 0) (sum / n).toFloat() else 0f
        }
        return FloatArray(w * h) { j -> sample(bias, cols, rows, (j % w).toFloat() / step, (j / w).toFloat() / step) }
    }

    private fun sample(src: FloatArray, w: Int, h: Int, x: Float, y: Float): Float {
        val x0 = x.toInt().coerceIn(0, w - 1); val y0 = y.toInt().coerceIn(0, h - 1)
        val x1 = minOf(x0 + 1, w - 1); val y1 = minOf(y0 + 1, h - 1)
        val ax = x - x0; val ay = y - y0
        return (src[y0 * w + x0] * (1 - ax) + src[y0 * w + x1] * ax) * (1 - ay) +
            (src[y1 * w + x0] * (1 - ax) + src[y1 * w + x1] * ax) * ay
    }
}
