package com.starfall.gaussianscanner.rendering

import kotlin.math.*

internal object CloudSampling {
    const val MAX_POINTS = 1_000_000

    /** Smoothly cap grid dimensions rather than jumping gridStep 1 -> 2 and
     * discarding 75% of points just above one million. */
    fun dimensions(width: Int, height: Int, requestedWidth: Int): Pair<Int, Int> {
        require(width > 0 && height > 0 && requestedWidth > 0)
        val aspect = width.toDouble() / height
        val w = minOf(requestedWidth, sqrt(MAX_POINTS * aspect).toInt().coerceAtLeast(1), MAX_POINTS)
        val h = minOf((w / aspect).toInt().coerceAtLeast(1), MAX_POINTS / w)
        return w to h
    }

    /** High-resolution depth needs a correspondingly fine display grid, or
     * the new detail is immediately undersampled by the ordinary 250k preset. */
    fun highResolutionWidth(width: Int, height: Int, requestedWidth: Int, depthW: Int, depthH: Int): Int {
        val count = minOf(MAX_POINTS.toLong(), depthW.toLong() * depthH)
        return maxOf(requestedWidth, sqrt(count * width.toDouble() / height).toInt().coerceAtLeast(1))
    }
}
