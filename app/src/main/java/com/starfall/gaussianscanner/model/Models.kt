package com.starfall.gaussianscanner.model

data class DepthMap(val data: FloatArray, val w: Int, val h: Int)
/** Interleaved xyz/rgb, without one JVM object and reference per point. */
class PointCloud(val data: FloatArray, val size: Int = data.size / STRIDE, val surface: SurfaceGrid? = null) {
    init {
        require(size >= 0 && size.toLong() * STRIDE <= data.size)
        require(surface == null || surface.slopes.size.toLong() == size.toLong() * 2)
    }

    companion object {
        const val STRIDE = 6
        val EMPTY = PointCloud(FloatArray(0))
    }
}

/** Original camera-grid geometry for projected per-point footprints. Two
 * signed-normalized slopes cost 4 bytes/point, not six extra float vectors. */
class SurfaceGrid(val stepOverFocal: Float, val slopes: ShortArray) {
    init { require(stepOverFocal.isFinite() && stepOverFocal > 0f) }
}
