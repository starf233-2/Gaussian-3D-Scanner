package com.starfall.gaussianscanner.model

data class DepthMap(val data: FloatArray, val w: Int, val h: Int)
data class ColoredPt(val x: Float, val y: Float, val z: Float, val r: Float, val g: Float, val b: Float)
