package com.starfall.gaussianscanner.rendering

import com.starfall.gaussianscanner.model.PointCloud
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class PointFootprintTest {
    private val position = floatArrayOf(0f, 0f, -2f)
    private val center = floatArrayOf(0f, 0f, -2f)
    // Standard perspective (50deg FOV) * camera-at-distance view * y rotation.
    private fun mvp(distance: Float, aspect: Float = 1f, yaw: Float = 0f): FloatArray {
        val p = 1f / tan(Math.toRadians(25.0)).toFloat()
        val c = cos(yaw); val s = sin(yaw)
        return floatArrayOf(p/aspect*c, 0f, s, s, 0f, p, 0f, 0f, p/aspect*s, 0f, -c, -c,
            0f, 0f, distance - .2f, distance)
    }
    private fun size(distance: Float, width: Int = 1080, height: Int = 1080, step: Float = .005f,
                     yaw: Float = 0f, p: FloatArray = position): Float =
        PointFootprint.projectedSize(p, center, mvp(distance, width.toFloat()/height, yaw), width, height, step)

    @Test fun sizeMatchesPixelSpacingFromPerspectiveProjection() {
        val expected = (1080f / (2 * tan(Math.toRadians(25.0)))) * (.005 * 2) / 3 * PointFootprint.COVERAGE
        assertEquals(expected.toFloat(), size(3f), .00001f)
    }
    @Test fun dollyZoomUsesInverseDistanceContinuouslyWithoutOldAnchorCliffs() {
        assertEquals(size(3f) * 2, size(1.5f), .00001f)
        val values = (10..100).map { size(it / 10f) }
        assertTrue(values.zipWithNext().all { (a, b) -> a >= b })
        assertTrue(abs(size(2.399f) - size(2.4f)) < .01f)
    }
    @Test fun densityViewportAndFocalLengthChangeFootprintCorrectly() {
        assertEquals(size(3f) * 2, size(3f, height = 2160, width = 2160), .00001f)
        assertEquals(size(3f) / 2, size(3f, step = .0025f), .00001f)
        assertEquals(size(3f), size(3f, width = 1920, height = 1080), .00001f)
    }
    @Test fun everyPointUsesItsOwnViewDepthAndRotation() {
        val point = floatArrayOf(1f, 0f, -2f)
        assertTrue(size(3f, yaw = -.6f, p = point) > size(3f, yaw = .6f, p = point))
        assertNotEquals(size(3f, p = floatArrayOf(0f, 0f, -1f)), size(3f, p = floatArrayOf(0f, 0f, -3f)))
    }
    @Test fun occlusionSlopesNeverBridgeTheDepthJump() {
        assertEquals(0.toShort(), PointFootprint.slope(1f, 1f, 8f, .01f))
        assertEquals(0.toShort(), PointFootprint.slope(1f, Float.NaN, 8f, .01f))
        assertEquals(0.toShort(), PointFootprint.slope(1f, 8f, 8f, .01f))
        val slope = PointFootprint.slope(1f, .99f, 1.01f, .01f)
        assertEquals(1f, slope / 32767f * PointFootprint.SLOPE_RANGE, .0002f)
    }
    @Test fun projectedSlopedSurfaceMatchesNeighbourFiniteDifferences() {
        val p = floatArrayOf(.8f, .3f, -2f)
        val sx = PointFootprint.slope(2f, 1.99f, 2.01f, .01f)
        val sy = PointFootprint.slope(2f, 2.005f, 1.995f, .01f)
        val matrix = mvp(3f, 1f, .4f)
        val predicted = PointFootprint.projectedSize(p, center, matrix, 1080, 1080, .005f, sx, sy)
        fun project(v: FloatArray): FloatArray {
            val x=v[0]-center[0]; val y=v[1]-center[1]; val z=v[2]-center[2]
            val w=matrix[3]*x+matrix[7]*y+matrix[11]*z+matrix[15]
            return floatArrayOf((matrix[0]*x+matrix[4]*y+matrix[8]*z+matrix[12])/w*540,
                (matrix[1]*x+matrix[5]*y+matrix[9]*z+matrix[13])/w*540)
        }
        val dx=sx/32767f*4*.01f; val dy=sy/32767f*4*.01f
        val tx=floatArrayOf(.01f+p[0]/2*dx,p[1]/2*dx,-dx)
        val ty=floatArrayOf(p[0]/2*dy,.01f+p[1]/2*dy,-dy)
        fun delta(t: FloatArray): FloatArray {
            val a=project(FloatArray(3){p[it]+t[it]*.5f})
            val b=project(FloatArray(3){p[it]-t[it]*.5f})
            return floatArrayOf(abs(a[0]-b[0]),abs(a[1]-b[1]))
        }
        val a=delta(tx); val b=delta(ty)
        assertEquals(maxOf(a[0]+b[0],a[1]+b[1])*PointFootprint.COVERAGE,predicted,.002f)
    }
    @Test fun nearPlaneAndHardwareLimitsStayFinite() {
        assertEquals(1f, size(.05f), 0f)
        assertEquals(64f, size(.101f, step = .5f), 0f)
    }
    @Test fun unorderedPlySpacingUsesGeometryNotPointOrder() {
        val vertices = (0 until 100).map { j -> floatArrayOf((j%10)*.1f,(j/10)*.1f,0f,1f,1f,1f) }
        fun cloud(points: List<FloatArray>) = PointCloud(points.flatMap { it.toList() }.toFloatArray())
        assertEquals(.1f, PointFootprint.estimateImportedSpacing(cloud(vertices.shuffled(kotlin.random.Random(10)))),.00001f)
        assertTrue(PointFootprint.estimateImportedSpacing(PointCloud.EMPTY).isFinite())
    }
    @Test fun highResolutionDisplayDoesNotFallBackToQuarterDensityAtCap() {
        val width = CloudSampling.highResolutionWidth(1200, 1600, 433, 1036, 1036)
        val (w,h) = CloudSampling.dimensions(1200,1600,width)
        assertTrue(w.toLong()*h in 990_000L..1_000_000L)
        val counts = (850..910).map { CloudSampling.dimensions(1200,1600,it).let { p -> p.first*p.second } }
        assertTrue(counts.zipWithNext().all { (a,b) -> b >= a })
        assertTrue(CloudSampling.dimensions(1, 100000, 100).let { it.first.toLong()*it.second } <= 1_000_000L)
    }
    @Test fun largeOrderedPlyDoesNotAliasIntoASingleColumn() {
        val n = 128 * 128
        val points = FloatArray(n * 6)
        for (j in 0 until n) { points[j*6] = (j%128)*.1f; points[j*6+1] = (j/128)*.1f }
        val spacing = PointFootprint.estimateImportedSpacing(PointCloud(points))
        assertTrue("estimated spacing=$spacing", spacing in .07f.. .15f)
    }
}
