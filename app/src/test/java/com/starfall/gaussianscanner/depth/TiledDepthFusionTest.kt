package com.starfall.gaussianscanner.depth

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class TiledDepthFusionTest {
    @Test fun doubleResolutionUsesNineEqualTilesAndCoversEveryPixel() {
        val plan = TiledDepthFusion.plan(518, 518, 1536, 2048)
        assertEquals(1036, plan.w); assertEquals(1036, plan.h)
        assertEquals(9, plan.tiles.size)
        assertEquals(listOf(0, 259, 518), plan.tiles.map { it.x }.distinct())
        assertTrue(plan.tiles.all { it.w == 518 && it.h == 518 })
        val coverage = IntArray(plan.w * plan.h)
        plan.tiles.forEach { t ->
            for (y in t.y until t.y + t.h) for (x in t.x until t.x + t.w) coverage[y * plan.w + x]++
        }
        assertTrue(coverage.all { it > 0 })
    }

    @Test fun sourcePixelsLimitResolutionInsteadOfInventingAnUpscaledInput() {
        assertTrue(TiledDepthFusion.plan(518, 518, 300, 1000).tiles.isEmpty())
        val plan = TiledDepthFusion.plan(518, 518, 800, 1200)
        assertEquals(800, plan.w); assertEquals(800, plan.h)
        assertTrue(plan.tiles.all { it.x + it.w <= plan.w && it.y + it.h <= plan.h })
    }

    @Test fun rectangularModelAndOddSizesHaveSymmetricCoverage() {
        val plan = TiledDepthFusion.plan(65, 47, 200, 200)
        assertEquals(130, plan.w); assertEquals(94, plan.h)
        assertEquals(9, plan.tiles.size)
        assertTrue(plan.tiles.all { it.w == 65 && it.h == 47 })
        assertEquals(65, plan.tiles.maxOf { it.x })
        assertEquals(47, plan.tiles.maxOf { it.y })
        val narrow = TiledDepthFusion.plan(2, 518, 3, 519)
        assertTrue(narrow.tiles.isNotEmpty())
        assertTrue(narrow.tiles.all { it.x == 0 && it.w == 2 })
    }

    private val plan = TiledDepthFusion.plan(128, 128, 256, 256)
    private fun reference(): FloatArray = FloatArray(256 * 256) { j -> 1f + j % 256 * .003f + j / 256 * .001f }
    private fun crop(src: FloatArray, t: TiledDepthFusion.Tile): FloatArray =
        FloatArray(t.w * t.h) { j -> src[(t.y + j / t.w) * 256 + t.x + j % t.w] }

    @Test fun perTileScaleAndOffsetAreRemovedWithoutSeams() {
        val ref = reference()
        val result = TiledDepthFusion.fuse(ref, plan) { tile, index ->
            val scale = .4f + index * .35f; val shift = index * 2f - 3f
            crop(ref, tile).map { it * scale + shift }.toFloatArray()
        }
        assertEquals(9, result.accepted)
        assertArrayEquals(ref, result.data, .00001f)
        assertEquals(0f, result.changedFraction, 0f)
    }

    @Test fun detailAbsentFromGlobalReferenceSurvivesAtMeasurableAmplitude() {
        val ref = reference()
        val detail = FloatArray(ref.size) { j -> .025f * sin((j % 256) * 2f * PI.toFloat() / 12f) * cos((j / 256) * 2f * PI.toFloat() / 16f) }
        val truth = FloatArray(ref.size) { ref[it] + detail[it] }
        val result = TiledDepthFusion.fuse(ref, plan) { tile, index ->
            crop(truth, tile).map { it * (.7f + index * .1f) + index }.toFloatArray()
        }
        var dot = 0.0; var energy = 0.0; var squaredError = 0.0
        for (y in 24 until 232) for (x in 24 until 232) {
            val j = y * 256 + x
            dot += (result.data[j] - ref[j]) * detail[j]
            energy += detail[j] * detail[j]
            squaredError += (result.data[j] - truth[j]).pow(2)
        }
        assertEquals(9, result.accepted)
        assertTrue("detail amplitude ${dot / energy}", dot / energy > .75)
        assertTrue("detail error ${squaredError / energy}", squaredError / energy < .15)
        assertTrue(result.changedFraction > .5f)
        println("Synthetic fine-detail amplitude retained=${dot / energy}; relative squared error=${squaredError / energy}")
    }

    @Test fun tileEdgesCanSharpenRatherThanBeingSuppressed() {
        val ref = FloatArray(256 * 256) { j ->
            val x = j % 256
            1f + j / 256 * .001f + .5f * ((x - 116f) / 24f).coerceIn(0f, 1f)
        }
        val truth = FloatArray(ref.size) { j -> 1f + j / 256 * .001f + if (j % 256 >= 128) .5f else 0f }
        val result = TiledDepthFusion.fuse(ref, plan) { tile, _ -> crop(truth, tile) }
        val y = 120
        val originalJump = ref[y * 256 + 129] - ref[y * 256 + 126]
        val fusedJump = result.data[y * 256 + 129] - result.data[y * 256 + 126]
        assertTrue("edge original=$originalJump fused=$fusedJump", fusedJump > originalJump * 3)
        assertTrue(result.data.all { it.isFinite() && it in .8f..2f })
        println("Synthetic edge contrast: global=$originalJump, fused=$fusedJump")
    }

    @Test fun oneFailedTileDoesNotDiscardOtherTiles() {
        val ref = reference()
        val truth = FloatArray(ref.size) { ref[it] + .02f * sin((it % 256) * .5f) }
        val result = TiledDepthFusion.fuse(ref, plan) { tile, index -> if (index == 0) null else crop(truth, tile) }
        assertEquals(8, result.accepted); assertEquals(1, result.rejected)
        assertEquals(ref[0], result.data[0], 0f)
        assertTrue(result.changedFraction > .25f)
    }

    @Test fun collapsedInvertedInvalidOrMissingPredictionsRetainGlobal() {
        val ref = reference()
        for (kind in 0..3) {
            val result = TiledDepthFusion.fuse(ref, plan) { tile, _ -> when (kind) {
                0 -> FloatArray(tile.w * tile.h) { 1f }
                1 -> crop(ref, tile).map { -it }.toFloatArray()
                2 -> FloatArray(tile.w * tile.h) { Float.NaN }
                else -> null
            } }
            assertEquals(0, result.accepted)
            assertArrayEquals(ref, result.data, 0f)
        }
    }

    @Test fun identicalOutputIsNotReportedAsDetailJustBecauseGridIsBigger() {
        val coarse = FloatArray(128 * 128) { it % 128 * .01f }
        val ref = TiledDepthFusion.upsample(coarse, 128, 128, 256, 256)
        val result = TiledDepthFusion.fuse(ref, plan) { tile, _ -> crop(ref, tile) }
        assertEquals(0f, result.changedFraction, 0f)
        assertEquals(0f, result.meanChange, .000001f)
    }

    @Test fun isolatedTileSpikesAreRemovedButContinuousEdgesRemain() {
        val ref = reference()
        val result = TiledDepthFusion.fuse(ref, plan) { tile, _ ->
            crop(ref, tile).also { raw ->
                val x = 100 - tile.x; val y = 100 - tile.y
                if (x in 0 until tile.w && y in 0 until tile.h) raw[y * tile.w + x] += .2f
            }
        }
        assertEquals(ref[100*256+100], result.data[100*256+100], .005f)
    }
}
