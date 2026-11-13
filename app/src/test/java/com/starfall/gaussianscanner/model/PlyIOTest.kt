package com.starfall.gaussianscanner.model

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PlyIOTest {
    private fun ascii(rows: String, count: Int, extra: String = ""): ByteArrayInputStream =
        ("ply\nformat ascii 1.0\nelement vertex $count\nproperty float x\nproperty float y\nproperty float z\n$extra" +
            "end_header\n$rows").byteInputStream()

    @Test fun asciiByteColorsAndCoordinatesRoundTrip() {
        val cloud = PointCloud(floatArrayOf(1f, -2f, 3.5f, 1f, 0.5f, 0f))
        val output = ByteArrayOutputStream()
        PlyIO.write(cloud, output)
        val parsed = PlyIO.parse(ByteArrayInputStream(output.toByteArray()))
        assertEquals(1, parsed.size)
        assertArrayEquals(floatArrayOf(1f, -2f, 3.5f, 1f, 127f / 255f, 0f), parsed.data, 0f)
    }

    @Test fun binaryIntegerColorsAreNormalizedJustLikeAscii() {
        for (order in listOf(ByteOrder.LITTLE_ENDIAN, ByteOrder.BIG_ENDIAN)) {
            val format = if (order == ByteOrder.LITTLE_ENDIAN) "binary_little_endian" else "binary_big_endian"
            val header = "ply\nformat $format 1.0\nelement vertex 1\nproperty uchar blue\nproperty float z\nproperty float x\nproperty uchar red\nproperty float y\nproperty uchar green\nend_header\n"
            val row = ByteBuffer.allocate(15).order(order)
                .put(64.toByte()).putFloat(3f).putFloat(1f).put(200.toByte()).putFloat(2f).put(128.toByte()).array()
            val result = PlyIO.parse(ByteArrayInputStream(header.toByteArray() + row))
            assertArrayEquals(floatArrayOf(1f, 2f, 3f, 200f / 255f, 128f / 255f, 64f / 255f), result.data, 0f)
        }
    }

    @Test fun floatingColorsStayNormalizedAndMissingColorsAreWhite() {
        val result = PlyIO.parse(ascii("1 2 3 0.25 0.5 0.75\n", 1,
            "property float red\nproperty float green\nproperty float blue\n"))
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 0.25f, 0.5f, 0.75f), result.data, 0f)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f), PlyIO.parse(ascii("1 2 3\n", 1)).data, 0f)
    }

    @Test fun downsamplingIsBoundedBeforeParsingAndUsesCeilingStep() {
        val rows = (0 until 10).joinToString("") { "$it 2 3\n" }
        val result = PlyIO.parse(ascii(rows, 10), maxPoints = 3)
        assertEquals(3, result.size)
        assertEquals(18, result.data.size)
        assertEquals(0f, result.data[0], 0f)
        assertEquals(4f, result.data[6], 0f)
        assertEquals(8f, result.data[12], 0f)
    }

    @Test fun nonFiniteCoordinatesAreNeverReturned() {
        val result = PlyIO.parse(ascii("NaN 2 3\n1 Infinity 3\n1 2 3\n", 3))
        assertEquals(1, result.size)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f), result.data.copyOf(6), 0f)
    }

    @Test fun malformedRowsCannotConsumeFaceRowsAsVertices() {
        assertThrows(IllegalArgumentException::class.java) { PlyIO.parse(ascii("1 2\n3 0 1 2\n", 1)) }
    }

    @Test fun truncatedDataAndMissingHeaderTerminatorAreRejected() {
        assertThrows(EOFException::class.java) { PlyIO.parse(ascii("1 2 3\n", 2)) }
        assertThrows(EOFException::class.java) { PlyIO.parse("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n".byteInputStream()) }
    }

    @Test fun hostileCountsAndListsAndUnknownTypesAreRejectedBeforeAllocation() {
        assertThrows(IllegalArgumentException::class.java) { PlyIO.parse(ascii("", Int.MAX_VALUE)) }
        assertThrows(IllegalArgumentException::class.java) { PlyIO.parse(ascii("", 1, "property list uchar float bogus\n")) }
        assertThrows(IllegalArgumentException::class.java) { PlyIO.parse(ascii("", 1, "property mystery value\n")) }
        assertThrows(IllegalArgumentException::class.java) { PlyIO.parse(("ply\ncomment " + "a".repeat(70000)).byteInputStream()) }
    }

    @Test fun crlfAndFaceListAfterVerticesAreAccepted() {
        val text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n1 2 3\n3 0 1 2\n"
        assertEquals(1, PlyIO.parse(text.replace("\n", "\r\n").byteInputStream()).size)
    }

    @Test fun emptyCloudExportAndInvalidPackedCapacity() {
        val output = ByteArrayOutputStream()
        PlyIO.write(PointCloud.EMPTY, output)
        assertEquals(0, PlyIO.parse(ByteArrayInputStream(output.toByteArray())).size)
        assertThrows(IllegalArgumentException::class.java) { PointCloud(FloatArray(6), 2) }
    }
}
