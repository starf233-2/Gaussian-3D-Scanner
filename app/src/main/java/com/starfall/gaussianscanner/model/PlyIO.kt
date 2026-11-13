package com.starfall.gaussianscanner.model

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.StringTokenizer

/** Bounded streaming PLY reader. No full-file allocation before downsampling. */
object PlyIO {
    private const val MAX_HEADER_BYTES = 64 * 1024
    private const val MAX_ROW_BYTES = 64 * 1024
    const val DEFAULT_MAX_POINTS = 1_500_000
    data class Header(val format: String, val vertexCount: Int, val props: List<Pair<String, String>>)

    fun export(points: PointCloud, file: File): Boolean = try {
        file.outputStream().buffered().use { write(points, it) }
        true
    } catch (_: Exception) { false }

    fun write(points: PointCloud, output: OutputStream) {
        val header = "ply\nformat ascii 1.0\nelement vertex ${points.size}\n" +
            "property float x\nproperty float y\nproperty float z\n" +
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        output.write(header.toByteArray(StandardCharsets.US_ASCII))
        val text = StringBuilder(64 * 1024)
        for (i in 0 until points.size) {
            val k = i * PointCloud.STRIDE
            text.append(points.data[k]).append(' ').append(points.data[k + 1]).append(' ')
                .append(points.data[k + 2]).append(' ')
            for (c in 3..5) {
                text.append((points.data[k + c] * 255f).toInt().coerceIn(0, 255))
                    .append(if (c == 5) '\n' else ' ')
            }
            if (text.length >= 64 * 1024) {
                output.write(text.toString().toByteArray(StandardCharsets.US_ASCII)); text.setLength(0)
            }
        }
        if (text.isNotEmpty()) output.write(text.toString().toByteArray(StandardCharsets.US_ASCII))
    }

    fun parse(input: InputStream, maxPoints: Int = DEFAULT_MAX_POINTS): PointCloud {
        require(maxPoints in 1..DEFAULT_MAX_POINTS)
        val stream = if (input is BufferedInputStream) input else BufferedInputStream(input)
        val header = readHeader(stream)
        val props = header.props
        val xyz = intArrayOf(props.indexOfFirst { it.first == "x" }, props.indexOfFirst { it.first == "y" }, props.indexOfFirst { it.first == "z" })
        require(xyz.all { it >= 0 }) { "PLY缺少xyz坐标" }
        val colors = arrayOf(setOf("red", "r", "diffuse_red"), setOf("green", "g", "diffuse_green"), setOf("blue", "b", "diffuse_blue"))
        val rgb = IntArray(3) { c -> props.indexOfFirst { it.first in colors[c] } }
        val colorDivisors = FloatArray(3) { c ->
            val idx = rgb[c]
            if (idx < 0 || props[idx].second in setOf("float", "float32", "double", "float64")) 1f else 255f
        }
        val step = ((header.vertexCount.toLong() + maxPoints - 1) / maxPoints).coerceAtLeast(1).toInt()
        val capacity = ((header.vertexCount.toLong() + step - 1) / step).toInt()
        val data = FloatArray(capacity * PointCloud.STRIDE)
        val values = FloatArray(props.size)
        val offsets = IntArray(props.size)
        var rowBytes = 0
        for (j in props.indices) { offsets[j] = rowBytes; rowBytes += typeSize(props[j].second) }
        require(rowBytes <= MAX_ROW_BYTES)
        val raw = ByteArray(rowBytes)
        val buffer = ByteBuffer.wrap(raw).order(if (header.format == "binary_big_endian") ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN)
        var written = 0
        for (i in 0 until header.vertexCount) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException("PLY导入已取消")
            val selected = i % step == 0
            if (header.format == "ascii") {
                var line = readLine(stream, MAX_ROW_BYTES) ?: throw EOFException("PLY顶点数据不完整")
                while (line.isBlank()) line = readLine(stream, MAX_ROW_BYTES) ?: throw EOFException("PLY顶点数据不完整")
                if (!selected) continue
                val tokens = StringTokenizer(line)
                for (j in props.indices) {
                    require(tokens.hasMoreTokens()) { "PLY顶点属性数量不足" }
                    values[j] = tokens.nextToken().toFloat()
                }
            } else {
                var read = 0
                while (read < rowBytes) {
                    val n = stream.read(raw, read, rowBytes - read)
                    if (n < 0) throw EOFException("PLY顶点数据不完整")
                    if (n == 0) { raw[read++] = stream.read().also { if (it < 0) throw EOFException() }.toByte() }
                    else read += n
                }
                if (!selected) continue
                for (j in props.indices) values[j] = readValue(buffer, offsets[j], props[j].second)
            }
            // Never feed NaN/Inf into centroid sums, matrices or GPU positions.
            if (xyz.any { !values[it].isFinite() }) continue
            val k = written++ * PointCloud.STRIDE
            for (c in 0..2) {
                data[k + c] = values[xyz[c]]
                val idx = rgb[c]
                val value = if (idx < 0) 1f else {
                    values[idx] / colorDivisors[c]
                }
                data[k + 3 + c] = if (value.isFinite()) value.coerceIn(0f, 1f) else 1f
            }
        }
        return if (written == 0) PointCloud.EMPTY else {
            val used = written * PointCloud.STRIDE
            // A mostly-invalid cloud should not retain its entire header-sized
            // reservation just to display a handful of valid points.
            PointCloud(if (used < data.size / 2) data.copyOf(used) else data, written)
        }
    }

    private fun readHeader(stream: BufferedInputStream): Header {
        require(readLine(stream, MAX_HEADER_BYTES) == "ply") { "不是PLY文件" }
        var format = ""
        var count = -1
        var inVertex = false
        var bytes = 4
        var vertexSeen = false
        val props = ArrayList<Pair<String, String>>()
        while (true) {
            val line = readLine(stream, MAX_HEADER_BYTES - bytes) ?: throw EOFException("PLY缺少end_header")
            bytes += line.length + 1
            require(bytes <= MAX_HEADER_BYTES) { "PLY文件头过大" }
            val parts = line.trim().split(Regex("\\s+"))
            when (parts[0]) {
                "end_header" -> break
                "format" -> {
                    require(parts.size == 3 && parts[2] == "1.0")
                    format = parts[1]
                    require(format in setOf("ascii", "binary_little_endian", "binary_big_endian")) { "不支持的PLY格式" }
                }
                "element" -> {
                    require(parts.size == 3)
                    val n = parts[2].toInt()
                    require(n in 0..50_000_000) { "PLY元素数量过大或无效" }
                    inVertex = parts[1] == "vertex"
                    if (inVertex) {
                        require(!vertexSeen) { "重复的vertex元素" }
                        count = n; vertexSeen = true
                    } else require(vertexSeen || n == 0) { "暂不支持vertex前含数据的PLY元素" }
                }
                "property" -> if (inVertex) {
                    require(parts.size == 3 && parts[1] != "list") { "暂不支持vertex的list属性" }
                    typeSize(parts[1])
                    require(props.size < 128 && props.none { it.first == parts[2] }) { "PLY属性过多或重复" }
                    props.add(parts[2] to parts[1])
                }
            }
        }
        require(format.isNotEmpty() && count >= 0 && props.isNotEmpty()) { "PLY文件头无效" }
        return Header(format, count, props)
    }

    private fun readLine(stream: InputStream, limit: Int): String? {
        require(limit > 0) { "PLY行或文件头过大" }
        val line = StringBuilder()
        while (true) {
            val b = stream.read()
            if (b < 0) return if (line.isEmpty()) null else line.toString().removeSuffix("\r")
            if (b == 10) return line.toString().removeSuffix("\r")
            require(line.length < limit) { "PLY行或文件头过大" }
            line.append(b.toChar())
        }
    }

    private fun typeSize(type: String): Int = when (type) {
        "float", "float32", "uint", "uint32", "int", "int32" -> 4
        "uchar", "uint8", "char", "int8" -> 1
        "ushort", "uint16", "short", "int16" -> 2
        "double", "float64" -> 8
        else -> throw IllegalArgumentException("不支持的PLY属性类型: $type")
    }
    private fun readValue(bb: ByteBuffer, off: Int, type: String): Float = when (type) {
        "uchar", "uint8" -> (bb.get(off).toInt() and 0xFF).toFloat()
        "char", "int8" -> bb.get(off).toFloat()
        "ushort", "uint16" -> (bb.getShort(off).toInt() and 0xFFFF).toFloat()
        "short", "int16" -> bb.getShort(off).toFloat()
        "uint", "uint32" -> (bb.getInt(off).toLong() and 0xFFFFFFFFL).toFloat()
        "int", "int32" -> bb.getInt(off).toFloat()
        "double", "float64" -> bb.getDouble(off).toFloat()
        else -> bb.getFloat(off)
    }
}
