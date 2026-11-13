package com.starfall.gaussianscanner.rendering

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.SystemClock
import com.starfall.gaussianscanner.model.PointCloud
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.lang.ref.WeakReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class PointCloudRenderer(private val onError: (String) -> Unit = {}) : GLSurfaceView.Renderer {
    private val vs = """#version 300 es
uniform mat4 uMVP; uniform vec3 uCenter; uniform float uPtSize;
uniform int uAutoSize; uniform vec2 uViewport;
uniform float uGridStep; uniform float uImportedSpacing; uniform float uMaxPtSize;
in vec3 aPos; in vec3 aCol; in vec2 aSlope; out vec3 vCol;
vec2 projectTangent(vec3 tangent, vec4 clip){
  vec4 v=uMVP*vec4(tangent,0.0);
  return (v.xy*clip.w-clip.xy*v.w)/(clip.w*clip.w)*uViewport*0.5;
}
void main(){
  vec4 clip=uMVP*vec4(aPos-uCenter,1.0);
  gl_Position=clip;
  float size=uPtSize;
  if(uAutoSize==1){
    size=1.0;
    if(clip.w>0.1){
      if(uGridStep>0.0 && aPos.z<0.0){
        float z=-aPos.z;
        float spacing=z*uGridStep;
        vec2 dz=aSlope*${PointFootprint.SLOPE_RANGE}*spacing;
        vec3 tx=vec3(spacing+aPos.x/z*dz.x,aPos.y/z*dz.x,-dz.x);
        vec3 ty=vec3(aPos.x/z*dz.y,spacing+aPos.y/z*dz.y,-dz.y);
        vec2 extent=abs(projectTangent(tx,clip))+abs(projectTangent(ty,clip));
        size=max(extent.x,extent.y)*${PointFootprint.COVERAGE};
      }else{
        // Isotropic fallback for imported, unordered points. The row length
        // of MVP is invariant to model rotation, unlike a single matrix entry.
        vec3 rowY=vec3(uMVP[0].y,uMVP[1].y,uMVP[2].y);
        size=length(rowY)*uViewport.y*0.5*uImportedSpacing/clip.w*${PointFootprint.COVERAGE};
      }
    }
  }
  gl_PointSize=clamp(size,1.0,uMaxPtSize);
  vCol=aCol;
}""".trimIndent()
    private val fs = """#version 300 es
precision mediump float;
uniform int uFuzzy; uniform float uFuzzyRadius; in vec3 vCol; out vec4 fc;
void main(){
  if(uFuzzy==0){fc=vec4(vCol,1.0);return;}
  float d=distance(gl_PointCoord,vec2(0.5));
  float sigma=0.45+uFuzzyRadius*0.35;
  float a=exp(-d*d/(2.0*sigma*sigma));
  if(a<0.05)discard;
  fc=vec4(vCol,a);
}""".trimIndent()

    @Volatile private var cloud = PointCloud.EMPTY
    private var uploaded: WeakReference<PointCloud>? = null
    private var program = 0
    private var vbo = 0
    private var slopeVbo = 0
    private var vao = 0
    private var mvpLoc = -1; private var sizeLoc = -1; private var centerLoc = -1
    private var fuzzyLoc = -1; private var radiusLoc = -1
    private var autoLoc = -1; private var viewportLoc = -1; private var gridLoc = -1
    private var importedLoc = -1; private var maximumLoc = -1
    private var viewportW = 1; private var viewportH = 1
    private var maxPointSize = 64f
    private var importedSpacing = .01f
    private val center = FloatArray(3)
    private val projection = FloatArray(16)
    private val rotation = FloatArray(16)
    private val view = FloatArray(16)
    private val modelView = FloatArray(16)
    private val mvp = FloatArray(16)
    // Only a small upload staging buffer, not two extra full-cloud copies.
    private val staging = ByteBuffer.allocateDirect(4096 * PointCloud.STRIDE * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val slopeStaging = ByteBuffer.allocateDirect(4096 * 4).order(ByteOrder.nativeOrder()).asShortBuffer()

    @Volatile var rx = 0f
    @Volatile var ry = 0f
    @Volatile var zoom = 3f
    @Volatile var ptSize = 6f
    @Volatile var autoPointSize = false
    @Volatile var fuzzy = false
    @Volatile var fuzzyRadius = 1f
    @Volatile var fps = 0; private set
    @Volatile var frameCount = 0L; private set
    private var fpsFrames = 0
    private var fpsTime = 0L

    fun setCloud(value: PointCloud) { cloud = value }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.03f, 0.03f, 0.06f, 1f)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        // IDs belonged to the previous EGL context. Re-upload retained data.
        vbo = 0; slopeVbo = 0; vao = 0; uploaded = null
        program = try { createProgram() } catch (t: Throwable) {
            onError(t.message ?: "点云渲染初始化失败")
            0
        }
        if (program == 0) return
        mvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        sizeLoc = GLES30.glGetUniformLocation(program, "uPtSize")
        centerLoc = GLES30.glGetUniformLocation(program, "uCenter")
        fuzzyLoc = GLES30.glGetUniformLocation(program, "uFuzzy")
        radiusLoc = GLES30.glGetUniformLocation(program, "uFuzzyRadius")
        autoLoc = GLES30.glGetUniformLocation(program, "uAutoSize")
        viewportLoc = GLES30.glGetUniformLocation(program, "uViewport")
        gridLoc = GLES30.glGetUniformLocation(program, "uGridStep")
        importedLoc = GLES30.glGetUniformLocation(program, "uImportedSpacing")
        maximumLoc = GLES30.glGetUniformLocation(program, "uMaxPtSize")
        val range = FloatArray(2)
        GLES30.glGetFloatv(GLES30.GL_ALIASED_POINT_SIZE_RANGE, range, 0)
        maxPointSize = if (range[1].isFinite()) range[1].coerceIn(1f, 64f) else 64f
        fpsTime = SystemClock.elapsedRealtime(); fpsFrames = 0
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        viewportW = w.coerceAtLeast(1); viewportH = h.coerceAtLeast(1)
        GLES30.glViewport(0, 0, w, h)
        Matrix.perspectiveM(projection, 0, 50f, w.toFloat() / h.coerceAtLeast(1), 0.1f, 50f)
    }

    override fun onDrawFrame(gl: GL10?) {
        frameCount++
        val now = SystemClock.elapsedRealtime()
        fpsFrames++
        if (now - fpsTime >= 500) {
            fps = (fpsFrames * 1000L / (now - fpsTime)).toInt()
            fpsFrames = 0; fpsTime = now
        }
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        if (program == 0) return
        val desired = cloud
        if (uploaded?.get() !== desired) {
            try { upload(desired) } catch (t: Throwable) {
                onError(t.message ?: "点云上传失败")
                if (vbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
                if (slopeVbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(slopeVbo), 0)
                if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
                GLES30.glBindVertexArray(0)
                vbo = 0; slopeVbo = 0; vao = 0
            }
        }
        if (desired.size == 0 || vbo == 0) return
        Matrix.setIdentityM(rotation, 0)
        Matrix.rotateM(rotation, 0, rx, 1f, 0f, 0f)
        Matrix.rotateM(rotation, 0, ry, 0f, 1f, 0f)
        Matrix.setLookAtM(view, 0, 0f, 0f, zoom, 0f, 0f, 0f, 0f, 1f, 0f)
        Matrix.multiplyMM(modelView, 0, view, 0, rotation, 0)
        Matrix.multiplyMM(mvp, 0, projection, 0, modelView, 0)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(mvpLoc, 1, false, mvp, 0)
        GLES30.glUniform3fv(centerLoc, 1, center, 0)
        GLES30.glUniform1f(sizeLoc, ptSize)
        GLES30.glUniform1i(autoLoc, if (autoPointSize) 1 else 0)
        GLES30.glUniform2f(viewportLoc, viewportW.toFloat(), viewportH.toFloat())
        GLES30.glUniform1f(gridLoc, desired.surface?.stepOverFocal ?: 0f)
        GLES30.glUniform1f(importedLoc, importedSpacing)
        GLES30.glUniform1f(maximumLoc, maxPointSize)
        GLES30.glUniform1i(fuzzyLoc, if (fuzzy) 1 else 0)
        GLES30.glUniform1f(radiusLoc, fuzzyRadius)
        if (fuzzy) {
            GLES30.glEnable(GLES30.GL_BLEND)
            GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        } else GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glBindVertexArray(vao)
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, desired.size)
        GLES30.glBindVertexArray(0)
    }

    private fun upload(value: PointCloud) {
        if (vbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
        if (slopeVbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(slopeVbo), 0)
        if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
        vbo = 0; slopeVbo = 0; vao = 0; uploaded = WeakReference(value)
        if (value.size == 0) return
        var x = 0.0; var y = 0.0; var z = 0.0
        for (i in 0 until value.size) {
            val k = i * PointCloud.STRIDE
            x += value.data[k]; y += value.data[k + 1]; z += value.data[k + 2]
        }
        center[0] = (x / value.size).toFloat(); center[1] = (y / value.size).toFloat(); center[2] = (z / value.size).toFloat()
        val ids = IntArray(1)
        GLES30.glGenBuffers(1, ids, 0); vbo = ids[0]
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        val floats = value.size * PointCloud.STRIDE
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, floats * 4, null, GLES30.GL_STATIC_DRAW)
        check(GLES30.glGetError() == GLES30.GL_NO_ERROR) { "GPU点云缓冲区分配失败" }
        var offset = 0
        while (offset < floats) {
            val count = minOf(staging.capacity(), floats - offset)
            staging.clear(); staging.put(value.data, offset, count); staging.flip()
            GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, offset * 4, count * 4, staging)
            offset += count
        }
        GLES30.glGenVertexArrays(1, ids, 0); vao = ids[0]
        GLES30.glBindVertexArray(vao)
        val pos = GLES30.glGetAttribLocation(program, "aPos")
        val col = GLES30.glGetAttribLocation(program, "aCol")
        val slope = GLES30.glGetAttribLocation(program, "aSlope")
        GLES30.glVertexAttribPointer(pos, 3, GLES30.GL_FLOAT, false, 24, 0)
        GLES30.glVertexAttribPointer(col, 3, GLES30.GL_FLOAT, false, 24, 12)
        GLES30.glEnableVertexAttribArray(pos); GLES30.glEnableVertexAttribArray(col)
        val surface = value.surface
        if (surface != null) {
            GLES30.glGenBuffers(1, ids, 0); slopeVbo = ids[0]
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, slopeVbo)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, surface.slopes.size * 2, null, GLES30.GL_STATIC_DRAW)
            check(GLES30.glGetError() == GLES30.GL_NO_ERROR) { "GPU点覆盖缓冲区分配失败" }
            var start = 0
            while (start < surface.slopes.size) {
                val count = minOf(slopeStaging.capacity(), surface.slopes.size - start)
                slopeStaging.clear(); slopeStaging.put(surface.slopes, start, count); slopeStaging.flip()
                GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, start * 2, count * 2, slopeStaging)
                start += count
            }
            GLES30.glVertexAttribPointer(slope, 2, GLES30.GL_SHORT, true, 4, 0)
            GLES30.glEnableVertexAttribArray(slope)
        } else {
            GLES30.glDisableVertexAttribArray(slope)
            GLES30.glVertexAttrib2f(slope, 0f, 0f)
            importedSpacing = PointFootprint.estimateImportedSpacing(value)
        }
        GLES30.glBindVertexArray(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        check(GLES30.glGetError() == GLES30.GL_NO_ERROR) { "GPU点云上传失败" }
    }

    private fun createProgram(): Int {
        fun shader(type: Int, source: String): Int {
            val id = GLES30.glCreateShader(type)
            GLES30.glShaderSource(id, source); GLES30.glCompileShader(id)
            val ok = IntArray(1)
            GLES30.glGetShaderiv(id, GLES30.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) {
                val message = GLES30.glGetShaderInfoLog(id)
                GLES30.glDeleteShader(id)
                error("点云着色器编译失败: $message")
            }
            return id
        }
        val vertex = shader(GLES30.GL_VERTEX_SHADER, vs)
        var fragment = 0
        val id = GLES30.glCreateProgram()
        try {
            fragment = shader(GLES30.GL_FRAGMENT_SHADER, fs)
            GLES30.glAttachShader(id, vertex); GLES30.glAttachShader(id, fragment)
            GLES30.glLinkProgram(id)
            val ok = IntArray(1)
            GLES30.glGetProgramiv(id, GLES30.GL_LINK_STATUS, ok, 0)
            check(ok[0] != 0) { "点云程序链接失败: ${GLES30.glGetProgramInfoLog(id)}" }
            return id
        } catch (t: Throwable) {
            GLES30.glDeleteProgram(id)
            throw t
        } finally {
            GLES30.glDeleteShader(vertex)
            if (fragment != 0) GLES30.glDeleteShader(fragment)
        }
    }
}
