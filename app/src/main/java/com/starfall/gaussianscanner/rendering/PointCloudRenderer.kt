package com.starfall.gaussianscanner.rendering

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.starfall.gaussianscanner.model.ColoredPt
import java.nio.ByteBuffer; import java.nio.ByteOrder; import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig; import javax.microedition.khronos.opengles.GL10

class PointCloudRenderer : GLSurfaceView.Renderer {

    private val vs = """#version 300 es
uniform mat4 uMVP; uniform float uPtSize; in vec3 aPos; in vec3 aCol; out vec3 vCol;
void main(){gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=uPtSize;vCol=aCol;}""".trimIndent()

    private val fs = """#version 300 es
precision mediump float; uniform int uFuzzy; uniform float uFuzzyRadius; in vec3 vCol; out vec4 fc;
void main(){
  float d=distance(gl_PointCoord,vec2(0.5));
  float sigma = (uFuzzy==1) ? (0.45+uFuzzyRadius*0.35) : 0.6;
  float a=exp(-d*d/(2.0*sigma*sigma));
  if(a<0.02)discard;
  fc=vec4(vCol,a);
}""".trimIndent()

    private var prog=0; private var n=0
    private var vBuf:FloatBuffer?=null; private var cBuf:FloatBuffer?=null
    private val proj=FloatArray(16); private val mvp=FloatArray(16)
    private var pending:List<ColoredPt>?=null

    var rx=0f; var ry=0f; var zoom=3f; var ptSize=6f
    var fuzzy = false
    var fuzzyRadius = 1.0f
    @Volatile var fps = 0; private set
    private var fpsFrames = 0
    private var fpsTime = 0L

    fun setCloud(pts:List<ColoredPt>){synchronized(this){pending=pts}}

    override fun onSurfaceCreated(gl:GL10?,c:EGLConfig?){
        GLES30.glClearColor(0.03f,0.03f,0.06f,1f); GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND); GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA,GLES30.GL_ONE_MINUS_SRC_ALPHA)
        prog=createProg()
    }

    override fun onSurfaceChanged(gl:GL10?,w:Int,h:Int){
        GLES30.glViewport(0,0,w,h); Matrix.perspectiveM(proj,0,50f,w.toFloat()/h,0.1f,50f)
    }

    override fun onDrawFrame(gl:GL10?){
        val now = System.currentTimeMillis()
        fpsFrames++
        if (now - fpsTime >= 500) {
            fps = (fpsFrames * 1000f / (now - fpsTime)).toInt()
            fpsFrames = 0; fpsTime = now
        }
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        synchronized(this){pending?.let{pts->
            n=pts.size; if(n>0){
                var cx=0f;var cy=0f;var cz=0f; for(p in pts){cx+=p.x;cy+=p.y;cz+=p.z}; cx/=n;cy/=n;cz/=n
                val vt=FloatArray(n*3);val cl=FloatArray(n*3)
                for(i in pts.indices){vt[i*3]=pts[i].x-cx;vt[i*3+1]=pts[i].y-cy;vt[i*3+2]=pts[i].z-cz;cl[i*3]=pts[i].r;cl[i*3+1]=pts[i].g;cl[i*3+2]=pts[i].b}
                vBuf=ByteBuffer.allocateDirect(vt.size*4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply{put(vt);position(0)}
                cBuf=ByteBuffer.allocateDirect(cl.size*4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply{put(cl);position(0)}
            }
            pending=null
        }}
        if(n==0)return
        GLES30.glUseProgram(prog)
        val rot=FloatArray(16); Matrix.setIdentityM(rot,0); Matrix.rotateM(rot,0,rx,1f,0f,0f); Matrix.rotateM(rot,0,ry,0f,1f,0f)
        val view=FloatArray(16); Matrix.setLookAtM(view,0,0f,0f,zoom,0f,0f,0f,0f,1f,0f)
        val rv=FloatArray(16); Matrix.multiplyMM(rv,0,view,0,rot,0)
        val t=FloatArray(16); Matrix.multiplyMM(t,0,proj,0,rv,0)
        Matrix.multiplyMM(mvp,0,t,0,FloatArray(16).also{Matrix.setIdentityM(it,0)},0)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(prog,"uMVP"),1,false,mvp,0)

        GLES30.glUniform1f(GLES30.glGetUniformLocation(prog,"uPtSize"),ptSize)
        GLES30.glUniform1i(GLES30.glGetUniformLocation(prog,"uFuzzy"), if (fuzzy) 1 else 0)
        GLES30.glUniform1f(GLES30.glGetUniformLocation(prog,"uFuzzyRadius"), fuzzyRadius)

        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        val aP=GLES30.glGetAttribLocation(prog,"aPos"); val aC=GLES30.glGetAttribLocation(prog,"aCol")
        vBuf?.position(0); GLES30.glVertexAttribPointer(aP,3,GLES30.GL_FLOAT,false,0,vBuf); GLES30.glEnableVertexAttribArray(aP)
        cBuf?.position(0); GLES30.glVertexAttribPointer(aC,3,GLES30.GL_FLOAT,false,0,cBuf); GLES30.glEnableVertexAttribArray(aC)
        GLES30.glDrawArrays(GLES30.GL_POINTS,0,n)
        GLES30.glDisableVertexAttribArray(aP); GLES30.glDisableVertexAttribArray(aC)
    }

    private fun createProg():Int{
        fun ld(t:Int,s:String)=GLES30.glCreateShader(t).also{GLES30.glShaderSource(it,s);GLES30.glCompileShader(it)}
        return GLES30.glCreateProgram().also{GLES30.glAttachShader(it,ld(GLES30.GL_VERTEX_SHADER,vs));GLES30.glAttachShader(it,ld(GLES30.GL_FRAGMENT_SHADER,fs));GLES30.glLinkProgram(it)}
    }
}
