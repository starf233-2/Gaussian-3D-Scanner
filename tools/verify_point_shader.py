"""Execute the app's GLES3 point-size shader via an installed Windows ANGLE.

python tools/verify_point_shader.py --angle-dir <folder containing libEGL.dll and libGLESv2.dll>

Uses a headless EGL pbuffer, not a browser. Transform feedback reads the REAL
vertex shader result, including normalized SHORT attributes. This validates
shader compilation/linkage and equations, not Qualcomm device compatibility.
"""
import argparse
import ctypes as c
import math
import os
from pathlib import Path
import re
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.angle_dir).resolve()
    directory_handle = os.add_dll_directory(str(folder))
    egl = c.WinDLL(str(folder / 'libEGL.dll'))
    gl = c.WinDLL(str(folder / 'libGLESv2.dll'))

    def bind(lib, name, result, *params):
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = params
        return function

    ptr = c.c_void_p; integer = c.c_int; uint = c.c_uint; floating = c.c_float
    ip = c.POINTER(integer)
    display = bind(egl, 'eglGetDisplay', ptr, ptr)(None)
    major = integer(); minor = integer()
    assert bind(egl, 'eglInitialize', uint, ptr, ip, ip)(display, c.byref(major), c.byref(minor)), 'eglInitialize'
    config = ptr(); configs = integer()
    attributes = (integer * 13)(0x3033, 1, 0x3040, 0x40, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3021, 8, 0x3038)
    assert bind(egl, 'eglChooseConfig', uint, ptr, ip, c.POINTER(ptr), integer, ip)(display, attributes, c.byref(config), 1, c.byref(configs)) and configs.value
    assert bind(egl, 'eglBindAPI', uint, uint)(0x30A0)
    context = bind(egl, 'eglCreateContext', ptr, ptr, ptr, ptr, ip)(display, config, None, (integer * 3)(0x3098, 3, 0x3038))
    surface = bind(egl, 'eglCreatePbufferSurface', ptr, ptr, ptr, ip)(display, config, (integer * 5)(0x3057, 16, 0x3056, 16, 0x3038))
    assert context and surface
    current = bind(egl, 'eglMakeCurrent', uint, ptr, ptr, ptr, ptr)
    assert current(display, surface, surface, context)
    print('GLES renderer:', bind(gl, 'glGetString', c.c_char_p, uint)(0x1F01).decode())

    create_shader = bind(gl, 'glCreateShader', uint, uint)
    shader_source = bind(gl, 'glShaderSource', None, uint, integer, c.POINTER(c.c_char_p), ip)
    compile_shader = bind(gl, 'glCompileShader', None, uint)
    get_shader = bind(gl, 'glGetShaderiv', None, uint, uint, ip)
    shader_log = bind(gl, 'glGetShaderInfoLog', None, uint, integer, ip, c.c_char_p)
    source_dir = Path(__file__).resolve().parents[1] / 'app/src/main/java/com/starfall/gaussianscanner/rendering'
    kotlin = (source_dir / 'PointCloudRenderer.kt').read_text(encoding='utf-8')
    constants = (source_dir / 'PointFootprint.kt').read_text(encoding='utf-8')
    shaders = re.findall(r'private val (?:vs|fs) = """(.*?)"""', kotlin, re.S)
    assert len(shaders) == 2
    for name in ('SLOPE_RANGE', 'COVERAGE'):
        value = re.search(r'const val ' + name + r' = ([0-9.]+)f', constants).group(1)
        value = str(float(value))  # Kotlin interpolates Float 4f as GLSL "4.0".
        shaders = [s.replace('${PointFootprint.' + name + '}', value) for s in shaders]
    # Expose the output only; do not alter the production footprint equation.
    shaders[0] = shaders[0].replace('void main(){', 'out float testPointSize;\nvoid main(){')
    shaders[0] = shaders[0].replace('vCol=aCol;', 'testPointSize=gl_PointSize;\nvCol=aCol;')
    ids = []
    for stage, source in zip((0x8B31, 0x8B30), shaders):
        shader = create_shader(stage); ids.append(shader)
        code = c.c_char_p(source.encode())
        shader_source(shader, 1, c.byref(code), None)
        compile_shader(shader)
        ok = integer(); get_shader(shader, 0x8B81, c.byref(ok))
        log = c.create_string_buffer(8192); shader_log(shader, 8192, None, log)
        assert ok.value, log.value.decode()
    program = bind(gl, 'glCreateProgram', uint)()
    attach = bind(gl, 'glAttachShader', None, uint, uint)
    for shader in ids:
        attach(program, shader)
    varying = c.c_char_p(b'testPointSize')
    bind(gl, 'glTransformFeedbackVaryings', None, uint, integer, c.POINTER(c.c_char_p), uint)(program, 1, c.byref(varying), 0x8C8C)
    bind(gl, 'glLinkProgram', None, uint)(program)
    ok = integer(); bind(gl, 'glGetProgramiv', None, uint, uint, ip)(program, 0x8B82, c.byref(ok))
    log = c.create_string_buffer(8192)
    bind(gl, 'glGetProgramInfoLog', None, uint, integer, ip, c.c_char_p)(program, 8192, None, log)
    assert ok.value, log.value.decode()
    bind(gl, 'glUseProgram', None, uint)(program)
    location = bind(gl, 'glGetUniformLocation', integer, uint, c.c_char_p)
    u1f = bind(gl, 'glUniform1f', None, integer, floating)
    u1i = bind(gl, 'glUniform1i', None, integer, integer)
    u2f = bind(gl, 'glUniform2f', None, integer, floating, floating)
    u3f = bind(gl, 'glUniform3f', None, integer, floating, floating, floating)
    matrix_uniform = bind(gl, 'glUniformMatrix4fv', None, integer, integer, c.c_ubyte, c.POINTER(floating))
    def loc(name):
        return location(program, name.encode())
    bind_buffer = bind(gl, 'glBindBuffer', None, uint, uint)
    buffer_data = bind(gl, 'glBufferData', None, uint, c.c_ssize_t, ptr, uint)
    buffers = (uint * 2)(); bind(gl, 'glGenBuffers', None, integer, c.POINTER(uint))(2, buffers)
    vao = uint(); bind(gl, 'glGenVertexArrays', None, integer, c.POINTER(uint))(1, c.byref(vao))
    bind(gl, 'glBindVertexArray', None, uint)(vao)
    bind_buffer(0x8892, buffers[0])
    attrib = bind(gl, 'glGetAttribLocation', integer, uint, c.c_char_p)
    pointer = bind(gl, 'glVertexAttribPointer', None, uint, integer, uint, c.c_ubyte, integer, ptr)
    enable = bind(gl, 'glEnableVertexAttribArray', None, uint)
    for name, count, kind, normalized, offset in ((b'aPos', 3, 0x1406, 0, 0), (b'aCol', 3, 0x1406, 0, 12), (b'aSlope', 2, 0x1402, 1, 24)):
        index = attrib(program, name)
        if index >= 0:
            pointer(index, count, kind, normalized, 28, ptr(offset)); enable(index)
    bind_buffer(0x8C8E, buffers[1]); buffer_data(0x8C8E, 4, None, 0x88E1)
    bind(gl, 'glBindBufferBase', None, uint, uint, uint)(0x8C8E, 0, buffers[1])
    bind(gl, 'glEnable', None, uint)(0x8C89)  # rasterizer discard
    begin = bind(gl, 'glBeginTransformFeedback', None, uint)
    draw = bind(gl, 'glDrawArrays', None, uint, integer, integer)
    end = bind(gl, 'glEndTransformFeedback', None)
    mapped = bind(gl, 'glMapBufferRange', ptr, uint, c.c_ssize_t, c.c_ssize_t, uint)
    unmap = bind(gl, 'glUnmapBuffer', c.c_ubyte, uint)
    get_error = bind(gl, 'glGetError', uint)
    cases = 0
    for distance in (.101, .5, 1.5, 3, 6, 18):
        for viewport in (720, 1080, 2160):
            for slope in (0, 8192, -4096):
                for automatic in (0, 1):
                    for grid in (0., .005):
                        p = 1 / math.tan(math.radians(25))
                        matrix = (floating * 16)(p, 0, 0, 0, 0, p, 0, 0, 0, 0, -1, -1, 0, 0, distance - .2, distance)
                        matrix_uniform(loc('uMVP'), 1, 0, matrix)
                        u3f(loc('uCenter'), 0, 0, -2)
                        u1i(loc('uAutoSize'), automatic); u1f(loc('uPtSize'), 6)
                        u1f(loc('uGridStep'), grid); u1f(loc('uImportedSpacing'), .01)
                        u2f(loc('uViewport'), viewport, viewport); u1f(loc('uMaxPtSize'), 64)
                        row = c.create_string_buffer(struct.pack('<6f2h', 0, 0, -2, 1, 1, 1, slope, slope))
                        bind_buffer(0x8892, buffers[0]); buffer_data(0x8892, 28, row, 0x88E4)
                        begin(0); draw(0, 0, 1); end()
                        bind_buffer(0x8C8E, buffers[1])
                        address = mapped(0x8C8E, 0, 4, 1)
                        assert address, 'transform feedback readback failed'
                        actual = c.cast(address, c.POINTER(floating))[0]; unmap(0x8C8E)
                        assert get_error() == 0
                        # At the centred vertex, perspective slope contribution
                        # is zero. This still exercises SHORT normalization/data ABI.
                        expected = min(64, max(1, p * viewport * .5 * .01 / distance * 1.12)) if automatic else 6
                        assert abs(actual - expected) < .002, (distance, viewport, automatic, grid, actual, expected)
                        cases += 1
    # Off-centre sloped cells under rotation: compare GPU derivatives against
    # independently projected neighbours (finite differences, not the formula).
    for distance in (2., 3., 8.):
        for yaw in (-.7, 0., .5):
            for depth in (1.2, 2., 3.):
                for slope in (-8192, 0, 12288):
                    p = 1 / math.tan(math.radians(25)); co = math.cos(yaw); si = math.sin(yaw)
                    matrix = (floating * 16)(p*co, 0, si, si, 0, p, 0, 0, p*si, 0, -co, -co, 0, 0, distance-.2, distance)
                    matrix_uniform(loc('uMVP'), 1, 0, matrix)
                    u1i(loc('uAutoSize'), 1); u1f(loc('uGridStep'), .005)
                    u2f(loc('uViewport'), 1080, 1080)
                    point = (.6, .4, -depth)
                    row = c.create_string_buffer(struct.pack('<6f2h', *point, 1, 1, 1, slope, -slope))
                    bind_buffer(0x8892, buffers[0]); buffer_data(0x8892, 28, row, 0x88E4)
                    begin(0); draw(0, 0, 1); end()
                    bind_buffer(0x8C8E, buffers[1]); address = mapped(0x8C8E, 0, 4, 1)
                    assert address
                    actual = c.cast(address, c.POINTER(floating))[0]; unmap(0x8C8E)
                    spacing = depth*.005; dz = slope/32767*4*spacing
                    tx = (spacing+point[0]/depth*dz, point[1]/depth*dz, -dz)
                    ty = (-point[0]/depth*dz, spacing-point[1]/depth*dz, dz)
                    def project(v):
                        x,y,z=v[0],v[1],v[2]+2
                        w=distance+si*x-co*z
                        return p*(co*x+si*z)/w*540, p*y/w*540
                    def extent(t):
                        epsilon=.001
                        a=project(tuple(point[j]+t[j]*epsilon for j in range(3)))
                        b=project(tuple(point[j]-t[j]*epsilon for j in range(3)))
                        return tuple(abs(a[j]-b[j])/(2*epsilon) for j in range(2))
                    a=extent(tx); b=extent(ty)
                    expected=min(64,max(1,max(a[0]+b[0],a[1]+b[1])*1.12))
                    assert abs(actual-expected)<.003, (distance,yaw,depth,slope,actual,expected)
                    assert get_error()==0
                    cases+=1
    print('Shader compile/link + real GLES transform-feedback cases PASS:', cases)
    current(display, None, None, None)
    bind(egl, 'eglDestroyContext', uint, ptr, ptr)(display, context)
    bind(egl, 'eglDestroySurface', uint, ptr, ptr)(display, surface)
    bind(egl, 'eglTerminate', uint, ptr)(display)
    directory_handle.close()


if __name__ == '__main__':
    main()
