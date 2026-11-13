#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_qnn_libs.py — copy the Qualcomm QNN (AI Engine Direct) runtime libraries
from a local QNN SDK zip/tree into the GaussianScanner Android project.

Usage from project root:
    python tools/setup_qnn_libs.py --sdk <qnn-sdk.zip or extracted dir> [--app .]

The QNN SDK can be downloaded from:
    https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/
        Qualcomm_AI_Runtime_Community/All/<version>/v<version>.zip
(a browser User-Agent is required; ~1.1-1.4 GB download).

The script also unpacks qtld-release.aar into app/libs/qtld.jar and copies its
JNI libs into jniLibs, so no Gradle repository changes are needed.
"""

import argparse
import os
import shutil
import zipfile

ROOT_INSIDE = "qairt"

# (name inside app/src/main/jniLibs/arm64-v8a, path fragment in the SDK)
RUNTIME_LIBS = [
    "libQnnSystem.so",       "lib/aarch64-android/libQnnSystem.so",
    "libQnnIr.so",           "lib/aarch64-android/libQnnIr.so",
    "libQnnGpu.so",          "lib/aarch64-android/libQnnGpu.so",
    "libQnnCpu.so",          "lib/aarch64-android/libQnnCpu.so",
    "libQnnDsp.so",          "lib/aarch64-android/libQnnDsp.so",
    "libQnnDspV66Stub.so",   "lib/aarch64-android/libQnnDspV66Stub.so",
]

# HTP runtime lives in app/jniLibsHtp (build flag -PwithoutHtp drops it).
HTP_RUNTIME_LIBS = [
    "libQnnHtp.so",          "lib/aarch64-android/libQnnHtp.so",
    "libQnnHtpPrepare.so",   "lib/aarch64-android/libQnnHtpPrepare.so",
    "libQnnHtpV68Stub.so",   "lib/aarch64-android/libQnnHtpV68Stub.so",
    "libQnnHtpV69Stub.so",   "lib/aarch64-android/libQnnHtpV69Stub.so",
    "libQnnHtpV73Stub.so",   "lib/aarch64-android/libQnnHtpV73Stub.so",
    "libQnnHtpV75Stub.so",   "lib/aarch64-android/libQnnHtpV75Stub.so",
    "libQnnHtpV79Stub.so",   "lib/aarch64-android/libQnnHtpV79Stub.so",
]

# DSP / HTP "Skel" libraries: these are pushed to the DSP over fastRPC.
# They are NOT dlopen'ed on the CPU, so they ship inside assets and are copied
# to a private app dir at runtime (the delegate's skel_library_dir option).
SKEL_LIBS = [
    "libQnnDspV66Skel.so",   "lib/hexagon-v66/unsigned/libQnnDspV66Skel.so",
]

# HTP skels live in app/assetsHtp (build flag -PwithoutHtp drops them).
HTP_SKEL_LIBS = [
    "libQnnHtpV68Skel.so",   "lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so",
    "libQnnHtpV69Skel.so",   "lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so",
    "libQnnHtpV73Skel.so",   "lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so",
    "libQnnHtpV75Skel.so",   "lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
    "libQnnHtpV79Skel.so",   "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
]

AAR_ENTRY = "lib/android/qtld-release.aar"


def find_entry(names, *fragments):
    for want in fragments:
        head, tail = want.split("/*/", 1)
        for n in names:
            if n.startswith(head + "/") and n.endswith("/" + tail):
                return n
    raise FileNotFoundError("not found in SDK: %s" % fragments)


def copy_from_zip(zf, src, dst):
    data = zf.read(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    print("  %-28s <- %s" % (os.path.basename(dst), src.split("/")[-1]))


def copy_from_tree(root, rel, dst):
    src = os.path.join(root, rel)
    if not os.path.isfile(src):
        raise FileNotFoundError("missing in SDK: %s" % src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print("  %-28s <- %s" % (os.path.basename(dst), os.path.basename(src)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", required=True, help="path to qnn-sdk zip or extracted qairt dir")
    ap.add_argument("--app", default=".", help="path to the Android app project (default: cwd)")
    args = ap.parse_args()

    sdk = args.sdk
    app = os.path.abspath(args.app)
    jni = os.path.join(app, "app", "src", "main", "jniLibs", "arm64-v8a")
    jni_htp = os.path.join(app, "app", "jniLibsHtp", "arm64-v8a")
    skel = os.path.join(app, "app", "src", "main", "assets", "qnn", "skels")
    skel_htp = os.path.join(app, "app", "assetsHtp", "qnn", "skels")
    libs = os.path.join(app, "app", "libs")

    print("== QNN runtime libs ->", jni, " (+HTP ->", jni_htp, ")")
    print("== QNN skel libs    ->", skel, " (+HTP ->", skel_htp, ")")
    print("== qtld AAR         ->", libs)

    if os.path.isdir(sdk):
        # sdk may point to the extracted "<ver>" folder or the "qairt" root
        candidates = [sdk]
        base = os.path.join(sdk, ROOT_INSIDE)
        if os.path.isdir(base):
            for sub in sorted(os.listdir(base)):
                candidates.append(os.path.join(base, sub))
        for c in candidates:
            if os.path.isdir(os.path.join(c, "lib", "aarch64-android")):
                root = c
                break
        else:
            raise SystemExit("cannot locate lib/aarch64-android in SDK dir %s" % sdk)

        for name, rel in zip(RUNTIME_LIBS[0::2], RUNTIME_LIBS[1::2]):
            copy_from_tree(root, rel, os.path.join(jni, name))
        for name, rel in zip(HTP_RUNTIME_LIBS[0::2], HTP_RUNTIME_LIBS[1::2]):
            copy_from_tree(root, rel, os.path.join(jni_htp, name))
        for name, rel in zip(SKEL_LIBS[0::2], SKEL_LIBS[1::2]):
            copy_from_tree(root, rel, os.path.join(skel, name))
        for name, rel in zip(HTP_SKEL_LIBS[0::2], HTP_SKEL_LIBS[1::2]):
            copy_from_tree(root, rel, os.path.join(skel_htp, name))
        aar_src = os.path.join(root, AAR_ENTRY)
        if not os.path.isfile(aar_src):
            raise SystemExit("qtld-release.aar missing at %s" % aar_src)
        aar_dst = os.path.join(libs, "qtld-release.aar")
        os.makedirs(libs, exist_ok=True)
        shutil.copy2(aar_src, aar_dst)

    elif zipfile.is_zipfile(sdk):
        with zipfile.ZipFile(sdk) as zf:
            names = zf.namelist()
            for name, rel in zip(RUNTIME_LIBS[0::2], RUNTIME_LIBS[1::2]):
                src = find_entry(names, ROOT_INSIDE + "/*/" + rel)
                copy_from_zip(zf, src, os.path.join(jni, name))
            for name, rel in zip(HTP_RUNTIME_LIBS[0::2], HTP_RUNTIME_LIBS[1::2]):
                src = find_entry(names, ROOT_INSIDE + "/*/" + rel)
                copy_from_zip(zf, src, os.path.join(jni_htp, name))
            for name, rel in zip(SKEL_LIBS[0::2], SKEL_LIBS[1::2]):
                src = find_entry(names, ROOT_INSIDE + "/*/" + rel)
                copy_from_zip(zf, src, os.path.join(skel, name))
            for name, rel in zip(HTP_SKEL_LIBS[0::2], HTP_SKEL_LIBS[1::2]):
                src = find_entry(names, ROOT_INSIDE + "/*/" + rel)
                copy_from_zip(zf, src, os.path.join(skel_htp, name))
            src = find_entry(names, ROOT_INSIDE + "/*/" + AAR_ENTRY)
            os.makedirs(libs, exist_ok=True)
            copy_from_zip(zf, src, os.path.join(libs, "qtld-release.aar"))
    else:
        raise SystemExit("--sdk must be a zip or an extracted directory")

    # unpack qtld-release.aar: classes.jar -> app/libs/qtld.jar, jni -> jniLibs
    aar = os.path.join(libs, "qtld-release.aar")
    with zipfile.ZipFile(aar) as a:
        for n in a.namelist():
            if n == "classes.jar":
                with open(os.path.join(libs, "qtld.jar"), "wb") as f:
                    f.write(a.read(n))
                print("  qtld.jar <- classes.jar (%d bytes)" % a.getinfo(n).file_size)
            elif n.startswith("jni/arm64-v8a/") and n.endswith(".so"):
                with open(os.path.join(jni, os.path.basename(n)), "wb") as f:
                    f.write(a.read(n))
                print("  %-28s <- %s" % (os.path.basename(n), n))

    print("\\nDone. Now build with:  ./gradlew assembleRelease")


if __name__ == "__main__":
    main()