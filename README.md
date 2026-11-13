# GaussianScanner v10.4 — QNN 硬件加速版

单张图片 → Depth Anything 3 Small（高通官方 fp32 模型）→ 深度图 → 3D 点云。

性能改良源码及验证范围见 [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md)。本轮未调整版本号或更新备份。

最新高分辨率/自动点大小重构见 [HIRES_POINTSIZE_REFACTOR.md](HIRES_POINTSIZE_REFACTOR.md)：
完整2倍模式共10次推理，逐点投影决定自动点大小，替代旧瓦片高频注入和固定缩放锚点。

### 源码压缩包	

`GaussianScanner-v10.4-source.zip` 只包含自行编写的源码、Gradle 配置、工具脚本和文档，
便于公开上传。以下内容因体积或许可原因**未包含**，请在构建前补齐：

| 缺失项 | 补齐方式 |
|---|---|
| `app/src/main/assets/da3_small_gpu_fp32.tflite`（143.7MB，超过GitHub单文件100MB限制） | 从 `https://hf-mirror.com/qualcomm/Depth-Anything-V3` 获取官方 float 导出，按原文件名放入 assets |
| `app/libs/qtld.jar`、`app/src/main/jniLibs/`、`app/jniLibsHtp/`、`app/assetsHtp/`（QNN 厂商二进制） | 下载 Qualcomm AI Runtime Community SDK 后执行 `python tools/setup_qnn_libs.py --sdk <SDK> --app .` |
| `local.properties` | 用 Android Studio 打开工程，或写入 `sdk.dir=C\:\\path\\to\\Android\\Sdk` |
| `release.jks`（签名密钥） | 不要上传个人密钥；把 `release.jks` 放在工程上级目录即可自动签名，或改用你自己的签名配置。缺失时 release APK 不签名，debug 包不受影响 |

`.gitignore` 已排除构建产物、本机路径、模型、APK、厂商 `.so`/`.skel` 与签名文件。

v7.0 的核心变化：把深度模型的推理从 CPU 卸载到高通的 **GPU / NPU**，
使用 **Qualcomm AI Engine Direct（QNN）SDK** 的 TFLite Delegate 完成。

---

## 1. 加速方案总览

推理引擎选择顺序（可在“设置 → 推理加速引擎”中更改，默认“自动”）：

| 优先级 | 后端 | 加载的 QNN 库 | 适用的 SoC |
|--------|------|----------------|------------|
| 1 (NPU) | QNN HTP (Hexagon Tensor Processor) | `libQnnHtp.so` + `libQnnHtpV{68,69,73,75,79}*.so` | 骁龙 888 (SM8350) 及以上的 HTP |
| 2 (NPU) | QNN DSP (Hexagon DSP / HVX v66) | `libQnnDsp.so` + `libQnnDspV66*.so` | 骁龙 865 / 870 (SM8250) |
| 3 (GPU) | QNN GPU (Adreno, OpenCL) | `libQnnGpu.so` | 所有 Adreno（.so 全机型同名） |
| 4 (回退) | TFLite GPU Delegate (OpenCL) | —— | 任意 |
| 5 (回退) | CPU (XNNPACK) | —— | 任意 |

- `自动`：有 HTP 先 HTP → 否则有 DSP v66 先 DSP → 再 GPU → CPU。
- `GPU`：直接 QNN GPU。
- `NPU`：优先 HTP，870 用 DSP。
- `仅CPU`：兼容模式，不加载任何 QNN 库。

### 骁龙 870（SM8250）具体行为

870 与 865 同属 SM8250（kona）平台，**没有 HTP 张量加速器**，其“NPU”是
Hexagon DSP（HVX v66）。在已锁 Bootloader 的零售手机上，向 CDSP 上传我们附带
的**未签名** Skel 可能被拒绝，此时程序会**自动落回 QNN GPU（Adreno 650, OpenCL）**，
完成加速；设备允许且算子兼容时仍可使用 DSP，并非保证所有机型一定走 GPU。

### 各代骁龙对应的 QNN HTP 版本（自动识别）

| SoC | 代号 | HTP lib | 说明 |
|-----|------|---------|------|
| SM8250 (865/870) | kona | 无 HTP，用 DSP v66 | 本机 |
| SM8350 (888) | lahaina | libQnnHtpV68 | HTP 第一代 |
| SM8450 / SM8475 (8/8+ Gen1) | taro | libQnnHtpV69 | |
| SM8550 (8 Gen2) | kalama | libQnnHtpV73 | |
| SM8650 (8 Gen3) | pineapple | libQnnHtpV75 | |
| SM8750 (8 Elite) | sun | libQnnHtpV79 | |

运行时由 QNN 库根据设备 SoC 自动加载对应 V-版本库，代码无需逐个指定。

---

## 2. 模型（重要）

App 现在默认使用 **Qualcomm 官方的 float 导出** `da3_small_gpu_fp32.tflite`
（来源：`https://hf-mirror.com/qualcomm/Depth-Anything-V3`，qairt 2.45 转换）：

- 输入 `[1, 518, 518, 3]` FLOAT32，**NHWC**；输出 `[1, 518, 518, 1]` FLOAT32
- 权重全部 fp32（首次修复：旧 fp16 权重模型会在 GPU 委托分区边界出错，产出
  平面/垃圾深度）
- 输入喂**原始像素 [0,1]**（ImageNet 均值/标准差已在模型内部完成，op0=SUB(mean)、
  op1=MUL(1/std)），App 不再做外部标准化
- 布局为 NHWC，是 TFLite/GPU 委托的标准约定，QNN GPU 可全图正确委托

> 旧版 NCHW fp16（`da3_small_gpu_fp16.tflite`）已移出主 App（备份中保留）。
> 若放入 NCHW/INT8 版本，代码会根据输入张量形状自动切换布局与预处理
> （NCHW + 外部 ImageNet 标准化；NHWC + 原始 [0,1]）。

体积提示：该模型约 140MB，故 APK 约 186MB。

### 可选 INT8 模型

888 的 HTP v68 不提供当前所请求的 FP16 运行时能力，需要量化模型及量化档；
硬件存在并不代表当前模型能在 NPU 上运行。App 会先探测所请求的能力，失败正常回退。
App 支持 INT8/UINT8 输入输出；从原始 ONNX 模型离线转换可参考
`tools/quantize_da3_to_int8.py`（需真实校准照片，完整转换与深度质量尚需验证）。
当前打包资产仍只有官方 fp32 模型，不会为了节省内存自动降低模型精度。

---

## 3. QNN SDK 获取与库文件部署

QNN SDK（Qualcomm AI Runtime Community）公开下载地址（无需登录，但需要浏览器
User-Agent；`<version>` 形如 `2.32.0.250228`）：

```
https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<version>/v<version>.zip
```

> 中国大陆可直接访问（CloudFront 落地北京/上海节点）。若失败可先
> `curl -H "User-Agent: Mozilla/5.0 ..."` 重试。

把解析后的 SDK 库部署进工程（幂等脚本，支持 zip 或已解压目录）：

```
python tools/setup_qnn_libs.py --sdk <qnn-sdk.zip 或解压目录> --app .
```

脚本会：

1. 复制 QNN 运行时 `.so` → `app/src/main/jniLibs/arm64-v8a/`
   （libQnnGpu / libQnnDsp / libQnnIr / libQnnSystem / libQnnDspV66Stub / libQnnCpu）
2. 复制 HTP (888+ NPU) 运行时 → `app/jniLibsHtp/arm64-v8a/`
   （libQnnHtp / libQnnHtpPrepare / libQnnHtpV68~V79 stub；构建时默认打包，
   `-PwithoutHtp=true` 可排除，省 ~69 MB）
3. 复制 DSP/HTP 的 **Skel** 库 → `app/src/main/assets/qnn/skels/` 与
    `app/assetsHtp/qnn/skels/`（尝试 NPU 时复制到私有目录并通过
   `skel_library_dir` 交给 fastRPC 上传到 DSP/HTP）
4. 解包 `qtld-release.aar` → `app/libs/qtld.jar` + 两个委托 JNI 库到 jniLibs
   （`com.qualcomm.qti.QnnDelegate` Java 绑定）

### 关键文件说明

| 文件 | 作用 |
|------|------|
| `libQnnTFLiteDelegate.so` | QNN 的 TFLite 委托（C API，被 Java 绑定调用） |
| `libqnn_delegate_jni.so` | Java <-> C 的 JNI 桥 |
| `libQnnGpu.so` | Adreno GPU 后端（OpenCL，所有骁龙通用） |
| `libQnnHtp.so` / `libQnnHtpPrepare.so` | HTP 后端，$(888+) NPU |
| `libQnnDsp.so` | Hexagon DSP 后端（865/870 NPU） |
| `libQnnHtpVxxStub.so` + `libQnnHtpVxxSkel.so` | HTP 各代代理/DSP固件 |
| `libQnnDspV66Stub.so` + `libQnnDspV66Skel.so` | DSP v66 代理/DSP固件 |

### 版本一致性（重要）

- 本项目固定 `org.tensorflow:tensorflow-lite:2.15.0` —— 这是 QNN TFLite Delegate
  （2.32.0.250228）官方开发/测试的版本。不要随意升级 TFLite 版本。
- 委托库与运行时库同属一个 SDK 版本，必须成套替换。

### 按官方文档强制要求的部署细节（已配置）

- `jniLibs.useLegacyPackaging = true`：QNN 官方要求 native 库必须解压成真实文件
  （`nativeLibraryDir`），委托才能 dlopen 到后端库。
- `AndroidManifest.xml` 声明 `<uses-native-library android:name="libcdsprpc.so"
  android:required="false"/>`：QNN 委托经 fastRPC 访问 DSP/HTP 需要该系统库。
- 仅按需加载当前后端的运行时；HTP 能力探测通过后才加载图准备库。
  Vxx stub 由后端按设备自动加载，不预加载所有架构。
- ProGuard：`-keep class com.qualcomm.qti.** { *; }`（见 app/proguard-rules.pro）。

---

## 4. 构建

需要：JDK 17+，Android SDK（`local.properties` 已指向
`C:/Android/Sdk`），内置 Gradle 8.5 可直接使用（无需在线下载 Gradle）。

```powershell
$env:JAVA_HOME = "C:\Program Files\Zulu\zulu-21"
& "C:\Gradle\gradle-8.5\bin\gradle.bat" :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleRelease --no-daemon
```

产物：`app/build/outputs/apk/release/app-release.apk` 与 `app/build/outputs/apk/debug/app-debug.apk`。
模型文件约 143.7MB。`-PwithoutHtp=true` 可排除 HTP 运行时/Skel；APK 体积不等于运行内存。

```powershell
& "C:\Gradle\gradle-8.5\bin\gradle.bat" :app:assembleRelease -PwithoutHtp=true --no-daemon
```

---

## 5. 已知限制与排错

- **NPU( DSP/HTP ) 在已锁零售机上的表现取决于机型**：向 DSP 上传未签名 Skel 需要
  设备 CDSP 开放或已有固件匹配。若失败，App 自动回退 GPU，不影响使用。
  开启“设置 → 调试模式 → 生成日志”后才输出 `debug_*.log`，
  其中列出所有后端尝试与失败原因。
- **第一次创建推理引擎较慢**：模型在第一张照片/图片推理时才加载，图编译可能需要几分钟。
  闲置 60 秒、退到后台或收到内存压力通知时释放模型；再次推理需要重新加载，可复用 QNN 缓存。
- **JNI 超时不能强杀驱动线程**：超时/取消后不并行分配第二份模型；迟到结果在原生线程清理。
  驱动若始终不返回，可能需要关闭/重启应用，不能承诺通过线程中断解决。
- **静止画面 0 FPS 是正常省电行为**：仅新模型、触摸旋转/缩放或设置变化触发渲染。
- 不支持 32 位 ABI：APK 已限制为 arm64-v8a。
- 本仓库在 `tools/` 中保留了 SDK 获取与部署的完整脚本，便于重放。

---

## 6. 目录结构（v7.0 新增部分）

```
app/src/main/java/com/starfall/gaussianscanner/qnn/QnnEngine.kt   # 设备识别+委托构建
app/src/main/java/com/starfall/gaussianscanner/depth/DepthEstimator.kt  # 引擎选择+推理
app/src/main/jniLibs/arm64-v8a/       # QNN 运行时 .so（GPU/DSP/IR/System)
app/jniLibsHtp/arm64-v8a/             # QNN HTP 运行时 .so（888+ NPU，可截除）
app/src/main/assets/qnn/skels/        # DSP skel（运行时上传）
app/assetsHtp/qnn/skels/              # HTP skel（888+ NPU，可截除）
app/libs/qtld.jar                     # QNN TFLite Delegate Java 绑定
tools/setup_qnn_libs.py               # SDK -> 工程 部署脚本
tools/quantize_da3_to_int8.py         # 可选 INT8 模型生成工具
```
