#!/usr/bin/env python3
"""Optional OFFLINE INT8 conversion; not used by the Android app automatically.

Requires the original fixed-shape ONNX model and real calibration photographs.
An existing TFLite flatbuffer cannot be re-quantized with TFLiteConverter.

    pip install tensorflow onnx onnx2tf numpy pillow
    python tools/quantize_da3_to_int8.py --onnx da3.onnx --images photos/

--normalization imagenet wraps external ImageNet normalization into the exported
graph. Use raw if normalization is already embedded in the original model.
The output always expects NHWC RGB [0,1] before quantization, matching the app.
HTP v68 (888) needs the quantized runtime; FP16 availability is device-dependent.
Transformer operators may prevent full INT8 conversion: this script reports that
failure rather than producing an unvalidated float model labelled as INT8.
"""

import argparse
import os
from pathlib import Path
import tempfile


def representative_dataset(paths, height, width, samples):
    """Finite batch-ONE calibration, float32 NHWC, never synthetic noise."""
    import numpy as np
    from PIL import Image

    def generate():
        for i in range(samples):
            with Image.open(paths[i % len(paths)]) as image:
                rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
            yield [array[None, ...]]

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="original, fixed-shape ONNX model")
    parser.add_argument("--images", required=True, help="representative calibration photographs")
    parser.add_argument("--output", default="da3_small_int8.tflite")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--normalization", choices=("raw", "imagenet"), default="raw")
    args = parser.parse_args()
    model = Path(args.onnx).resolve()
    output = Path(args.output).resolve()
    images = Path(args.images)
    if not model.is_file() or not images.is_dir() or not 1 <= args.samples <= 10000:
        parser.error("需要有效的ONNX文件、校准图片目录和1..10000个样本")
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not paths:
        parser.error("校准目录没有图片；不会用随机噪声替代真实数据")
    if output.exists():
        parser.error("输出文件已存在，请指定新的--output（不覆盖已有模型）")
    try:
        import tensorflow as tf
        import numpy as np
        from onnx2tf import convert
    except ImportError as error:
        parser.error(f"缺少离线转换依赖: {error}; 安装 tensorflow onnx onnx2tf numpy pillow")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="da3_int8_", dir=output.parent) as work:
        saved = os.path.join(work, "saved_model")
        convert(input_onnx_file_path=str(model), output_folder_path=saved)
        source = tf.saved_model.load(saved)
        signature = source.signatures.get("serving_default")
        if signature is None:
            parser.error("转换结果缺少serving_default签名")
        inputs = signature.structured_input_signature[1]
        if len(inputs) != 1:
            parser.error("仅支持单张RGB输入的模型")
        name, spec = next(iter(inputs.items()))
        shape = spec.shape.as_list()
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3 or any(v is None or v <= 0 for v in shape):
            parser.error(f"转换后的输入必须是固定[1,H,W,3] NHWC，实际: {shape}")

        class RawInputModel(tf.Module):
            def __init__(self):
                super().__init__()
                self.source = source  # Track all variables captured by the function.

            @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name="image")])
            def run(self, image):
                if args.normalization == "imagenet":
                    image = (image - tf.constant([.485, .456, .406])) / tf.constant([.229, .224, .225])
                return self.source.signatures["serving_default"](**{name: image})

        wrapper = RawInputModel()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([wrapper.run.get_concrete_function()], wrapper)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(paths, shape[1], shape[2], args.samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        result = converter.convert()
        # Verify ABI and quantization before publishing the output.
        interpreter = tf.lite.Interpreter(model_content=result)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            parser.error("结果不是单输入/单输出深度模型")
        if input_details[0]["dtype"] != np.int8 or output_details[0]["dtype"] != np.int8:
            parser.error("转换结果不是INT8输入/输出")
        if int(np.prod(output_details[0]["shape"])) != shape[1] * shape[2]:
            parser.error("深度输出尺寸与输入不匹配")
        with output.open("xb") as file:
            file.write(result)
    print(f"Wrote {output} ({len(result)} bytes). Depth quality and HTP operator support still need device testing.")


if __name__ == "__main__":
    main()
