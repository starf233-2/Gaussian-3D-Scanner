#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_da3_base_to_tflite.py

把 Depth-Anything-V3-BASE（官方架构 https://github.com/ByteDance-Seed/depth-anything-3）
转换为单图相对深度 TFLite 模型，用于本工程（GaussianScanner）的相机成像管线。

背景
----
- 该 checkpoint 是完整 DA3（DINOv2-B 主干 + DualDPT 解码头 + 相机位姿分支）。
  单图推理时（不提供 extrinsics）模型内部会自动估计位姿并输出相对深度，
  与高通 small 版的外观一致，App 端无需改动预处理（输入仍是原始像素 [0,1]，
  本脚本把 ImageNet 归一化嵌入图内，NHWC 输出 518x518 深度图）。

用法
----
  python tools/convert_da3_base_to_tflite.py \
      --checkpoint model.safetensors \
      --src ../depth-anything-3          # 官方源码目录
      --out models \
      --size 518

产物
----
  models/da3_base_int8.tflite   (INT8 量化，≈120MB，适合 CPU / NPU 推理)
  models/da3_base_fp16.tflite   (FP16，≈240MB，适合 GPU 推理)

前提（中国大陆请用清华镜像安装）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
      torch torchvision einops omegaconf addict safetensors huggingface_hub pillow \
      numpy<2 onnx onnxruntime ai-edge-litert onnx2tf
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from addict import Dict
from omegaconf import DictConfig, OmegaConf

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(src_dir, cfg_path):
    """Build the single-image depth module from the repo's code + checkpoint config."""
    sys.path.insert(0, src_dir)
    sys.path.insert(0, os.path.join(src_dir, "src"))
    from depth_anything_3.cfg import create_object  # noqa: E402
    from depth_anything_3.model.da3 import DepthAnything3Net  # noqa: E402

    if cfg_path.endswith(".yaml"):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        net = create_object(cfg.net)
        head = create_object(cfg.head)
        cam_enc = create_object(cfg.cam_enc) if "cam_enc" in cfg else None
        cam_dec = create_object(cfg.cam_dec) if "cam_dec" in cfg else None
    else:
        raw = json.load(open(cfg_path, encoding="utf-8"))
        cfg = raw["config"]
        net = create_object(DictConfig(cfg["net"]))
        head = create_object(DictConfig(cfg["head"]))
        cam_enc = create_object(DictConfig(cfg["cam_enc"])) if "cam_enc" in cfg else None
        cam_dec = create_object(DictConfig(cfg["cam_dec"])) if "cam_dec" in cfg else None
    model = DepthAnything3Net(net=net, head=head, cam_enc=cam_enc, cam_dec=cam_dec)
    model.eval()
    return model


def load_state(model, ckpt_path):
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    # checkpoint keys carry a leading "model." prefix; strip it for the inner module
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ok = len(missing) == 0 and len(unexpected) == 0
    print("missing keys:", len(missing), "| unexpected keys:", len(unexpected))
    if not ok:
        print("missing sample:", list(missing)[:5])
        print("unexpected sample:", list(unexpected)[:5])
    return model


def make_export_wrapper(model, size):
    """A pure-functional forward: raw [0,1] RGB (1,3,H,W) -> depth (1,1,H,W)."""
    m = [MEAN[0], MEAN[1], MEAN[2]]
    s = [STD[0], STD[1], STD[2]]

    def forward(x: torch.Tensor) -> torch.Tensor:
        # embed ImageNet normalization so the graph consumes raw [0,1] pixels
        mean = x.new_tensor(m).reshape(1, 3, 1, 1)
        std = x.new_tensor(s).reshape(1, 3, 1, 1)
        xn = (x - mean) / std          # (1,3,H,W)
        x5 = xn.unsqueeze(1)           # (1,1,3,H,W)  [B,S,C,H,W]
        feats, _ = model.backbone(
            x5,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        out = model.head(feats, size, size, patch_start_idx=0)
        # DualDPT depth head already outputs (B, S=1, 1, H, W) depth
        depth = out.depth[:, 0, 0]      # (B, H, W)
        return depth.unsqueeze(1)       # (B, 1, H, W) NCHW at full resolution

    return forward


class ExportWrapper(torch.nn.Module):
    """A functional module: raw [0,1] RGB (1,3,H,W) -> depth (1,1,H,W)."""

    def __init__(self, model: torch.nn.Module, size: int):
        super().__init__()
        self.m = model
        self.size = size
        self.register_buffer("mean", torch.tensor(MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("up", torch.tensor([0], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        m = self.mean
        s = self.std
        xn = (x - m) / s              # embed ImageNet normalization
        x5 = xn.unsqueeze(1)          # (1,1,3,H,W)  [B,S,C,H,W]
        feats, _ = self.m.backbone(
            x5,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        out = self.m.head(feats, self.size, self.size, patch_start_idx=0)
        # DualDPT depth head outputs (B, S=1, H, W) depth
        depth = out.depth[:, 0]       # (B, H, W)
        return depth.unsqueeze(1)     # (1, 1, H, W) NCHW

def export_onnx(model, size, out_path):
    model.eval()
    wrapper = ExportWrapper(model, size).eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        _ = wrapper(x)  # warm up
        torch.onnx.export(
            wrapper, x, out_path,
            input_names=["image"], output_names=["depth"],
            opset_version=17,
            dynamic_axes=None,
            do_constant_folding=True,
        )
    print("ONNX exported:", out_path)


def convert_to_tflite(onnx_path, out_int8, out_fp16, calib_images=None):
    """ONNX -> TFLite (int8 + fp16). Requires ai-edge-litert + onnx2tf."""
    from ai_edge_litert.convert import Converter, SourceType

    # fp16
    print("converting fp16 ...")
    c = Converter(source_type=SourceType.ONNX, model_path=onnx_path)
    c.convert(output_path=f"{out_fp16}.tmp")
    os.replace(f"{out_fp16}.tmp", out_fp16)

    # int8 with calibration
    print("converting int8 ...")
    c2 = Converter(source_type=SourceType.ONNX, model_path=onnx_path)
    c2.inference_input_type = 8   # INT8
    c2.inference_output_type = 0  # FP32 output to keep app dequantization simple
    c2.representative_dataset = calib_dataset(calib_images, size=518)
    c2.convert(output_path=f"{out_int8}.tmp")
    os.replace(f"{out_int8}.tmp", out_int8)
    print("TFLite ready:", out_int8)


def calib_dataset(images_dir, size=518, batch=1, count=32):
    import cv2
    paths = []
    if images_dir and os.path.isdir(images_dir):
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(("jpg", "jpeg", "png")):
                paths.append(os.path.join(images_dir, f))
    def gen():
        if not paths:
            rng = np.random.default_rng(0)
            for _ in range(count):
                yield [rng.uniform(0, 1, (size, size, 3)).astype(np.float32)]  # NHWC raw
        else:
            import random
            for _ in range(count):
                p = random.choice(paths)
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
                yield [img]  # NHWC raw [0,1]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to model.safetensors")
    ap.add_argument("--src", default="depth-anything-3", help="cloned repo dir")
    ap.add_argument("--out", default="models")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--onnx-only", action="store_true", help="stop after ONNX export")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # locate config:
    # 1) repo yaml 2) repo yaml (alt path) 3) config.json next to the checkpoint
    candidates = [
        os.path.join(args.src, "src", "depth_anything_3", "configs", "da3-base.yaml"),
        os.path.join(args.src, "configs", "da3-base.yaml"),
        os.path.join(os.path.dirname(args.checkpoint), "config.json"),
    ]
    cfg_path = next((p for p in candidates if os.path.exists(p)), None)
    if cfg_path is None:
        raise SystemExit("cannot locate model config (yaml or config.json)")
    print("config:", cfg_path)
    model = build_model(args.src, cfg_path)
    load_state(model, args.checkpoint)

    onnx_path = os.path.join(args.out, "da3_base.onnx")
    export_onnx(model, args.size, onnx_path)
    if args.onnx_only:
        print("ONNX done (tflite skipped). Next: install onnx2tf, then re-run without --onnx-only.")
        return
    convert_to_tflite(
        onnx_path,
        os.path.join(args.out, "da3_base_int8.tflite"),
        os.path.join(args.out, "da3_base_fp16.tflite"),
        calib_images="",
    )
    print("done.")


if __name__ == "__main__":
    main()