"""CLIP 이미지 임베딩 (앱과 같은 Xenova/clip-vit-base-patch32 양자화 모델).
모델 폴더는 첫 번째 명령줄 인자(sys.argv[1])로 받는다."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_sess = None


def _session():
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(str(Path(sys.argv[1]) / "vision_model_quantized.onnx"))
    return _sess


def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((224, 224), Image.BICUBIC)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def image_embed(img):
    v = _session().run(None, {"pixel_values": preprocess(img)})[0][0]
    return v / (np.linalg.norm(v) + 1e-9)
