"""모아둔 얼굴 사진(tools/.cache/faces)으로 인물별 이미지 임베딩을 만드는 공용 함수."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

FACES = Path(__file__).resolve().parent / ".cache" / "faces"


def key(title):
    return hashlib.sha1(title.encode()).hexdigest()[:16]


def load_index():
    p = FACES / "index.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def face_images(title, index):
    info = index.get(title) or {}
    d = FACES / key(title)
    return [(f["src"], Image.open(d / f["file"]).convert("RGB")) for f in info.get("faces", [])]


def robust_mean(embs):
    """다른 사람이 섞인 사진(분류 오류·단체 사진)을 걸러내고 평균.
    평균과의 유사도가 무리에서 크게 벗어나는 사진은 제외."""
    E = np.stack(embs)
    if len(E) < 3:
        m = E.mean(0)
        return m / np.linalg.norm(m), list(range(len(E)))
    keep = list(range(len(E)))
    for _ in range(2):
        m = E[keep].mean(0)
        m /= np.linalg.norm(m)
        sims = E @ m
        ks = sims[keep]
        thr = max(ks.mean() - 2 * ks.std(), 0.55)
        keep = [i for i in range(len(E)) if sims[i] >= thr]
        if len(keep) < 2:
            keep = list(np.argsort(-sims)[:2])
    m = E[keep].mean(0)
    return m / np.linalg.norm(m), keep


_detector = None


def face_crop(img, yunet_path, scale=1.7, min_size=40, allow_multi=False):
    """얼굴 주변 정사각형(얼굴 크기 × scale)을 256×256으로. 얼굴이 없거나(단체 사진이면) None.
    앱에서 사용자가 원 안에 얼굴을 맞춘 구도와 비슷하게 맞춘다."""
    import cv2
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.8)
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(arr)
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4]
    if fw < min_size:
        return None
    if not allow_multi and len(faces) > 1 and faces[1][2] * faces[1][3] > 0.35 * fw * fh:
        return None
    cx, cy, side = x + fw / 2, y + fh / 2 + fh * 0.05, max(fw, fh) * scale
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (128, 128, 128))
    canvas.paste(img.convert("RGB"), (-box[0], -box[1]))
    return canvas.resize((256, 256), Image.BICUBIC)
