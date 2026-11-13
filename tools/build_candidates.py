"""기본 후보 데이터(candidates.json)를 만드는 스크립트.

- 후보마다 CLIP 텍스트 임베딩(이름 → 벡터)을 계산
- 위키백과 대표 사진을 받아 CLIP 이미지 임베딩도 계산
- 앱이 쓰는 것과 같은 모델(Xenova/clip-vit-base-patch32, 8bit 양자화 ONNX)을 사용

사용법: python tools/build_candidates.py <모델 폴더>
  모델 폴더에는 tokenizer.json, text_model_quantized.onnx, vision_model_quantized.onnx 가 있어야 함
"""
import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from candidates_source import CATEGORIES, PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "lookalike-app-builder/1.0 (open-source hobby project)"}
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

model_dir = Path(sys.argv[1])
tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
text_sess = ort.InferenceSession(str(model_dir / "text_model_quantized.onnx"))
vis_sess = ort.InferenceSession(str(model_dir / "vision_model_quantized.onnx"))


def norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def text_embed(text):
    ids = tok.encode(text).ids[:77]
    out = text_sess.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
    return norm(out)


def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    # 앱과 같게: 가운데 정사각형 → 224 리사이즈
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((224, 224), Image.BICUBIC)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def image_embed(img):
    return norm(vis_sess.run(None, {"pixel_values": preprocess(img)})[0][0])


def get(url, **kw):
    for attempt in range(6):
        r = requests.get(url, headers=UA, timeout=30, **kw)
        if r.status_code != 429:
            return r
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    return r


def wiki_thumbs(titles):
    """위키백과 대표 사진 URL을 50개씩 묶어서 조회 (캐시 사용)"""
    cache = CACHE / "wiki.json"
    data = json.loads(cache.read_text("utf-8")) if cache.exists() else {}
    todo = [t for t in dict.fromkeys(titles) if t not in data]
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        j = get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
            "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 400, "pilicense": "any",
            "titles": "|".join(chunk),
        }).json()["query"]
        alias = {t: t for t in chunk}
        for key in ("normalized", "redirects"):
            for n in j.get(key, []):
                for t, cur in list(alias.items()):
                    if cur == n["from"]:
                        alias[t] = n["to"]
        pages = {pg["title"]: (pg.get("thumbnail") or {}).get("source") for pg in j["pages"]}
        for t in chunk:
            data[t] = pages.get(alias[t])
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        time.sleep(1)
    return data


def fetch_image(url):
    f = CACHE / ("img_" + hashlib.sha1(url.encode()).hexdigest())
    if not f.exists():
        r = get(url)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(0.5)
    return Image.open(io.BytesIO(f.read_bytes()))


def pack(v):
    """정규화 벡터 → int8 base64 (용량 절약). 복원: int8 / 127 * scale"""
    scale = float(np.abs(v).max())
    q = np.round(v / scale * 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), round(scale, 6)


print("tokenizer check:", tok.encode("a photo of a cat.").ids)

THUMBS = wiki_thumbs([w or en for _, _, items in CATEGORIES for _, en, w in items])
out = []
report = []
for cat, kind, items in CATEGORIES:
    entries = []
    for ko, en, wiki in items:
        t = norm(np.mean([text_embed(p.format(en)) for p in PROMPTS[kind]], axis=0))
        thumb = THUMBS.get(wiki or en)
        img_e = None
        if thumb:
            try:
                img_e = image_embed(fetch_image(thumb))
            except Exception as e:  # noqa: BLE001
                print("  이미지 실패:", ko, e)
        entries.append(dict(ko=ko, en=en, thumb=thumb, t=t, i=img_e))

    # 검증: 위키 사진을 보여줬을 때 AI가 같은 카테고리에서 그 이름을 몇 위로 고르는지
    T = np.stack([e["t"] for e in entries])
    for idx, e in enumerate(entries):
        if e["i"] is None:
            e["rank"] = None
            continue
        sims = T @ e["i"]
        e["rank"] = int((sims > sims[idx]).sum()) + 1
    ranks = [e["rank"] for e in entries if e["rank"]]
    top3 = sum(r <= 3 for r in ranks)
    report.append(f"{cat}: {len(entries)}개, 사진 {len(ranks)}개, AI가 3위 안에 맞힘 {top3}/{len(ranks)}")
    for e in entries:
        flag = "" if e["rank"] and e["rank"] <= 3 else "  <-- " + ("사진 없음" if not e["rank"] else f"{e['rank']}위")
        print(f"[{cat}] {e['ko']}: {e['rank']}{flag}")
        if e["i"] is None:  # 사진이 없으면 결과 화면에 보여줄 수 없으니 제외
            continue
        tb, ts = pack(e["t"])
        rec = {"name": e["ko"], "en": e["en"], "category": cat, "img": e["thumb"], "t": tb, "ts": ts}
        rec["i"], rec["is"] = pack(e["i"])
        out.append(rec)

dest = ROOT / "candidates.json"
dest.write_text(json.dumps({"model": "Xenova/clip-vit-base-patch32", "candidates": out}, ensure_ascii=False), "utf-8")
print("\n".join(report))
print(f"저장: {dest} ({dest.stat().st_size // 1024} KB, {len(out)}개)")
