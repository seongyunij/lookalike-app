"""정확도 측정: '처음 보는 사진'으로 그 사람을 몇 위에 찾는지.

각 인물의 얼굴 사진을 하나씩 질문 사진으로 쓰고(그 사진은 후보 쪽에서 제외 = leave-one-out),
같은 카테고리 후보들 중 정답 인물의 순위를 잰다. 앱과 같은 점수식(텍스트·이미지 z점수 가중합)을 사용.

비교:
  기존  = 위키백과 대표 사진 1장(가운데 자르기) [candidates.json 의 i]
  개선  = 커먼즈 얼굴 사진 여러 장의 평균 (질문 사진 제외)

사용법: python tools/eval_faces.py <CLIP 모델 폴더>
"""
import base64
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_helpers import image_embed  # noqa: E402
from face_gallery import face_images, load_index, robust_mean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def unpack(b64, scale):
    return np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32) / 127 * scale


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def fname(url):
    # .../commons/thumb/a/ab/File_name.jpg/500px-File_name.jpg → File_name.jpg
    parts = unquote(url.split("?")[0]).split("/")
    return parts[-2] if "thumb" in parts else parts[-1]


cands = json.loads((ROOT / "candidates.json").read_text("utf-8"))["candidates"]
index = load_index()
by_title = {v["name"]: t for t, v in index.items()}

cats = {}
for c in cands:
    if c.get("g"):  # 사람 카테고리만
        cats.setdefault(c["category"], []).append(c)

W_TEXT_OLD = {"한국 연예인": 0.3, "한국 배우": 0.3}
results = {}
for cat, items in cats.items():
    T = np.stack([unpack(c["t"], c["ts"]) for c in items])
    I_old = np.stack([unpack(c["i"], c["is"]) for c in items])
    faces = {}
    for k, c in enumerate(items):
        title = by_title.get(c["name"])
        imgs = face_images(title, index) if title else []
        faces[k] = [(src, image_embed(im)) for src, im in imgs]
    full = {}
    for k, fl in faces.items():
        full[k] = robust_mean([e for _, e in fl])[0] if len(fl) >= 2 else I_old[k]

    stats = {name: [] for name in ["기존", "개선(text0.3)", "개선(text0.15)", "개선(text0)"]}
    for k, c in enumerate(items):
        fl = faces[k]
        if len(fl) < 3:
            continue
        wiki_file = fname(c["img"])
        for qi, (src, q) in enumerate(fl):
            if fname(src) == wiki_file:
                continue  # 위키 대표 사진과 같은 사진은 기존 방식에 불공평하게 유리하므로 제외
            zt = z(T @ q)
            wt_old = W_TEXT_OLD.get(cat, 0.5)
            s_old = wt_old * zt + (1 - wt_old) * z(I_old @ q)
            stats["기존"].append(int((s_old > s_old[k]).sum()) + 1)
            I_new = np.stack([full[j] for j in range(len(items))])
            I_new[k] = robust_mean([e for i, (_, e) in enumerate(fl) if i != qi])[0]
            zi = z(I_new @ q)
            for wt, name in [(0.3, "개선(text0.3)"), (0.15, "개선(text0.15)"), (0.0, "개선(text0)")]:
                s = wt * zt + (1 - wt) * zi
                stats[name].append(int((s > s[k]).sum()) + 1)
    results[cat] = stats
    n = len(stats["기존"])
    print(f"\n[{cat}] 후보 {len(items)}명, 질문 사진 {n}장")
    for name, ranks in stats.items():
        r = np.array(ranks)
        print(f"  {name:14s} 1위 {np.mean(r == 1) * 100:5.1f}%  3위안 {np.mean(r <= 3) * 100:5.1f}%  5위안 {np.mean(r <= 5) * 100:5.1f}%  평균순위 {r.mean():.1f}")
