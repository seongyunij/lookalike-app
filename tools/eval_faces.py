"""정확도 측정: AI가 '처음 보는 사진'으로 그 사람을 몇 위에 찾는지.

질문 사진 = 커먼즈에서 모은 그 사람의 얼굴 사진(위키백과 대표 사진과 같은 사진은 제외).
같은 카테고리 후보들 중 정답 인물의 순위를 잰다. 점수식은 앱과 동일(텍스트·이미지 z점수 가중합),
질문 사진은 앱처럼 4가지(원본·좌우반전·확대 2종) 평균으로 분석.

비교하는 후보 이미지 벡터:
  A 개선 전   = 위키백과 대표 사진 전체 1장       (tools/.cache/candidates_before.json)
  B 현재 배포 = 위키백과 대표 사진의 얼굴만 1장   (candidates.json)
  C 다중 평균 = 커먼즈 얼굴 사진 여러 장 + B의 평균 (질문 사진은 제외하고 평균 = leave-one-out)

사용법: python tools/eval_faces.py <CLIP 모델 폴더>
"""
import base64
import re
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from build_helpers import image_embed  # noqa: E402
from face_gallery import face_images, load_index, robust_mean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W_TEXT = {"한국 연예인": 0.3, "한국 배우": 0.3}


def unpack(b64, scale):
    return np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float32) / 127 * scale


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def fname(url):
    parts = unquote(url.split("?")[0]).split("/")
    return parts[-2] if "thumb" in parts else parts[-1]


def event_key(url):
    """같은 행사에서 찍은 연속 사진 묶기: 파일 이름 끝의 번호·확장자를 떼어낸 앞부분"""
    n = fname(url).rsplit(".", 1)[0]
    n = re.sub(r"[\s_\-]*\(?\d{1,4}\)?$", "", n)
    return re.sub(r"[\s_\-]+", "_", n).lower()


def tta(img):
    """앱과 같은 4가지 평균"""
    vs = []
    for flip in (False, True):
        for zoom in (1.0, 0.9):
            im = ImageOps.mirror(img) if flip else img
            n = im.size[0]
            s = n * zoom
            o = (n - s) / 2
            vs.append(image_embed(im.crop((o, o, o + s, o + s))))
    m = np.mean(vs, 0)
    return m / np.linalg.norm(m)


new = {c["name"]: c for c in json.loads((ROOT / "candidates.json").read_text("utf-8"))["candidates"]}
old = {c["name"]: c for c in json.loads((ROOT / "tools/.cache/candidates_before.json").read_text("utf-8"))["candidates"]}
index = load_index()
title_of = {v["name"]: t for t, v in index.items()}

cats = {}
for c in new.values():
    if c.get("g"):
        cats.setdefault(c["category"], []).append(c["name"])

summary = []
for cat, names in cats.items():
    T = np.stack([unpack(new[n]["t"], new[n]["ts"]) for n in names])
    A = np.stack([unpack(old[n]["i"], old[n]["is"]) for n in names])
    B = np.stack([unpack(new[n]["i"], new[n]["is"]) for n in names])
    faces = []
    for n in names:
        imgs = face_images(title_of[n], index) if n in title_of else []
        faces.append([(src, img) for src, img in imgs])
    gal = [[image_embed(img) for _, img in fl] for fl in faces]  # 갤러리용(단일)
    C_full = np.stack([robust_mean(g + [B[k]])[0] if len(g) >= 2 else B[k] for k, g in enumerate(gal)])
    wt = W_TEXT.get(cat, 0.5)
    ranks = {"A 개선 전": [], "B 현재 배포": [], "C 다중 평균": [], "C 엄격(같은 행사 제외)": []}
    people = 0
    for k, n in enumerate(names):
        fl = faces[k]
        if len(fl) < 2:
            continue
        people += 1
        wiki_file = fname(new[n]["img"])
        for qi, (src, img) in enumerate(fl):
            if fname(src) == wiki_file:
                continue
            q = tta(img)
            zt = z(T @ q)
            for key, G in (("A 개선 전", A), ("B 현재 배포", B)):
                s = wt * zt + (1 - wt) * z(G @ q)
                ranks[key].append(int((s > s[k]).sum()) + 1)
            C = C_full.copy()
            rest = [e for i, e in enumerate(gal[k]) if i != qi]
            C[k] = robust_mean(rest + [B[k]])[0]
            s = wt * zt + (1 - wt) * z(C @ q)
            ranks["C 다중 평균"].append(int((s > s[k]).sum()) + 1)
            ek = event_key(src)
            strict = [e for i, e in enumerate(gal[k]) if i != qi and event_key(fl[i][0]) != ek]
            C2 = C_full.copy()
            C2[k] = robust_mean(strict + [B[k]])[0] if strict else B[k]
            s2 = wt * zt + (1 - wt) * z(C2 @ q)
            ranks["C 엄격(같은 행사 제외)"].append(int((s2 > s2[k]).sum()) + 1)
    nq = len(ranks["A 개선 전"])
    line = [f"\n[{cat}] 후보 {len(names)}명 · 시험 인물 {people}명 · 처음 보는 사진 {nq}장"]
    for key, r in ranks.items():
        r = np.array(r)
        if len(r):
            line.append(f"  {key:14s}  1위 적중 {np.mean(r == 1) * 100:5.1f}%   3위 안 {np.mean(r <= 3) * 100:5.1f}%   5위 안 {np.mean(r <= 5) * 100:5.1f}%   평균 순위 {r.mean():4.1f}")
    print("\n".join(line), flush=True)
    summary.append({"category": cat, "people": people, "queries": nq,
                    **{k: {"top1": float(np.mean(np.array(v) == 1)) if v else None,
                           "top3": float(np.mean(np.array(v) <= 3)) if v else None,
                           "top5": float(np.mean(np.array(v) <= 5)) if v else None,
                           "mean_rank": float(np.mean(v)) if v else None} for k, v in ranks.items()}})

(ROOT / "tools/.cache/eval_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
