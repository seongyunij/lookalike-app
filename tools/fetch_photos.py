"""인물 후보마다 위키미디어 커먼즈에서 사진을 여러 장 모으고 얼굴만 잘라 저장한다.

- 위키백과 문서 → 위키데이터(P373) → 커먼즈 분류의 사진 목록
- 얼굴 검출(OpenCV YuNet)로 얼굴이 하나만 또렷한 사진만 남기고, 얼굴 주변을 정사각형으로 잘라 저장
  (앱에서 사용자가 원 안에 얼굴을 맞추는 것과 비슷한 구도)

사용법: python tools/fetch_photos.py <yunet.onnx 경로> [인물당 최대 장수=10]
결과: tools/.cache/faces/<위키 제목 해시>/*.jpg  (+ index.json)
"""
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from candidates_source import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / ".cache"
FACES = CACHE / "faces"
RAW = CACHE / "raw"
FACES.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "lookalike-app-builder/1.0 (https://github.com/seongyunij/lookalike-app; low-rate batch for face matching data)"}
MAX_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 10
detector = cv2.FaceDetectorYN.create(sys.argv[1], "", (320, 320), 0.8)


def get(url, **kw):
    for attempt in range(6):
        r = requests.get(url, headers=UA, timeout=30, **kw)
        if r.status_code != 429:
            return r
        print(f"  (요청 제한 429 → {30 * (attempt + 1)}초 대기) {url[:70]}", flush=True)
        time.sleep(30 * (attempt + 1))
    return r


def key(title):
    return hashlib.sha1(title.encode()).hexdigest()[:16]


def commons_categories(titles):
    """위키백과 제목 → 커먼즈 분류 이름 (위키데이터 P373)"""
    out = {}
    for k in range(0, len(titles), 50):
        chunk = titles[k:k + 50]
        j = get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
            "prop": "pageprops", "ppprop": "wikibase_item", "titles": "|".join(chunk),
        }).json()["query"]
        alias = {t: t for t in chunk}
        for kk in ("normalized", "redirects"):
            for n in j.get(kk, []):
                for t, cur in list(alias.items()):
                    if cur == n["from"]:
                        alias[t] = n["to"]
        qid = {pg["title"]: (pg.get("pageprops") or {}).get("wikibase_item") for pg in j["pages"]}
        ids = [q for q in qid.values() if q]
        claims = get("https://www.wikidata.org/w/api.php", params={
            "action": "wbgetentities", "format": "json", "props": "claims", "ids": "|".join(ids),
        }).json().get("entities", {})
        for t in chunk:
            q = qid.get(alias[t])
            p = (claims.get(q, {}).get("claims", {}).get("P373") or [{}])[0]
            out[t] = (((p.get("mainsnak") or {}).get("datavalue") or {}).get("value"))
        time.sleep(1)
    return out


SKIP = ("signature", "video", "audio", "logo", "album", "cover", "art", "fan", "tattoo", "house", "concert tour",
        "filmography", "family", "uniform", "costume", "wax", "statue", "drawing", "poster", "screenshot")


def commons_files(cat, want=25, max_cats=8):
    """커먼즈 분류와 하위 분류(연도별·행사별)를 넓게 훑어 사진 URL을 모음"""
    queue, seen, titles = ["Category:" + cat], set(), []
    while queue and len(seen) < max_cats and len(titles) < want:
        c = queue.pop(0)
        if c in seen:
            continue
        seen.add(c)
        j = get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": 2, "list": "categorymembers",
            "cmtitle": c, "cmtype": "subcat|file", "cmlimit": 100, "cmprop": "title|type",
        }).json()
        subs = []
        for m in (j.get("query") or {}).get("categorymembers", []):
            t = m["title"]
            if m["type"] == "file" and t.lower().endswith((".jpg", ".jpeg", ".png")):
                titles.append(t)
            elif m["type"] == "subcat" and not any(k in t.lower() for k in SKIP):
                subs.append(t)
        # 최근 연도 분류를 먼저 (요즘 얼굴과 비슷하도록)
        queue.extend(sorted(subs, reverse=True))
        time.sleep(1.0)
    urls = []
    for k in range(0, min(len(titles), want), 50):
        j = get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo", "iiprop": "url|mime|size", "titles": "|".join(titles[k:k + 50]),
        }).json()
        for pg in (j.get("query") or {}).get("pages", []):
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png") or (ii.get("width") or 0) < 600:
                continue
            # 원본 파일은 요청 제한(429)이 심해서, 표준 크기 썸네일 주소로 받음
            #   .../commons/a/ab/Name.jpg → .../commons/thumb/a/ab/Name.jpg/500px-Name.jpg
            url = ii["url"].split("?")[0]
            base, name = url.rsplit("/", 1)
            urls.append(base.replace("/commons/", "/commons/thumb/", 1) + f"/{name}/500px-{name}")
        time.sleep(1.0)
    return urls


def face_crop(img):
    """얼굴이 하나만 또렷하면 얼굴 주변 정사각형(얼굴 크기의 1.7배)을 돌려줌"""
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(arr)
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4]
    if fw < 50:
        return None
    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.35 * fw * fh:
        return None  # 비슷한 크기 얼굴이 여럿 → 단체 사진은 제외
    cx, cy, side = x + fw / 2, y + fh / 2 + fh * 0.05, max(fw, fh) * 1.7
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    # 사진 밖으로 나가면 회색으로 채움
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (128, 128, 128))
    canvas.paste(img, (-box[0], -box[1]))
    return canvas.resize((256, 256), Image.BICUBIC)


ONLY = [c for c in (sys.argv[3].split(",") if len(sys.argv) > 3 else []) if c]  # 예: "한국 연예인,한국 배우"
people = [(cat, ko, wiki or en) for cat, kind, items in CATEGORIES if kind == "person" and (not ONLY or cat in ONLY)
          for ko, en, wiki in items]
idx_path = FACES / "index.json"
index = json.loads(idx_path.read_text("utf-8")) if idx_path.exists() else {}
print('위키데이터에서 커먼즈 분류 찾는 중…', flush=True)
cats = commons_categories([t for _, _, t in people])
print('커먼즈 분류', sum(1 for v in cats.values() if v), '/', len(cats), flush=True)

for n, (cat, ko, title) in enumerate(people, 1):
    d = FACES / key(title)
    if title in index and index[title].get("done"):
        continue
    d.mkdir(exist_ok=True)
    kept = []
    urls = commons_files(cats[title]) if cats.get(title) else []
    for url in urls:
        if len(kept) >= MAX_PER:
            break
        raw = RAW / hashlib.sha1(url.encode()).hexdigest()
        if not raw.exists():
            r = get(url)
            if not r.ok:
                continue
            raw.write_bytes(r.content)
            time.sleep(1.5)
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw.read_bytes()))).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        crop = face_crop(img)
        if crop is None:
            continue
        f = d / f"{len(kept):02d}.jpg"
        crop.save(f, quality=90)
        kept.append({"file": f.name, "src": url})
    index[title] = {"name": ko, "category": cat, "commons": cats.get(title), "faces": kept, "done": True}
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), "utf-8")
    print(f"[{n}/{len(people)}] {ko}: 커먼즈 {len(urls)}장 → 얼굴 {len(kept)}장", flush=True)

total = sum(len(v["faces"]) for v in index.values())
print("완료: 인물", len(index), "명, 얼굴 사진", total, "장")
