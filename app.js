// 닮은꼴 찾기 — 브라우저 안에서 CLIP 모델로 이미지 유사도를 계산합니다.
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const IMG_SIZE = 256; // 저장/분석용 정사각형 크기
const $ = (id) => document.getElementById(id);

/* ---------------- AI 모델 ---------------- */
let processor, model;
const modelReady = (async () => {
  const status = $('modelStatus');
  try {
    processor = await AutoProcessor.from_pretrained(MODEL_ID);
    model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
          status.textContent = `AI 모델 다운로드 중… ${Math.round(p.progress)}%`;
        }
      },
    });
    status.textContent = 'AI 준비 완료!';
    status.className = 'status ok';
    setTimeout(() => (status.hidden = true), 1500);
  } catch (e) {
    console.error(e);
    status.textContent = 'AI 모델을 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침 해주세요.';
    status.className = 'status err';
    throw e;
  }
})();

// 이미지(dataURL) → 정규화된 임베딩 벡터(Float32Array)
async function embed(dataUrl) {
  await modelReady;
  const image = await RawImage.read(dataUrl);
  const inputs = await processor(image);
  const { image_embeds } = await model(inputs);
  const v = Float32Array.from(image_embeds.data);
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

const cosine = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/* ---------------- 저장소 (IndexedDB) ---------------- */
const dbp = new Promise((resolve, reject) => {
  const req = indexedDB.open('lookalike', 1);
  req.onupgradeneeded = () =>
    req.result.createObjectStore('candidates', { keyPath: 'id', autoIncrement: true });
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});
async function store(mode) {
  return (await dbp).transaction('candidates', mode).objectStore('candidates');
}
const wrap = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });
const getAll = async () => wrap((await store('readonly')).getAll());
const addCand = async (c) => wrap((await store('readwrite')).add(c));
const delCand = async (id) => wrap((await store('readwrite')).delete(id));

/* ---------------- 이미지 유틸 ---------------- */
// 가운데를 정사각형으로 잘라 IMG_SIZE로 줄인 JPEG dataURL
function squareCrop(source, sw, sh, mirror = false, zoom = 1) {
  const c = $('work');
  c.width = c.height = IMG_SIZE;
  const ctx = c.getContext('2d');
  const side = Math.min(sw, sh) / zoom;
  const sx = (sw - side) / 2, sy = (sh - side) / 2;
  ctx.save();
  if (mirror) { ctx.translate(IMG_SIZE, 0); ctx.scale(-1, 1); }
  ctx.drawImage(source, sx, sy, side, side, 0, 0, IMG_SIZE, IMG_SIZE);
  ctx.restore();
  return c.toDataURL('image/jpeg', 0.88);
}

function fileToImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { resolve(img); URL.revokeObjectURL(url); };
    img.onerror = reject;
    img.src = url;
  });
}

const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.crossOrigin = 'anonymous'; i.referrerPolicy = 'no-referrer'; i.onload = () => res(i); i.onerror = rej; i.src = src; });

/* ---------------- 탭 ---------------- */
document.querySelectorAll('.tab').forEach((t) =>
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === t));
    $('tab-find').hidden = t.dataset.tab !== 'find';
    $('tab-manage').hidden = t.dataset.tab !== 'manage';
  })
);

/* ---------------- 카메라 ---------------- */
let stream = null;
const video = $('video');

$('btnCamera').addEventListener('click', async () => {
  if (stream) return stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    video.hidden = false;
    $('preview').hidden = true;
    $('cameraPlaceholder').hidden = true;
    $('btnShot').disabled = false;
    $('btnCamera').textContent = '카메라 끄기';
  } catch (e) {
    console.error(e);
    alert('카메라를 켤 수 없어요. 브라우저에서 카메라 권한을 허용했는지 확인해주세요.\n(카메라는 https 주소나 localhost에서만 동작해요)');
  }
});

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $('btnShot').disabled = true;
  $('btnCamera').textContent = '카메라 켜기';
}

$('btnShot').addEventListener('click', () => {
  // 가이드 원 크기(가운데 76%)에 맞춰 살짝 확대해서 얼굴 위주로 자름
  const dataUrl = squareCrop(video, video.videoWidth, video.videoHeight, true, 1.25);
  stopCamera();
  analyze(dataUrl);
});

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  stopCamera();
  const img = await fileToImage(file);
  analyze(squareCrop(img, img.naturalWidth, img.naturalHeight));
});

/* ---------------- 닮은꼴 분석 ---------------- */
let lastResult = null;

async function analyze(dataUrl) {
  video.hidden = true;
  const preview = $('preview');
  preview.src = dataUrl;
  preview.hidden = false;
  $('cameraPlaceholder').hidden = true;

  const results = $('results');
  const list = $('resultList');
  results.hidden = false;
  list.innerHTML = '<p class="hint">AI가 열심히 비교하는 중… 🤔</p>';
  results.scrollIntoView({ behavior: 'smooth' });

  let me;
  try { me = await embed(dataUrl); } catch { list.innerHTML = '<p class="hint">AI 모델이 아직 준비되지 않았어요.</p>'; return; }

  const groups = [];
  // 1) 기본 후보: 카테고리별로 "이름(텍스트)과의 유사도" + "대표 사진과의 유사도"를 합산
  for (const [category, items] of builtinByCategory(await builtinReady)) {
    const wText = KOREAN.has(category) ? 0.3 : 0.5; // 한국 스타는 AI가 이름을 잘 몰라서 사진 비중↑
    const zt = zscores(items.map((c) => cosine(me, c.t)));
    const zi = zscores(items.map((c) => cosine(me, c.i)));
    const scored = items.map((c, k) => ({ name: c.name, img: c.img, score: wText * zt[k] + (1 - wText) * zi[k] }));
    groups.push({ category, ranked: rankGroup(scored) });
  }

  // 2) 내가 직접 추가한 후보 (같은 이름 사진이 여러 장이면 가장 비슷한 사진 기준)
  const mine = await getAll();
  if (mine.length) {
    const byName = new Map();
    for (const c of mine) {
      const sim = cosine(me, c.embedding);
      const g = byName.get(c.name);
      if (!g || sim > g.sim) byName.set(c.name, { name: c.name, img: c.image, sim });
    }
    const arr = [...byName.values()];
    const z = arr.length > 2 ? zscores(arr.map((g) => g.sim)) : arr.map((g) => (g.sim - 0.6) * 10);
    groups.push({ category: '내가 추가한 후보', ranked: rankGroup(arr.map((g, k) => ({ ...g, score: z[k] }))) });
  }

  lastResult = { me: dataUrl, groups };
  list.innerHTML = '';
  groups.forEach((g, gi) => {
    const [top, ...rest] = g.ranked;
    const el = document.createElement('div');
    el.className = 'result';
    el.innerHTML = `
      <img alt="" crossorigin="anonymous" referrerpolicy="no-referrer" />
      <div class="info">
        <div class="cat"></div>
        <div class="name"></div>
        <div class="bar"><span style="width:${top.pct}%"></span></div>
        <div><span class="pct">${top.pct}%</span> 닮음</div>
        <div class="comment">${comment(top.pct, g.category)}</div>
        <div class="rank"></div>
        <button class="btn small save">결과 이미지 저장</button>
      </div>`;
    el.querySelector('img').src = top.img;
    el.querySelector('.cat').textContent = g.category;
    el.querySelector('.name').textContent = g.category === '동물' ? `${top.name}상` : top.name;
    el.querySelector('.rank').textContent = rest.map((r, i) => `${i + 2}위 ${r.name} ${r.pct}%`).join(' · ');
    el.querySelector('.save').addEventListener('click', () => saveImage(gi));
    list.appendChild(el);
  });
}

const KOREAN = new Set(['한국 연예인', '한국 배우']);

function zscores(arr) {
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  const sd = Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length) || 1;
  return arr.map((x) => (x - m) / sd);
}

// 점수 → 상위 3개 + 보기 좋은 퍼센트 (평균보다 얼마나 튀게 닮았는지 기준)
function rankGroup(scored) {
  const z = scored.length > 2 ? zscores(scored.map((s) => s.score)) : scored.map((s) => s.score);
  return scored
    .map((s, k) => ({ ...s, pct: Math.round(Math.max(30, Math.min(98, 55 + 12 * z[k]))) }))
    .sort((a, b) => b.pct - a.pct || b.score - a.score)
    .slice(0, 3);
}

function comment(p, category) {
  if (category === '동물') {
    if (p >= 85) return '🐾 빼박이에요. 전생에 이 동물이었을지도?';
    if (p >= 70) return '🐾 친구들이 다 인정할 동물상!';
    return '🐾 은은하게 풍기는 동물상이에요.';
  }
  if (p >= 90) return '🤯 이 정도면 도플갱어! 혹시 본인 아니에요?';
  if (p >= 80) return '😲 길에서 사인 요청 받을 수도 있겠는데요?';
  if (p >= 70) return '😆 분위기가 꽤 비슷해요!';
  return '🙂 자세히 보면 살짝 닮았어요.';
}

$('btnRetry').addEventListener('click', () => {
  $('results').hidden = true;
  $('preview').hidden = true;
  $('cameraPlaceholder').hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// 내 사진 + 해당 카테고리 1위 사진을 나란히 붙인 결과 이미지 저장
async function saveImage(groupIndex) {
  const g = lastResult.groups[groupIndex];
  const top = g.ranked[0];
  const label = g.category === '동물' ? `${top.name}상` : top.name;
  try {
    const [a, b] = await Promise.all([loadImg(lastResult.me), loadImg(top.img)]);
    const c = document.createElement('canvas');
    const S = 400, P = 24;
    c.width = S * 2 + P * 3;
    c.height = S + 150;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff7f0';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(a, P, P, S, S);
    // 상대 사진은 가운데를 정사각형으로 잘라서
    const s = Math.min(b.naturalWidth, b.naturalHeight);
    ctx.drawImage(b, (b.naturalWidth - s) / 2, (b.naturalHeight - s) / 2, s, s, S + P * 2, P, S, S);
    ctx.fillStyle = '#2b2230';
    ctx.textAlign = 'center';
    ctx.font = 'bold 34px "Malgun Gothic", sans-serif';
    ctx.fillText(`나 ≈ ${label}  (${top.pct}% 닮음)`, c.width / 2, S + P + 60);
    ctx.font = '22px "Malgun Gothic", sans-serif';
    ctx.fillStyle = '#7a6f80';
    ctx.fillText(`닮은꼴 찾기 · ${g.category}`, c.width / 2, S + P + 104);
    const link = document.createElement('a');
    link.download = `닮은꼴_${label}.png`;
    link.href = c.toDataURL('image/png');
    link.click();
  } catch (e) {
    console.error(e);
    alert('이미지를 만들지 못했어요. 화면을 캡처해서 공유해주세요!');
  }
}

/* ---------------- 기본 후보 (candidates.json) ---------------- */
function unpack(b64, scale) {
  const bin = atob(b64);
  const v = new Float32Array(bin.length);
  for (let k = 0; k < bin.length; k++) {
    const x = bin.charCodeAt(k);
    v[k] = ((x > 127 ? x - 256 : x) / 127) * scale;
  }
  return v;
}

const builtinReady = fetch('candidates.json')
  .then((r) => r.json())
  .then((d) =>
    d.candidates.map((c) => ({ name: c.name, category: c.category, img: c.img, t: unpack(c.t, c.ts), i: unpack(c.i, c.is) }))
  )
  .catch((e) => { console.error(e); return []; });

function builtinByCategory(list) {
  const m = new Map();
  for (const c of list) {
    if (!m.has(c.category)) m.set(c.category, []);
    m.get(c.category).push(c);
  }
  return m;
}

/* ---------------- 후보 관리 ---------------- */
$('candFiles').addEventListener('change', async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  const name = $('candName').value.trim();
  const category = $('candCat').value;
  const status = $('addStatus');
  if (!name) { alert('이름을 먼저 입력해주세요!'); return; }
  if (!files.length) return;

  let done = 0;
  for (const f of files) {
    status.textContent = `분석 중… (${done + 1}/${files.length})`;
    try {
      const img = await fileToImage(f);
      const image = squareCrop(img, img.naturalWidth, img.naturalHeight);
      const embedding = Array.from(await embed(image));
      await addCand({ name, category, image, embedding, model: MODEL_ID });
      done++;
    } catch (err) {
      console.error(err);
    }
  }
  status.textContent = `✅ "${name}" 사진 ${done}장 추가 완료!`;
  renderCands();
});

async function renderCands() {
  const cands = await getAll();
  $('count').textContent = new Set(cands.map((c) => c.name)).size;
  const list = $('candList');
  list.innerHTML = '';
  if (!cands.length) {
    list.innerHTML = '<div class="empty">아직 후보가 없어요. 위에서 추가해보세요!</div>';
    return;
  }
  for (const c of cands) {
    const el = document.createElement('div');
    el.className = 'cand';
    el.innerHTML = `<img alt="" /><button class="del" title="삭제">✕</button><div class="n"></div>`;
    el.querySelector('img').src = c.image;
    el.querySelector('.n').textContent = c.name;
    el.querySelector('.del').addEventListener('click', async () => {
      if (!confirm(`"${c.name}" 사진 1장을 삭제할까요?`)) return;
      await delCand(c.id);
      renderCands();
    });
    list.appendChild(el);
  }
}

$('btnExport').addEventListener('click', async () => {
  const cands = (await getAll()).map(({ id, ...rest }) => rest);
  if (!cands.length) { alert('내보낼 후보가 없어요.'); return; }
  const blob = new Blob([JSON.stringify({ version: 1, candidates: cands })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '닮은꼴_후보목록.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

$('importFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const list = data.candidates || [];
    let n = 0;
    for (const c of list) {
      if (!c.name || !c.image) continue;
      // 임베딩이 없거나 다른 모델로 만든 경우 다시 계산
      const embedding = c.model === MODEL_ID && c.embedding?.length ? c.embedding : Array.from(await embed(c.image));
      await addCand({ name: String(c.name).slice(0, 40), category: c.category || '기타', image: c.image, embedding, model: MODEL_ID });
      n++;
    }
    alert(`${n}장의 후보 사진을 가져왔어요!`);
    renderCands();
  } catch (err) {
    console.error(err);
    alert('파일을 읽지 못했어요. 이 앱에서 내보낸 파일인지 확인해주세요.');
  }
});

renderCands();
