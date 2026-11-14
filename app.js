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

function fileToImage(file, keepUrl = false) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { resolve(img); if (!keepUrl) URL.revokeObjectURL(url); };
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
  closeEditor();
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
  // 가이드 원(가운데 76%) 부분만 잘라서 분석
  const dataUrl = squareCrop(video, video.videoWidth, video.videoHeight, true, 1 / GUIDE);
  stopCamera();
  analyze(dataUrl);
});

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  stopCamera();
  openEditor(await fileToImage(file, true));
});

// PC: 사진 파일을 상자에 끌어다 놓기
['dragenter', 'dragover'].forEach((t) => document.querySelector('.camera-box').addEventListener(t, (e) => {
  e.preventDefault();
  document.querySelector('.camera-box').classList.add('drop');
}));
['dragleave', 'drop'].forEach((t) => document.querySelector('.camera-box').addEventListener(t, () => document.querySelector('.camera-box').classList.remove('drop')));
document.querySelector('.camera-box').addEventListener('drop', async (e) => {
  e.preventDefault();
  const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
  if (!file) return;
  stopCamera();
  openEditor(await fileToImage(file, true));
});

/* ---------------- 사진 위치·크기 조정 (원 안에 얼굴 맞추기) ---------------- */
// 원(가이드)은 상자의 가운데 76% 영역. 분석에는 원을 감싸는 정사각형 부분만 사용.
const GUIDE = 0.76;
const box = document.querySelector('.camera-box');
const editImg = $('editImg');
const ed = { img: null, s: 1, base: 1, tx: 0, ty: 0, pointers: new Map(), pinch: null };

function openEditor(img) {
  if (ed.img?.src.startsWith('blob:')) URL.revokeObjectURL(ed.img.src);
  ed.img = img;
  const W = box.clientWidth;
  // 처음엔 사진 짧은 쪽이 원에 딱 맞게
  ed.base = (W * GUIDE) / Math.min(img.naturalWidth, img.naturalHeight);
  ed.s = ed.base;
  ed.tx = (W - img.naturalWidth * ed.s) / 2;
  ed.ty = (W - img.naturalHeight * ed.s) / 2;
  editImg.src = img.src;
  editImg.style.width = img.naturalWidth + 'px';
  editImg.style.height = img.naturalHeight + 'px';
  editImg.hidden = false;
  video.hidden = true;
  $('preview').hidden = true;
  $('cameraPlaceholder').hidden = true;
  $('results').hidden = true;
  $('editControls').hidden = false;
  box.classList.add('editing');
  $('zoom').value = 1;
  applyEdit();
}

function closeEditor() {
  editImg.hidden = true;
  $('editControls').hidden = true;
  box.classList.remove('editing');
  ed.pointers.clear();
  ed.pinch = null;
}

// 원 영역이 항상 사진으로 꽉 차도록 위치를 제한
function clampEdit() {
  const W = box.clientWidth;
  const g0 = (W * (1 - GUIDE)) / 2, g1 = g0 + W * GUIDE;
  const iw = ed.img.naturalWidth * ed.s, ih = ed.img.naturalHeight * ed.s;
  ed.tx = Math.min(g0, Math.max(g1 - iw, ed.tx));
  ed.ty = Math.min(g0, Math.max(g1 - ih, ed.ty));
}

function applyEdit() {
  clampEdit();
  editImg.style.transform = `translate(${ed.tx}px, ${ed.ty}px) scale(${ed.s})`;
}

// (cx, cy) 지점을 기준으로 확대/축소
function zoomTo(newS, cx, cy) {
  newS = Math.max(ed.base, Math.min(ed.base * 4, newS));
  ed.tx = cx - ((cx - ed.tx) * newS) / ed.s;
  ed.ty = cy - ((cy - ed.ty) * newS) / ed.s;
  ed.s = newS;
  $('zoom').value = ed.s / ed.base;
  applyEdit();
}

const local = (e) => { const r = box.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

box.addEventListener('pointerdown', (e) => {
  if (editImg.hidden) return;
  box.setPointerCapture(e.pointerId);
  ed.pointers.set(e.pointerId, local(e));
  ed.pinch = null;
});
box.addEventListener('pointermove', (e) => {
  if (editImg.hidden || !ed.pointers.has(e.pointerId)) return;
  const prev = ed.pointers.get(e.pointerId);
  const cur = local(e);
  ed.pointers.set(e.pointerId, cur);
  if (ed.pointers.size === 1) {
    ed.tx += cur.x - prev.x;
    ed.ty += cur.y - prev.y;
    applyEdit();
  } else if (ed.pointers.size === 2) {
    const [a, b] = [...ed.pointers.values()];
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    if (ed.pinch) {
      ed.tx += mid.x - ed.pinch.mid.x;
      ed.ty += mid.y - ed.pinch.mid.y;
      zoomTo(ed.s * (dist / ed.pinch.dist), mid.x, mid.y);
    }
    ed.pinch = { dist, mid };
  }
});
const endPointer = (e) => { ed.pointers.delete(e.pointerId); ed.pinch = null; };
box.addEventListener('pointerup', endPointer);
box.addEventListener('pointercancel', endPointer);
box.addEventListener('wheel', (e) => {
  if (editImg.hidden) return;
  e.preventDefault();
  const p = local(e);
  zoomTo(ed.s * Math.exp(-e.deltaY * 0.0015), p.x, p.y);
}, { passive: false });

$('zoom').addEventListener('input', (e) => {
  const W = box.clientWidth;
  zoomTo(ed.base * Number(e.target.value), W / 2, W / 2);
});

$('btnEditCancel').addEventListener('click', () => {
  closeEditor();
  $('cameraPlaceholder').hidden = false;
});

$('btnEditOk').addEventListener('click', () => {
  const W = box.clientWidth;
  const g0 = (W * (1 - GUIDE)) / 2;
  const c = $('work');
  c.width = c.height = IMG_SIZE;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  // 원을 감싸는 정사각형 영역 → 원본 사진 좌표
  ctx.drawImage(ed.img, (g0 - ed.tx) / ed.s, (g0 - ed.ty) / ed.s, (W * GUIDE) / ed.s, (W * GUIDE) / ed.s, 0, 0, IMG_SIZE, IMG_SIZE);
  const dataUrl = c.toDataURL('image/jpeg', 0.9);
  closeEditor();
  analyze(dataUrl);
});

/* ---------------- 닮은꼴 분석 ---------------- */
let lastResult = null;

let lastMe = null; // 마지막으로 분석한 내 사진의 임베딩 (성별 필터 바꿀 때 재사용)

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

  try { lastMe = { dataUrl, vec: await embed(dataUrl) }; } catch { list.innerHTML = '<p class="hint">AI 모델이 아직 준비되지 않았어요.</p>'; return; }
  await renderResults();
}

// 성별 필터: all | m | f  (사람 카테고리에만 적용)
let genderFilter = 'all';
try { genderFilter = localStorage.getItem('gender') || 'all'; } catch {}
document.querySelectorAll('.gender-btn').forEach((b) => {
  b.classList.toggle('active', b.dataset.g === genderFilter);
  b.addEventListener('click', () => {
    genderFilter = b.dataset.g;
    try { localStorage.setItem('gender', genderFilter); } catch {}
    document.querySelectorAll('.gender-btn').forEach((x) => x.classList.toggle('active', x === b));
    if (lastMe && !$('results').hidden) renderResults();
  });
});

async function renderResults() {
  const me = lastMe.vec;
  const list = $('resultList');
  const groups = [];
  // 1) 기본 후보: 카테고리별로 "이름(텍스트)과의 유사도" + "대표 사진과의 유사도"를 합산
  for (const [category, all] of builtinByCategory(await builtinReady)) {
    const items = genderFilter === 'all' ? all : all.filter((c) => !c.g || c.g === genderFilter);
    if (items.length < 3) continue;
    const wText = KOREAN.has(category) ? 0.3 : 0.5; // 한국 스타는 AI가 이름을 잘 몰라서 사진 비중↑
    const zt = zscores(items.map((c) => cosine(me, c.t)));
    const zi = zscores(items.map((c) => cosine(me, c.i)));
    const scored = items.map((c, k) => ({ name: c.name, img: c.img, wiki: c.wiki, score: wText * zt[k] + (1 - wText) * zi[k] }));
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

  lastResult = { me: lastMe.dataUrl, groups };
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
        <div class="row card-actions">
          <button class="btn small save">이 결과만 저장</button>
          <a class="credit" target="_blank" rel="noopener" hidden>사진 출처</a>
        </div>
      </div>`;
    el.querySelector('img').src = top.img;
    el.querySelector('.cat').textContent = g.category;
    el.querySelector('.name').textContent = displayName(top, g.category);
    el.querySelector('.rank').textContent = rest.map((r, i) => `${i + 2}위 ${r.name} ${r.pct}%`).join(' · ');
    el.querySelector('.save').addEventListener('click', () => saveImage(gi));
    if (top.wiki) {
      const a = el.querySelector('.credit');
      a.href = 'https://en.wikipedia.org/wiki/' + encodeURIComponent(top.wiki.replace(/ /g, '_'));
      a.hidden = false;
    }
    list.appendChild(el);
  });
  renderCompare();
}

const displayName = (r, category) => (category === '동물' ? `${r.name}상` : r.name);

/* ---------------- 인스타 스토리용 결과 이미지 (1080×1920) ---------------- */
const SITE_URL = 'seongyunij.github.io/lookalike-app';

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCover(ctx, img, x, y, w, h) {
  const s = Math.min(img.naturalWidth / w, img.naturalHeight / h);
  const sw = w * s, sh = h * s;
  ctx.drawImage(img, (img.naturalWidth - sw) / 2, (img.naturalHeight - sh) / 2, sw, sh, x, y, w, h);
}

async function makeStoryImage() {
  const { me, groups } = lastResult;
  const W = 1080, H = 1920;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#ffb199');
  grad.addColorStop(1, '#a18cd1');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  const FONT = '"Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#fff';
  ctx.font = `800 76px ${FONT}`;
  ctx.fillText('🔍 나의 닮은꼴 결과', W / 2, 150);

  // 내 사진 (원형)
  const meImg = await loadImg(me);
  ctx.save();
  ctx.beginPath();
  ctx.arc(W / 2, 400, 190, 0, Math.PI * 2);
  ctx.closePath();
  ctx.clip();
  drawCover(ctx, meImg, W / 2 - 190, 210, 380, 380);
  ctx.restore();
  ctx.lineWidth = 12;
  ctx.strokeStyle = '#fff';
  ctx.beginPath();
  ctx.arc(W / 2, 400, 190, 0, Math.PI * 2);
  ctx.stroke();

  // 카테고리별 1위 카드
  const shown = groups.slice(0, 6);
  const top = 650, gap = 22;
  const cardH = Math.min(190, (H - top - 190 - gap * (shown.length - 1)) / shown.length);
  const imgs = await Promise.all(shown.map((g) => loadImg(g.ranked[0].img).catch(() => null)));
  shown.forEach((g, k) => {
    const r = g.ranked[0];
    const y = top + k * (cardH + gap);
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    roundRect(ctx, 70, y, W - 140, cardH, 36);
    ctx.fill();
    const s = cardH - 30;
    if (imgs[k]) {
      ctx.save();
      roundRect(ctx, 90, y + 15, s, s, 26);
      ctx.clip();
      drawCover(ctx, imgs[k], 90, y + 15, s, s);
      ctx.restore();
    }
    const tx = 90 + s + 36;
    ctx.textAlign = 'left';
    ctx.fillStyle = '#845ef7';
    ctx.font = `700 34px ${FONT}`;
    ctx.fillText(g.category, tx, y + cardH / 2 - 28);
    ctx.fillStyle = '#2b2230';
    ctx.font = `800 54px ${FONT}`;
    let name = displayName(r, g.category);
    while (ctx.measureText(name).width > W - 140 - (tx - 70) - 190 && name.length > 2) name = name.slice(0, -2) + '…';
    ctx.fillText(name, tx, y + cardH / 2 + 38);
    ctx.textAlign = 'right';
    ctx.fillStyle = '#ff6b6b';
    ctx.font = `800 60px ${FONT}`;
    ctx.fillText(`${r.pct}%`, W - 100, y + cardH / 2 + 22);
  });

  ctx.textAlign = 'center';
  ctx.fillStyle = '#fff';
  ctx.font = `800 48px ${FONT}`;
  ctx.fillText('너는 누구 닮았어? 👇', W / 2, H - 120);
  ctx.font = `700 38px ${FONT}`;
  ctx.fillText(SITE_URL, W / 2, H - 60);
  return c;
}

const canvasToBlob = (c) => new Promise((res) => c.toBlob(res, 'image/png'));

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$('btnStory').addEventListener('click', async () => {
  if (!lastResult) return;
  const btn = $('btnStory');
  btn.disabled = true;
  try {
    const blob = await canvasToBlob(await makeStoryImage());
    const file = new File([blob], '닮은꼴_결과.png', { type: 'image/png' });
    // 폰: 공유 시트(인스타그램·카톡 등)로 바로 보내기 / PC: 파일 저장
    if (navigator.canShare?.({ files: [file] })) {
      try { await navigator.share({ files: [file], title: '나의 닮은꼴 결과' }); return; } catch (e) { if (e.name === 'AbortError') return; }
    }
    downloadBlob(blob, file.name);
  } catch (e) {
    console.error(e);
    alert('이미지를 만들지 못했어요. 화면을 캡처해서 공유해주세요!');
  } finally {
    btn.disabled = false;
  }
});

$('btnShareLink').addEventListener('click', async () => {
  const url = resultLink();
  const nick = cleanNick($('nick').value);
  const top = bestOverall();
  const text = top
    ? `${nick ? nick + ' ' : '나 '}${top.name} ${top.pct}% 닮았대 ㅋㅋ 너는 누구 닮았어?`
    : '사진 한 장으로 닮은꼴 연예인·캐릭터·동물 찾기!';
  if (navigator.share) {
    try { await navigator.share({ title: '닮은꼴 찾기', text, url }); return; } catch (e) { if (e.name === 'AbortError') return; }
  }
  try {
    await navigator.clipboard.writeText(`${text}\n${url}`);
    alert('내 결과 링크를 복사했어요! 친구에게 붙여넣기 해주세요.');
  } catch {
    prompt('아래 링크를 복사해주세요', url);
  }
});

/* ---------------- 카카오톡 공유 ---------------- */
// JavaScript 키는 웹에 공개되는 키 (카카오 개발자 콘솔에 등록한 도메인에서만 동작)
const KAKAO_JS_KEY = '02b72b1c17405579051dbf3e6e210d7c';

function kakaoReady() {
  const K = window.Kakao;
  if (!K) return null;
  if (!K.isInitialized()) K.init(KAKAO_JS_KEY);
  return K;
}

$('btnKakao').addEventListener('click', () => {
  const K = kakaoReady();
  if (!K) { alert('카카오톡 공유를 불러오지 못했어요. 링크 공유를 이용해주세요.'); return; }
  const url = resultLink();
  const nick = cleanNick($('nick').value);
  const top = bestOverall();
  const others = shareableGroups()
    .filter((g) => g.ranked[0] !== top)
    .map((g) => `${displayName(g.ranked[0], g.category)} ${g.ranked[0].pct}%`)
    .join(' · ');
  try {
    K.Share.sendDefault({
      objectType: 'feed',
      content: {
        title: top ? `${nick || '나'}, ${top.name} ${top.pct}% 닮았대 ㅋㅋ` : '너는 누구 닮았어? AI 닮은꼴 찾기',
        description: others || '사진 한 장으로 닮은 연예인·캐릭터·동물상 찾기',
        imageUrl: top?.img || 'https://' + SITE_URL + '/og.png',
        link: { mobileWebUrl: url, webUrl: url },
      },
      buttons: [
        { title: '결과 보기', link: { mobileWebUrl: url, webUrl: url } },
        { title: '나도 해보기', link: { mobileWebUrl: 'https://' + SITE_URL + '/', webUrl: 'https://' + SITE_URL + '/' } },
      ],
    });
  } catch (e) {
    console.error(e);
    alert('카카오톡 공유를 열지 못했어요. 링크 공유를 이용해주세요.');
  }
});

/* ---------------- 결과 링크 (친구가 열면 내 결과 + 나도 해보기) ---------------- */
// 링크에는 카테고리별 1위의 이름과 퍼센트만 담고, 사진은 절대 담지 않음
const cleanNick = (v) => String(v || '').replace(/[\u0000-\u001f<>]/g, '').trim().slice(0, 12);

function shareableGroups() {
  return (lastResult?.groups || []).filter((g) => g.category !== '내가 추가한 후보');
}

function bestOverall() {
  const gs = shareableGroups().filter((g) => g.category !== '동물');
  return gs.map((g) => g.ranked[0]).sort((a, b) => b.pct - a.pct)[0] || null;
}

function resultLink() {
  const base = 'https://' + SITE_URL + '/';
  const gs = shareableGroups();
  if (!gs.length) return base;
  const p = new URLSearchParams();
  p.set('r', gs.map((g) => `${g.ranked[0].name}:${g.ranked[0].pct}`).join('|'));
  const nick = cleanNick($('nick').value);
  if (nick) p.set('n', nick);
  return base + '?' + p.toString();
}

let friendResult = null; // [{ name, pct, category, img }]

async function loadFriendResult() {
  const p = new URLSearchParams(location.search);
  const r = p.get('r');
  if (!r) return;
  const byName = new Map((await builtinReady).map((c) => [c.name, c]));
  const items = r.split('|').slice(0, 8).map((part) => {
    const k = part.lastIndexOf(':');
    const c = byName.get(part.slice(0, k));
    const pct = Math.round(Number(part.slice(k + 1)));
    return c && pct >= 0 && pct <= 100 ? { name: c.name, pct, category: c.category, img: c.img } : null;
  }).filter(Boolean);
  if (!items.length) return;
  friendResult = items;
  const nick = cleanNick(p.get('n'));
  $('friendTitle').textContent = `${nick || '친구'}의 닮은꼴 결과 👀`;
  const list = $('friendList');
  list.innerHTML = '';
  for (const it of items) {
    const el = document.createElement('div');
    el.className = 'friend-item';
    el.innerHTML = '<img alt="" referrerpolicy="no-referrer" /><div class="fc"></div><div class="fn"></div><div class="fp"></div>';
    el.querySelector('img').src = it.img;
    el.querySelector('.fc').textContent = it.category;
    el.querySelector('.fn').textContent = displayName(it, it.category);
    el.querySelector('.fp').textContent = it.pct + '%';
    list.appendChild(el);
  }
  $('friendCard').hidden = false;
}

$('btnTryMe').addEventListener('click', () => {
  const card = $('findCard');
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  card.classList.remove('highlight');
  void card.offsetWidth;
  card.classList.add('highlight');
});

// 내 결과가 나오면 링크를 보내준 친구 결과와 비교
function renderCompare() {
  const box = $('compare');
  if (!friendResult || !lastResult) { box.hidden = true; return; }
  const mine = new Map(shareableGroups().map((g) => [g.category, g.ranked[0]]));
  const same = friendResult.filter((f) => mine.get(f.category)?.name === f.name);
  const nick = cleanNick(new URLSearchParams(location.search).get('n')) || '친구';
  // 받침 있으면 '이랑', 없으면 '랑' (예: 준혁이랑, 민수랑)
  const code = nick.charCodeAt(nick.length - 1) - 0xac00;
  const with_ = nick + (code >= 0 && code < 11172 && code % 28 ? '이랑' : '랑');
  const lines = same.length
    ? same.map((f) => `🤝 ${with_} 둘 다 <b></b> 닮은꼴!`)
    : [`${with_} 겹치는 닮은꼴은 없어요. 완전 다른 스타일! 😆`];
  box.innerHTML = lines.map((l) => `<div>${l}</div>`).join('');
  same.forEach((f, i) => { box.children[i].querySelector('b').textContent = displayName(f, f.category); });
  box.hidden = false;
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
    d.candidates.map((c) => ({ name: c.name, category: c.category, img: c.img, wiki: c.wiki, g: c.g, t: unpack(c.t, c.ts), i: unpack(c.i, c.is) }))
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
loadFriendResult();
try { $('nick').value = localStorage.getItem('nick') || ''; } catch {}
$('nick').addEventListener('change', () => { try { localStorage.setItem('nick', cleanNick($('nick').value)); } catch {} });
