# 닮은꼴 찾기

사진을 찍으면 AI(CLIP)가 미리 넣어둔 후보(연예인·배우·애니 캐릭터 등) 중 가장 닮은 것을 찾아주는 웹앱.
모든 분석은 브라우저 안에서 이뤄지고 사진은 업로드되지 않는다.

로컬 실행: `start.bat` 더블클릭 → http://localhost:8765

## 기본 후보
연예인·배우·해외 스타·캐릭터·동물 약 230개가 `candidates.json`에 들어 있다.
후보마다 CLIP 텍스트 임베딩(이름)과 위키백과 대표 사진의 이미지 임베딩을 미리 계산해 두었고, 사진은 위키백과에서 바로 불러온다.

후보를 바꾸려면 `tools/candidates_source.py`를 고친 뒤 다시 생성한다:

```
pip install onnxruntime tokenizers numpy pillow requests
python tools/build_candidates.py <CLIP 모델 폴더>
```

## 배포할 때 주의
GitHub Pages는 파일을 약 10분간 캐시한다. `app.js`나 `style.css`를 고치면 `index.html`(과 `privacy.html`)의 `?v=숫자`를 1 올려야 사용자에게 새 파일이 바로 적용된다.

## 얼굴 인식
- 앱: Google MediaPipe BlazeFace(`models/blaze_face_short_range.tflite`, Apache-2.0)로 사용자 사진의 얼굴을 찾아 원 안에 자동으로 맞춘다.
- 후보 데이터: OpenCV YuNet으로 위키 사진에서 얼굴만 잘라 임베딩한다 (`python tools/build_candidates.py <CLIP 폴더> <yunet.onnx>`).
- 두 검출기의 상자 크기 차이는 실측(0.817배)으로 보정했다 (`app.js`의 `FACE_BOX_SCALE`).

## 정확도 측정 (2026-11-14, `python tools/eval_faces.py <CLIP 폴더>`)
처음 보는 사진(커먼즈 얼굴 사진, 위키 대표 사진과 같은 행사 사진은 비교 대상에서 제외)으로 정답 인물을 1위로 찾는 비율.

| 카테고리 | 개선 전 (위키 사진 1장) | 얼굴 크롭 1장 | 얼굴 여러 장 평균 (현재) |
|---|---|---|---|
| 한국 연예인 (212장) | 38.7% | 42.0% | **53.8%** |
| 한국 배우 (197장) | 29.4% | 31.5% | **41.6%** |

후보 사진 수집: `python tools/fetch_photos.py <yunet.onnx> 5 "한국 연예인,한국 배우"` (위키미디어 커먼즈, 1.5초에 1장, 연락처 포함 UA)
