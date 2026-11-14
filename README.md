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
