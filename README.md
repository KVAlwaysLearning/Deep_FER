# DeepFER: Facial Emotion Recognition System

Real-time facial emotion recognition — a fully static, client-side web
application built for **Deep Learning for Computer Vision**. No backend
server, no API keys, no data ever leaves the browser: all face detection
and emotion classification run on-device via TensorFlow.js.

See `PROJECT_BLUEPRINT.txt` for the full project specification, and
`DeepFER_Project_Report.docx` for the complete write-up covering all
objectives, repository structure, dataset analysis, and training results.

---

## Project Overview

DeepFER classifies faces into seven emotions — **angry, disgust, fear,
happy, neutral, sad, surprise** — using the FER-2013 dataset (35,887
images). The project spans eight specific objectives:

1. **Data Collection & Preprocessing** — FER-2013 dataset curation, class-weight balancing, augmentation
2. **Model Development** — a Custom 4-Block CNN and a MobileNetV2 Transfer Learning model
3. **Training & Evaluation** — 40-epoch training with EarlyStopping, full classification reports
4. **Real-Time Processing** — on-device webcam/image inference with temporal smoothing
5. **Application Development** — a 5-tab web app for HCI and mental-health-monitoring use cases
6. **Performance Optimization** — head-to-head model comparison and selection
7. **Documentation & Reporting** — this README, the blueprint, and the project report
8. **Deployment & Testing** — static, privacy-first deployment (Vercel/GitHub Pages)

### Current Status

Both architectures have been trained and evaluated on the same held-out
7,178-image test set:

| Architecture | Test Accuracy | Weighted F1 | Status |
|---|---|---|---|
| Custom 4-Block CNN | 52.67% | 0.519 | Baseline contender |
| **MobileNetV2 Transfer Learning** | **54.35%** | **0.534** | **Deployed** (higher weighted F1) |

The MobileNetV2 model is the one currently wired into the live
application's real-time inference pipeline. Full per-class metrics,
confusion matrices, and training curves are in `public/results/*.json`
and in the project report.

---

## Repository Structure

```
Deep_FER/
├── PROJECT_BLUEPRINT.txt / .md   # Full project specification (11 sections)
├── README.md                     # This file
├── requirements.txt              # Python deps for training
├── package.json                  # Node deps for the web app
├── scripts/
│   └── train_fer_pipeline.py     # Full training pipeline (Objectives 1, 2, 3, 6)
├── public/
│   ├── model/                    # Deployed TF.js model (model.json + .bin shards)
│   └── results/                  # Real training/evaluation results (JSON)
└── src/
    ├── components/                # 5 app tabs (Detector, Mood, Explainer, Benchmarks, Docs)
    ├── data/                      # Dataset stats + sample images
    └── utils/                     # ML engine, benchmark loader, CNN explainer math
```

See `DeepFER_Project_Report.docx` for a complete file-by-file description.

---

## Run Locally

**Prerequisites:** Node.js (v18+)

```bash
npm install
npm run dev
```

Build for production (static output in `dist/`):
```bash
npm run build
npm run preview   # preview the production build locally
```

Deploy the contents of `dist/` to any static host — GitHub Pages, Vercel,
or Netlify. No environment variables or server config needed (Section 8.1
of the blueprint).

---

## Training Your Own Model

The app ships with a trained MobileNetV2 model already in `public/model/`,
so it works out of the box. To retrain or experiment with a new model:

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python scripts/train_fer_pipeline.py --data_dir /path/to/archive-3 --architecture mobilenetv2
python scripts/train_fer_pipeline.py --data_dir /path/to/archive-3 --architecture custom_cnn
```
`--data_dir` should point at a folder containing `train/<class>/*.jpg` and
`test/<class>/*.jpg` subfolders (the FER-2013 layout). This:
- excludes `__MACOSX/` and hidden `._*` files automatically,
- computes class weights to offset the severe Disgust-class imbalance,
- trains for up to 40 epochs with EarlyStopping (patience=8),
- evaluates on the held-out test set and writes `results/<architecture>_results.json`,
- attempts to auto-convert the trained model to TensorFlow.js in `./public/model`.

### 3. ⚠️ Important: fix the Keras 3 → TensorFlow.js export format

Models trained with Keras 3.x export `model.json` in a format that
**TensorFlow.js's loader does not fully support out of the box**. If you
see `An InputLayer should be passed either a 'batchInputShape' or an
'inputShape'` (or the model silently fails to load with no visible
error), the exported `model.json` needs a one-time post-processing fix:

- `InputLayer` configs use the key `batch_shape` — TF.js expects `batch_input_shape`
- Every layer's `dtype` field is serialized as a nested `DTypePolicy` object (`{"class_name": "DTypePolicy", "config": {"name": "float32"}}`) instead of the plain string `"float32"` that TF.js expects

Both must be patched in the converted `model.json` before deploying it.
A short Python script to do this:

```python
import json

path = "public/model/model.json"
with open(path) as f:
    d = json.load(f)

def fix(obj):
    if isinstance(obj, dict):
        if obj.get('class_name') == 'InputLayer' and 'batch_shape' in obj.get('config', {}):
            obj['config']['batch_input_shape'] = obj['config'].pop('batch_shape')
        if 'dtype' in obj and isinstance(obj['dtype'], dict) and obj['dtype'].get('class_name') == 'DTypePolicy':
            obj['dtype'] = obj['dtype']['config']['name']
        for v in obj.values():
            fix(v)
    elif isinstance(obj, list):
        for v in obj:
            fix(v)

fix(d['modelTopology'])
with open(path, 'w') as f:
    json.dump(d, f)
```

Run this once against any newly converted `model.json` before copying it
into `public/model/`.

### 4. Copy results into the web app
```bash
cp results/*.json public/results/
```

### 5. Reload the app

`emotionDetector.ts` auto-detects `public/model/model.json` at load time
and identifies which architecture it is (by scanning for a nested
`mobilenetv2` layer), switching real-time detection and the CNN Explainer
tab to use it automatically. The Benchmarks tab shows whichever model is
actually deployed as "DEPLOYED," independent of which one scores a higher
weighted F1 — so a manual choice always overrides the automatic ranking
if you deploy the lower-scoring model on purpose.

See `public/model/README.md` and `public/results/README.md` for further
detail on the expected file layout.

---

## Architecture Notes

- **Custom 4-Block CNN**: input `48×48×1` (grayscale), ~1.48M parameters, trained from scratch with in-model augmentation (flip/rotate/zoom/translate).
- **MobileNetV2 Transfer Learning**: input `48×48×3`, ImageNet-pretrained backbone with the top 30 layers fine-tuned, internally upsampled to `96×96` before the backbone.
- Both models are trained on identical data splits (stratified 90/10 train/val, held-out test set) for a fair comparison.
- Only **one** model's converted files live in `public/model/` at a time — that's what's actually deployed. Both models' evaluation results can coexist in `public/results/` for comparison purposes without affecting what's deployed.

---

## Privacy & Deployment

DeepFER is deployed as a fully static site with **no backend server**.
All face detection (via face-api.js) and emotion classification (via the
deployed TF.js model) run entirely inside the browser — no webcam frame,
image, or biometric data is ever transmitted to or stored on an external
server. This was a deliberate choice given the mental-health-monitoring
use case's privacy sensitivity (see Section 8.1 of the blueprint for the
full rationale).

---

## What's Real vs. Placeholder

- **Real:** face/landmark detection, on-device inference for both trained
  architectures, 5-frame EMA temporal smoothing, mood analytics + CSV/JSON
  export, the CNN Explainer's convolution-math walkthrough and (where
  extractable) real feature-map activations, static deployment, and the
  training results shown in the Benchmarks tab.
- **Fallback only:** if `public/model/` or `public/results/` are ever
  emptied, the app gracefully falls back to face-api.js's generic
  pretrained model and clearly-labeled placeholder benchmark data, rather
  than breaking.
