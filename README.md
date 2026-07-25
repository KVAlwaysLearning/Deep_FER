# DeepFER: Facial Emotion Recognition System

Real-time facial emotion recognition — fully static, client-side web app.
No backend server, no API keys required. All inference runs on-device via
TensorFlow.js. See `PROJECT_BLUEPRINT.txt` for the full spec this build
implements.

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:
   ```
   npm install
   ```
2. Run the dev server:
   ```
   npm run dev
   ```
3. Build for production (static output in `dist/`):
   ```
   npm run build
   npm run preview   # to preview the production build locally
   ```

Deploy the contents of `dist/` to any static host — GitHub Pages, Vercel,
or Netlify (Section 8.1 of the blueprint).

## Using your own trained model (optional but recommended)

Out of the box, the app uses face-api.js's generic pretrained model for
face detection and expression classification, so it works immediately
with no training required. To use a model trained on your own FER-2013
dataset instead (Objectives 1-3 & 6 of the blueprint):

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train (pick one architecture, or run both to compare):
   ```
   python scripts/train_fer_pipeline.py --data_dir /path/to/archive-3 --architecture mobilenetv2
   python scripts/train_fer_pipeline.py --data_dir /path/to/archive-3 --architecture custom_cnn
   ```
   This trains, evaluates on the held-out test set, and:
   - writes `results/<architecture>_results.json` (real metrics/history/confusion matrix)
   - attempts to auto-convert the trained model to TF.js in `./public/model`
     (falls back to printed manual instructions if `tensorflowjs` isn't installed)
3. Copy the results JSON files into the web app so the Benchmarks tab uses
   real data instead of placeholders:
   ```
   cp results/*.json public/results/
   ```
4. Reload the app. `emotionDetector.ts` auto-detects `public/model/model.json`
   and switches real-time detection + the CNN Explainer tab to use it,
   automatically selecting whichever architecture scored the higher
   weighted F1 (Section 2.3 selection rule) if you trained both.

See `public/model/README.md` and `public/results/README.md` for details.

## What's real vs placeholder right now

- **Real:** face/landmark detection, on-device inference pipeline, 5-frame
  EMA temporal smoothing, mood analytics + CSV/JSON export, the CNN
  Explainer's convolution-math walkthrough, static deployment.
- **Placeholder until you train:** the exact training curves, confusion
  matrix, and per-class F1 scores shown in the Benchmarks tab — clearly
  labeled in the UI, and automatically replaced once you populate
  `public/results/` as described above.
