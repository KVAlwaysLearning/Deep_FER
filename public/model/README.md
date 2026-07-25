# public/model/

This folder is where your trained model goes **after** conversion to
TensorFlow.js format, so the app can load it at runtime and switch from
face-api.js's generic pretrained model to your own trained Custom CNN /
MobileNetV2 Transfer Learning model (Section 2.3 of the project blueprint).

## How to populate this folder

1. Train a model:
   ```
   python scripts/train_fer_pipeline.py --data_dir /path/to/archive-3 --architecture mobilenetv2
   ```
   (or `--architecture custom_cnn`)

2. The script automatically attempts to convert the trained model to TF.js
   format and write it here via `--output_dir ./public/model` (the
   default). If the automatic conversion step fails (e.g. `tensorflowjs`
   isn't installed), run it manually:
   ```
   pip install tensorflowjs
   tensorflowjs_converter --input_format=keras ./best_fer_model_<architecture>.h5 ./public/model
   ```

3. You should end up with files like:
   ```
   public/model/model.json
   public/model/group1-shard1of3.bin
   public/model/group1-shard2of3.bin
   ...
   ```

4. Reload the app — `emotionDetector.ts`'s `loadOnDeviceModels()` will
   automatically detect `public/model/model.json` and use it for
   classification instead of face-api's generic fallback. No code changes
   needed. The CNN Explainer tab (Section 5.3) will also automatically
   switch from simulated to real activations once this is present.

## Why this is empty by default

No model has been trained yet as of this build — see
`scripts/train_fer_pipeline.py`. Until this folder is populated, the app
runs on face-api.js's pretrained expression net so it's still fully
functional, just not using a model trained on your specific FER-2013
dataset.
