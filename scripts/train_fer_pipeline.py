"""
Facial Emotion Recognition System — Model Training & Export Pipeline
FER-2013 Dataset Preprocessing, Custom CNN & MobileNetV2 Transfer Learning

This script performs Objectives 1, 2, 3, and 6:
  1. Data Preprocessing & Augmentation (35,887 images, __MACOSX exclusion,
     stratified 90/10 train/val split, class weight balancing)
  2. Model Architectures:
      - Model A: Custom 4-Block ConvNet (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout)
      - Model B: Pretrained MobileNetV2 Transfer Learning with fine-tuned top layers
  3. Training & Evaluation (40 Epochs, Adam, ReduceLROnPlateau, EarlyStopping,
     classification report, confusion matrix) — results are written to
     results/<architecture>_results.json for the web app to consume.
  4. Optimization & Export: Convert best model to TensorFlow.js format for
     zero-latency in-browser edge inference (public/model/).

Usage:
  python scripts/train_fer_pipeline.py --data_dir /content/dataset/archive-3 --architecture mobilenetv2
  python scripts/train_fer_pipeline.py --data_dir /content/dataset/archive-3 --architecture custom_cnn
"""

import os
import glob
import json
import argparse
import subprocess
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Target Classes & Constants (Blueprint Section 1.1 — confirmed dataset structure)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)
IMG_SIZE = (48, 48)  # Standard FER-2013 input dimensions
BATCH_SIZE = 64
EPOCHS = 40
VAL_SPLIT = 0.10
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train FER-2013 Emotion Recognition Models")
    parser.add_argument("--data_dir", type=str, default="./dataset/archive-3",
                         help="Path to the dataset root containing train/ and test/ "
                              "subfolders (e.g. /content/dataset/archive-3)")
    parser.add_argument("--architecture", type=str, choices=["custom_cnn", "mobilenetv2"],
                         default="mobilenetv2")
    parser.add_argument("--output_dir", type=str, default="./public/model",
                         help="Target export path for the converted TF.js model")
    parser.add_argument("--results_dir", type=str, default="./results",
                         help="Where to write the JSON training/evaluation results "
                              "consumed by the web app's Benchmarks tab")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Objective 1: Data Collection & Preprocessing
# --------------------------------------------------------------------------

def clean_macosx_and_load_paths(split_dir):
    """
    Objective 1.1: Index every real image under split_dir/<class>/*.jpg while
    filtering out __MACOSX/ and AppleDouble "._" shadow files.
    """
    print(f"[Dataset] Indexing files in {split_dir} ...")
    valid_filepaths = []
    labels = []

    for idx, emo in enumerate(EMOTION_CLASSES):
        search_pattern = os.path.join(split_dir, emo, "*.*")
        for filepath in glob.glob(search_pattern):
            if "__MACOSX" in filepath or os.path.basename(filepath).startswith("."):
                continue
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_filepaths.append(filepath)
                labels.append(idx)

    print(f"[Dataset] Indexed {len(valid_filepaths)} valid images across {NUM_CLASSES} emotion categories.")
    if len(valid_filepaths) == 0:
        raise RuntimeError(
            f"No images found under {split_dir}. Check --data_dir points at the "
            f"folder containing train/<class>/*.jpg and test/<class>/*.jpg."
        )
    return np.array(valid_filepaths), np.array(labels)


def compute_dataset_class_weights(labels):
    """
    Objective 1.1: Inverse class weights to offset the severe Disgust
    class imbalance (~6-17x smaller than the other classes).
    """
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    print(f"[Dataset] Calculated Class Weights: {class_weight_dict}")
    return class_weight_dict


def _load_and_preprocess_image(filepath, label, img_size, channels):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def build_tf_dataset(filepaths, labels, img_size, channels, batch_size, shuffle=True):
    """
    Objective 1.3: Builds a batched tf.data pipeline. Random flip/rotation/zoom
    augmentation is applied inside the model itself (see build_custom_cnn /
    build_mobilenetv2_transfer) so it only runs during training, not eval.
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda fp, lb: _load_and_preprocess_image(fp, lb, img_size, channels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# --------------------------------------------------------------------------
# Objective 2: Model Development
# --------------------------------------------------------------------------

def build_custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Objective 2.1: Custom 4-Block ConvNet Architecture
    Block 1: Conv2D(32) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.25)
    Block 2: Conv2D(64) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.25)
    Block 3: Conv2D(128) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.30)
    Block 4: Conv2D(256) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.40)
    Classifier: Flatten -> Dense(512) -> BatchNorm -> Dropout(0.50) -> Softmax(7)
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.40)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="emotions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Custom_4Block_CNN")


def build_mobilenetv2_transfer(input_shape=(48, 48, 3), num_classes=7):
    """
    Objective 2.2: MobileNetV2 Transfer Learning Architecture.
    Phase 1 (this build): base frozen except top ~30 layers, trained with the
    classification head. Phase 2 fine-tuning at a lower LR is a documented
    follow-up run (see README) rather than baked into a single call, since it
    needs its own optimizer/LR schedule per the blueprint's two-phase plan.
    """
    inputs = keras.Input(shape=input_shape)

    x = layers.Resizing(96, 96)(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="emotions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_FER2013")


# --------------------------------------------------------------------------
# Objective 3: Training & Evaluation
# --------------------------------------------------------------------------

def train_and_evaluate(args):
    print(f"\n=======================================================")
    print(f"Starting Training Pipeline: Architecture={args.architecture}")
    print(f"=======================================================\n")

    channels = 1 if args.architecture == "custom_cnn" else 3
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    # --- Objective 1: index files, exclude __MACOSX/._ junk, compute class weights ---
    filepaths, labels = clean_macosx_and_load_paths(train_dir)
    class_weight_dict = compute_dataset_class_weights(labels)

    train_fp, val_fp, train_lb, val_lb = train_test_split(
        filepaths, labels, test_size=VAL_SPLIT, random_state=SEED, stratify=labels
    )
    print(f"[Dataset] Train: {len(train_fp)} images | Validation: {len(val_fp)} images (stratified {int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)} split)")

    test_fp, test_lb = clean_macosx_and_load_paths(test_dir)
    print(f"[Dataset] Test (held out, untouched until final evaluation): {len(test_fp)} images")

    train_ds = build_tf_dataset(train_fp, train_lb, IMG_SIZE, channels, args.batch_size, shuffle=True)
    val_ds = build_tf_dataset(val_fp, val_lb, IMG_SIZE, channels, args.batch_size, shuffle=False)
    test_ds = build_tf_dataset(test_fp, test_lb, IMG_SIZE, channels, args.batch_size, shuffle=False)

    # --- Objective 2: build selected architecture ---
    if args.architecture == "custom_cnn":
        model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], channels), num_classes=NUM_CLASSES)
        learning_rate = 1e-3
    else:
        model = build_mobilenetv2_transfer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], channels), num_classes=NUM_CLASSES)
        learning_rate = 1e-4

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )

    model_path = f"./best_fer_model_{args.architecture}.h5"
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    print(f"\n[Training Loop] Running up to {args.epochs} epochs (EarlyStopping patience=8)...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # --- Objective 3.2: Evaluation on the held-out test set ---
    print("\n[Evaluation] Running inference on the held-out test set...")
    y_true, y_pred = [], []
    for images, batch_labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    report_text = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print("\n[Evaluation] Classification Report:")
    print(report_text)
    print("[Evaluation] Confusion Matrix:")
    print(np.array(cm))

    # --- Write results JSON consumed by the web app's Benchmarks tab ---
    os.makedirs(args.results_dir, exist_ok=True)
    results = {
        "isPlaceholder": False,
        "architecture": args.architecture,
        "trainedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "epochsRun": len(history.history["loss"]),
        "classes": EMOTION_CLASSES,
        "classWeights": class_weight_dict,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }
    results_path = os.path.join(args.results_dir, f"{args.architecture}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Wrote real evaluation results to {results_path}")
    print("Copy/merge this file into the web app (see README) so the Benchmarks")
    print("tab shows real numbers instead of placeholder data.")

    # --- Objective 6: Export to TensorFlow.js ---
    print("\n=======================================================")
    print("Objective 6: Converting Model to TensorFlow.js")
    print("=======================================================")
    os.makedirs(args.output_dir, exist_ok=True)
    convert_cmd = ["tensorflowjs_converter", "--input_format=keras", model_path, args.output_dir]
    try:
        print(f"[Export] Running: {' '.join(convert_cmd)}")
        subprocess.run(convert_cmd, check=True)
        print(f"[Export] TF.js model successfully exported to {args.output_dir}")
        print(f"[Export] Drop the resulting model.json + weight shards into the web app's")
        print(f"[Export] public/model/ folder — emotionDetector.ts will auto-detect and use it.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[Export] Automatic conversion failed ({e}).")
        print("Run manually after installing the converter:")
        print("  pip install tensorflowjs")
        print(f"  tensorflowjs_converter --input_format=keras {model_path} {args.output_dir}")

    print("=======================================================\n")
    return model, results


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
