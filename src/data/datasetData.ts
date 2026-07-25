import { EpochMetric, ClassPerformance } from "../types";

// DATASET_STATS below is REAL confirmed data from the actual dataset
// inspection (Blueprint Section 1.1) — class counts, directory structure,
// and exclusion rules are all accurate, not placeholder.
export const DATASET_STATS = {
  title: "FER-2013 Pre-Extracted Directory Dataset Curation",
  baseDirectory: "/content/dataset/archive-3/",
  totalImages: 35887,
  splits: {
    train: { count: 25839, percentage: 72 },
    val: { count: 2870, percentage: 8 },
    test: { count: 7178, percentage: 20 },
  },
  exclusionRules: [
    "__MACOSX/ shadow folder strictly excluded",
    "._* hidden AppleDouble system files filtered",
  ],
  classDistribution: [
    { emotion: "Angry", train: 3995, test: 958, total: 4953, percent: 13.8, weight: 1.03 },
    { emotion: "Disgust", train: 436, test: 111, total: 547, percent: 1.5, weight: 9.38, minority: true },
    { emotion: "Fear", train: 4097, test: 1024, total: 5121, percent: 14.3, weight: 1.00 },
    { emotion: "Happy", train: 7215, test: 1774, total: 8989, percent: 25.0, weight: 0.57 },
    { emotion: "Neutral", train: 4965, test: 1233, total: 6198, percent: 17.3, weight: 0.83 },
    { emotion: "Sad", train: 4830, test: 1247, total: 6077, percent: 16.9, weight: 0.84 },
    { emotion: "Surprise", train: 3171, test: 831, total: 4002, percent: 11.2, weight: 1.28 },
  ],
};

/**
 * PLACEHOLDER DATA FROM HERE DOWN — training curves, confusion matrix,
 * per-class metrics, and the model comparison table below are illustrative
 * (matching the SHAPE of what a real training run produces) but are NOT
 * measured from an actual trained model. They exist so the UI has
 * something sensible to render before training has happened.
 *
 * Once scripts/train_fer_pipeline.py has been run and its
 * results/<architecture>_results.json output has been copied into
 * public/results/, src/utils/benchmarkResults.ts will load the REAL
 * results at runtime and the Benchmarks tab will use those instead — see
 * loadBenchmarkResults() there. Do not treat the numbers below as evidence
 * of actual model performance.
 */
export const DATASET_BENCHMARKS_ARE_PLACEHOLDER = true;

// 40 Epochs training history comparison between Custom 4-block CNN & Fine-Tuned MobileNetV2
export const TRAINING_EPOCHS_DATA: EpochMetric[] = Array.from({ length: 40 }, (_, i) => {
  const ep = i + 1;
  // Custom CNN curve progression
  const cnnTL = Math.max(0.42, 1.85 * Math.exp(-0.08 * ep) + 0.15 * Math.sin(ep * 0.4));
  const cnnVL = Math.max(0.58, 1.88 * Math.exp(-0.07 * ep) + 0.08 + (ep > 25 ? 0.003 * (ep - 25) : 0));
  const cnnTA = Math.min(88.4, 32 + 54 * (1 - Math.exp(-0.09 * ep)));
  const cnnVA = Math.min(74.2, 30 + 43 * (1 - Math.exp(-0.08 * ep)));

  // Transfer Learning MobileNetV2 curve progression
  const tlTL = Math.max(0.28, 1.62 * Math.exp(-0.11 * ep) + 0.08 * Math.sin(ep * 0.3));
  const tlVL = Math.max(0.44, 1.65 * Math.exp(-0.10 * ep) + 0.04);
  const tlTA = Math.min(94.1, 38 + 55 * (1 - Math.exp(-0.12 * ep)));
  const tlVA = Math.min(81.6, 36 + 45 * (1 - Math.exp(-0.10 * ep)));

  return {
    epoch: ep,
    cnnTrainLoss: Number(cnnTL.toFixed(3)),
    cnnValLoss: Number(cnnVL.toFixed(3)),
    cnnTrainAcc: Number(cnnTA.toFixed(1)),
    cnnValAcc: Number(cnnVA.toFixed(1)),
    tlTrainLoss: Number(tlTL.toFixed(3)),
    tlValLoss: Number(tlVL.toFixed(3)),
    tlTrainAcc: Number(tlTA.toFixed(1)),
    tlValAcc: Number(tlVA.toFixed(1)),
  };
});

export const CLASS_PERFORMANCE_METRICS: ClassPerformance[] = [
  { emotion: "Happy", precision: 0.88, recall: 0.89, f1Score: 0.88, support: 1774, weight: 0.57 },
  { emotion: "Surprise", precision: 0.84, recall: 0.82, f1Score: 0.83, support: 831, weight: 1.28 },
  { emotion: "Neutral", precision: 0.78, recall: 0.76, f1Score: 0.77, support: 1233, weight: 0.83 },
  { emotion: "Angry", precision: 0.72, recall: 0.70, f1Score: 0.71, support: 958, weight: 1.03 },
  { emotion: "Sad", precision: 0.68, recall: 0.69, f1Score: 0.68, support: 1247, weight: 0.84 },
  { emotion: "Fear", precision: 0.66, recall: 0.62, f1Score: 0.64, support: 1024, weight: 1.00 },
  { emotion: "Disgust", precision: 0.64, recall: 0.59, f1Score: 0.61, support: 111, weight: 9.38 },
];

export const CONFUSION_MATRIX_LABELS = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Neutral",
  "Sad",
  "Surprise",
];

// 7x7 confusion matrix counts for Test set (7178 total)
export const CONFUSION_MATRIX_DATA = [
  // Actual \ Predicted
  [671,  18,  82,  34,  51,  88,  14], // Angry (958)
  [ 12,  65,   8,   5,   7,  11,   3], // Disgust (111)
  [ 68,   6, 635,  28,  72, 138,  77], // Fear (1024)
  [ 21,   2,  18, 1579, 78,  52,  24], // Happy (1774)
  [ 42,   3,  48,  82, 937, 102,  19], // Neutral (1233)
  [ 74,   5, 112,  56, 118, 860,  22], // Sad (1247)
  [ 16,   1,  64,  28,  22,  18, 682], // Surprise (831)
];

export const MODEL_COMPARISON_TABLE = [
  {
    architecture: "Custom 4-Block CNN",
    inputShape: "(48, 48, 1)",
    params: "1,482,951",
    sizeMb: "5.9 MB",
    inferenceLatencyMs: "12 ms / frame",
    fps: "83 FPS",
    top1Acc: "74.2%",
    top2Acc: "88.6%",
    weightedF1: "0.738",
    status: "Baseline Contender",
  },
  {
    architecture: "MobileNetV2 Transfer Learning",
    inputShape: "(224, 224, 3)",
    params: "2,318,279",
    sizeMb: "9.2 MB",
    inferenceLatencyMs: "18 ms / frame",
    fps: "55 FPS",
    top1Acc: "81.6%",
    top2Acc: "93.4%",
    weightedF1: "0.812",
    status: "SELECTED WINNER (Wired to Real-Time Pipeline)",
  },
];
