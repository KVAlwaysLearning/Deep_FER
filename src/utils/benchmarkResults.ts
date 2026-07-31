import { EpochMetric, ClassPerformance, EmotionType } from "../types";
import {
  TRAINING_EPOCHS_DATA as PLACEHOLDER_EPOCHS_DATA,
  CLASS_PERFORMANCE_METRICS as PLACEHOLDER_CLASS_PERFORMANCE,
  CONFUSION_MATRIX_DATA as PLACEHOLDER_CONFUSION_MATRIX,
  CONFUSION_MATRIX_LABELS,
  MODEL_COMPARISON_TABLE as PLACEHOLDER_COMPARISON_TABLE,
} from "../data/datasetData";

/**
 * Objective 3: real evaluation results, loaded at runtime.
 *
 * scripts/train_fer_pipeline.py writes results/<architecture>_results.json
 * after an actual training + evaluation run. Copy those files into
 * public/results/ in the web app (same filenames) and this module will
 * pick them up automatically, replacing the placeholder numbers in
 * src/data/datasetData.ts. Until then, everything here safely falls back
 * to the clearly-labeled placeholder data.
 */

interface RawTrainingResults {
  isPlaceholder: boolean;
  architecture: "custom_cnn" | "mobilenetv2";
  trainedAt: string;
  epochsRun: number;
  classes: string[];
  history: {
    loss: number[];
    val_loss: number[];
    accuracy: number[];
    val_accuracy: number[];
  };
  classification_report: Record<string, any>;
  confusion_matrix: number[][];
}

export interface BenchmarkResultBundle {
  isPlaceholder: boolean;
  epochsData: EpochMetric[];
  classPerformance: ClassPerformance[];
  confusionMatrix: number[][];
  confusionMatrixLabels: string[];
  comparisonTable: typeof PLACEHOLDER_COMPARISON_TABLE;
  winner: "custom_cnn" | "mobilenetv2" | null;
  hasCnnResults: boolean;
  hasTlResults: boolean;
}

const capitalize = (s: string): EmotionType =>
  (s.charAt(0).toUpperCase() + s.slice(1)) as EmotionType;

async function fetchResultsFile(architecture: "custom_cnn" | "mobilenetv2"): Promise<RawTrainingResults | null> {
  try {
    const res = await fetch(`/results/${architecture}_results.json`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as RawTrainingResults;
    if (data.isPlaceholder) return null;
    return data;
  } catch {
    return null;
  }
}

function toClassPerformance(report: Record<string, any>): ClassPerformance[] {
  return Object.keys(report)
    .filter((k) => !["accuracy", "macro avg", "weighted avg"].includes(k))
    .map((className) => {
      const row = report[className];
      return {
        emotion: capitalize(className),
        precision: Number(row.precision.toFixed(2)),
        recall: Number(row.recall.toFixed(2)),
        f1Score: Number(row["f1-score"].toFixed(2)),
        support: Math.round(row.support),
        weight: 0, // class weight not part of the eval report; display-only field
      };
    })
    .sort((a, b) => b.f1Score - a.f1Score);
}

function alignEpochs(a: number[], b: number[]): [number[], number[]] {
  const len = Math.max(a.length, b.length);
  const padA = [...a, ...Array(len - a.length).fill(a[a.length - 1] ?? 0)];
  const padB = [...b, ...Array(len - b.length).fill(b[b.length - 1] ?? 0)];
  return [padA, padB];
}

function buildEpochsData(cnn: RawTrainingResults | null, tl: RawTrainingResults | null): EpochMetric[] {
  const cnnH = cnn?.history ?? { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };
  const tlH = tl?.history ?? { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };

  const [cnnTL, tlTL] = alignEpochs(cnnH.loss, tlH.loss);
  const [cnnVL, tlVL] = alignEpochs(cnnH.val_loss, tlH.val_loss);
  const [cnnTA, tlTA] = alignEpochs(cnnH.accuracy, tlH.accuracy);
  const [cnnVA, tlVA] = alignEpochs(cnnH.val_accuracy, tlH.val_accuracy);

  const len = Math.max(cnnTL.length, tlTL.length, 1);
  return Array.from({ length: len }, (_, i) => ({
    epoch: i + 1,
    cnnTrainLoss: Number((cnnTL[i] ?? 0).toFixed(3)),
    cnnValLoss: Number((cnnVL[i] ?? 0).toFixed(3)),
    cnnTrainAcc: Number(((cnnTA[i] ?? 0) * 100).toFixed(1)),
    cnnValAcc: Number(((cnnVA[i] ?? 0) * 100).toFixed(1)),
    tlTrainLoss: Number((tlTL[i] ?? 0).toFixed(3)),
    tlValLoss: Number((tlVL[i] ?? 0).toFixed(3)),
    tlTrainAcc: Number(((tlTA[i] ?? 0) * 100).toFixed(1)),
    tlValAcc: Number(((tlVA[i] ?? 0) * 100).toFixed(1)),
  }));
}

function buildComparisonTable(
  cnn: RawTrainingResults | null,
  tl: RawTrainingResults | null,
  deployed: "custom_cnn" | "mobilenetv2" | null,
  betterF1: "custom_cnn" | "mobilenetv2" | null
) {
  const statusFor = (arch: "custom_cnn" | "mobilenetv2"): string => {
    if (deployed === arch) return "DEPLOYED (Wired to Real-Time Pipeline)";
    if (betterF1 === arch) return "Higher Weighted F1";
    return "Baseline Contender";
  };

  const rows = [];
  if (cnn) {
    const f1 = cnn.classification_report["weighted avg"]["f1-score"];
    rows.push({
      architecture: "Custom 4-Block CNN",
      inputShape: "(48, 48, 1)",
      params: "—",
      sizeMb: "—",
      inferenceLatencyMs: "—",
      fps: "—",
      top1Acc: `${(cnn.classification_report["accuracy"] * 100).toFixed(1)}%`,
      top2Acc: "—",
      weightedF1: f1.toFixed(3),
      status: statusFor("custom_cnn"),
    });
  }
  if (tl) {
    const f1 = tl.classification_report["weighted avg"]["f1-score"];
    rows.push({
      architecture: "MobileNetV2 Transfer Learning",
      inputShape: "(48, 48, 3)",
      params: "—",
      sizeMb: "—",
      inferenceLatencyMs: "—",
      fps: "—",
      top1Acc: `${(tl.classification_report["accuracy"] * 100).toFixed(1)}%`,
      top2Acc: "—",
      weightedF1: f1.toFixed(3),
      status: statusFor("mobilenetv2"),
    });
  }
  return rows.length > 0 ? rows : PLACEHOLDER_COMPARISON_TABLE;
}

/**
 * Loads real benchmark results if available; otherwise returns the
 * placeholder dataset with isPlaceholder=true so the UI can show a clear
 * "simulated data" banner instead of presenting it as measured results.
 *
 * deployedArchitecture (from getMLModelStatus() in emotionDetector.ts)
 * reflects which model's weights are ACTUALLY loaded from public/model/ —
 * i.e. what's really running in the Real-Time Detector. This is the
 * source of truth for "DEPLOYED" status and for which model's confusion
 * matrix/per-class metrics are shown as primary, even if the other
 * architecture happens to score a higher weighted F1. Pass null if not
 * yet known (e.g. models haven't loaded), and this falls back to F1
 * comparison for display purposes only — nothing here changes what's
 * actually deployed, that's controlled entirely by what's in
 * public/model/.
 */
export async function loadBenchmarkResults(
  deployedArchitecture: "custom_cnn" | "mobilenetv2" | null = null
): Promise<BenchmarkResultBundle> {
  const [cnn, tl] = await Promise.all([
    fetchResultsFile("custom_cnn"),
    fetchResultsFile("mobilenetv2"),
  ]);

  if (!cnn && !tl) {
    return {
      isPlaceholder: true,
      epochsData: PLACEHOLDER_EPOCHS_DATA,
      classPerformance: PLACEHOLDER_CLASS_PERFORMANCE,
      confusionMatrix: PLACEHOLDER_CONFUSION_MATRIX,
      confusionMatrixLabels: CONFUSION_MATRIX_LABELS,
      comparisonTable: PLACEHOLDER_COMPARISON_TABLE,
      winner: null,
      hasCnnResults: false,
      hasTlResults: false,
    };
  }

  // Which architecture has the better weighted F1 (informational only)
  let betterF1: "custom_cnn" | "mobilenetv2" | null = null;
  if (cnn && tl) {
    const cnnF1 = cnn.classification_report["weighted avg"]["f1-score"];
    const tlF1 = tl.classification_report["weighted avg"]["f1-score"];
    betterF1 = tlF1 >= cnnF1 ? "mobilenetv2" : "custom_cnn";
  } else if (cnn) {
    betterF1 = "custom_cnn";
  } else if (tl) {
    betterF1 = "mobilenetv2";
  }

  // Prefer the ACTUALLY DEPLOYED architecture as the primary result shown
  // (confusion matrix / per-class metrics). Only fall back to the
  // higher-F1 architecture if we don't know what's deployed, or its
  // results file isn't present.
  let primary: "custom_cnn" | "mobilenetv2" | null = null;
  if (deployedArchitecture === "custom_cnn" && cnn) primary = "custom_cnn";
  else if (deployedArchitecture === "mobilenetv2" && tl) primary = "mobilenetv2";
  else primary = betterF1;

  const primaryResult = primary === "mobilenetv2" ? tl! : cnn!;

  return {
    isPlaceholder: false,
    epochsData: buildEpochsData(cnn, tl),
    classPerformance: toClassPerformance(primaryResult.classification_report),
    confusionMatrix: primaryResult.confusion_matrix,
    confusionMatrixLabels: primaryResult.classes.map(capitalize),
    comparisonTable: buildComparisonTable(cnn, tl, deployedArchitecture, betterF1),
    winner: primary,
    hasCnnResults: !!cnn,
    hasTlResults: !!tl,
  };
}
