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
  winner: "custom_cnn" | "mobilenetv2" | null
) {
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
      status: winner === "custom_cnn" ? "SELECTED WINNER (Wired to Real-Time Pipeline)" : "Baseline Contender",
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
      status: winner === "mobilenetv2" ? "SELECTED WINNER (Wired to Real-Time Pipeline)" : "Baseline Contender",
    });
  }
  return rows.length > 0 ? rows : PLACEHOLDER_COMPARISON_TABLE;
}

/**
 * Loads real benchmark results if available; otherwise returns the
 * placeholder dataset with isPlaceholder=true so the UI can show a clear
 * "simulated data" banner instead of presenting it as measured results.
 */
export async function loadBenchmarkResults(): Promise<BenchmarkResultBundle> {
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
    };
  }

  // Section 2.3 selection rule: weighted F1 is the tiebreaker
  let winner: "custom_cnn" | "mobilenetv2" | null = null;
  if (cnn && tl) {
    const cnnF1 = cnn.classification_report["weighted avg"]["f1-score"];
    const tlF1 = tl.classification_report["weighted avg"]["f1-score"];
    winner = tlF1 >= cnnF1 ? "mobilenetv2" : "custom_cnn";
  } else if (cnn) {
    winner = "custom_cnn";
  } else if (tl) {
    winner = "mobilenetv2";
  }

  const winningResult = winner === "mobilenetv2" ? tl! : cnn!;

  return {
    isPlaceholder: false,
    epochsData: buildEpochsData(cnn, tl),
    classPerformance: toClassPerformance(winningResult.classification_report),
    confusionMatrix: winningResult.confusion_matrix,
    confusionMatrixLabels: winningResult.classes.map(capitalize),
    comparisonTable: buildComparisonTable(cnn, tl, winner),
    winner,
  };
}
