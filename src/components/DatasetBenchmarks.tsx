import React, { useState, useEffect } from "react";
import {
  Database,
  BarChart,
  Layers,
  Award,
  TrendingUp,
  Info,
  CheckCircle2,
  Table,
  AlertTriangle,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart as ReBarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { DATASET_STATS } from "../data/datasetData";
import { loadBenchmarkResults, BenchmarkResultBundle } from "../utils/benchmarkResults";

export const DatasetBenchmarks: React.FC = () => {
  const [activeModelCurve, setActiveModelCurve] = useState<"both" | "cnn" | "tl">("both");
  const [hoveredCell, setHoveredCell] = useState<{ actual: string; pred: string; count: number } | null>(null);
  const [bundle, setBundle] = useState<BenchmarkResultBundle | null>(null);

  useEffect(() => {
    loadBenchmarkResults().then(setBundle);
  }, []);

  if (!bundle) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-slate-400 text-sm">
        Loading benchmark results...
      </div>
    );
  }

  const {
    epochsData: TRAINING_EPOCHS_DATA,
    classPerformance: CLASS_PERFORMANCE_METRICS,
    confusionMatrixLabels: CONFUSION_MATRIX_LABELS,
    confusionMatrix: CONFUSION_MATRIX_DATA,
    comparisonTable: MODEL_COMPARISON_TABLE,
  } = bundle;

  return (
    <div className="space-y-6">
      {!bundle.isPlaceholder ? null : (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-4 flex items-start gap-3 text-xs text-amber-300">
          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <span>
            <strong>Simulated data.</strong> No real results found at{" "}
            <code className="text-amber-200">public/results/*.json</code> yet. The numbers below
            illustrate the expected shape of the output, not a measured model. Run{" "}
            <code className="text-amber-200">scripts/train_fer_pipeline.py</code>, then copy its{" "}
            <code className="text-amber-200">results/*.json</code> output into{" "}
            <code className="text-amber-200">public/results/</code> to see real numbers here.
          </span>
        </div>
      )}
      {/* Top Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-sky-400" />
            Objectives 1, 2, 3 & 6: Dataset Curation & Model Benchmarks
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            FER-2013 Dataset • Custom 4-Block CNN vs MobileNetV2 Transfer Learning • 7×7 Confusion Matrix
          </p>
        </div>

        <div className="bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 text-xs font-mono text-slate-300">
          Dataset Path: <strong className="text-sky-300">{DATASET_STATS.baseDirectory}</strong>
        </div>
      </div>

      {/* Dataset Overview & Class Distribution Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Dataset Breakdown Cards */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white">Dataset Assembly & Preprocessing (Objective 1)</h3>
            <p className="text-xs text-slate-400 mt-0.5">35,887 grayscale facial images pre-extracted into folders</p>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase block">Train Set</span>
              <span className="font-bold text-white text-sm mt-0.5 block">{DATASET_STATS.splits.train.count.toLocaleString()}</span>
              <span className="text-[10px] text-sky-400">72% Split</span>
            </div>

            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase block">Val Set</span>
              <span className="font-bold text-white text-sm mt-0.5 block">{DATASET_STATS.splits.val.count.toLocaleString()}</span>
              <span className="text-[10px] text-amber-400">10% Stratified</span>
            </div>

            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 uppercase block">Test Set</span>
              <span className="font-bold text-white text-sm mt-0.5 block">{DATASET_STATS.splits.test.count.toLocaleString()}</span>
              <span className="text-[10px] text-emerald-400">20% Holdout</span>
            </div>
          </div>

          {/* Exclusion Rules */}
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1.5">
            <span className="font-semibold text-slate-200 block text-[11px]">Preprocessing Exclusion Rules:</span>
            {DATASET_STATS.exclusionRules.map((rule, idx) => (
              <div key={idx} className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span>{rule}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Class Distribution Bar Chart */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <BarChart className="w-4 h-4 text-sky-400" />
              Class Distribution & Minority Class Weights
            </h3>
            <span className="text-xs text-rose-400 bg-rose-950/50 px-2 py-0.5 rounded border border-rose-800">
              Disgust Minority Class (~1.5%)
            </span>
          </div>

          <div className="h-56 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <ReBarChart data={DATASET_STATS.classDistribution}>
                <XAxis dataKey="emotion" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                />
                <Bar dataKey="total" name="Total Image Count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </ReBarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Architectures Comparison Table (Objective 2 & 6) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-400" />
            Model Benchmark Comparison (Custom CNN vs Fine-Tuned MobileNetV2)
          </h3>
          <span className="text-xs text-emerald-400 font-semibold">Objective 6 Selection</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Input Resolution</th>
                <th className="py-3 px-4">Parameters</th>
                <th className="py-3 px-4">Model Size</th>
                <th className="py-3 px-4">Inference Latency</th>
                <th className="py-3 px-4">Top-1 Accuracy</th>
                <th className="py-3 px-4">Weighted F1</th>
                <th className="py-3 px-4">Pipeline Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {MODEL_COMPARISON_TABLE.map((row, idx) => {
                const isWinner = row.status.includes("SELECTED WINNER");
                return (
                <tr key={idx} className={isWinner ? "bg-emerald-950/20 font-semibold" : "hover:bg-slate-800/40"}>
                  <td className="py-3 px-4 text-white font-bold">{row.architecture}</td>
                  <td className="py-3 px-4 font-mono">{row.inputShape}</td>
                  <td className="py-3 px-4 font-mono">{row.params}</td>
                  <td className="py-3 px-4 font-mono">{row.sizeMb}</td>
                  <td className="py-3 px-4 font-mono text-amber-300">{row.inferenceLatencyMs}</td>
                  <td className="py-3 px-4 font-bold text-sky-400">{row.top1Acc}</td>
                  <td className="py-3 px-4 font-bold text-emerald-400">{row.weightedF1}</td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2.5 py-1 rounded-full text-[10px] font-bold border ${
                        isWinner
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-slate-800 text-slate-400 border-slate-700"
                      }`}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              );})}
            </tbody>
          </table>
        </div>
      </div>

      {/* 40-Epoch Training Curves */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-sky-400" />
            40-Epoch Training & Validation Loss/Accuracy Curves (Objective 3)
          </h3>

          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setActiveModelCurve("both")}
              className={`px-2.5 py-1 rounded ${activeModelCurve === "both" ? "bg-sky-500 text-white font-semibold" : "text-slate-400"}`}
            >
              Compare Both
            </button>
            <button
              onClick={() => setActiveModelCurve("tl")}
              className={`px-2.5 py-1 rounded ${activeModelCurve === "tl" ? "bg-sky-500 text-white font-semibold" : "text-slate-400"}`}
            >
              MobileNetV2
            </button>
            <button
              onClick={() => setActiveModelCurve("cnn")}
              className={`px-2.5 py-1 rounded ${activeModelCurve === "cnn" ? "bg-sky-500 text-white font-semibold" : "text-slate-400"}`}
            >
              Custom CNN
            </button>
          </div>
        </div>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={TRAINING_EPOCHS_DATA}>
              <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />

              {(activeModelCurve === "both" || activeModelCurve === "tl") && (
                <>
                  <Line type="monotone" dataKey="tlValAcc" name="MobileNetV2 Val Acc (%)" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="tlTrainAcc" name="MobileNetV2 Train Acc (%)" stroke="#34d399" strokeDasharray="3 3" dot={false} />
                </>
              )}

              {(activeModelCurve === "both" || activeModelCurve === "cnn") && (
                <>
                  <Line type="monotone" dataKey="cnnValAcc" name="Custom CNN Val Acc (%)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="cnnTrainAcc" name="Custom CNN Train Acc (%)" stroke="#60a5fa" strokeDasharray="3 3" dot={false} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 7x7 Confusion Matrix & Per-Class Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* 7x7 Confusion Matrix Heatmap */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Table className="w-4 h-4 text-emerald-400" />
              7×7 Confusion Matrix (Holdout Test Set)
            </h3>
            <span className="text-xs text-slate-400">7,178 Total Predictions</span>
          </div>

          <div className="overflow-x-auto">
            <div className="min-w-[420px]">
              {/* Columns Header */}
              <div className="grid grid-cols-8 text-center text-[10px] font-bold text-slate-400 pb-2">
                <div>Actual \ Pred</div>
                {CONFUSION_MATRIX_LABELS.map((lbl) => (
                  <div key={lbl} className="truncate px-1">{lbl}</div>
                ))}
              </div>

              {/* Rows */}
              {CONFUSION_MATRIX_DATA.map((row, rIdx) => {
                const actualLabel = CONFUSION_MATRIX_LABELS[rIdx];
                const rowSum = row.reduce((a, b) => a + b, 0);

                return (
                  <div key={actualLabel} className="grid grid-cols-8 text-center items-center py-1">
                    <div className="text-[10px] font-bold text-slate-300 text-left truncate pr-1">
                      {actualLabel}
                    </div>

                    {row.map((val, cIdx) => {
                      const predLabel = CONFUSION_MATRIX_LABELS[cIdx];
                      const isDiagonal = rIdx === cIdx;
                      const ratio = val / rowSum;

                      return (
                        <div
                          key={cIdx}
                          onMouseEnter={() => setHoveredCell({ actual: actualLabel, pred: predLabel, count: val })}
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{
                            backgroundColor: isDiagonal
                              ? `rgba(16, 185, 129, ${Math.max(0.2, ratio)})`
                              : `rgba(244, 63, 94, ${Math.min(0.7, ratio * 2)})`,
                          }}
                          className={`py-2 text-[10px] font-mono rounded cursor-pointer transition-all ${
                            isDiagonal ? "text-white font-bold" : "text-slate-300"
                          }`}
                        >
                          {val}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>

          {hoveredCell && (
            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
              <span>
                Actual: <strong className="text-white">{hoveredCell.actual}</strong> → Predicted: <strong className="text-sky-300">{hoveredCell.pred}</strong>
              </span>
              <span className="font-mono text-emerald-400 font-bold">{hoveredCell.count} images</span>
            </div>
          )}
        </div>

        {/* Per-Class Metrics Table */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white">Per-Class Classification Report</h3>
            <p className="text-xs text-slate-400 mt-0.5">Precision, Recall, F1-Score & Minority Class Weight</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Class</th>
                  <th className="py-2.5 px-3">Precision</th>
                  <th className="py-2.5 px-3">Recall</th>
                  <th className="py-2.5 px-3">F1-Score</th>
                  <th className="py-2.5 px-3">Weight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {CLASS_PERFORMANCE_METRICS.map((row) => (
                  <tr key={row.emotion} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 font-bold text-white">{row.emotion}</td>
                    <td className="py-2 px-3 font-mono text-slate-300">{row.precision}</td>
                    <td className="py-2 px-3 font-mono text-slate-300">{row.recall}</td>
                    <td className="py-2 px-3 font-mono font-bold text-emerald-400">{row.f1Score}</td>
                    <td className="py-2 px-3 font-mono text-amber-300">{row.weight}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
