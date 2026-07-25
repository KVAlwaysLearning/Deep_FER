import React, { useEffect, useState } from "react";
import {
  FileText,
  ShieldCheck,
  BookOpen,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";
import { loadOnDeviceModels, getMLModelStatus } from "../utils/emotionDetector";
import { loadBenchmarkResults, BenchmarkResultBundle } from "../utils/benchmarkResults";

interface ObjectiveCardProps {
  number: number;
  title: string;
  description: string;
  done: boolean;
  pendingNote?: string;
}

const ObjectiveCard: React.FC<ObjectiveCardProps> = ({ number, title, description, done, pendingNote }) => (
  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
    <div className="flex items-center justify-between">
      <span className="font-bold text-sky-400 uppercase text-[11px]">Objective {number}</span>
      <span
        className={`px-2 py-0.5 rounded text-[10px] border ${
          done
            ? "bg-emerald-950 text-emerald-300 border-emerald-800"
            : "bg-amber-950 text-amber-300 border-amber-800"
        }`}
      >
        {done ? "Completed" : "Pending Real Training Run"}
      </span>
    </div>
    <h4 className="font-semibold text-white">{title}</h4>
    <p className="text-slate-400 text-[11px] leading-relaxed">{description}</p>
    {!done && pendingNote && (
      <p className="text-amber-300/90 text-[10px] leading-relaxed flex items-start gap-1 pt-1 border-t border-slate-800">
        <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
        {pendingNote}
      </p>
    )}
  </div>
);

export const ProjectDocumentation: React.FC = () => {
  const [bundle, setBundle] = useState<BenchmarkResultBundle | null>(null);
  const [usingCustomModel, setUsingCustomModel] = useState(false);

  useEffect(() => {
    loadOnDeviceModels().then(() => {
      setUsingCustomModel(getMLModelStatus().usingCustomModel);
    });
    loadBenchmarkResults().then(setBundle);
  }, []);

  const trainingDone = !!bundle && !bundle.isPlaceholder;
  const winnerLabel =
    bundle?.winner === "mobilenetv2"
      ? "MobileNetV2 Transfer Learning"
      : bundle?.winner === "custom_cnn"
      ? "Custom 4-Block CNN"
      : "not yet determined (train both architectures to decide)";

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-sky-400" />
            Objectives 7 & 8: System Specification & Final Implementation Report
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Complete Architectural Blueprint Specification • Ethics & Privacy • Client-Side Deployment
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5" />
            On-Device Inference — No Server, No Data Leaves the Browser
          </span>
        </div>
      </div>

      {/* 8 Objectives Blueprint Mapping */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-6">
        <div className="border-b border-slate-800 pb-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-sky-400" />
            Full Blueprint Mapping across 8 Core Objectives
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Status reflects PROJECT_BLUEPRINT.txt fulfillment — Objectives 1/2/3/6 status is
            determined live from whether real training results exist in{" "}
            <code className="text-sky-300">public/results/</code>, not hardcoded.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <ObjectiveCard
            number={1}
            title="Data Collection and Preprocessing"
            description="FER-2013 35,887 image dataset, folder-per-class layout under /content/dataset/archive-3/, with __MACOSX/ and ._* exclusion enforced in scripts/train_fer_pipeline.py's clean_macosx_and_load_paths()."
            done={trainingDone}
            pendingNote="Preprocessing code exists and is wired up, but has not yet been run against the real dataset. Run scripts/train_fer_pipeline.py to execute it."
          />

          <ObjectiveCard
            number={2}
            title="Model Development (Custom CNN & Transfer Learning)"
            description="Custom 4-Block CNN (~1.48M params) and MobileNetV2 Transfer Learning architectures are both implemented in scripts/train_fer_pipeline.py, with class weighting for the Disgust minority class."
            done={trainingDone}
            pendingNote="Architectures are defined and buildable, but neither has been trained yet — no weights exist in public/model/ (the app currently falls back to face-api.js's generic pretrained model)."
          />

          <ObjectiveCard
            number={3}
            title="Training and Evaluation Framework"
            description="40-epoch Adam training with ReduceLROnPlateau/EarlyStopping, classification report, and confusion matrix evaluation, implemented in scripts/train_fer_pipeline.py."
            done={trainingDone}
            pendingNote="No training run has completed yet, so no classification report or confusion matrix exists. The Benchmarks tab is showing clearly-labeled placeholder data."
          />

          <ObjectiveCard
            number={4}
            title="Real-Time Processing Pipeline"
            description="On-device face detection + landmark extraction + expression classification via TensorFlow.js, with 5-frame EMA temporal smoothing to eliminate inter-frame flicker."
            done={true}
            pendingNote={
              usingCustomModel
                ? undefined
                : "Currently classifying with face-api.js's generic pretrained model, not a model trained on your dataset — this will switch automatically once public/model/ is populated."
            }
          />

          <ObjectiveCard
            number={5}
            title="Application Development (HCI & Mental Health)"
            description="Full-featured web application with live webcam stream, sample image catalog, interactive CNN Explainer (with Live Mode) with 3×3 conv math walkthrough, and a Mental Health Mood Analytics dashboard with CSV/JSON export."
            done={true}
          />

          <ObjectiveCard
            number={6}
            title="System Integration and Optimization"
            description={`Model selection rule (Section 2.3): whichever architecture scores higher weighted F1 is wired into the pipeline. Current winner: ${winnerLabel}.`}
            done={trainingDone}
            pendingNote="Selection can't be made until both architectures have real evaluation results — see Objective 3."
          />

          <ObjectiveCard
            number={7}
            title="Documentation and Reporting"
            description="This tab, README.md, and inline code documentation cover dataset curation, architecture, and the training pipeline. Loss/accuracy curves and confusion matrices populate automatically once training results exist."
            done={true}
          />

          <ObjectiveCard
            number={8}
            title="Static Client-Side Edge Deployment"
            description="Deployed as a fully static, client-side web application (GitHub Pages / Vercel / Netlify) with no backend server. All video frame processing occurs 100% inside the client browser."
            done={true}
          />
        </div>
      </div>

      {/* Ethics & Privacy Assurance Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          Ethical AI & Privacy Guarantee
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          Facial emotion recognition systems require strict privacy protections. This application
          executes model inference locally inside the client browser. No video frames, biometric
          templates, or user face snapshots are transmitted to or stored on external servers —
          there is no backend server in this deployment at all (Section 8.1).
        </p>
        {!trainingDone && (
          <p className="text-amber-300/90 text-[11px] leading-relaxed flex items-start gap-2 pt-2 border-t border-slate-800">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            Honesty note: this build has not yet been trained on your dataset — see Objectives
            1-3 and 6 above. Real-time detection currently works via face-api.js's generic
            pretrained model, not a model trained on your FER-2013 data.
          </p>
        )}
        {trainingDone && (
          <p className="text-emerald-300/90 text-[11px] leading-relaxed flex items-start gap-2 pt-2 border-t border-slate-800">
            <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            Real training results detected in public/results/ — Objectives 1-3 and 6 are backed
            by an actual training run, not placeholder data.
          </p>
        )}
      </div>
    </div>
  );
};
