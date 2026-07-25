import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  BrainCircuit,
  Layers,
  Sparkles,
  ChevronRight,
  Calculator,
  ArrowRight,
  Info,
  Sliders,
  Video,
  VideoOff,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import {
  CNN_LAYERS_SPEC,
  DEMO_3X3_KERNEL,
  DEMO_INPUT_PATCH_5X5,
  compute3x3ConvStep,
  generatePlaceholderFeatureMap,
  computeRealFeatureMap,
} from "../utils/cnnExplainerMath";
import { getMLModelStatus } from "../utils/emotionDetector";
import { SAMPLE_IMAGES } from "../data/sampleImages";

// Maps the illustrative layer-diagram ids to the Nth Conv2D layer found in
// whichever real model is loaded (see computeRealFeatureMap in
// cnnExplainerMath.ts for why this works for both the Custom CNN and the
// MobileNetV2 transfer model).
const CONV_LAYER_INDEX: Record<string, number> = {
  conv1: 0,
  conv2: 1,
  conv3: 2,
  conv4: 3,
};

interface FeatureMapResult {
  grid: number[][];
  isReal: boolean;
}

export const CnnExplainer: React.FC = () => {
  const [selectedLayerId, setSelectedLayerId] = useState<string>("conv1");
  const [selectedSampleId, setSelectedSampleId] = useState<string>("sample-happy");

  // Step position for 3x3 convolution math walkthrough (0..2 for row, 0..2 for col)
  const [patchRow, setPatchRow] = useState<number>(1);
  const [patchCol, setPatchCol] = useState<number>(1);

  // Section 5.3 "Live Mode": feed the user's own webcam frame into the explainer
  const [liveMode, setLiveMode] = useState<boolean>(false);
  const [tick, setTick] = useState<number>(0);
  const [featureMaps, setFeatureMaps] = useState<FeatureMapResult[]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const modelInfo = getMLModelStatus();
  const selectedLayer = CNN_LAYERS_SPEC.find((l) => l.id === selectedLayerId) || CNN_LAYERS_SPEC[1];

  const convStepDetail = compute3x3ConvStep(
    DEMO_INPUT_PATCH_5X5,
    DEMO_3X3_KERNEL,
    patchRow,
    patchCol
  );

  // Draws whichever source is currently active (live webcam frame or the
  // selected sample image) onto a hidden working canvas that feeds the
  // real-activation computation.
  const drawCurrentSource = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (liveMode) {
      if (videoRef.current && videoRef.current.readyState >= 2) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        setTick((t) => t + 1);
      }
      return;
    }

    const sample = SAMPLE_IMAGES.find((s) => s.id === selectedSampleId);
    if (!sample) return;
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      setTick((t) => t + 1);
    };
    img.src = sample.url;
  }, [liveMode, selectedSampleId]);

  // Start/stop the webcam stream when Live Mode is toggled
  useEffect(() => {
    if (liveMode) {
      navigator.mediaDevices
        .getUserMedia({ video: { width: 320, height: 320 } })
        .then((stream) => {
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((err) => {
          console.warn("[CNN Explainer] Webcam access denied:", err);
          setLiveMode(false);
        });
    } else {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [liveMode]);

  // Redraw whenever the sample changes, or right after switching modes
  useEffect(() => {
    drawCurrentSource();
  }, [selectedSampleId, liveMode, drawCurrentSource]);

  // In Live Mode, periodically capture the current webcam frame (2 fps is
  // plenty for an explainer view — this isn't the real-time detector)
  useEffect(() => {
    if (!liveMode) return;
    const interval = setInterval(drawCurrentSource, 500);
    return () => clearInterval(interval);
  }, [liveMode, drawCurrentSource]);

  // Recompute the 8 displayed feature maps whenever the selected layer or
  // the current source frame changes. Uses REAL activations from the
  // trained model when available (public/model/ has been populated),
  // otherwise falls back to clearly-labeled placeholder data.
  useEffect(() => {
    const canvas = canvasRef.current;
    const convIndex = CONV_LAYER_INDEX[selectedLayer.id];
    const maps: FeatureMapResult[] = Array.from({ length: 8 }).map((_, fIdx) => {
      if (canvas && convIndex !== undefined) {
        const real = computeRealFeatureMap(canvas, convIndex, fIdx);
        if (real) return { grid: real, isReal: true };
      }
      return { grid: generatePlaceholderFeatureMap(selectedLayer.id, fIdx, 8), isReal: false };
    });
    setFeatureMaps(maps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLayerId, tick]);

  const usingRealActivations = featureMaps.length > 0 && featureMaps[0].isReal;

  return (
    <div className="space-y-6">
      {/* Hidden working canvas + video element — not rendered visibly, used only to feed real activations */}
      <canvas ref={canvasRef} width={160} height={160} className="hidden" />
      <video ref={videoRef} autoPlay muted playsInline className="hidden" />

      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-sky-400" />
            Objective 5.3: Interactive CNN Model Explainer & Feature Map Inspector
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            CNN Architecture Visualization • Feature Map Activations • 3×3 Kernel Conv Arithmetic
          </p>
        </div>

        {/* Sample Image Input Picker, Live Mode Toggle & Model Status */}
        <div className="flex flex-wrap items-center gap-3">
          {!liveMode && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Input Sample:</span>
              <select
                value={selectedSampleId}
                onChange={(e) => setSelectedSampleId(e.target.value)}
                className="bg-slate-950 border border-slate-800 text-xs text-white rounded-xl px-3 py-1.5 focus:ring-sky-500"
              >
                {SAMPLE_IMAGES.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={() => setLiveMode((v) => !v)}
            className={`text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-xl border transition-colors ${
              liveMode
                ? "bg-rose-500/10 border-rose-500/30 text-rose-300 hover:bg-rose-500/20"
                : "bg-emerald-500/10 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20"
            }`}
            title="Feed your own webcam frame into the explainer (Section 5.3 Live Mode)"
          >
            {liveMode ? <VideoOff className="w-3.5 h-3.5" /> : <Video className="w-3.5 h-3.5" />}
            {liveMode ? "Stop Live Mode" : "Live Mode (Webcam)"}
          </button>

          <div
            className={`text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-xl border ${
              usingRealActivations
                ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-300"
                : "bg-amber-500/10 border-amber-500/20 text-amber-300"
            }`}
            title={
              usingRealActivations
                ? modelInfo.modelSource
                : "No trained model found at public/model/ yet — run scripts/train_fer_pipeline.py and export there to see real activations."
            }
          >
            {usingRealActivations ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5" />
            )}
            {usingRealActivations ? "Real Activations" : "Simulated (model not trained yet)"}
          </div>
        </div>
      </div>

      {/* Layer Architecture Pipeline Node Diagram */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-400" />
            {modelInfo.usingCustomModel ? "Trained Model Architecture Node Pipeline" : "Custom 4-Block CNN Architecture Node Pipeline (reference diagram)"}
          </h3>
          <span className="text-xs text-slate-400">Total Params: 1,482,951</span>
        </div>

        {!modelInfo.usingCustomModel && (
          <div className="flex items-start gap-2 text-[11px] text-amber-300 bg-amber-500/5 border border-amber-500/20 rounded-xl px-3 py-2">
            <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <span>
              This diagram shows the planned Custom CNN architecture (Section 2.1) for reference. No
              trained model has been loaded from <code className="text-amber-200">public/model/</code>{" "}
              yet, so feature maps below are simulated. Run{" "}
              <code className="text-amber-200">scripts/train_fer_pipeline.py</code> and export the
              result there to see this tab switch to real activations automatically.
            </span>
          </div>
        )}

        <div className="flex items-center gap-2 overflow-x-auto pb-3 pt-1 scrollbar-none">
          {CNN_LAYERS_SPEC.map((layer, idx) => {
            const isSelected = layer.id === selectedLayerId;
            return (
              <React.Fragment key={layer.id}>
                <button
                  onClick={() => setSelectedLayerId(layer.id)}
                  className={`flex-shrink-0 p-3 rounded-xl border transition-all text-left group ${
                    isSelected
                      ? "bg-sky-500/20 border-sky-500 text-white ring-1 ring-sky-500 shadow-lg"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                >
                  <div className="text-[10px] uppercase font-bold text-sky-400">{layer.type}</div>
                  <div className="text-xs font-semibold text-white mt-0.5 truncate max-w-[130px]">
                    {layer.name}
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono mt-1">
                    {layer.outputShape}
                  </div>
                </button>
                {idx < CNN_LAYERS_SPEC.length - 1 && (
                  <ChevronRight className="w-4 h-4 text-slate-700 flex-shrink-0" />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Selected Layer Feature Maps & Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Layer Spec & Description */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <span className="text-[10px] uppercase font-bold text-sky-400">{selectedLayer.type} Layer</span>
            <h3 className="text-base font-bold text-white mt-0.5">{selectedLayer.name}</h3>
            <p className="text-xs text-slate-400 mt-1">{selectedLayer.description}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-500 text-[10px] uppercase block">Output Tensor Shape</span>
              <span className="font-mono font-bold text-white text-sm mt-0.5 block">{selectedLayer.outputShape}</span>
            </div>

            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <span className="text-slate-500 text-[10px] uppercase block">Learnable Parameters</span>
              <span className="font-mono font-bold text-sky-400 text-sm mt-0.5 block">
                {selectedLayer.params.toLocaleString()}
              </span>
            </div>
          </div>

          {/* Feature Map Activations Grid */}
          <div className="space-y-2 pt-2 border-t border-slate-800">
            <h4 className="text-xs font-semibold text-white flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              {usingRealActivations ? "Real Feature Map Activations" : "Simulated Feature Map Filters"}
            </h4>
            <div className="grid grid-cols-4 gap-2">
              {featureMaps.map((result, fIdx) => (
                <div key={fIdx} className="bg-slate-950 p-1.5 rounded-lg border border-slate-800 text-center space-y-1">
                  <span className="text-[9px] text-slate-500 font-mono">F#{fIdx + 1}</span>
                  <div className="grid grid-cols-8 gap-0.5 aspect-square bg-slate-900 rounded overflow-hidden">
                    {result.grid.flat().map((val, cellIdx) => (
                      <div
                        key={cellIdx}
                        style={{ backgroundColor: `rgba(56, 189, 248, ${val})` }}
                        className="w-full h-full"
                        title={`Activation: ${val}`}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 3x3 Kernel Convolution Walkthrough */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Calculator className="w-4 h-4 text-emerald-400" />
              Interactive 3×3 Convolution Arithmetic Walkthrough
            </h3>
            <span className="text-xs text-slate-400">Step-by-Step Math</span>
          </div>

          <p className="text-xs text-slate-400">
            Slide the 3×3 filter over the 5×5 input feature patch to observe pixel-wise multiplication, sum, bias addition, and ReLU activation function execution.
          </p>

          {/* Position Selectors */}
          <div className="flex items-center gap-4 bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs">
            <span className="text-slate-400 flex items-center gap-1">
              <Sliders className="w-3.5 h-3.5 text-sky-400" />
              Filter Position:
            </span>

            <div className="flex items-center gap-2">
              <span>Row Offset:</span>
              <input
                type="range"
                min={0}
                max={2}
                value={patchRow}
                onChange={(e) => setPatchRow(Number(e.target.value))}
                className="w-20 accent-sky-500"
              />
              <span className="font-mono text-white font-bold">{patchRow}</span>
            </div>

            <div className="flex items-center gap-2">
              <span>Col Offset:</span>
              <input
                type="range"
                min={0}
                max={2}
                value={patchCol}
                onChange={(e) => setPatchCol(Number(e.target.value))}
                className="w-20 accent-sky-500"
              />
              <span className="font-mono text-white font-bold">{patchCol}</span>
            </div>
          </div>

          {/* Interactive Math Grid comparison */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            {/* 5x5 Input Patch with 3x3 highlight */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
              <span className="text-[11px] font-semibold text-slate-300 block">
                Input Image Patch (5×5)
              </span>
              <div className="grid grid-cols-5 gap-1 font-mono text-[10px] text-center">
                {DEMO_INPUT_PATCH_5X5.map((row, r) =>
                  row.map((val, c) => {
                    const isHovered =
                      r >= patchRow && r < patchRow + 3 && c >= patchCol && c < patchCol + 3;
                    return (
                      <div
                        key={`${r}-${c}`}
                        className={`p-1.5 rounded transition-all ${
                          isHovered
                            ? "bg-sky-500 text-slate-950 font-bold border border-sky-300"
                            : "bg-slate-900 text-slate-400 border border-slate-800"
                        }`}
                      >
                        {val}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* 3x3 Sobel Edge Kernel Filter */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
              <span className="text-[11px] font-semibold text-slate-300 block">
                3×3 Kernel Filter Weights
              </span>
              <div className="grid grid-cols-3 gap-1 font-mono text-[10px] text-center">
                {DEMO_3X3_KERNEL.map((row, r) =>
                  row.map((kVal, c) => (
                    <div key={`${r}-${c}`} className="p-2.5 bg-amber-500/20 text-amber-300 font-bold border border-amber-500/40 rounded">
                      {kVal}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Multiply Products Matrix */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
              <span className="text-[11px] font-semibold text-slate-300 block">
                Hadamard Product (3×3)
              </span>
              <div className="grid grid-cols-3 gap-1 font-mono text-[10px] text-center">
                {convStepDetail.products.map((row, r) =>
                  row.map((pVal, c) => (
                    <div key={`${r}-${c}`} className="p-2.5 bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40 rounded">
                      {pVal}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Arithmetic Step Calculation Summary */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2 font-mono text-xs text-slate-300">
            <div className="flex items-center justify-between text-slate-400 text-[11px]">
              <span>Dot Product Sum:</span>
              <span className="text-white font-bold">{convStepDetail.sum}</span>
            </div>
            <div className="flex items-center justify-between text-slate-400 text-[11px]">
              <span>Add Layer Bias (+0.05):</span>
              <span className="text-white font-bold">{(convStepDetail.sum + convStepDetail.bias).toFixed(2)}</span>
            </div>
            <div className="flex items-center justify-between border-t border-slate-800 pt-2 text-sky-400">
              <span className="font-bold flex items-center gap-1">
                <ArrowRight className="w-3.5 h-3.5" />
                ReLU Activation Output max(0, x):
              </span>
              <span className="text-base font-bold text-emerald-400 bg-emerald-950/60 px-2.5 py-0.5 rounded border border-emerald-800">
                {convStepDetail.reluOutput}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
