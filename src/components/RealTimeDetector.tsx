import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  Camera,
  Upload,
  RefreshCw,
  Sliders,
  CheckCircle2,
  Sparkles,
  Zap,
  Activity,
  PlusCircle,
  HelpCircle,
  Video,
} from "lucide-react";
import { DetectionResult, EmotionType, SessionRecord } from "../types";
import { SAMPLE_IMAGES } from "../data/sampleImages";
import {
  processRealTimeFaceDetection,
  loadOnDeviceModels,
  getMLModelStatus,
  resetTemporalSmoothingCache,
  ModelStatus,
} from "../utils/emotionDetector";

interface RealTimeDetectorProps {
  onNewDetection: (result: DetectionResult) => void;
  onAddSessionRecord: (record: SessionRecord) => void;
}

export const RealTimeDetector: React.FC<RealTimeDetectorProps> = ({
  onNewDetection,
  onAddSessionRecord,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [activeInputMode, setActiveInputMode] = useState<"webcam" | "sample" | "upload">("sample");
  const [selectedSampleId, setSelectedSampleId] = useState<string>("sample-happy");
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);

  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  const [useSmoothing, setUseSmoothing] = useState<boolean>(true);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [currentDetection, setCurrentDetection] = useState<DetectionResult | null>(null);

  const [mlEngineInfo, setMlEngineInfo] = useState<{
    status: ModelStatus;
    backend: string;
    error: string | null;
  }>({ status: "idle", backend: "CPU", error: null });

  // Initialize on-device ML models on component mount
  useEffect(() => {
    let isMounted = true;
    loadOnDeviceModels().then(() => {
      if (isMounted) {
        setMlEngineInfo(getMLModelStatus());
      }
    });
    return () => {
      isMounted = false;
    };
  }, []);

  // Reusable processing runner for current image source
  const processCurrentFrame = useCallback(async () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (activeInputMode === "webcam" && videoRef.current && isWebcamActive) {
      const video = videoRef.current;
      if (video.readyState >= 2) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        const result = await processRealTimeFaceDetection(video, canvas, useSmoothing);
        setCurrentDetection(result);
        onNewDetection(result);

        // Render bounding box overlay
        drawOverlay(ctx, canvas.width, canvas.height, result);
      }
    } else if (activeInputMode === "sample") {
      const sample = SAMPLE_IMAGES.find((s) => s.id === selectedSampleId) || SAMPLE_IMAGES[0];
      const img = new Image();
      img.src = sample.url;
      img.onload = async () => {
        canvas.width = 224;
        canvas.height = 224;
        const result = await processRealTimeFaceDetection(img, canvas, useSmoothing, sample.trueScores);
        result.sampleName = sample.name;
        setCurrentDetection(result);
        onNewDetection(result);
        drawOverlay(ctx, 224, 224, result);
      };
    } else if (activeInputMode === "upload" && uploadedImageSrc) {
      const img = new Image();
      img.src = uploadedImageSrc;
      img.onload = async () => {
        canvas.width = img.width || 400;
        canvas.height = img.height || 400;
        const result = await processRealTimeFaceDetection(img, canvas, useSmoothing);
        result.source = "upload";
        setCurrentDetection(result);
        onNewDetection(result);
        drawOverlay(ctx, canvas.width, canvas.height, result);
      };
    }
  }, [activeInputMode, isWebcamActive, selectedSampleId, uploadedImageSrc, useSmoothing, onNewDetection]);

  // Canvas Overlay Drawing (Bounding Box, Landmarks, Label)
  const drawOverlay = (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    res: DetectionResult
  ) => {
    if (!res.faceDetected) {
      // Render subtle target guide when no face is found
      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 2;
      ctx.strokeRect(w * 0.25, h * 0.2, w * 0.5, h * 0.6);
      
      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.fillRect(w * 0.2, h * 0.45, w * 0.6, 32);
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No Face Detected — Position face in camera view", w * 0.5, h * 0.45 + 20);
      ctx.textAlign = "left";
      return;
    }

    const box = res.boundingBox;
    ctx.lineWidth = 3;

    // Emotion theme color selection
    let strokeColor = "#38bdf8"; // sky
    if (res.primaryEmotion === "Happy") strokeColor = "#10b981"; // emerald
    else if (res.primaryEmotion === "Angry") strokeColor = "#f43f5e"; // rose
    else if (res.primaryEmotion === "Surprise") strokeColor = "#f59e0b"; // amber
    else if (res.primaryEmotion === "Sad") strokeColor = "#818cf8"; // indigo
    else if (res.primaryEmotion === "Fear") strokeColor = "#a855f7"; // purple

    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Draw Corner Reticles
    const len = 16;
    ctx.lineWidth = 4;
    // Top-Left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + len);
    ctx.lineTo(box.x, box.y);
    ctx.lineTo(box.x + len, box.y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - len, box.y);
    ctx.lineTo(box.x + box.width, box.y);
    ctx.lineTo(box.x + box.width, box.y + len);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + box.height - len);
    ctx.lineTo(box.x, box.y + box.height);
    ctx.lineTo(box.x + len, box.y + box.height);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(box.x + box.width - len, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height);
    ctx.lineTo(box.x + box.width, box.y + box.height - len);
    ctx.stroke();

    // Draw facial landmark dots if enabled
    if (showLandmarks && res.landmarks) {
      ctx.fillStyle = "#38bdf8";
      const lm = res.landmarks;
      [lm.leftEye, lm.rightEye, lm.nose, lm.mouth].forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Top Label Banner
    const labelText = `${res.primaryEmotion} (${res.confidence}%)`;
    ctx.font = "bold 13px system-ui, sans-serif";
    const textWidth = ctx.measureText(labelText).width;

    ctx.fillStyle = strokeColor;
    ctx.fillRect(box.x, Math.max(0, box.y - 28), textWidth + 16, 26);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(labelText, box.x + 8, Math.max(18, box.y - 10));
  };

  // Start Webcam stream
  const startWebcam = async () => {
    try {
      setWebcamError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsWebcamActive(true);
      }
    } catch (err: any) {
      console.error("Webcam error:", err);
      setWebcamError(err.message || "Could not access camera. Please check browser permissions.");
      setIsWebcamActive(false);
    }
  };

  // Stop Webcam stream
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  // Webcam live loop
  useEffect(() => {
    let animationFrameId: number;
    const loop = () => {
      if (activeInputMode === "webcam" && isWebcamActive) {
        processCurrentFrame();
        animationFrameId = requestAnimationFrame(loop);
      }
    };

    if (activeInputMode === "webcam" && isWebcamActive) {
      animationFrameId = requestAnimationFrame(loop);
    } else {
      processCurrentFrame();
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [activeInputMode, isWebcamActive, processCurrentFrame]);

  // Mode switch handler
  const handleModeChange = (mode: "webcam" | "sample" | "upload") => {
    setActiveInputMode(mode);
    resetTemporalSmoothingCache();
    if (mode === "webcam") {
      startWebcam();
    } else {
      stopWebcam();
    }
  };

  // File Upload Handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setUploadedImageSrc(event.target.result as string);
          setActiveInputMode("upload");
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // Add current result to session history
  const handleRecordToSession = () => {
    if (!currentDetection) return;
    const record: SessionRecord = {
      id: `rec-${Date.now()}`,
      time: currentDetection.timestamp,
      emotion: currentDetection.primaryEmotion,
      confidence: currentDetection.confidence,
      stress: currentDetection.stressIndex,
      valence: currentDetection.valenceIndex,
    };
    onAddSessionRecord(record);
  };

  const getEmotionBarColor = (emo: EmotionType) => {
    switch (emo) {
      case "Happy": return "bg-emerald-500";
      case "Neutral": return "bg-sky-500";
      case "Surprise": return "bg-amber-500";
      case "Sad": return "bg-indigo-500";
      case "Angry": return "bg-rose-500";
      case "Fear": return "bg-purple-500";
      case "Disgust": return "bg-teal-500";
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Controller Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Camera className="w-5 h-5 text-sky-400" />
            Objective 4 & 5: Real-Time Inference & HCI Engine
          </h2>
          <div className="flex flex-wrap items-center gap-2 mt-2">
            <p className="text-xs text-slate-400">
              MobileNetV2 Edge Model • 7 Emotions Classification • Temporal Smoothing EMA Filter
            </p>
            {mlEngineInfo.status === "ready" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                On-Device {mlEngineInfo.backend} ML Active
              </span>
            )}
            {mlEngineInfo.status === "loading" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Loading TensorFlow.js Weights...
              </span>
            )}
            {mlEngineInfo.status === "error" && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
                Using Local Feature Inference
              </span>
            )}
          </div>
        </div>

        {/* Input Mode Selector */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-medium self-start md:self-auto">
          <button
            onClick={() => handleModeChange("sample")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeInputMode === "sample"
                ? "bg-sky-500 text-white shadow-sm font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Sample Faces (7)</span>
          </button>

          <button
            onClick={() => handleModeChange("webcam")}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
              activeInputMode === "webcam"
                ? "bg-sky-500 text-white shadow-sm font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>Live Webcam</span>
          </button>

          <label
            className={`px-3 py-1.5 rounded-lg cursor-pointer transition-all flex items-center gap-1.5 ${
              activeInputMode === "upload"
                ? "bg-sky-500 text-white shadow-sm font-semibold"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Image</span>
            <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
          </label>
        </div>
      </div>

      {/* Main Grid: Left Stream/Canvas Display + Right Emotion Meters */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (Canvas Preview & Control Panel) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3 relative overflow-hidden">
            {/* Header info bar */}
            <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800/80 pb-2">
              <span className="font-mono text-slate-300 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-sky-400 animate-pulse" />
                Input Stream: <strong className="text-white uppercase">{activeInputMode}</strong>
              </span>

              {currentDetection && (
                <div className="flex items-center gap-3">
                  <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px]">
                    Latency: <strong className="text-emerald-400">{currentDetection.inferenceTimeMs}ms</strong>
                  </span>
                  <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px]">
                    FPS: <strong className="text-sky-300">{currentDetection.fps}</strong>
                  </span>
                </div>
              )}
            </div>

            {/* Hidden Video element for webcam stream */}
            <video ref={videoRef} className="hidden" playsInline muted />

            {/* Canvas Display View */}
            <div className="relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-800/80 flex items-center justify-center group">
              <canvas ref={canvasRef} className="w-full h-full object-contain" />

              {/* Webcam state overlay if not connected */}
              {activeInputMode === "webcam" && !isWebcamActive && (
                <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center space-y-3">
                  <Camera className="w-12 h-12 text-slate-600 animate-bounce" />
                  <h3 className="text-sm font-semibold text-white">Camera Offline</h3>
                  <p className="text-xs text-slate-400 max-w-xs">
                    Click start below to initiate webcam video capture for real-time facial emotion inference.
                  </p>
                  {webcamError && (
                    <div className="text-xs text-rose-400 bg-rose-950/50 p-2 rounded border border-rose-800 max-w-sm">
                      {webcamError}
                    </div>
                  )}
                  <button
                    onClick={startWebcam}
                    className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-bold shadow-lg transition-all"
                  >
                    Enable Webcam Camera
                  </button>
                </div>
              )}

              {/* Primary Emotion Banner overlay */}
              {currentDetection && (
                <div className="absolute bottom-3 left-3 right-3 bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl p-3 flex items-center justify-between text-xs">
                  <div>
                    <span className="text-[10px] uppercase text-slate-400 tracking-wider">Primary Classification</span>
                    <div className="text-base font-bold text-white flex items-center gap-2">
                      <span>{currentDetection.primaryEmotion}</span>
                      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/30">
                        {currentDetection.confidence}% Confidence
                      </span>
                    </div>
                  </div>

                  <button
                    onClick={handleRecordToSession}
                    className="px-3 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-xs font-semibold flex items-center gap-1.5 transition-all"
                  >
                    <PlusCircle className="w-3.5 h-3.5" />
                    <span>Log Session</span>
                  </button>
                </div>
              )}
            </div>

            {/* Filter & Smoothing Controls */}
            <div className="flex items-center justify-between pt-2 text-xs text-slate-300">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={useSmoothing}
                  onChange={(e) => setUseSmoothing(e.target.checked)}
                  className="rounded bg-slate-800 border-slate-700 text-sky-500 focus:ring-sky-500"
                />
                <span className="flex items-center gap-1 font-medium">
                  <Sliders className="w-3.5 h-3.5 text-sky-400" />
                  5-Frame EMA Temporal Smoothing (Eliminates Flicker)
                </span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showLandmarks}
                  onChange={(e) => setShowLandmarks(e.target.checked)}
                  className="rounded bg-slate-800 border-slate-700 text-sky-500 focus:ring-sky-500"
                />
                <span className="text-slate-400">Landmarks</span>
              </label>
            </div>
          </div>

          {/* Sample Images Selector Bar */}
          {activeInputMode === "sample" && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-sky-400" />
                  Select Test Sample Expression (7 Distinct Emotions)
                </h3>
                <span className="text-[11px] text-slate-400">Click to evaluate</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                {SAMPLE_IMAGES.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => {
                      setSelectedSampleId(sample.id);
                      resetTemporalSmoothingCache();
                    }}
                    className={`p-2 rounded-xl border text-left transition-all flex flex-col items-center gap-1.5 group ${
                      selectedSampleId === sample.id
                        ? "bg-sky-500/20 border-sky-500 text-white shadow-md ring-1 ring-sky-500"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                    }`}
                  >
                    <div className="w-12 h-12 rounded-lg overflow-hidden bg-slate-900 border border-slate-800">
                      <img src={sample.url} alt={sample.name} className="w-full h-full object-cover" />
                    </div>
                    <span className="text-[11px] font-semibold truncate max-w-full">
                      {sample.emotion}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column (7-Emotion Probability Breakdown & HCI Reactive Panel) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Emotion Probability Gauges */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                Live Emotion Probabilities (7 Categories)
              </h3>
              <span className="text-[11px] text-slate-400">Softmax Output</span>
            </div>

            {currentDetection ? (
              <div className="space-y-3">
                {Object.entries(currentDetection.scores).map(([emo, score]) => {
                  const emoType = emo as EmotionType;
                  const scoreVal = typeof score === "number" ? score : Number(score);
                  const pct = Math.round(scoreVal * 100);
                  const isPrimary = emoType === currentDetection.primaryEmotion;

                  return (
                    <div key={emo} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span
                          className={`font-semibold flex items-center gap-1.5 ${
                            isPrimary ? "text-white" : "text-slate-400"
                          }`}
                        >
                          {emoType}
                          {isPrimary && (
                            <CheckCircle2 className="w-3.5 h-3.5 text-sky-400 inline" />
                          )}
                        </span>
                        <span className={`font-mono ${isPrimary ? "text-sky-400 font-bold" : "text-slate-400"}`}>
                          {pct}%
                        </span>
                      </div>

                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${getEmotionBarColor(emoType)}`}
                          style={{ width: `${Math.max(2, pct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="py-12 text-center text-xs text-slate-500">
                Evaluating frame probabilities...
              </div>
            )}
          </div>

          {/* HCI & Mental Health Wellness Reaction Card */}
          {currentDetection && (
            <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  HCI & Mental Health Indicator
                </span>
                <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Domain Objective 5
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase">Stress Index</span>
                  <div className="text-lg font-bold text-white mt-0.5 flex items-baseline gap-1">
                    <span>{currentDetection.stressIndex}</span>
                    <span className="text-xs text-slate-500 font-normal">/ 100</span>
                  </div>
                  <p className="text-[10px] text-slate-400 mt-1">
                    {currentDetection.stressIndex > 60
                      ? "Elevated stress signals"
                      : "Balanced stress level"}
                  </p>
                </div>

                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase">Valence (Affect)</span>
                  <div className="text-lg font-bold text-white mt-0.5 flex items-baseline gap-1">
                    <span>
                      {currentDetection.valenceIndex > 0 ? `+${currentDetection.valenceIndex}` : currentDetection.valenceIndex}
                    </span>
                    <span className="text-xs text-slate-500 font-normal">pts</span>
                  </div>
                  <p className="text-[10px] text-slate-400 mt-1">
                    {currentDetection.valenceIndex > 20
                      ? "Positive emotional valence"
                      : currentDetection.valenceIndex < -20
                      ? "Negative valence signals"
                      : "Neutral affective state"}
                  </p>
                </div>
              </div>

              <div className="bg-sky-950/30 border border-sky-800/40 rounded-xl p-3 text-xs text-sky-200">
                <strong className="font-semibold block mb-0.5">Adaptive HCI Action:</strong>
                {currentDetection.primaryEmotion === "Happy" &&
                  "System interface optimized for active learning & engagement. High positive affect recorded."}
                {currentDetection.primaryEmotion === "Sad" &&
                  "Activating gentle supportive theme & suggesting brief wellness pause or breathing exercise."}
                {currentDetection.primaryEmotion === "Angry" &&
                  "Activating de-escalation interface styling and reducing notification frequency."}
                {currentDetection.primaryEmotion === "Surprise" &&
                  "Capturing peak focus moment for interactive task feedback."}
                {currentDetection.primaryEmotion === "Neutral" &&
                  "Standard calm baseline. Ideal state for focused productivity."}
                {currentDetection.primaryEmotion === "Fear" &&
                  "Reassurance theme activated with simplified navigation options."}
                {currentDetection.primaryEmotion === "Disgust" &&
                  "Neutralizing interface theme to promote clarity and focus."}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
