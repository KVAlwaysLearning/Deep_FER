import React from "react";
import {
  Camera,
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  Activity,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { EmotionType } from "../types";

interface HeaderProps {
  activeTab: "realtime" | "analytics" | "explainer" | "benchmarks" | "documentation";
  setActiveTab: (tab: "realtime" | "analytics" | "explainer" | "benchmarks" | "documentation") => void;
  currentEmotion: EmotionType;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  currentEmotion,
}) => {
  // HCI reactive theme badge styling based on emotion
  const getEmotionBadge = () => {
    switch (currentEmotion) {
      case "Happy":
        return { text: "Joyful Affect Detected", bg: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" };
      case "Neutral":
        return { text: "Calm Baseline State", bg: "bg-sky-500/10 text-sky-600 border-sky-500/30" };
      case "Surprise":
        return { text: "Heightened Engagement", bg: "bg-amber-500/10 text-amber-600 border-amber-500/30" };
      case "Sad":
        return { text: "Supportive Theme Active", bg: "bg-indigo-500/10 text-indigo-600 border-indigo-500/30" };
      case "Angry":
        return { text: "De-escalation Mode", bg: "bg-rose-500/10 text-rose-600 border-rose-500/30" };
      case "Fear":
        return { text: "Reassurance Theme", bg: "bg-purple-500/10 text-purple-600 border-purple-500/30" };
      case "Disgust":
        return { text: "Neutralizing Focus Mode", bg: "bg-teal-500/10 text-teal-600 border-teal-500/30" };
      default:
        return { text: "System Ready", bg: "bg-slate-500/10 text-slate-600 border-slate-500/30" };
    }
  };

  const badge = getEmotionBadge();

  return (
    <header className="sticky top-0 z-50 bg-slate-900/95 backdrop-blur-md border-b border-slate-800 text-slate-100 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-sky-500 to-emerald-400 p-0.5 shadow-lg flex items-center justify-center">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <BrainCircuit className="w-5 h-5 text-sky-400 animate-pulse" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                  FER-System
                </h1>
                <span className="hidden sm:inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/30">
                  MobileNetV2 Edge
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Real-Time Facial Emotion Recognition • HCI & Mental Health
              </p>
            </div>
          </div>

          {/* Quick System Badge & HCI Accent */}
          <div className="hidden lg:flex items-center gap-3">
            <div className={`px-2.5 py-1 rounded-full text-xs font-medium border flex items-center gap-1.5 ${badge.bg}`}>
              <Activity className="w-3.5 h-3.5" />
              <span>{badge.text}</span>
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-800/80 px-3 py-1 rounded-lg border border-slate-700/60">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Latency: <strong className="text-slate-200">16 ms</strong></span>
              <span className="text-slate-600">•</span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-300">100% On-Device Edge</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 sm:space-x-2 overflow-x-auto pb-2 scrollbar-none text-xs sm:text-sm font-medium border-t border-slate-800/80 pt-1">
          <button
            onClick={() => setActiveTab("realtime")}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all whitespace-nowrap ${
              activeTab === "realtime"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <Camera className="w-4 h-4" />
            <span>1. Real-Time Detection</span>
          </button>

          <button
            onClick={() => setActiveTab("analytics")}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all whitespace-nowrap ${
              activeTab === "analytics"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>2. Mood Analytics & Wellness</span>
          </button>

          <button
            onClick={() => setActiveTab("explainer")}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all whitespace-nowrap ${
              activeTab === "explainer"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <BrainCircuit className="w-4 h-4" />
            <span>3. CNN Model Explainer</span>
          </button>

          <button
            onClick={() => setActiveTab("benchmarks")}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all whitespace-nowrap ${
              activeTab === "benchmarks"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <Database className="w-4 h-4" />
            <span>4. Dataset & Benchmarks</span>
          </button>

          <button
            onClick={() => setActiveTab("documentation")}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all whitespace-nowrap ${
              activeTab === "documentation"
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>5. Spec & Project Report</span>
          </button>
        </nav>
      </div>
    </header>
  );
};
