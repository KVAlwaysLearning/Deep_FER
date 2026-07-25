import React, { useState } from "react";
import { Header } from "./components/Header";
import { RealTimeDetector } from "./components/RealTimeDetector";
import { MoodAnalyticsDashboard } from "./components/MoodAnalyticsDashboard";
import { CnnExplainer } from "./components/CnnExplainer";
import { DatasetBenchmarks } from "./components/DatasetBenchmarks";
import { ProjectDocumentation } from "./components/ProjectDocumentation";
import { DetectionResult, SessionRecord, EmotionType } from "./types";

export function App() {
  const [activeTab, setActiveTab] = useState<
    "realtime" | "analytics" | "explainer" | "benchmarks" | "documentation"
  >("realtime");

  const [currentDetection, setCurrentDetection] = useState<DetectionResult | null>(null);

  // Initial demo session records
  const [sessionRecords, setSessionRecords] = useState<SessionRecord[]>([
    {
      id: "rec-1",
      time: "09:15:02",
      emotion: "Happy",
      confidence: 96.4,
      stress: 12,
      valence: 88,
    },
    {
      id: "rec-2",
      time: "09:18:44",
      emotion: "Neutral",
      confidence: 92.1,
      stress: 20,
      valence: 15,
    },
    {
      id: "rec-3",
      time: "09:22:10",
      emotion: "Surprise",
      confidence: 94.0,
      stress: 28,
      valence: 45,
    },
    {
      id: "rec-4",
      time: "09:25:30",
      emotion: "Sad",
      confidence: 89.2,
      stress: 65,
      valence: -42,
    },
  ]);

  const handleNewDetection = (result: DetectionResult) => {
    setCurrentDetection(result);
  };

  const handleAddSessionRecord = (record: SessionRecord) => {
    setSessionRecords((prev) => [record, ...prev]);
  };

  const handleClearRecords = () => {
    setSessionRecords([]);
  };

  const currentEmotion: EmotionType = currentDetection ? currentDetection.primaryEmotion : "Neutral";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500 selection:text-slate-950">
      {/* Header Bar */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentEmotion={currentEmotion}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "realtime" && (
          <RealTimeDetector
            onNewDetection={handleNewDetection}
            onAddSessionRecord={handleAddSessionRecord}
          />
        )}

        {activeTab === "analytics" && (
          <MoodAnalyticsDashboard
            sessionRecords={sessionRecords}
            onClearRecords={handleClearRecords}
          />
        )}

        {activeTab === "explainer" && <CnnExplainer />}

        {activeTab === "benchmarks" && <DatasetBenchmarks />}

        {activeTab === "documentation" && <ProjectDocumentation />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-xs text-slate-500 text-center">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Real-Time Facial Emotion Recognition System • HCI & Mental Health Platform</span>
          <span className="font-mono text-[11px] text-slate-400">Edge MobileNetV2 • FER-2013 • Sub-25ms SLA</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
