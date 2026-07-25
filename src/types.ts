export type EmotionType =
  | "Angry"
  | "Disgust"
  | "Fear"
  | "Happy"
  | "Neutral"
  | "Sad"
  | "Surprise";

export interface EmotionScores {
  Angry: number;
  Disgust: number;
  Fear: number;
  Happy: number;
  Neutral: number;
  Sad: number;
  Surprise: number;
}

export interface BoundingBox {
  x: number; // percentage or pixels
  y: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  id: string;
  timestamp: string;
  primaryEmotion: EmotionType;
  confidence: number;
  scores: EmotionScores;
  boundingBox: BoundingBox;
  inferenceTimeMs: number;
  fps: number;
  faceDetected: boolean;
  landmarks?: {
    leftEye: { x: number; y: number };
    rightEye: { x: number; y: number };
    nose: { x: number; y: number };
    mouth: { x: number; y: number };
  };
  source: "webcam" | "upload" | "sample";
  sampleName?: string;
  stressIndex: number; // 0 to 100
  valenceIndex: number; // -100 (negative affect) to +100 (positive affect)
}

export interface SessionRecord {
  id: string;
  time: string;
  emotion: EmotionType;
  confidence: number;
  stress: number;
  valence: number;
}

export interface LayerSpec {
  id: string;
  name: string;
  type: "Input" | "Conv2D" | "BatchNorm" | "ReLU" | "MaxPool2D" | "Dropout" | "Flatten" | "Dense" | "Softmax";
  outputShape: string;
  filters?: number;
  kernelSize?: string;
  params: number;
  description: string;
}

export interface EpochMetric {
  epoch: number;
  cnnTrainLoss: number;
  cnnValLoss: number;
  cnnTrainAcc: number;
  cnnValAcc: number;
  tlTrainLoss: number;
  tlValLoss: number;
  tlTrainAcc: number;
  tlValAcc: number;
}

export interface ClassPerformance {
  emotion: EmotionType;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
  weight: number;
}

export interface SampleImageItem {
  id: string;
  name: string;
  emotion: EmotionType;
  description: string;
  url: string;
  trueScores: EmotionScores;
  roi: BoundingBox;
}
