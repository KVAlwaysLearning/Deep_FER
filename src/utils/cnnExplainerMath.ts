import { LayerSpec } from "../types";
import * as tf from "@tensorflow/tfjs";
import { getCustomModel } from "./emotionDetector";

export const CNN_LAYERS_SPEC: LayerSpec[] = [
  {
    id: "input",
    name: "Input Image",
    type: "Input",
    outputShape: "48 × 48 × 1",
    params: 0,
    description: "Normalized 48x48 pixel grayscale face patch cropped from face detector ROI.",
  },
  {
    id: "conv1",
    name: "Conv Block 1 (32 Filters)",
    type: "Conv2D",
    outputShape: "48 × 48 × 32",
    filters: 32,
    kernelSize: "3 × 3",
    params: 320,
    description: "Extracts low-level edge features, high-contrast facial contours, and line orientation.",
  },
  {
    id: "pool1",
    name: "MaxPool 1 (2x2)",
    type: "MaxPool2D",
    outputShape: "24 × 24 × 32",
    params: 0,
    description: "Downsamples spatial resolution by factor of 2, providing minor spatial translation invariance.",
  },
  {
    id: "conv2",
    name: "Conv Block 2 (64 Filters)",
    type: "Conv2D",
    outputShape: "24 × 24 × 64",
    filters: 64,
    kernelSize: "3 × 3",
    params: 18496,
    description: "Detects mid-level facial features like eye corners, eyebrow curves, and mouth angles.",
  },
  {
    id: "pool2",
    name: "MaxPool 2 (2x2)",
    type: "MaxPool2D",
    outputShape: "12 × 12 × 64",
    params: 0,
    description: "Downsamples to 12x12 spatial grids, reducing memory and computation cost.",
  },
  {
    id: "conv3",
    name: "Conv Block 3 (128 Filters)",
    type: "Conv2D",
    outputShape: "12 × 12 × 128",
    filters: 128,
    kernelSize: "3 × 3",
    params: 73856,
    description: "Combines mid-level features into facial sub-expressions (e.g. smile wrinkles, frown furrows).",
  },
  {
    id: "pool3",
    name: "MaxPool 3 (2x2)",
    type: "MaxPool2D",
    outputShape: "6 × 6 × 128",
    params: 0,
    description: "Downsamples to 6x6 spatial resolution.",
  },
  {
    id: "conv4",
    name: "Conv Block 4 (256 Filters)",
    type: "Conv2D",
    outputShape: "6 × 6 × 256",
    filters: 256,
    kernelSize: "3 × 3",
    params: 295168,
    description: "High-level abstract emotion representation across entire facial geometry.",
  },
  {
    id: "flatten",
    name: "Flatten",
    type: "Flatten",
    outputShape: "9216",
    params: 0,
    description: "Unrolls 6 × 6 × 256 feature volume into a single 1D vector of 9,216 neurons.",
  },
  {
    id: "dense1",
    name: "Dense Fully Connected",
    type: "Dense",
    outputShape: "512",
    params: 4719104,
    description: "Dense representation with Dropout (0.5) to prevent overfitting.",
  },
  {
    id: "softmax",
    name: "Softmax Output",
    type: "Softmax",
    outputShape: "7",
    params: 3591,
    description: "Outputs probability distribution over the 7 target facial emotion categories.",
  },
];

// Sample 3x3 Convolution Kernel (e.g., Sobel / Edge Detection Filter)
export const DEMO_3X3_KERNEL = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
];

// Sample 5x5 Input Matrix Patch for interactive walk-through
export const DEMO_INPUT_PATCH_5X5 = [
  [0.12, 0.45, 0.78, 0.82, 0.35],
  [0.18, 0.52, 0.89, 0.91, 0.42],
  [0.22, 0.60, 0.95, 0.88, 0.50],
  [0.15, 0.48, 0.82, 0.75, 0.38],
  [0.10, 0.35, 0.65, 0.60, 0.28],
];

export interface ConvStepDetail {
  patch3x3: number[][];
  kernel3x3: number[][];
  products: number[][];
  sum: number;
  bias: number;
  reluOutput: number;
}

/**
 * Computes step-by-step 3x3 convolution arithmetic for a given top-left corner in input matrix
 */
export function compute3x3ConvStep(
  inputMatrix5x5: number[][],
  kernel3x3: number[][],
  startRow: number,
  startCol: number,
  bias: number = 0.05
): ConvStepDetail {
  const patch3x3: number[][] = [];
  const products: number[][] = [];
  let sum = 0;

  for (let r = 0; r < 3; r++) {
    const patchRow: number[] = [];
    const prodRow: number[] = [];
    for (let c = 0; c < 3; c++) {
      const inputVal = inputMatrix5x5[startRow + r][startCol + c];
      const kVal = kernel3x3[r][c];
      const prod = inputVal * kVal;

      patchRow.push(Number(inputVal.toFixed(2)));
      prodRow.push(Number(prod.toFixed(2)));
      sum += prod;
    }
    patch3x3.push(patchRow);
    products.push(prodRow);
  }

  const rawResult = sum + bias;
  const reluOutput = Math.max(0, Number(rawResult.toFixed(2)));

  return {
    patch3x3,
    kernel3x3,
    products,
    sum: Number(sum.toFixed(2)),
    bias,
    reluOutput,
  };
}

/**
 * Generates PLACEHOLDER feature map data (sine/cosine patterns) used only
 * when no trained model is available yet. Clearly not real activations —
 * callers should label this as simulated in the UI. Once
 * scripts/train_fer_pipeline.py has been run and exported to
 * public/model/, computeRealFeatureMap() below should be used instead.
 */
export function generatePlaceholderFeatureMap(
  layerId: string,
  filterIdx: number,
  gridSize: number
): number[][] {
  const map: number[][] = [];
  const phase = filterIdx * 0.7;

  for (let r = 0; r < gridSize; r++) {
    const row: number[] = [];
    for (let c = 0; c < gridSize; c++) {
      let val = 0;
      if (layerId.includes("conv1")) {
        val = 0.5 + 0.4 * Math.sin(r * 0.5 + phase) * Math.cos(c * 0.5);
      } else if (layerId.includes("conv2")) {
        val = 0.5 + 0.45 * Math.sin((r + c) * 0.6 + phase);
      } else if (layerId.includes("conv3")) {
        const dist = Math.hypot(r - gridSize / 2, c - gridSize / 2);
        val = Math.max(0, 1 - dist / (gridSize / 1.8));
      } else {
        val = Math.random() > 0.4 ? Math.random() * 0.9 : 0;
      }
      row.push(Number(Math.min(1, Math.max(0, val)).toFixed(2)));
    }
    map.push(row);
  }

  return map;
}

// Kept as an alias so any existing callers still compile; new code should
// call generatePlaceholderFeatureMap directly to be explicit about intent.
export const generateSyntheticFeatureMap = generatePlaceholderFeatureMap;

/**
 * Section 5.3 "Implementation Approach": uses tf.js intermediate-layer
 * models (equivalent to Keras Model(inputs, outputs=layer.output)) to
 * extract REAL per-layer activations for the current input, straight from
 * the model's own trained weights — no hand-authored demo data.
 *
 * convLayerIndex is the Nth Conv2D layer found in the model (0-indexed).
 * For the Custom CNN (Section 2.1, 4 conv blocks) this lines up 1:1 with
 * conv1..conv4. For the MobileNetV2 transfer model (Section 2.2), this
 * naturally samples across the backbone's many conv layers — matching the
 * blueprint's own scoping note to show "a few early, middle, and late
 * backbone layers" rather than every single one.
 *
 * Returns null if no trained model is loaded, or if the requested conv
 * layer / filter index doesn't exist — callers should fall back to
 * generatePlaceholderFeatureMap() in that case.
 */
/**
 * Finds every Conv2D layer in a model, including ones nested inside a
 * sub-model. This matters specifically for the MobileNetV2 transfer model
 * (Section 2.2): its backbone is a single "Functional" layer in the outer
 * model's top-level layers list, with all of its real Conv2D layers living
 * one level deeper inside it. A flat model.layers.filter(...) finds zero
 * Conv2D layers for MobileNetV2 and silently falls back to placeholder
 * data — this recursive walk is what makes real activations work for both
 * the Custom CNN (flat) and MobileNetV2 (nested) architectures.
 */
function findAllConv2DLayers(model: tf.LayersModel): tf.layers.Layer[] {
  const found: tf.layers.Layer[] = [];
  const visit = (layer: tf.layers.Layer) => {
    if (layer.getClassName() === "Conv2D") {
      found.push(layer);
    }
    const nestedLayers = (layer as unknown as { layers?: tf.layers.Layer[] }).layers;
    if (Array.isArray(nestedLayers)) {
      nestedLayers.forEach(visit);
    }
  };
  model.layers.forEach(visit);
  return found;
}

export function computeRealFeatureMap(
  inputCanvas: HTMLCanvasElement,
  convLayerIndex: number,
  filterIdx: number
): number[][] | null {
  const model = getCustomModel();
  if (!model) return null;

  const convLayers = findAllConv2DLayers(model);
  const targetLayer = convLayers[convLayerIndex];
  if (!targetLayer) return null;

  try {
    return tf.tidy(() => {
      const inputShape = model.inputs[0].shape; // [null, H, W, C]
      const targetH = (inputShape[1] as number) || 48;
      const targetW = (inputShape[2] as number) || 48;
      const channels = (inputShape[3] as number) || 1;

      let inputTensor = tf.browser
        .fromPixels(inputCanvas, channels === 1 ? 1 : 3)
        .resizeBilinear([targetH, targetW])
        .toFloat()
        .div(255.0)
        .expandDims(0);

      const activationModel = tf.model({
        inputs: model.inputs,
        outputs: targetLayer.output as tf.SymbolicTensor,
      });
      const activation = activationModel.predict(inputTensor) as tf.Tensor4D;
      const numFilters = activation.shape[3] ?? 1;
      const safeFilterIdx = Math.min(filterIdx, numFilters - 1);

      const singleChannel = activation
        .slice([0, 0, 0, safeFilterIdx], [1, activation.shape[1], activation.shape[2], 1])
        .squeeze() as tf.Tensor2D;

      const minVal = singleChannel.min();
      const maxVal = singleChannel.max();
      const range = maxVal.sub(minVal).add(1e-6);
      const normalized = singleChannel.sub(minVal).div(range);

      const grid = normalized.arraySync() as number[][];
      return grid.map((row) => row.map((v) => Number(v.toFixed(2))));
    });
  } catch (err) {
    console.warn("[CNN Explainer] Real activation extraction failed, falling back to placeholder:", err);
    return null;
  }
}
