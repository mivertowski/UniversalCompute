// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

using ILGPU.Apple.NeuralEngine.Native;
using System;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// High-level operations for Apple Neural Engine.
    /// </summary>
    public static class ANEOperations
    {
        #region Convolution Operations

        /// <summary>
        /// Executes a convolution operation on the Apple Neural Engine.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="weights">Convolution weights.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="config">Convolution configuration.</param>
        /// <param name="context">ANE context handle.</param>
        public static unsafe void ExecuteConvolution(
            float* input,
            ArrayView<float> weights,
            float* output,
            ANEConvolutionConfig config,
            IntPtr context)
        {
            // Validate inputs
            if (input == null || output == null || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for ANE convolution");

            // TODO: Implement proper ANE convolution execution
            throw new NotSupportedException("ANE convolution operations not fully implemented");
        }

        /// <summary>
        /// Executes a depthwise separable convolution on ANE.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="depthwiseWeights">Depthwise convolution weights.</param>
        /// <param name="pointwiseWeights">Pointwise convolution weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Convolution configuration.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteDepthwiseSeparableConvolution(
            float* input,
            ArrayView<float> depthwiseWeights,
            ArrayView<float> pointwiseWeights,
            float* output,
            ANEConvolutionConfig config,
            IntPtr context) =>
            // TODO: Implement proper ANE depthwise separable convolution
            throw new NotSupportedException("ANE depthwise separable convolution not fully implemented");

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Executes matrix multiplication on ANE.
        /// </summary>
        /// <param name="a">Matrix A data.</param>
        /// <param name="b">Matrix B data.</param>
        /// <param name="c">Result matrix C data.</param>
        /// <param name="m">Number of rows in A.</param>
        /// <param name="n">Number of columns in B.</param>
        /// <param name="k">Number of columns in A / rows in B.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteMatrixMultiply(
            float* a, float* b, float* c,
            int m, int n, int k,
            IntPtr context)
        {
            if (a == null || b == null || c == null || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for ANE matrix multiply");

            var inputSize = m * k + k * n; // Size of both input matrices
            var outputSize = m * n; // Size of output matrix

            ANENative.ExecuteMatMul(a, c, inputSize, outputSize, context);
        }

        /// <summary>
        /// Executes batched matrix multiplication on ANE.
        /// </summary>
        /// <param name="a">Batch of A matrices.</param>
        /// <param name="b">Batch of B matrices.</param>
        /// <param name="c">Batch of result C matrices.</param>
        /// <param name="batchSize">Number of matrices in batch.</param>
        /// <param name="m">Matrix dimension M.</param>
        /// <param name="n">Matrix dimension N.</param>
        /// <param name="k">Matrix dimension K.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteBatchedMatrixMultiply(
            float* a, float* b, float* c,
            int batchSize, int m, int n, int k,
            IntPtr context)
        {
            var matrixASize = m * k;
            var matrixBSize = k * n;
            var matrixCSize = m * n;

            for (int batch = 0; batch < batchSize; batch++)
            {
                var aPtr = a + batch * matrixASize;
                var bPtr = b + batch * matrixBSize;
                var cPtr = c + batch * matrixCSize;

                ExecuteMatrixMultiply(aPtr, bPtr, cPtr, m, n, k, context);
            }
        }

        #endregion

        #region Attention Operations

        /// <summary>
        /// Executes scaled dot-product attention on ANE.
        /// </summary>
        /// <param name="queries">Query tensor.</param>
        /// <param name="keys">Key tensor.</param>
        /// <param name="values">Value tensor.</param>
        /// <param name="output">Output attention tensor.</param>
        /// <param name="config">Attention configuration.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteAttention(
            float* queries, float* keys, float* values, float* output,
            ANEAttentionConfig config,
            IntPtr context)
        {
            if (queries == null || keys == null || values == null || output == null)
                throw new ArgumentException("Invalid input parameters for ANE attention");

            // TODO: ANEAttentionConfig properties not implemented
            throw new NotSupportedException("ANE attention config properties not available");
        }

        /// <summary>
        /// Executes multi-head attention on ANE.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="weightQ">Query projection weights.</param>
        /// <param name="weightK">Key projection weights.</param>
        /// <param name="weightV">Value projection weights.</param>
        /// <param name="weightO">Output projection weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Attention configuration.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteMultiHeadAttention(
            float* input,
            float* weightQ, float* weightK, float* weightV, float* weightO,
            float* output,
            ANEAttentionConfig config,
            IntPtr context) =>
            // TODO: ANEAttentionConfig properties not implemented
            throw new NotSupportedException("ANE attention config properties not available");

        #endregion

        #region Transformer Operations

        /// <summary>
        /// Executes a complete transformer block on ANE.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="attentionWeights">Multi-head attention weights.</param>
        /// <param name="ffnWeights">Feed-forward network weights.</param>
        /// <param name="layerNormWeights">Layer normalization weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Transformer configuration.</param>
        /// <param name="context">ANE context.</param>
        public static unsafe void ExecuteTransformerBlock(
            float* input,
            float* attentionWeights,
            float* ffnWeights,
            float* layerNormWeights,
            float* output,
            ANETransformerConfig config,
            IntPtr context) =>
            // TODO: ANEAttentionConfig properties not available
            throw new NotSupportedException("ANE transformer operations not implemented - config properties unavailable");

        #endregion

        #region Helper Functions

        /// <summary>
        /// Calculates the total size of a tensor.
        /// </summary>
        private static long GetTensorSize((int N, int C, int H, int W) shape) => (long)shape.N * shape.C * shape.H * shape.W;

        /// <summary>
        /// Applies activation function to tensor data.
        /// </summary>
        private static unsafe void ApplyActivation(float* data, long size, ANEActivation activation)
        {
            switch (activation)
            {
                case ANEActivation.ReLU:
                    for (long i = 0; i < size; i++)
                        data[i] = Math.Max(0.0f, data[i]);
                    break;

                case ANEActivation.GELU:
                    for (long i = 0; i < size; i++)
                        data[i] = data[i] * 0.5f * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (data[i] + 0.044715f * data[i] * data[i] * data[i])));
                    break;

                case ANEActivation.Swish:
                    for (long i = 0; i < size; i++)
                        data[i] = data[i] / (1.0f + MathF.Exp(-data[i]));
                    break;

                case ANEActivation.Sigmoid:
                    for (long i = 0; i < size; i++)
                        data[i] = 1.0f / (1.0f + MathF.Exp(-data[i]));
                    break;

                case ANEActivation.Tanh:
                    for (long i = 0; i < size; i++)
                        data[i] = MathF.Tanh(data[i]);
                    break;
            }
        }

        /// <summary>
        /// Applies scaling to tensor data.
        /// </summary>
        private static unsafe void ApplyScaling(float* data, long size, float scale)
        {
            for (long i = 0; i < size; i++)
            {
                data[i] *= scale;
            }
        }

        /// <summary>
        /// Applies layer normalization to tensor data.
        /// </summary>
        private static unsafe void ApplyLayerNorm(float* input, float* output, float* weights, long totalSize, int hiddenSize)
        {
            int batchSeqLen = (int)(totalSize / hiddenSize);
            
            for (int i = 0; i < batchSeqLen; i++)
            {
                var inputPtr = input + i * hiddenSize;
                var outputPtr = output + i * hiddenSize;
                
                // Calculate mean
                float mean = 0.0f;
                for (int j = 0; j < hiddenSize; j++)
                    mean += inputPtr[j];
                mean /= hiddenSize;
                
                // Calculate variance
                float variance = 0.0f;
                for (int j = 0; j < hiddenSize; j++)
                {
                    float diff = inputPtr[j] - mean;
                    variance += diff * diff;
                }
                variance /= hiddenSize;
                
                // Normalize
                float invStd = 1.0f / MathF.Sqrt(variance + 1e-5f);
                for (int j = 0; j < hiddenSize; j++)
                {
                    outputPtr[j] = (inputPtr[j] - mean) * invStd * weights[j] + weights[hiddenSize + j];
                }
            }
        }

        /// <summary>
        /// Adds two tensors element-wise.
        /// </summary>
        private static unsafe void AddTensors(float* a, float* b, float* output, long size)
        {
            for (long i = 0; i < size; i++)
            {
                output[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// Executes feed-forward network.
        /// </summary>
        private static unsafe void ExecuteFeedForward(
            float* input, float* weights, float* output,
            int batchSize, int seqLen, int dModel, int ffnDim,
            IntPtr context)
        {
            var hiddenSize = batchSize * seqLen * ffnDim;
            var hidden = stackalloc float[Math.Min(hiddenSize, 512 * 1024)];
            
            // First linear layer
            ExecuteMatrixMultiply(input, weights, hidden, batchSize * seqLen, ffnDim, dModel, context);
            
            // Apply ReLU activation
            ApplyActivation(hidden, hiddenSize, ANEActivation.ReLU);
            
            // Second linear layer
            ExecuteMatrixMultiply(hidden, weights + dModel * ffnDim, output, batchSize * seqLen, dModel, ffnDim, context);
        }

        #endregion
    }

    /// <summary>
    /// Configuration for transformer operations on ANE.
    /// </summary>
    public sealed class ANETransformerConfig
    {
        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; } = 1;

        /// <summary>
        /// Gets or sets the sequence length.
        /// </summary>
        public int SequenceLength { get; set; }

        /// <summary>
        /// Gets or sets the model dimension.
        /// </summary>
        public int ModelDimension { get; set; } = 768;

        /// <summary>
        /// Gets or sets the number of attention heads.
        /// </summary>
        public int NumAttentionHeads { get; set; } = 12;

        /// <summary>
        /// Gets or sets the feed-forward network dimension.
        /// </summary>
        public int FFNDimension { get; set; } = 3072;

        /// <summary>
        /// Gets or sets the dropout probability.
        /// </summary>
        public float DropoutProbability { get; set; } = 0.1f;

        /// <summary>
        /// Creates a default transformer configuration.
        /// </summary>
        public static ANETransformerConfig Default => new();
    }
}