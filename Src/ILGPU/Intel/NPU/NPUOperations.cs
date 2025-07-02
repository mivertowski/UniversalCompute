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

using ILGPU.Intel.NPU.Native;
using ILGPU.Runtime;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// High-level operations for Intel Neural Processing Unit.
    /// </summary>
    public static class NPUOperations
    {
        #region Convolution Operations

        /// <summary>
        /// Executes a convolution operation on the Intel NPU.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="weights">Convolution weights.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="config">Convolution configuration.</param>
        /// <param name="context">NPU context handle.</param>
        public static unsafe void ExecuteConvolution(
            float* input,
            ArrayView<float> weights,
            float* output,
            NPUConvolutionConfig config,
            IntPtr context)
        {
            // Validate inputs
            if (input == null || output == null || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for NPU convolution");

            var inputSize = GetTensorSize(config.InputShape);
            var outputSize = GetTensorSize(config.OutputShape);

            // Pin weights for the duration of the operation
            fixed (float* weightsPtr = weights.SubView(0, (int)weights.Length).AsSpan())
            {
                // Execute native NPU convolution through OpenVINO
                NPUNative.ExecuteConvolution(input, output, inputSize, outputSize, context);
                
                // Apply activation if specified
                if (config.Activation != NPUActivation.None)
                {
                    ApplyActivation(output, outputSize, config.Activation);
                }
            }
        }

        /// <summary>
        /// Executes a depthwise separable convolution on NPU.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="depthwiseWeights">Depthwise convolution weights.</param>
        /// <param name="pointwiseWeights">Pointwise convolution weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Convolution configuration.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteDepthwiseSeparableConvolution(
            float* input,
            ArrayView<float> depthwiseWeights,
            ArrayView<float> pointwiseWeights,
            float* output,
            NPUConvolutionConfig config,
            IntPtr context)
        {
            var inputSize = GetTensorSize(config.InputShape);
            var outputSize = GetTensorSize(config.OutputShape);
            
            // Allocate intermediate buffer
            var intermediateSize = inputSize; // Same size for depthwise
            var intermediate = stackalloc float[(int)Math.Min(intermediateSize, 1024 * 1024)];

            // Step 1: Depthwise convolution
            fixed (float* dwWeights = depthwiseWeights.SubView(0, (int)depthwiseWeights.Length).AsSpan())
            {
                NPUNative.ExecuteConvolution(input, intermediate, inputSize, intermediateSize, context);
            }

            // Step 2: Pointwise convolution (1x1)
            fixed (float* pwWeights = pointwiseWeights.SubView(0, (int)pointwiseWeights.Length).AsSpan())
            {
                NPUNative.ExecuteConvolution(intermediate, output, intermediateSize, outputSize, context);
            }

            // Apply activation
            if (config.Activation != NPUActivation.None)
            {
                ApplyActivation(output, outputSize, config.Activation);
            }
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Executes matrix multiplication on NPU.
        /// </summary>
        /// <param name="a">Matrix A data.</param>
        /// <param name="b">Matrix B data.</param>
        /// <param name="c">Result matrix C data.</param>
        /// <param name="m">Number of rows in A.</param>
        /// <param name="n">Number of columns in B.</param>
        /// <param name="k">Number of columns in A / rows in B.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteMatrixMultiply(
            float* a, float* b, float* c,
            int m, int n, int k,
            IntPtr context)
        {
            if (a == null || b == null || c == null || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for NPU matrix multiply");

            var inputSize = m * k + k * n; // Size of both input matrices
            var outputSize = m * n; // Size of output matrix

            NPUNative.ExecuteMatMul(a, c, inputSize, outputSize, context);
        }

        /// <summary>
        /// Executes batched matrix multiplication on NPU.
        /// </summary>
        /// <param name="a">Batch of A matrices.</param>
        /// <param name="b">Batch of B matrices.</param>
        /// <param name="c">Batch of result C matrices.</param>
        /// <param name="batchSize">Number of matrices in batch.</param>
        /// <param name="m">Matrix dimension M.</param>
        /// <param name="n">Matrix dimension N.</param>
        /// <param name="k">Matrix dimension K.</param>
        /// <param name="context">NPU context.</param>
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

        #region Quantized Operations

        /// <summary>
        /// Executes quantized inference using INT8 precision.
        /// </summary>
        /// <param name="input">Input data in INT8 format.</param>
        /// <param name="weights">Quantized weights.</param>
        /// <param name="output">Output data in FP32 format.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteQuantizedInference(
            sbyte* input,
            sbyte* weights,
            float* output,
            NPUQuantizationConfig config,
            IntPtr context)
        {
            if (input == null || weights == null || output == null || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for NPU quantized inference");

            var inputSize = config.InputElements;
            var outputSize = config.OutputElements;

            // Execute quantized operation through OpenVINO
            NPUNative.ExecuteQuantizedInference(
                input, weights, output,
                inputSize, outputSize,
                config.InputScale, config.WeightScale, config.OutputScale,
                context);
        }

        /// <summary>
        /// Executes mixed precision inference (INT8 input, FP16 weights).
        /// </summary>
        /// <param name="input">Input data in INT8 format.</param>
        /// <param name="weights">Weights in FP16 format.</param>
        /// <param name="output">Output data in FP32 format.</param>
        /// <param name="inputScale">Input quantization scale.</param>
        /// <param name="outputScale">Output dequantization scale.</param>
        /// <param name="inputSize">Size of input data.</param>
        /// <param name="outputSize">Size of output data.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteMixedPrecisionInference(
            sbyte* input,
            ushort* weights, // FP16 as ushort
            float* output,
            float inputScale,
            float outputScale,
            long inputSize,
            long outputSize,
            IntPtr context)
        {
            // Execute mixed precision operation
            NPUNative.ExecuteMixedPrecisionInference(
                input, weights, output,
                inputSize, outputSize,
                inputScale, outputScale,
                context);
        }

        #endregion

        #region Model Execution

        /// <summary>
        /// Executes a pre-compiled OpenVINO model on NPU.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <param name="output">Output data.</param>
        /// <param name="model">OpenVINO model handle.</param>
        /// <param name="config">Execution configuration.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteOpenVINOModel(
            float* input,
            float* output,
            IntPtr model,
            NPUExecutionConfig config,
            IntPtr context)
        {
            if (input == null || output == null || model == IntPtr.Zero || context == IntPtr.Zero)
                throw new ArgumentException("Invalid input parameters for NPU OpenVINO model execution");

            // Set execution parameters
            NPUNative.SetExecutionConfig(context, config.BatchSize, config.Precision, config.CacheMode);

            // Execute model inference
            NPUNative.ExecuteOpenVINOInference(
                input, output,
                config.InputSize, config.OutputSize,
                model, context);
        }

        /// <summary>
        /// Executes a dynamic batch inference on NPU.
        /// </summary>
        /// <param name="inputs">Batch of input tensors.</param>
        /// <param name="outputs">Batch of output tensors.</param>
        /// <param name="batchSize">Dynamic batch size.</param>
        /// <param name="model">OpenVINO model handle.</param>
        /// <param name="context">NPU context.</param>
        public static unsafe void ExecuteDynamicBatchInference(
            float** inputs,
            float** outputs,
            int batchSize,
            IntPtr model,
            IntPtr context)
        {
            for (int i = 0; i < batchSize; i++)
            {
                // Execute each item in the batch
                // In a real implementation, this could be optimized to process the entire batch at once
                NPUNative.ExecuteOpenVINOInference(
                    inputs[i], outputs[i],
                    1, 1, // Single item sizes - would be computed based on model
                    model, context);
            }
        }

        #endregion

        #region Performance Optimization

        /// <summary>
        /// Enables NPU performance optimizations.
        /// </summary>
        /// <param name="context">NPU context.</param>
        /// <param name="optimizations">Optimization flags.</param>
        public static void EnableOptimizations(IntPtr context, NPUOptimizationFlags optimizations)
        {
            NPUNative.SetOptimizationFlags(context, (uint)optimizations);
        }

        /// <summary>
        /// Sets NPU power management mode.
        /// </summary>
        /// <param name="context">NPU context.</param>
        /// <param name="powerMode">Power management mode.</param>
        public static void SetPowerMode(IntPtr context, NPUPowerMode powerMode)
        {
            NPUNative.SetPowerMode(context, (int)powerMode);
        }

        /// <summary>
        /// Enables NPU telemetry and profiling.
        /// </summary>
        /// <param name="context">NPU context.</param>
        /// <param name="enableProfiling">Whether to enable profiling.</param>
        public static void EnableProfiling(IntPtr context, bool enableProfiling)
        {
            NPUNative.EnableProfiling(context, enableProfiling);
        }

        #endregion

        #region Helper Functions

        /// <summary>
        /// Calculates the total size of a tensor.
        /// </summary>
        private static long GetTensorSize((int N, int C, int H, int W) shape)
        {
            return (long)shape.N * shape.C * shape.H * shape.W;
        }

        /// <summary>
        /// Applies activation function to tensor data.
        /// </summary>
        private static unsafe void ApplyActivation(float* data, long size, NPUActivation activation)
        {
            switch (activation)
            {
                case NPUActivation.ReLU:
                    for (long i = 0; i < size; i++)
                        data[i] = Math.Max(0.0f, data[i]);
                    break;

                case NPUActivation.GELU:
                    for (long i = 0; i < size; i++)
                        data[i] = data[i] * 0.5f * (1.0f + MathF.Tanh(MathF.Sqrt(2.0f / MathF.PI) * (data[i] + 0.044715f * data[i] * data[i] * data[i])));
                    break;

                case NPUActivation.Swish:
                    for (long i = 0; i < size; i++)
                        data[i] = data[i] / (1.0f + MathF.Exp(-data[i]));
                    break;

                case NPUActivation.Sigmoid:
                    for (long i = 0; i < size; i++)
                        data[i] = 1.0f / (1.0f + MathF.Exp(-data[i]));
                    break;

                case NPUActivation.Tanh:
                    for (long i = 0; i < size; i++)
                        data[i] = MathF.Tanh(data[i]);
                    break;

                case NPUActivation.LeakyReLU:
                    for (long i = 0; i < size; i++)
                        data[i] = data[i] > 0 ? data[i] : 0.01f * data[i];
                    break;
            }
        }

        /// <summary>
        /// Optimizes tensor layout for NPU execution.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="output">Optimized tensor data.</param>
        /// <param name="shape">Tensor shape.</param>
        /// <param name="targetLayout">Target layout format.</param>
        public static unsafe void OptimizeTensorLayout(
            float* input,
            float* output,
            (int N, int C, int H, int W) shape,
            NPUTensorLayout targetLayout)
        {
            switch (targetLayout)
            {
                case NPUTensorLayout.NCHW:
                    // Already in optimal format for most NPU operations
                    var totalSize = shape.N * shape.C * shape.H * shape.W;
                    Buffer.MemoryCopy(input, output, totalSize * sizeof(float), totalSize * sizeof(float));
                    break;

                case NPUTensorLayout.NHWC:
                    // Transform NCHW to NHWC for NPU-specific optimizations
                    TransformNCHWToNHWC(input, output, shape);
                    break;

                case NPUTensorLayout.Blocked:
                    // Create blocked layout for cache optimization
                    CreateBlockedLayout(input, output, shape);
                    break;
            }
        }

        /// <summary>
        /// Transforms tensor from NCHW to NHWC format.
        /// </summary>
        private static unsafe void TransformNCHWToNHWC(
            float* input,
            float* output,
            (int N, int C, int H, int W) shape)
        {
            for (int n = 0; n < shape.N; n++)
            {
                for (int h = 0; h < shape.H; h++)
                {
                    for (int w = 0; w < shape.W; w++)
                    {
                        for (int c = 0; c < shape.C; c++)
                        {
                            var nchwIndex = n * shape.C * shape.H * shape.W + c * shape.H * shape.W + h * shape.W + w;
                            var nhwcIndex = n * shape.H * shape.W * shape.C + h * shape.W * shape.C + w * shape.C + c;
                            output[nhwcIndex] = input[nchwIndex];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates a blocked layout for cache optimization.
        /// </summary>
        private static unsafe void CreateBlockedLayout(
            float* input,
            float* output,
            (int N, int C, int H, int W) shape)
        {
            const int blockSize = 16; // 16x16 blocks for cache optimization
            
            for (int n = 0; n < shape.N; n++)
            {
                for (int c = 0; c < shape.C; c++)
                {
                    for (int bh = 0; bh < shape.H; bh += blockSize)
                    {
                        for (int bw = 0; bw < shape.W; bw += blockSize)
                        {
                            // Process 16x16 block
                            for (int h = bh; h < Math.Min(bh + blockSize, shape.H); h++)
                            {
                                for (int w = bw; w < Math.Min(bw + blockSize, shape.W); w++)
                                {
                                    var inputIndex = n * shape.C * shape.H * shape.W + c * shape.H * shape.W + h * shape.W + w;
                                    var outputIndex = inputIndex; // Simplified - would use proper blocked indexing
                                    output[outputIndex] = input[inputIndex];
                                }
                            }
                        }
                    }
                }
            }
        }

        #endregion
    }
}