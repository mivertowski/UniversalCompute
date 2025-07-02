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

using ILGPU.Intel.AMX.Native;
using ILGPU.Runtime;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Advanced AI/ML operations using Intel AMX tiles.
    /// </summary>
    public static class AMXAIOperations
    {
        #region GEMM Operations

        /// <summary>
        /// Performs batched GEMM operations using AMX for AI workloads.
        /// </summary>
        /// <param name="batchCount">Number of matrices in the batch.</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="alpha">Scaling factor for A*B.</param>
        /// <param name="a">Batch of A matrices.</param>
        /// <param name="strideA">Stride between A matrices.</param>
        /// <param name="b">Batch of B matrices.</param>
        /// <param name="strideB">Stride between B matrices.</param>
        /// <param name="beta">Scaling factor for C.</param>
        /// <param name="c">Batch of C matrices.</param>
        /// <param name="strideC">Stride between C matrices.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void BatchedGEMM(
            int batchCount,
            int m, int n, int k,
            float alpha,
            float* a, int strideA,
            float* b, int strideB,
            float beta,
            float* c, int strideC,
            AMXTileConfiguration config)
        {
            for (int batch = 0; batch < batchCount; batch++)
            {
                var aPtr = a + batch * strideA;
                var bPtr = b + batch * strideB;
                var cPtr = c + batch * strideC;
                
                AMXOperations.MatrixMultiplyFP32(aPtr, bPtr, cPtr, m, n, k, config);
                
                // Apply scaling factors
                if (alpha != 1.0f || beta != 0.0f)
                {
                    ApplyScaling(cPtr, m, n, alpha, beta);
                }
            }
        }

        /// <summary>
        /// Performs mixed-precision GEMM with BF16 inputs and FP32 accumulation.
        /// </summary>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="a">Matrix A in BF16 format.</param>
        /// <param name="b">Matrix B in BF16 format.</param>
        /// <param name="c">Matrix C in FP32 format (output).</param>
        /// <param name="config">AMX tile configuration for BF16.</param>
        public static unsafe void MixedPrecisionGEMM(
            int m, int n, int k,
            ushort* a, ushort* b, float* c,
            AMXTileConfiguration config)
        {
            if (config.DataType != AMXDataType.BFloat16)
                throw new ArgumentException("Configuration must be set for BFloat16");

            AMXOperations.MatrixMultiplyBF16(a, b, c, m, n, k, config);
        }

        /// <summary>
        /// Performs quantized INT8 GEMM with FP32 accumulation.
        /// </summary>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="a">Matrix A in INT8 format.</param>
        /// <param name="b">Matrix B in INT8 format.</param>
        /// <param name="c">Matrix C in INT32 format (output).</param>
        /// <param name="scaleA">Scaling factor for matrix A.</param>
        /// <param name="scaleB">Scaling factor for matrix B.</param>
        /// <param name="config">AMX tile configuration for INT8.</param>
        public static unsafe void QuantizedGEMM(
            int m, int n, int k,
            sbyte* a, sbyte* b, int* c,
            float scaleA, float scaleB,
            AMXTileConfiguration config)
        {
            if (config.DataType != AMXDataType.Int8)
                throw new ArgumentException("Configuration must be set for INT8");

            AMXOperations.MatrixMultiplyINT8(a, b, c, m, n, k, config);
            
            // Apply quantization scaling
            ApplyQuantizationScaling(c, m, n, scaleA * scaleB);
        }

        #endregion

        #region Convolution Operations

        /// <summary>
        /// Performs 2D convolution using AMX tiles with im2col transformation.
        /// </summary>
        /// <param name="input">Input tensor (NCHW format).</param>
        /// <param name="weights">Weight tensor (OIHW format).</param>
        /// <param name="output">Output tensor (NOHW format).</param>
        /// <param name="batchSize">Batch size (N).</param>
        /// <param name="inputChannels">Input channels (C).</param>
        /// <param name="inputHeight">Input height (H).</param>
        /// <param name="inputWidth">Input width (W).</param>
        /// <param name="outputChannels">Output channels (O).</param>
        /// <param name="kernelHeight">Kernel height.</param>
        /// <param name="kernelWidth">Kernel width.</param>
        /// <param name="strideH">Vertical stride.</param>
        /// <param name="strideW">Horizontal stride.</param>
        /// <param name="padH">Vertical padding.</param>
        /// <param name="padW">Horizontal padding.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void Convolution2D(
            float* input, float* weights, float* output,
            int batchSize, int inputChannels, int inputHeight, int inputWidth,
            int outputChannels, int kernelHeight, int kernelWidth,
            int strideH, int strideW, int padH, int padW,
            AMXTileConfiguration config)
        {
            int outputHeight = (inputHeight + 2 * padH - kernelHeight) / strideH + 1;
            int outputWidth = (inputWidth + 2 * padW - kernelWidth) / strideW + 1;
            
            // Calculate dimensions for im2col
            int colHeight = kernelHeight * kernelWidth * inputChannels;
            int colWidth = outputHeight * outputWidth;
            int totalElements = batchSize * colHeight * colWidth;
            
            // Allocate temporary buffer for im2col transformation
            var colBuffer = stackalloc float[Math.Min(totalElements, 1024 * 1024)]; // Cap at 4MB
            
            for (int batch = 0; batch < batchSize; batch++)
            {
                var batchInput = input + batch * inputChannels * inputHeight * inputWidth;
                var batchOutput = output + batch * outputChannels * outputHeight * outputWidth;
                
                // Transform input to column format (im2col)
                Im2Col(batchInput, colBuffer, 
                      inputChannels, inputHeight, inputWidth,
                      kernelHeight, kernelWidth, 
                      outputHeight, outputWidth,
                      strideH, strideW, padH, padW);
                
                // Perform GEMM: weights × colBuffer = output
                AMXOperations.MatrixMultiplyFP32(
                    weights, colBuffer, batchOutput,
                    outputChannels, colWidth, colHeight, config);
            }
        }

        /// <summary>
        /// Performs depthwise separable convolution using AMX.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="depthwiseWeights">Depthwise convolution weights.</param>
        /// <param name="pointwiseWeights">Pointwise convolution weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="channels">Number of channels.</param>
        /// <param name="height">Input height.</param>
        /// <param name="width">Input width.</param>
        /// <param name="kernelSize">Depthwise kernel size.</param>
        /// <param name="outputChannels">Output channels for pointwise.</param>
        /// <param name="config">AMX configuration.</param>
        public static unsafe void DepthwiseSeparableConvolution(
            float* input, float* depthwiseWeights, float* pointwiseWeights, float* output,
            int batchSize, int channels, int height, int width,
            int kernelSize, int outputChannels,
            AMXTileConfiguration config)
        {
            // Allocate intermediate buffer for depthwise output
            int intermediateSize = batchSize * channels * height * width;
            var intermediate = stackalloc float[Math.Min(intermediateSize, 512 * 1024)];
            
            // Step 1: Depthwise convolution
            DepthwiseConvolution(input, depthwiseWeights, intermediate,
                               batchSize, channels, height, width, kernelSize);
            
            // Step 2: Pointwise convolution using AMX
            for (int batch = 0; batch < batchSize; batch++)
            {
                var batchIntermediate = intermediate + batch * channels * height * width;
                var batchOutput = output + batch * outputChannels * height * width;
                
                AMXOperations.MatrixMultiplyFP32(
                    pointwiseWeights, batchIntermediate, batchOutput,
                    outputChannels, height * width, channels, config);
            }
        }

        #endregion

        #region Attention Mechanisms

        /// <summary>
        /// Computes scaled dot-product attention using AMX tiles.
        /// </summary>
        /// <param name="queries">Query matrix (batch_size × seq_len × d_model).</param>
        /// <param name="keys">Key matrix (batch_size × seq_len × d_model).</param>
        /// <param name="values">Value matrix (batch_size × seq_len × d_model).</param>
        /// <param name="output">Output attention matrix.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="seqLength">Sequence length.</param>
        /// <param name="dModel">Model dimension.</param>
        /// <param name="config">AMX configuration.</param>
        public static unsafe void ScaledDotProductAttention(
            float* queries, float* keys, float* values, float* output,
            int batchSize, int seqLength, int dModel,
            AMXTileConfiguration config)
        {
            float scale = 1.0f / MathF.Sqrt(dModel);
            int attentionSize = seqLength * seqLength;
            
            // Allocate temporary buffer for attention scores
            var attentionScores = stackalloc float[Math.Min(attentionSize, 256 * 1024)];
            
            for (int batch = 0; batch < batchSize; batch++)
            {
                var batchQueries = queries + batch * seqLength * dModel;
                var batchKeys = keys + batch * seqLength * dModel;
                var batchValues = values + batch * seqLength * dModel;
                var batchOutput = output + batch * seqLength * dModel;
                
                // Compute Q × K^T
                AMXOperations.MatrixMultiplyTransposedFP32(
                    batchQueries, batchKeys, attentionScores,
                    seqLength, seqLength, dModel, config);
                
                // Apply scaling and softmax
                ApplyScaledSoftmax(attentionScores, seqLength, seqLength, scale);
                
                // Compute attention × V
                AMXOperations.MatrixMultiplyFP32(
                    attentionScores, batchValues, batchOutput,
                    seqLength, dModel, seqLength, config);
            }
        }

        /// <summary>
        /// Computes multi-head attention using AMX tiles.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="wq">Query weight matrix.</param>
        /// <param name="wk">Key weight matrix.</param>
        /// <param name="wv">Value weight matrix.</param>
        /// <param name="wo">Output weight matrix.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="seqLength">Sequence length.</param>
        /// <param name="dModel">Model dimension.</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="config">AMX configuration.</param>
        public static unsafe void MultiHeadAttention(
            float* input, float* wq, float* wk, float* wv, float* wo, float* output,
            int batchSize, int seqLength, int dModel, int numHeads,
            AMXTileConfiguration config)
        {
            int dHead = dModel / numHeads;
            int qkvSize = batchSize * seqLength * dModel;
            
            // Allocate temporary buffers
            var queries = stackalloc float[Math.Min(qkvSize, 512 * 1024)];
            var keys = stackalloc float[Math.Min(qkvSize, 512 * 1024)];
            var values = stackalloc float[Math.Min(qkvSize, 512 * 1024)];
            var headOutput = stackalloc float[Math.Min(qkvSize, 512 * 1024)];
            
            // Compute Q, K, V projections
            BatchedGEMM(batchSize, seqLength, dModel, dModel, 1.0f,
                       input, seqLength * dModel,
                       wq, dModel * dModel, 0.0f,
                       queries, seqLength * dModel, config);
                       
            BatchedGEMM(batchSize, seqLength, dModel, dModel, 1.0f,
                       input, seqLength * dModel,
                       wk, dModel * dModel, 0.0f,
                       keys, seqLength * dModel, config);
                       
            BatchedGEMM(batchSize, seqLength, dModel, dModel, 1.0f,
                       input, seqLength * dModel,
                       wv, dModel * dModel, 0.0f,
                       values, seqLength * dModel, config);
            
            // Process each head
            for (int head = 0; head < numHeads; head++)
            {
                int headOffset = head * dHead;
                
                for (int batch = 0; batch < batchSize; batch++)
                {
                    var batchQ = queries + batch * seqLength * dModel + headOffset;
                    var batchK = keys + batch * seqLength * dModel + headOffset;
                    var batchV = values + batch * seqLength * dModel + headOffset;
                    var batchHeadOut = headOutput + batch * seqLength * dHead;
                    
                    ScaledDotProductAttention(batchQ, batchK, batchV, batchHeadOut,
                                            1, seqLength, dHead, config);
                }
            }
            
            // Concatenate heads and apply output projection
            // This is a simplified version - actual implementation would handle head concatenation
            BatchedGEMM(batchSize, seqLength, dModel, dModel, 1.0f,
                       headOutput, seqLength * dModel,
                       wo, dModel * dModel, 0.0f,
                       output, seqLength * dModel, config);
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Applies scaling factors to a matrix.
        /// </summary>
        private static unsafe void ApplyScaling(float* matrix, int rows, int cols, float alpha, float beta)
        {
            int totalElements = rows * cols;
            var span = new Span<float>(matrix, totalElements);
            
            for (int i = 0; i < totalElements; i++)
            {
                span[i] = alpha * span[i] + beta;
            }
        }

        /// <summary>
        /// Applies quantization scaling to INT32 results.
        /// </summary>
        private static unsafe void ApplyQuantizationScaling(int* matrix, int rows, int cols, float scale)
        {
            int totalElements = rows * cols;
            var span = new Span<int>(matrix, totalElements);
            
            for (int i = 0; i < totalElements; i++)
            {
                span[i] = (int)(span[i] * scale);
            }
        }

        /// <summary>
        /// Transforms input tensor to column format for convolution (im2col).
        /// </summary>
        private static unsafe void Im2Col(
            float* input, float* colBuffer,
            int channels, int height, int width,
            int kernelH, int kernelW,
            int outputH, int outputW,
            int strideH, int strideW,
            int padH, int padW)
        {
            int colIndex = 0;
            
            for (int c = 0; c < channels; c++)
            {
                for (int kh = 0; kh < kernelH; kh++)
                {
                    for (int kw = 0; kw < kernelW; kw++)
                    {
                        for (int oh = 0; oh < outputH; oh++)
                        {
                            for (int ow = 0; ow < outputW; ow++)
                            {
                                int ih = oh * strideH + kh - padH;
                                int iw = ow * strideW + kw - padW;
                                
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIndex = c * height * width + ih * width + iw;
                                    colBuffer[colIndex] = input[inputIndex];
                                }
                                else
                                {
                                    colBuffer[colIndex] = 0.0f; // Padding
                                }
                                colIndex++;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Performs depthwise convolution (simplified implementation).
        /// </summary>
        private static unsafe void DepthwiseConvolution(
            float* input, float* weights, float* output,
            int batchSize, int channels, int height, int width,
            int kernelSize)
        {
            // Simplified depthwise convolution implementation
            // In practice, this would be optimized with SIMD or other acceleration
            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            float sum = 0.0f;
                            for (int kh = 0; kh < kernelSize; kh++)
                            {
                                for (int kw = 0; kw < kernelSize; kw++)
                                {
                                    int ih = h + kh - kernelSize / 2;
                                    int iw = w + kw - kernelSize / 2;
                                    
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int inputIdx = batch * channels * height * width + c * height * width + ih * width + iw;
                                        int weightIdx = c * kernelSize * kernelSize + kh * kernelSize + kw;
                                        sum += input[inputIdx] * weights[weightIdx];
                                    }
                                }
                            }
                            
                            int outputIdx = batch * channels * height * width + c * height * width + h * width + w;
                            output[outputIdx] = sum;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Applies scaled softmax to attention scores.
        /// </summary>
        private static unsafe void ApplyScaledSoftmax(float* scores, int rows, int cols, float scale)
        {
            for (int row = 0; row < rows; row++)
            {
                var rowPtr = scores + row * cols;
                
                // Scale and find max for numerical stability
                float maxVal = float.MinValue;
                for (int col = 0; col < cols; col++)
                {
                    rowPtr[col] *= scale;
                    maxVal = Math.Max(maxVal, rowPtr[col]);
                }
                
                // Compute exp and sum
                float sum = 0.0f;
                for (int col = 0; col < cols; col++)
                {
                    rowPtr[col] = MathF.Exp(rowPtr[col] - maxVal);
                    sum += rowPtr[col];
                }
                
                // Normalize
                float invSum = 1.0f / sum;
                for (int col = 0; col < cols; col++)
                {
                    rowPtr[col] *= invSum;
                }
            }
        }

        #endregion
    }
}