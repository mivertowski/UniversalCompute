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

using ILGPU;
using ILGPU.Runtime;
using System;

namespace ILGPU.AI.Quantization
{
    /// <summary>
    /// ILGPU kernels for AI model quantization and dequantization operations.
    /// </summary>
    /// <remarks>
    /// These kernels provide efficient GPU-accelerated quantization for reducing
    /// model size and improving inference performance in AI workloads.
    /// 
    /// Supported quantization formats:
    /// - INT8 symmetric and asymmetric quantization
    /// - INT4 block-wise quantization
    /// - BFloat16 conversion
    /// - Dynamic range quantization
    /// </remarks>
    public static class QuantizationKernels
    {
        #region INT8 Quantization Kernels

        /// <summary>
        /// Quantizes FP32 values to INT8 using symmetric quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="output">Output INT8 array.</param>
        /// <param name="scale">Quantization scale factor.</param>
        public static void QuantizeToInt8Symmetric(
            Index1D index,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            float scale)
        {
            if (index >= input.Length) return;

            var value = input[index];
            var quantized = IntrinsicMath.CPUOnly.Floor(0.5f + value / scale);
            
            // Clamp to INT8 range [-128, 127]
            quantized = IntrinsicMath.Clamp(quantized, -128.0f, 127.0f);
            
            output[index] = (sbyte)quantized;
        }

        /// <summary>
        /// Quantizes FP32 values to INT8 using asymmetric quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="output">Output INT8 array.</param>
        /// <param name="scale">Quantization scale factor.</param>
        /// <param name="zeroPoint">Zero point offset.</param>
        public static void QuantizeToInt8Asymmetric(
            Index1D index,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            float scale,
            sbyte zeroPoint)
        {
            if (index >= input.Length) return;

            var value = input[index];
            var quantized = IntrinsicMath.CPUOnly.Floor(0.5f + value / scale) + zeroPoint;
            
            // Clamp to INT8 range [-128, 127]
            quantized = IntrinsicMath.Clamp(quantized, -128.0f, 127.0f);
            
            output[index] = (sbyte)quantized;
        }

        /// <summary>
        /// Dequantizes INT8 values to FP32 using symmetric quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input INT8 array.</param>
        /// <param name="output">Output FP32 array.</param>
        /// <param name="scale">Dequantization scale factor.</param>
        public static void DequantizeFromInt8Symmetric(
            Index1D index,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            float scale)
        {
            if (index >= input.Length) return;

            var quantized = input[index];
            output[index] = quantized * scale;
        }

        /// <summary>
        /// Dequantizes INT8 values to FP32 using asymmetric quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input INT8 array.</param>
        /// <param name="output">Output FP32 array.</param>
        /// <param name="scale">Dequantization scale factor.</param>
        /// <param name="zeroPoint">Zero point offset.</param>
        public static void DequantizeFromInt8Asymmetric(
            Index1D index,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            float scale,
            sbyte zeroPoint)
        {
            if (index >= input.Length) return;

            var quantized = input[index];
            output[index] = (quantized - zeroPoint) * scale;
        }

        #endregion

        #region INT4 Block Quantization Kernels

        /// <summary>
        /// Quantizes FP32 values to INT4 using block-wise quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="output">Output packed INT4 array (2 values per byte).</param>
        /// <param name="scales">Scale factors per block.</param>
        /// <param name="blockSize">Size of each quantization block.</param>
        public static void QuantizeToInt4Block(
            Index1D index,
            ArrayView<float> input,
            ArrayView<byte> output,
            ArrayView<float> scales,
            int blockSize)
        {
            var blockIndex = index / blockSize;
            var localIndex = index % blockSize;
            
            if (index >= input.Length || blockIndex >= scales.Length) return;

            var value = input[index];
            var scale = scales[blockIndex];
            var quantized = IntrinsicMath.CPUOnly.Floor(0.5f + value / scale);
            
            // Clamp to INT4 range [-8, 7]
            quantized = IntrinsicMath.Clamp(quantized, -8.0f, 7.0f);
            var int4Value = (sbyte)quantized;
            
            // Pack two INT4 values into one byte
            var outputIndex = index / 2;
            if (outputIndex >= output.Length) return;
            
            if (localIndex % 2 == 0)
            {
                // Lower 4 bits
                output[outputIndex] = (byte)((output[outputIndex] & 0xF0) | (int4Value & 0x0F));
            }
            else
            {
                // Upper 4 bits
                output[outputIndex] = (byte)((output[outputIndex] & 0x0F) | ((int4Value & 0x0F) << 4));
            }
        }

        /// <summary>
        /// Dequantizes INT4 values to FP32 using block-wise quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input packed INT4 array.</param>
        /// <param name="output">Output FP32 array.</param>
        /// <param name="scales">Scale factors per block.</param>
        /// <param name="blockSize">Size of each quantization block.</param>
        public static void DequantizeFromInt4Block(
            Index1D index,
            ArrayView<byte> input,
            ArrayView<float> output,
            ArrayView<float> scales,
            int blockSize)
        {
            var blockIndex = index / blockSize;
            var localIndex = index % blockSize;
            
            if (index >= output.Length || blockIndex >= scales.Length) return;

            var inputIndex = index / 2;
            if (inputIndex >= input.Length) return;
            
            var packedByte = input[inputIndex];
            sbyte int4Value;
            
            if (localIndex % 2 == 0)
            {
                // Lower 4 bits
                int4Value = (sbyte)(packedByte & 0x0F);
                // Sign extend
                if ((int4Value & 0x08) != 0)
                    int4Value |= unchecked((sbyte)0xF0);
            }
            else
            {
                // Upper 4 bits
                int4Value = (sbyte)((packedByte >> 4) & 0x0F);
                // Sign extend
                if ((int4Value & 0x08) != 0)
                    int4Value |= unchecked((sbyte)0xF0);
            }
            
            var scale = scales[blockIndex];
            output[index] = int4Value * scale;
        }

        #endregion

        #region Dynamic Range Quantization

        /// <summary>
        /// Computes quantization parameters for dynamic range quantization.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="minValues">Output minimum values per block.</param>
        /// <param name="maxValues">Output maximum values per block.</param>
        /// <param name="blockSize">Size of each quantization block.</param>
        public static void ComputeQuantizationParams(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> minValues,
            ArrayView<float> maxValues,
            int blockSize)
        {
            var blockIndex = index;
            if (blockIndex >= minValues.Length) return;

            var startIndex = blockIndex * blockSize;
            var endIndex = IntrinsicMath.Min(startIndex + blockSize, input.Length);
            
            if (startIndex >= input.Length) return;

            var minVal = float.MaxValue;
            var maxVal = float.MinValue;
            
            for (var i = startIndex; i < endIndex; i++)
            {
                var value = input[i];
                minVal = IntrinsicMath.Min(minVal, value);
                maxVal = IntrinsicMath.Max(maxVal, value);
            }
            
            minValues[blockIndex] = minVal;
            maxValues[blockIndex] = maxVal;
        }

        /// <summary>
        /// Performs dynamic range quantization to INT8.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="output">Output INT8 array.</param>
        /// <param name="minValues">Minimum values per block.</param>
        /// <param name="maxValues">Maximum values per block.</param>
        /// <param name="blockSize">Size of each quantization block.</param>
        public static void DynamicRangeQuantize(
            Index1D index,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            ArrayView<float> minValues,
            ArrayView<float> maxValues,
            int blockSize)
        {
            if (index >= input.Length) return;

            var blockIndex = index / blockSize;
            if (blockIndex >= minValues.Length) return;
            
            var value = input[index];
            var minVal = minValues[blockIndex];
            var maxVal = maxValues[blockIndex];
            
            // Avoid division by zero
            var range = maxVal - minVal;
            if (range < 1e-8f)
            {
                output[index] = 0;
                return;
            }
            
            // Map to [0, 1] then to [-128, 127]
            var normalized = (value - minVal) / range;
            var quantized = IntrinsicMath.CPUOnly.Floor(0.5f + normalized * 255.0f - 128.0f);
            quantized = IntrinsicMath.Clamp(quantized, -128.0f, 127.0f);
            
            output[index] = (sbyte)quantized;
        }

        /// <summary>
        /// Performs dynamic range dequantization from INT8.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input INT8 array.</param>
        /// <param name="output">Output FP32 array.</param>
        /// <param name="minValues">Minimum values per block.</param>
        /// <param name="maxValues">Maximum values per block.</param>
        /// <param name="blockSize">Size of each quantization block.</param>
        public static void DynamicRangeDequantize(
            Index1D index,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            ArrayView<float> minValues,
            ArrayView<float> maxValues,
            int blockSize)
        {
            if (index >= input.Length) return;

            var blockIndex = index / blockSize;
            if (blockIndex >= minValues.Length) return;
            
            var quantized = input[index];
            var minVal = minValues[blockIndex];
            var maxVal = maxValues[blockIndex];
            
            // Map from [-128, 127] to [0, 1] then to [minVal, maxVal]
            var normalized = (quantized + 128.0f) / 255.0f;
            output[index] = minVal + normalized * (maxVal - minVal);
        }

        #endregion

        #region BFloat16 Conversion Kernels

        /// <summary>
        /// Converts FP32 values to BFloat16 format.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input FP32 array.</param>
        /// <param name="output">Output BFloat16 array (as ushort).</param>
        public static void ConvertToBFloat16(
            Index1D index,
            ArrayView<float> input,
            ArrayView<ushort> output)
        {
            if (index >= input.Length) return;

            var value = input[index];
            
            // Convert FP32 to BFloat16 by truncating mantissa
            var bits = BitOperations.FloatAsInt(value);
            var bfloat16 = (ushort)(bits >> 16);
            
            output[index] = bfloat16;
        }

        /// <summary>
        /// Converts BFloat16 values to FP32 format.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input BFloat16 array (as ushort).</param>
        /// <param name="output">Output FP32 array.</param>
        public static void ConvertFromBFloat16(
            Index1D index,
            ArrayView<ushort> input,
            ArrayView<float> output)
        {
            if (index >= input.Length) return;

            var bfloat16 = input[index];
            
            // Convert BFloat16 to FP32 by expanding mantissa with zeros
            var bits = (uint)bfloat16 << 16;
            var value = BitOperations.IntAsFloat((int)bits);
            
            output[index] = value;
        }

        #endregion

        #region Quantized Matrix Operations

        /// <summary>
        /// Performs quantized matrix multiplication (INT8 x INT8 -> INT32).
        /// </summary>
        /// <param name="index">Thread index (row, col).</param>
        /// <param name="a">Matrix A (INT8).</param>
        /// <param name="b">Matrix B (INT8).</param>
        /// <param name="c">Result matrix C (INT32).</param>
        /// <param name="m">Number of rows in A.</param>
        /// <param name="n">Number of columns in B.</param>
        /// <param name="k">Number of columns in A / rows in B.</param>
        public static void QuantizedMatMul(
            Index2D index,
            ArrayView2D<sbyte, Stride2D.DenseX> a,
            ArrayView2D<sbyte, Stride2D.DenseX> b,
            ArrayView2D<int, Stride2D.DenseX> c,
            int m, int n, int k)
        {
            var row = index.X;
            var col = index.Y;
            
            if (row >= m || col >= n) return;

            var sum = 0;
            for (var i = 0; i < k; i++)
            {
                sum += a[row, i] * b[i, col];
            }
            
            c[row, col] = sum;
        }

        /// <summary>
        /// Applies scale and bias to quantized matrix multiplication result.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input INT32 array.</param>
        /// <param name="output">Output FP32 array.</param>
        /// <param name="scale">Scale factor.</param>
        /// <param name="bias">Bias value.</param>
        public static void ApplyScaleAndBias(
            Index1D index,
            ArrayView<int> input,
            ArrayView<float> output,
            float scale,
            float bias)
        {
            if (index >= input.Length) return;

            var value = input[index];
            output[index] = value * scale + bias;
        }

        #endregion

        #region Vector Quantization

        /// <summary>
        /// Performs vector quantization using k-means clustering.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="input">Input vectors.</param>
        /// <param name="codebook">Quantization codebook.</param>
        /// <param name="output">Output quantized indices.</param>
        /// <param name="vectorDim">Dimension of each vector.</param>
        /// <param name="numClusters">Number of clusters in codebook.</param>
        public static void VectorQuantize(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> codebook,
            ArrayView<int> output,
            int vectorDim,
            int numClusters)
        {
            if (index >= output.Length) return;

            var vectorStart = index * vectorDim;
            if (vectorStart + vectorDim > input.Length) return;

            var bestCluster = 0;
            var bestDistance = float.MaxValue;

            for (var cluster = 0; cluster < numClusters; cluster++)
            {
                var clusterStart = cluster * vectorDim;
                var distance = 0.0f;

                for (var dim = 0; dim < vectorDim; dim++)
                {
                    var diff = input[vectorStart + dim] - codebook[clusterStart + dim];
                    distance += diff * diff;
                }

                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestCluster = cluster;
                }
            }

            output[index] = bestCluster;
        }

        /// <summary>
        /// Reconstructs vectors from quantized indices.
        /// </summary>
        /// <param name="index">Thread index.</param>
        /// <param name="indices">Quantized indices.</param>
        /// <param name="codebook">Quantization codebook.</param>
        /// <param name="output">Reconstructed vectors.</param>
        /// <param name="vectorDim">Dimension of each vector.</param>
        public static void VectorDequantize(
            Index1D index,
            ArrayView<int> indices,
            ArrayView<float> codebook,
            ArrayView<float> output,
            int vectorDim)
        {
            var vectorIndex = index / vectorDim;
            var dimIndex = index % vectorDim;
            
            if (vectorIndex >= indices.Length || index >= output.Length) return;

            var clusterIndex = indices[vectorIndex];
            var codebookIndex = clusterIndex * vectorDim + dimIndex;
            
            if (codebookIndex >= codebook.Length) return;

            output[index] = codebook[codebookIndex];
        }

        #endregion
    }

    /// <summary>
    /// Utility class for bit operations in quantization.
    /// </summary>
    internal static class BitOperations
    {
        /// <summary>
        /// Reinterprets a float as an int.
        /// </summary>
        public static int FloatAsInt(float value)
        {
            unsafe
            {
                return *(int*)&value;
            }
        }

        /// <summary>
        /// Reinterprets an int as a float.
        /// </summary>
        public static float IntAsFloat(int value)
        {
            unsafe
            {
                return *(float*)&value;
            }
        }
    }
}