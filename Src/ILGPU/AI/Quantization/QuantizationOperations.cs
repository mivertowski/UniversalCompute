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

using ILGPU.Runtime;
using System;

namespace ILGPU.AI.Quantization
{
    /// <summary>
    /// High-level quantization operations for AI model compression.
    /// </summary>
    public static class QuantizationOperations
    {
        #region Symmetric INT8 Quantization

        /// <summary>
        /// Quantizes a tensor to INT8 using symmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input FP32 tensor.</param>
        /// <param name="output">Output INT8 tensor.</param>
        /// <param name="scale">Quantization scale factor.</param>
        public static void QuantizeSymmetricInt8(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            float scale)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<sbyte>, float>(
                QuantizationKernels.QuantizeToInt8Symmetric);

            kernel(new Index1D(input.IntLength), input, output, scale);
        }

        /// <summary>
        /// Dequantizes an INT8 tensor using symmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input INT8 tensor.</param>
        /// <param name="output">Output FP32 tensor.</param>
        /// <param name="scale">Dequantization scale factor.</param>
        public static void DequantizeSymmetricInt8(
            Accelerator accelerator,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            float scale)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<sbyte>, ArrayView<float>, float>(
                QuantizationKernels.DequantizeFromInt8Symmetric);

            kernel(new Index1D(input.IntLength), input, output, scale);
        }

        #endregion

        #region Asymmetric INT8 Quantization

        /// <summary>
        /// Quantizes a tensor to INT8 using asymmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input FP32 tensor.</param>
        /// <param name="output">Output INT8 tensor.</param>
        /// <param name="scale">Quantization scale factor.</param>
        /// <param name="zeroPoint">Zero point offset.</param>
        public static void QuantizeAsymmetricInt8(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            float scale,
            sbyte zeroPoint)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<sbyte>, float, sbyte>(
                QuantizationKernels.QuantizeToInt8Asymmetric);

            kernel(new Index1D(input.IntLength), input, output, scale, zeroPoint);
        }

        /// <summary>
        /// Dequantizes an INT8 tensor using asymmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input INT8 tensor.</param>
        /// <param name="output">Output FP32 tensor.</param>
        /// <param name="scale">Dequantization scale factor.</param>
        /// <param name="zeroPoint">Zero point offset.</param>
        public static void DequantizeAsymmetricInt8(
            Accelerator accelerator,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            float scale,
            sbyte zeroPoint)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<sbyte>, ArrayView<float>, float, sbyte>(
                QuantizationKernels.DequantizeFromInt8Asymmetric);

            kernel(new Index1D(input.IntLength), input, output, scale, zeroPoint);
        }

        #endregion

        #region Dynamic Range Quantization

        /// <summary>
        /// Performs dynamic range quantization on a tensor.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input FP32 tensor.</param>
        /// <param name="output">Output INT8 tensor.</param>
        /// <param name="blockSize">Size of quantization blocks.</param>
        public static void DynamicRangeQuantize(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<sbyte> output,
            int blockSize = 256)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var numBlocks = (input.Length + blockSize - 1) / blockSize;
            
            using var minValues = accelerator.Allocate1D<float>(numBlocks);
            using var maxValues = accelerator.Allocate1D<float>(numBlocks);

            // Compute quantization parameters
            var paramsKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                QuantizationKernels.ComputeQuantizationParams);

            paramsKernel(new Index1D((int)numBlocks), input, minValues.View, maxValues.View, blockSize);

            // Perform quantization
            var quantizeKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<sbyte>, ArrayView<float>, ArrayView<float>, int>(
                QuantizationKernels.DynamicRangeQuantize);

            quantizeKernel(new Index1D(input.IntLength), input, output, minValues.View, maxValues.View, blockSize);
        }

        /// <summary>
        /// Performs dynamic range dequantization on a tensor.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input INT8 tensor.</param>
        /// <param name="output">Output FP32 tensor.</param>
        /// <param name="minValues">Minimum values per block.</param>
        /// <param name="maxValues">Maximum values per block.</param>
        /// <param name="blockSize">Size of quantization blocks.</param>
        public static void DynamicRangeDequantize(
            Accelerator accelerator,
            ArrayView<sbyte> input,
            ArrayView<float> output,
            ArrayView<float> minValues,
            ArrayView<float> maxValues,
            int blockSize = 256)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<sbyte>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                QuantizationKernels.DynamicRangeDequantize);

            kernel(new Index1D(input.IntLength), input, output, minValues, maxValues, blockSize);
        }

        #endregion

        #region BFloat16 Operations

        /// <summary>
        /// Converts FP32 tensor to BFloat16 format.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input FP32 tensor.</param>
        /// <param name="output">Output BFloat16 tensor (as ushort).</param>
        public static void ConvertToBFloat16(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<ushort> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<ushort>>(
                QuantizationKernels.ConvertToBFloat16);

            kernel(new Index1D(input.IntLength), input, output);
        }

        /// <summary>
        /// Converts BFloat16 tensor to FP32 format.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input BFloat16 tensor (as ushort).</param>
        /// <param name="output">Output FP32 tensor.</param>
        public static void ConvertFromBFloat16(
            Accelerator accelerator,
            ArrayView<ushort> input,
            ArrayView<float> output)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<ushort>, ArrayView<float>>(
                QuantizationKernels.ConvertFromBFloat16);

            kernel(new Index1D(input.IntLength), input, output);
        }

        #endregion

        #region Quantized Matrix Operations

        /// <summary>
        /// Performs quantized matrix multiplication with INT8 inputs.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="a">Matrix A (INT8).</param>
        /// <param name="b">Matrix B (INT8).</param>
        /// <param name="c">Result matrix C (INT32).</param>
        /// <param name="m">Number of rows in A.</param>
        /// <param name="n">Number of columns in B.</param>
        /// <param name="k">Number of columns in A / rows in B.</param>
        public static void QuantizedMatrixMultiply(
            Accelerator accelerator,
            ArrayView2D<sbyte, Stride2D.DenseX> a,
            ArrayView2D<sbyte, Stride2D.DenseX> b,
            ArrayView2D<int, Stride2D.DenseX> c,
            int m, int n, int k)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<sbyte, Stride2D.DenseX>, ArrayView2D<sbyte, Stride2D.DenseX>,
                ArrayView2D<int, Stride2D.DenseX>, int, int, int>(
                QuantizationKernels.QuantizedMatMul);

            kernel(new Index2D(m, n), a, b, c, m, n, k);
        }

        /// <summary>
        /// Applies scale and bias to quantized matrix multiplication result.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="input">Input INT32 tensor.</param>
        /// <param name="output">Output FP32 tensor.</param>
        /// <param name="scale">Scale factor.</param>
        /// <param name="bias">Bias value.</param>
        public static void ApplyScaleAndBias(
            Accelerator accelerator,
            ArrayView<int> input,
            ArrayView<float> output,
            float scale,
            float bias)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<float>, float, float>(
                QuantizationKernels.ApplyScaleAndBias);

            kernel(new Index1D(input.IntLength), input, output, scale, bias);
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Computes the optimal scale factor for symmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="data">Input data to analyze.</param>
        /// <returns>Optimal scale factor.</returns>
        public static float ComputeSymmetricScale(Accelerator accelerator, ArrayView<float> data)
        {
            // Find absolute maximum value
            using var absData = accelerator.Allocate1D<float>(data.Length);
            
            // Compute absolute values (simplified - would use kernel for efficiency)
            var hostData = data.GetAsArray();
            var maxAbs = 0.0f;
            
            for (int i = 0; i < hostData.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(hostData[i]));
            }

            // Scale to utilize full INT8 range [-127, 127]
            return maxAbs / 127.0f;
        }

        /// <summary>
        /// Computes optimal scale and zero point for asymmetric quantization.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="data">Input data to analyze.</param>
        /// <returns>Tuple of (scale, zeroPoint).</returns>
        public static (float scale, sbyte zeroPoint) ComputeAsymmetricParams(
            Accelerator accelerator, 
            ArrayView<float> data)
        {
            // Find min and max values (simplified - would use reduction kernels)
            var hostData = data.GetAsArray();
            var minVal = float.MaxValue;
            var maxVal = float.MinValue;
            
            for (int i = 0; i < hostData.Length; i++)
            {
                minVal = Math.Min(minVal, hostData[i]);
                maxVal = Math.Max(maxVal, hostData[i]);
            }

            // Compute scale and zero point for range [-128, 127]
            var scale = (maxVal - minVal) / 255.0f;
            var zeroPoint = (sbyte)Math.Round(-minVal / scale - 128.0f);
            
            return (scale, zeroPoint);
        }

        #endregion
    }
}