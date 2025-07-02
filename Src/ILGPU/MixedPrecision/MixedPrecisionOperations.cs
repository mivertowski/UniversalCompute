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
using ILGPU.TensorCores;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ILGPU.MixedPrecision
{
    /// <summary>
    /// Defines precision modes for mixed-precision operations.
    /// </summary>
    public enum PrecisionMode
    {
        /// <summary>
        /// Full precision (FP32).
        /// </summary>
        Full,

        /// <summary>
        /// Half precision (FP16).
        /// </summary>
        Half,

        /// <summary>
        /// BFloat16 precision.
        /// </summary>
        BFloat16,

        /// <summary>
        /// TensorFloat-32 precision.
        /// </summary>
        TensorFloat32,

        /// <summary>
        /// 8-bit integer quantization.
        /// </summary>
        INT8,

        /// <summary>
        /// Mixed precision (FP16 compute, FP32 accumulate).
        /// </summary>
        Mixed,

        /// <summary>
        /// Automatic precision selection based on hardware.
        /// </summary>
        Auto
    }

    /// <summary>
    /// Configuration for mixed-precision operations.
    /// </summary>
    public sealed class MixedPrecisionConfig
    {
        /// <summary>
        /// Gets or sets the compute precision for forward pass.
        /// </summary>
        public PrecisionMode ComputePrecision { get; set; } = PrecisionMode.Half;

        /// <summary>
        /// Gets or sets the storage precision for weights and activations.
        /// </summary>
        public PrecisionMode StoragePrecision { get; set; } = PrecisionMode.Half;

        /// <summary>
        /// Gets or sets the accumulation precision for gradients.
        /// </summary>
        public PrecisionMode AccumulationPrecision { get; set; } = PrecisionMode.Full;

        /// <summary>
        /// Gets or sets whether to use automatic loss scaling.
        /// </summary>
        public bool EnableAutoLossScaling { get; set; } = true;

        /// <summary>
        /// Gets or sets the initial loss scale factor.
        /// </summary>
        public float LossScale { get; set; } = 65536.0f;

        /// <summary>
        /// Gets or sets the loss scale update interval.
        /// </summary>
        public int LossScaleUpdateInterval { get; set; } = 2000;

        /// <summary>
        /// Gets or sets the minimum loss scale value.
        /// </summary>
        public float MinLossScale { get; set; } = 1.0f;

        /// <summary>
        /// Gets or sets the maximum loss scale value.
        /// </summary>
        public float MaxLossScale { get; set; } = 65536.0f;

        /// <summary>
        /// Gets the default mixed precision configuration.
        /// </summary>
        public static MixedPrecisionConfig Default => new();

        /// <summary>
        /// Creates a configuration optimized for inference.
        /// </summary>
        /// <returns>Inference-optimized configuration.</returns>
        public static MixedPrecisionConfig ForInference() => new()
        {
            ComputePrecision = PrecisionMode.Half,
            StoragePrecision = PrecisionMode.Half,
            AccumulationPrecision = PrecisionMode.Half,
            EnableAutoLossScaling = false
        };

        /// <summary>
        /// Creates a configuration optimized for training.
        /// </summary>
        /// <returns>Training-optimized configuration.</returns>
        public static MixedPrecisionConfig ForTraining() => new()
        {
            ComputePrecision = PrecisionMode.Half,
            StoragePrecision = PrecisionMode.Half,
            AccumulationPrecision = PrecisionMode.Full,
            EnableAutoLossScaling = true,
            LossScale = 65536.0f
        };
    }

    /// <summary>
    /// Provides mixed-precision operations for AI/ML workloads.
    /// </summary>
    public static class MixedPrecisionOperations
    {
        #region Precision Conversion

        /// <summary>
        /// Converts FP32 to FP16 with proper rounding.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="source">Source FP32 array.</param>
        /// <param name="destination">Destination FP16 array.</param>
        public static void ConvertFP32ToFP16(
            Accelerator accelerator,
            ArrayView<float> source,
            ArrayView<Half> destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<Half>>(
                ConvertFP32ToFP16Kernel);
            kernel(new Index1D(source.IntLength), source, destination);
        }

        /// <summary>
        /// Converts FP16 to FP32.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="source">Source FP16 array.</param>
        /// <param name="destination">Destination FP32 array.</param>
        public static void ConvertFP16ToFP32(
            Accelerator accelerator,
            ArrayView<Half> source,
            ArrayView<float> destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Half>, ArrayView<float>>(
                ConvertFP16ToFP32Kernel);
            kernel(new Index1D(source.IntLength), source, destination);
        }

        /// <summary>
        /// Converts FP32 to BFloat16.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="source">Source FP32 array.</param>
        /// <param name="destination">Destination BF16 array.</param>
        public static void ConvertFP32ToBF16(
            Accelerator accelerator,
            ArrayView<float> source,
            ArrayView<BFloat16> destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, ArrayView<BFloat16>>(
                ConvertFP32ToBF16Kernel);
            kernel(source, destination);
        }

        /// <summary>
        /// Quantizes FP32 values to INT8 with scaling.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="source">Source FP32 array.</param>
        /// <param name="destination">Destination INT8 array.</param>
        /// <param name="scale">Quantization scale factor.</param>
        /// <param name="zeroPoint">Zero point for quantization.</param>
        public static void QuantizeFP32ToINT8(
            Accelerator accelerator,
            ArrayView<float> source,
            ArrayView<sbyte> destination,
            float scale,
            sbyte zeroPoint)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, ArrayView<sbyte>, float, sbyte>(
                QuantizeFP32ToINT8Kernel);
            kernel(source, destination, scale, zeroPoint);
        }

        /// <summary>
        /// Dequantizes INT8 values to FP32.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="source">Source INT8 array.</param>
        /// <param name="destination">Destination FP32 array.</param>
        /// <param name="scale">Dequantization scale factor.</param>
        /// <param name="zeroPoint">Zero point for dequantization.</param>
        public static void DequantizeINT8ToFP32(
            Accelerator accelerator,
            ArrayView<sbyte> source,
            ArrayView<float> destination,
            float scale,
            sbyte zeroPoint)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination arrays must have the same length");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<sbyte>, ArrayView<float>, float, sbyte>(
                DequantizeINT8ToFP32Kernel);
            kernel(source, destination, scale, zeroPoint);
        }

        #endregion

        #region Mixed Precision GEMM

        /// <summary>
        /// Performs mixed-precision GEMM: C = alpha * A * B + beta * C.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="alpha">Scaling factor for A*B.</param>
        /// <param name="a">Matrix A (FP16).</param>
        /// <param name="lda">Leading dimension of A.</param>
        /// <param name="b">Matrix B (FP16).</param>
        /// <param name="ldb">Leading dimension of B.</param>
        /// <param name="beta">Scaling factor for C.</param>
        /// <param name="c">Matrix C (FP32).</param>
        /// <param name="ldc">Leading dimension of C.</param>
        /// <param name="config">Mixed precision configuration.</param>
        public static void MixedPrecisionGEMM(
            Accelerator accelerator,
            AcceleratorStream stream,
            int m, int n, int k,
            float alpha,
            ArrayView2D<Half, Stride2D.DenseX> a, int lda,
            ArrayView2D<Half, Stride2D.DenseX> b, int ldb,
            float beta,
            ArrayView2D<float, Stride2D.DenseX> c, int ldc,
            MixedPrecisionConfig config)
        {
            if (accelerator.SupportsTensorCores())
            {
                // Use tensor cores for mixed precision
                var tensorConfig = new TensorOperations.TensorConfig
                {
                    TileSize = 16,
                    Precision = TensorPrecision.FP16,
                    UseMixedPrecision = true
                };

                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    int, int, int, float,
                    ArrayView2D<Half, Stride2D.DenseX>, int,
                    ArrayView2D<Half, Stride2D.DenseX>, int,
                    float, ArrayView2D<float, Stride2D.DenseX>, int,
                    TensorOperations.TensorConfig>(MixedPrecisionTensorGEMMKernel);

                kernel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, tensorConfig);
            }
            else
            {
                // Fallback to software mixed precision
                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    int, int, int, float,
                    ArrayView2D<Half, Stride2D.DenseX>, int,
                    ArrayView2D<Half, Stride2D.DenseX>, int,
                    float, ArrayView2D<float, Stride2D.DenseX>, int>(SoftwareMixedPrecisionGEMMKernel);

                kernel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
        }

        #endregion

        #region Loss Scaling

        /// <summary>
        /// Scales gradients to prevent underflow in FP16 training.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="gradients">Gradient arrays to scale.</param>
        /// <param name="lossScale">Loss scale factor.</param>
        public static void ScaleGradients(
            Accelerator accelerator,
            ArrayView<float> gradients,
            float lossScale)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, float>(
                ScaleGradientsKernel);
            kernel(gradients, lossScale);
        }

        /// <summary>
        /// Unscales gradients after loss scaling.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="gradients">Gradient arrays to unscale.</param>
        /// <param name="lossScale">Loss scale factor.</param>
        public static void UnscaleGradients(
            Accelerator accelerator,
            ArrayView<float> gradients,
            float lossScale)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, float>(
                UnscaleGradientsKernel);
            kernel(gradients, lossScale);
        }

        /// <summary>
        /// Checks for infinite or NaN values in gradients.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="gradients">Gradients to check.</param>
        /// <param name="hasInfNaN">Output flag indicating presence of inf/NaN.</param>
        public static void CheckForInfNaN(
            Accelerator accelerator,
            ArrayView<float> gradients,
            ArrayView<int> hasInfNaN)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, ArrayView<int>>(
                CheckInfNaNKernel);
            kernel(gradients, hasInfNaN);
        }

        #endregion

        #region Kernel Implementations

        /// <summary>
        /// Kernel for FP32 to FP16 conversion.
        /// </summary>
        private static void ConvertFP32ToFP16Kernel(Index1D index, ArrayView<float> source, ArrayView<Half> destination)
        {
            if (index < source.Length)
            {
                destination[index] = (Half)source[index];
            }
        }

        /// <summary>
        /// Kernel for FP16 to FP32 conversion.
        /// </summary>
        private static void ConvertFP16ToFP32Kernel(Index1D index, ArrayView<Half> source, ArrayView<float> destination)
        {
            if (index < source.Length)
            {
                destination[index] = (float)source[index];
            }
        }

        /// <summary>
        /// Kernel for FP32 to BFloat16 conversion.
        /// </summary>
        private static void ConvertFP32ToBF16Kernel(Index1D index, ArrayView<float> source, ArrayView<BFloat16> destination)
        {
            if (index < source.Length)
            {
                destination[index] = (BFloat16)source[index];
            }
        }

        /// <summary>
        /// Kernel for FP32 to INT8 quantization.
        /// </summary>
        private static void QuantizeFP32ToINT8Kernel(
            Index1D index,
            ArrayView<float> source,
            ArrayView<sbyte> destination,
            float scale,
            sbyte zeroPoint)
        {
            if (index < source.Length)
            {
                var quantized = XMath.Round(source[index] / scale) + zeroPoint;
                destination[index] = (sbyte)IntrinsicMath.Clamp(quantized, -128, 127);
            }
        }

        /// <summary>
        /// Kernel for INT8 to FP32 dequantization.
        /// </summary>
        private static void DequantizeINT8ToFP32Kernel(
            Index1D index,
            ArrayView<sbyte> source,
            ArrayView<float> destination,
            float scale,
            sbyte zeroPoint)
        {
            if (index < source.Length)
            {
                destination[index] = (source[index] - zeroPoint) * scale;
            }
        }

        /// <summary>
        /// Kernel for mixed precision tensor GEMM.
        /// </summary>
        private static void MixedPrecisionTensorGEMMKernel(
            Index2D index,
            int m, int n, int k,
            float alpha,
            ArrayView2D<Half, Stride2D.DenseX> a, int lda,
            ArrayView2D<Half, Stride2D.DenseX> b, int ldb,
            float beta,
            ArrayView2D<float, Stride2D.DenseX> c, int ldc,
            TensorOperations.TensorConfig config)
        {
            var row = index.X;
            var col = index.Y;

            if (row >= m || col >= n) return;

            // Load tensor fragments
            // This is a simplified version - actual implementation would use proper WMMA intrinsics
            float sum = 0.0f;
            for (int ki = 0; ki < k; ki++)
            {
                sum += (float)a[row, ki] * (float)b[ki, col];
            }

            c[row, col] = alpha * sum + beta * c[row, col];
        }

        /// <summary>
        /// Kernel for software mixed precision GEMM.
        /// </summary>
        private static void SoftwareMixedPrecisionGEMMKernel(
            Index2D index,
            int m, int n, int k,
            float alpha,
            ArrayView2D<Half, Stride2D.DenseX> a, int lda,
            ArrayView2D<Half, Stride2D.DenseX> b, int ldb,
            float beta,
            ArrayView2D<float, Stride2D.DenseX> c, int ldc)
        {
            var row = index.X;
            var col = index.Y;

            if (row >= m || col >= n) return;

            float sum = 0.0f;
            for (int ki = 0; ki < k; ki++)
            {
                // Convert to FP32 for computation, accumulate in FP32
                sum += (float)a[row, ki] * (float)b[ki, col];
            }

            c[row, col] = alpha * sum + beta * c[row, col];
        }

        /// <summary>
        /// Kernel for scaling gradients.
        /// </summary>
        private static void ScaleGradientsKernel(Index1D index, ArrayView<float> gradients, float lossScale)
        {
            if (index < gradients.Length)
            {
                gradients[index] *= lossScale;
            }
        }

        /// <summary>
        /// Kernel for unscaling gradients.
        /// </summary>
        private static void UnscaleGradientsKernel(Index1D index, ArrayView<float> gradients, float lossScale)
        {
            if (index < gradients.Length)
            {
                gradients[index] /= lossScale;
            }
        }

        /// <summary>
        /// Kernel for checking infinite or NaN values.
        /// </summary>
        private static void CheckInfNaNKernel(Index1D index, ArrayView<float> gradients, ArrayView<int> hasInfNaN)
        {
            if (index < gradients.Length)
            {
                var value = gradients[index];
                if (float.IsInfinity(value) || float.IsNaN(value))
                {
                    hasInfNaN[0] = 1;
                }
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Gets the optimal precision mode for the given accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <returns>Optimal precision mode.</returns>
        public static PrecisionMode GetOptimalPrecisionMode(Accelerator accelerator)
        {
            if (accelerator.SupportsTensorCores())
            {
                var supportedPrecisions = accelerator.GetSupportedTensorPrecisions();
                
                if (supportedPrecisions.Contains(TensorPrecision.BF16))
                    return PrecisionMode.BFloat16;
                if (supportedPrecisions.Contains(TensorPrecision.FP16))
                    return PrecisionMode.Mixed;
                if (supportedPrecisions.Contains(TensorPrecision.TF32))
                    return PrecisionMode.TensorFloat32;
            }

            return PrecisionMode.Full;
        }

        /// <summary>
        /// Calculates quantization parameters for a given array.
        /// </summary>
        /// <param name="values">Values to analyze.</param>
        /// <returns>Scale and zero point for quantization.</returns>
        public static (float scale, sbyte zeroPoint) CalculateQuantizationParams(ReadOnlySpan<float> values)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            foreach (var value in values)
            {
                if (!float.IsNaN(value))
                {
                    min = Math.Min(min, value);
                    max = Math.Max(max, value);
                }
            }

            // Calculate scale and zero point for symmetric quantization
            float scale = Math.Max(Math.Abs(min), Math.Abs(max)) / 127.0f;
            sbyte zeroPoint = 0; // Symmetric quantization

            return (scale, zeroPoint);
        }

        #endregion
    }
}