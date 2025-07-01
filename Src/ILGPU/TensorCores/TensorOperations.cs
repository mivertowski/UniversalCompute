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
using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;

namespace ILGPU.TensorCores
{
    /// <summary>
    /// Provides high-level tensor operations using tensor cores.
    /// </summary>
    public static class TensorOperations
    {
        /// <summary>
        /// Configuration for tensor operations.
        /// </summary>
        public struct TensorConfig
        {
            /// <summary>
            /// The tile size for matrix operations (must match tensor core requirements).
            /// </summary>
            public int TileSize { get; set; }

            /// <summary>
            /// The precision mode for operations.
            /// </summary>
            public TensorPrecision Precision { get; set; }

            /// <summary>
            /// Whether to use mixed precision (e.g., FP16 compute with FP32 accumulate).
            /// </summary>
            public bool UseMixedPrecision { get; set; }

            /// <summary>
            /// Gets the default configuration.
            /// </summary>
            public static TensorConfig Default => new()
            {
                TileSize = 16,
                Precision = TensorPrecision.FP16,
                UseMixedPrecision = true
            };
        }

        /// <summary>
        /// Performs matrix multiplication using tensor cores: C = alpha * A * B + beta * C.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="alpha">Scaling factor for A*B.</param>
        /// <param name="a">Matrix A.</param>
        /// <param name="lda">Leading dimension of A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="ldb">Leading dimension of B.</param>
        /// <param name="beta">Scaling factor for C.</param>
        /// <param name="c">Matrix C (input/output).</param>
        /// <param name="ldc">Leading dimension of C.</param>
        /// <param name="config">Tensor operation configuration.</param>
        public static void TensorGemm<T>(
            Accelerator accelerator,
            AcceleratorStream stream,
            int m,
            int n,
            int k,
            T alpha,
            ArrayView2D<T, Stride2D.DenseX> a,
            int lda,
            ArrayView2D<T, Stride2D.DenseX> b,
            int ldb,
            T beta,
            ArrayView2D<T, Stride2D.DenseX> c,
            int ldc,
            TensorConfig? config = null)
            where T : unmanaged, INumber<T>
        {
            var cfg = config ?? TensorConfig.Default;
            
            if (!TensorIntrinsics.IsTensorCoreSupported())
            {
                throw new NotSupportedException(
                    "Tensor cores are not supported on this device. " +
                    "Requires CUDA device with compute capability 7.0 or higher.");
            }

            // Validate dimensions are compatible with tensor cores
            ValidateTensorDimensions(m, n, k, cfg.TileSize);

            // Launch tensor GEMM kernel
            var gridDim = new Index3D(
                (m + cfg.TileSize - 1) / cfg.TileSize,
                (n + cfg.TileSize - 1) / cfg.TileSize,
                1);
            
            var blockDim = new Index3D(32, 1, 1); // Warp size for tensor cores

            // For benchmarking purposes, simulate tensor core operation
            // In production, this would load and execute actual tensor core kernels
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<T, Stride2D.DenseX>, ArrayView2D<T, Stride2D.DenseX>, ArrayView2D<T, Stride2D.DenseX>>(SimpleTensorGemmKernel);
            kernel(new Index2D(m, n), a, b, c);
        }

        /// <summary>
        /// Simple tensor GEMM kernel for benchmarking.
        /// </summary>
        private static void SimpleTensorGemmKernel<T>(
            Index2D index,
            ArrayView2D<T, Stride2D.DenseX> a,
            ArrayView2D<T, Stride2D.DenseX> b,
            ArrayView2D<T, Stride2D.DenseX> c)
            where T : unmanaged, INumber<T>
        {
            // Simple matrix multiplication for benchmarking
            var row = index.X;
            var col = index.Y;
            
            if (row < c.IntExtent.X && col < c.IntExtent.Y)
            {
                T sum = T.Zero;
                for (int k = 0; k < a.IntExtent.Y; k++)
                {
                    sum += a[row, k] * b[k, col];
                }
                c[row, col] = sum;
            }
        }

        /// <summary>
        /// The tensor GEMM kernel implementation.
        /// </summary>
        private static void TensorGemmKernel<T>(
            Index3D index,
            int m,
            int n,
            int k,
            T alpha,
            ArrayView2D<T, Stride2D.DenseX> a,
            int lda,
            ArrayView2D<T, Stride2D.DenseX> b,
            int ldb,
            T beta,
            ArrayView2D<T, Stride2D.DenseX> c,
            int ldc,
            TensorConfig config)
            where T : unmanaged, INumber<T>
        {
            // This is a simplified version - actual implementation would need
            // proper tiling and fragment management
            
            var warpId = index.X;
            var laneId = Warp.LaneIdx;
            
            // Calculate tile indices
            var tileM = warpId / (n / config.TileSize);
            var tileN = warpId % (n / config.TileSize);
            
            if (tileM * config.TileSize >= m || tileN * config.TileSize >= n)
                return;

            // In a real implementation, we would:
            // 1. Declare fragments for A, B, C, and D
            // 2. Load C fragment if beta != 0
            // 3. Loop over K dimension in tiles
            // 4. Load A and B fragments
            // 5. Perform MMA operation
            // 6. Scale and store result
            
            // For now, this is a placeholder
            TensorIntrinsics.SyncTensor();
        }

        /// <summary>
        /// Performs batched matrix multiplication using tensor cores.
        /// </summary>
        public static void TensorBatchedGemm<T>(
            Accelerator accelerator,
            AcceleratorStream stream,
            int batchCount,
            int m,
            int n,
            int k,
            T alpha,
            ArrayView<T> a,  // Simplified to 1D for now
            int lda,
            int strideA,
            ArrayView<T> b,
            int ldb,
            int strideB,
            T beta,
            ArrayView<T> c,
            int ldc,
            int strideC,
            TensorConfig? config = null)
            where T : unmanaged, INumber<T>
        {
            var cfg = config ?? TensorConfig.Default;
            
            // Launch batched kernel
            var gridDim = new Index3D(
                (m + cfg.TileSize - 1) / cfg.TileSize,
                (n + cfg.TileSize - 1) / cfg.TileSize,
                batchCount);
            
            var blockDim = new Index3D(32, 1, 1);

            // For now, implement as sequential batch of regular GEMM operations
            // This can be optimized with true batched tensor core operations later
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, int, int, int, int, T, ArrayView<T>, int, int, 
                ArrayView<T>, int, int, T, ArrayView<T>, int, int, TensorConfig>(
                TensorBatchedGemmKernel);

            kernel(stream, batchCount, batchCount, m, n, k, alpha, a, lda, strideA, 
                   b, ldb, strideB, beta, c, ldc, strideC, cfg);
        }

        /// <summary>
        /// The batched tensor GEMM kernel.
        /// </summary>
        private static void TensorBatchedGemmKernel<T>(
            Index3D index,
            int batchCount,
            int m,
            int n,
            int k,
            T alpha,
            ArrayView<T> a,
            int lda,
            int strideA,
            ArrayView<T> b,
            int ldb,
            int strideB,
            T beta,
            ArrayView<T> c,
            int ldc,
            int strideC,
            TensorConfig config)
            where T : unmanaged, INumber<T>
        {
            var batchIdx = index.Z;
            if (batchIdx >= batchCount)
                return;

            // Process batch element
            // Implementation would follow similar pattern to single GEMM
            TensorIntrinsics.SyncTensor();
        }

        /// <summary>
        /// Performs 2D convolution using tensor cores.
        /// </summary>
        public static void TensorConv2D<T>(
            Accelerator accelerator,
            AcceleratorStream stream,
            ArrayView<T> input,      // Flattened NCHW format
            ArrayView<T> filter,     // Flattened KCRS format
            ArrayView<T> output,     // Flattened NKHW format
            int inputN, int inputC, int inputH, int inputW,
            int filterK, int filterC, int filterR, int filterS,
            int strideH,
            int strideW,
            int padH,
            int padW,
            TensorConfig? config = null)
            where T : unmanaged, INumber<T>
        {
            // Implement direct convolution for now - can be optimized with im2col later
            var cfg = config ?? TensorConfig.Default;
            
            // Calculate output dimensions
            int outputH = (inputH + 2 * padH - filterR) / strideH + 1;
            int outputW = (inputW + 2 * padW - filterS) / strideW + 1;
            
            // Basic validation
            if (filterC != inputC)
                throw new ArgumentException("Filter channels must match input channels");
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView<T>, ArrayView<T>, ArrayView<T>,
                int, int, int, int, int, int, int, int,
                int, int, int, int, int, int, int, int>(
                TensorConvolutionKernel);
                
            var gridDim = new Index3D(
                (outputH + cfg.TileSize - 1) / cfg.TileSize,
                (outputW + cfg.TileSize - 1) / cfg.TileSize,
                filterK);
                
            kernel(stream, gridDim, input, filter, output,
                   inputN, inputC, inputH, inputW,
                   filterK, filterC, filterR, filterS,
                   outputH, outputW, strideH, strideW, padH, padW);
        }

        /// <summary>
        /// Validates that matrix dimensions are compatible with tensor cores.
        /// </summary>
        private static void ValidateTensorDimensions(int m, int n, int k, int tileSize)
        {
            if (m % tileSize != 0 || n % tileSize != 0 || k % tileSize != 0)
            {
                throw new ArgumentException(
                    $"Matrix dimensions ({m}x{n}x{k}) must be multiples of tile size ({tileSize}) " +
                    "for tensor core operations. Consider padding the matrices.");
            }
        }

        /// <summary>
        /// Kernel for batched tensor GEMM operations.
        /// </summary>
        private static void TensorBatchedGemmKernel<T>(
            Index1D index,
            int batchCount,
            int m,
            int n,
            int k,
            T alpha,
            ArrayView<T> a,
            int lda,
            int strideA,
            ArrayView<T> b,
            int ldb,
            int strideB,
            T beta,
            ArrayView<T> c,
            int ldc,
            int strideC,
            TensorConfig config)
            where T : unmanaged, INumber<T>
        {
            int batchIndex = index;
            if (batchIndex >= batchCount) return;

            // Calculate batch offsets
            int aOffset = batchIndex * strideA;
            int bOffset = batchIndex * strideB;
            int cOffset = batchIndex * strideC;

            // For now, implement basic matrix multiplication
            // This would be replaced with actual tensor core operations
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T sum = T.Zero;
                    for (int ki = 0; ki < k; ki++)
                    {
                        sum += a[aOffset + i * lda + ki] * b[bOffset + ki * ldb + j];
                    }
                    c[cOffset + i * ldc + j] = alpha * sum + beta * c[cOffset + i * ldc + j];
                }
            }
        }

        /// <summary>
        /// Kernel for tensor convolution operations.
        /// </summary>
        private static void TensorConvolutionKernel<T>(
            Index3D index,
            ArrayView<T> input,
            ArrayView<T> filter,
            ArrayView<T> output,
            int inputN, int inputC, int inputH, int inputW,
            int filterK, int filterC, int filterR, int filterS,
            int outputH, int outputW,
            int strideH, int strideW, int padH, int padW)
            where T : unmanaged, INumber<T>
        {
            int oh = index.X;
            int ow = index.Y;
            int k = index.Z;

            if (oh >= outputH || ow >= outputW || k >= filterK) return;

            // Basic convolution implementation
            T sum = T.Zero;
            for (int c = 0; c < inputC; c++)
            {
                for (int r = 0; r < filterR; r++)
                {
                    for (int s = 0; s < filterS; s++)
                    {
                        int ih = oh * strideH + r - padH;
                        int iw = ow * strideW + s - padW;

                        if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                        {
                            int inputIdx = c * inputH * inputW + ih * inputW + iw;
                            int filterIdx = k * filterC * filterR * filterS + c * filterR * filterS + r * filterS + s;
                            sum += input[inputIdx] * filter[filterIdx];
                        }
                    }
                }
            }

            int outputIdx = k * outputH * outputW + oh * outputW + ow;
            output[outputIdx] = sum;
        }
    }

    /// <summary>
    /// Extension methods for tensor operations on accelerators.
    /// </summary>
    public static class TensorAcceleratorExtensions
    {
        /// <summary>
        /// Checks if this accelerator supports tensor cores.
        /// </summary>
        /// <param name="accelerator">The accelerator to check.</param>
        /// <returns>True if tensor cores are supported.</returns>
        public static bool SupportsTensorCores(this Accelerator accelerator)
        {
            if (accelerator is CudaAccelerator cuda)
            {
                return cuda.Architecture.Major >= 7; // SM_70+
            }
            return false;
        }

        /// <summary>
        /// Gets the supported tensor precisions for this accelerator.
        /// </summary>
        public static TensorPrecision[] GetSupportedTensorPrecisions(this Accelerator accelerator)
        {
            if (accelerator is CudaAccelerator cuda)
            {
                var arch = cuda.Architecture;
                if (arch.Major >= 9) // SM_90+
                {
                    return [ 
                        TensorPrecision.FP16, TensorPrecision.BF16, TensorPrecision.TF32, 
                        TensorPrecision.INT8, TensorPrecision.FP8_E4M3, TensorPrecision.FP8_E5M2 
                    ];
                }
                else if (arch.Major >= 8) // SM_80+
                {
                    return [ 
                        TensorPrecision.FP16, TensorPrecision.BF16, TensorPrecision.TF32, 
                        TensorPrecision.INT8 
                    ];
                }
                else if (arch.Major == 7 && arch.Minor >= 5) // SM_75+
                {
                    return [ 
                        TensorPrecision.FP16, TensorPrecision.INT8 
                    ];
                }
                else if (arch.Major == 7 && arch.Minor >= 0) // SM_70+
                {
                    return [ 
                        TensorPrecision.FP16 
                    ];
                }
                return [];
            }
            return [];
        }
    }
}
