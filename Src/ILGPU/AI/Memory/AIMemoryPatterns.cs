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
using System.Runtime.CompilerServices;

namespace ILGPU.AI.Memory
{
    /// <summary>
    /// Memory layout patterns optimized for AI/ML workloads.
    /// </summary>
    public enum AIMemoryLayout
    {
        /// <summary>
        /// Standard row-major layout (NCHW for tensors).
        /// </summary>
        RowMajor,

        /// <summary>
        /// Column-major layout (NHWC for tensors).
        /// </summary>
        ColumnMajor,

        /// <summary>
        /// Blocked layout for cache optimization.
        /// </summary>
        Blocked,

        /// <summary>
        /// Tiled layout for tensor operations.
        /// </summary>
        Tiled,

        /// <summary>
        /// Interleaved layout for SIMD operations.
        /// </summary>
        Interleaved,

        /// <summary>
        /// Packed layout for quantized data.
        /// </summary>
        Packed,

        /// <summary>
        /// Sparse layout for sparse tensors.
        /// </summary>
        Sparse,

        /// <summary>
        /// Z-order (Morton order) layout for 2D locality.
        /// </summary>
        ZOrder
    }

    /// <summary>
    /// Memory access patterns for different AI/ML operations.
    /// </summary>
    public enum AIAccessPattern
    {
        /// <summary>
        /// Sequential access pattern.
        /// </summary>
        Sequential,

        /// <summary>
        /// Strided access pattern.
        /// </summary>
        Strided,

        /// <summary>
        /// Random access pattern.
        /// </summary>
        Random,

        /// <summary>
        /// Blocked access pattern.
        /// </summary>
        Blocked,

        /// <summary>
        /// Tiled access pattern for convolutions.
        /// </summary>
        Tiled,

        /// <summary>
        /// Streaming access pattern.
        /// </summary>
        Streaming,

        /// <summary>
        /// Gather/scatter pattern for sparse operations.
        /// </summary>
        GatherScatter
    }

    /// <summary>
    /// Configuration for AI memory optimization.
    /// </summary>
    public sealed class AIMemoryConfig
    {
        /// <summary>
        /// Gets or sets the preferred memory layout.
        /// </summary>
        public AIMemoryLayout Layout { get; set; } = AIMemoryLayout.RowMajor;

        /// <summary>
        /// Gets or sets the access pattern hint.
        /// </summary>
        public AIAccessPattern AccessPattern { get; set; } = AIAccessPattern.Sequential;

        /// <summary>
        /// Gets or sets the tile size for tiled layouts.
        /// </summary>
        public (int Height, int Width) TileSize { get; set; } = (16, 16);

        /// <summary>
        /// Gets or sets the block size for blocked layouts.
        /// </summary>
        public int BlockSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets whether to enable memory prefetching.
        /// </summary>
        public bool EnablePrefetching { get; set; } = true;

        /// <summary>
        /// Gets or sets the cache line size hint.
        /// </summary>
        public int CacheLineSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets whether to align memory allocations.
        /// </summary>
        public bool AlignMemory { get; set; } = true;

        /// <summary>
        /// Gets or sets the memory alignment in bytes.
        /// </summary>
        public int MemoryAlignment { get; set; } = 32;

        /// <summary>
        /// Creates a default AI memory configuration.
        /// </summary>
        public static AIMemoryConfig Default => new();

        /// <summary>
        /// Creates a configuration optimized for convolution operations.
        /// </summary>
        /// <returns>Convolution-optimized configuration.</returns>
        public static AIMemoryConfig ForConvolution() => new()
        {
            Layout = AIMemoryLayout.Tiled,
            AccessPattern = AIAccessPattern.Tiled,
            TileSize = (16, 16),
            EnablePrefetching = true,
            AlignMemory = true,
            MemoryAlignment = 64
        };

        /// <summary>
        /// Creates a configuration optimized for matrix operations.
        /// </summary>
        /// <returns>Matrix-optimized configuration.</returns>
        public static AIMemoryConfig ForMatrixOperations() => new()
        {
            Layout = AIMemoryLayout.Blocked,
            AccessPattern = AIAccessPattern.Blocked,
            BlockSize = 64,
            EnablePrefetching = true,
            AlignMemory = true,
            MemoryAlignment = 32
        };

        /// <summary>
        /// Creates a configuration optimized for attention mechanisms.
        /// </summary>
        /// <returns>Attention-optimized configuration.</returns>
        public static AIMemoryConfig ForAttention() => new()
        {
            Layout = AIMemoryLayout.RowMajor,
            AccessPattern = AIAccessPattern.Strided,
            EnablePrefetching = true,
            AlignMemory = true,
            MemoryAlignment = 64
        };
    }

    /// <summary>
    /// AI-optimized memory patterns and layouts for efficient data access.
    /// </summary>
    public static class AIMemoryPatterns
    {
        #region Tensor Layout Transformations

        /// <summary>
        /// Transforms tensor from NCHW to NHWC layout.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input tensor in NCHW format.</param>
        /// <param name="output">Output tensor in NHWC format.</param>
        /// <param name="n">Batch size.</param>
        /// <param name="c">Channels.</param>
        /// <param name="h">Height.</param>
        /// <param name="w">Width.</param>
        public static void TransformNCHWToNHWC(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            int n, int c, int h, int w)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(
                TransformNCHWToNHWCKernel);
            kernel(new Index1D(n * c * h * w), input, output, n, c, h, w);
        }

        /// <summary>
        /// Transforms tensor from NHWC to NCHW layout.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input tensor in NHWC format.</param>
        /// <param name="output">Output tensor in NCHW format.</param>
        /// <param name="n">Batch size.</param>
        /// <param name="c">Channels.</param>
        /// <param name="h">Height.</param>
        /// <param name="w">Width.</param>
        public static void TransformNHWCToNCHW(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            int n, int c, int h, int w)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(
                TransformNHWCToNCHWKernel);
            kernel(new Index1D(n * c * h * w), input, output, n, c, h, w);
        }

        /// <summary>
        /// Creates a tiled memory layout for improved cache locality.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input data in linear layout.</param>
        /// <param name="output">Output data in tiled layout.</param>
        /// <param name="height">Matrix height.</param>
        /// <param name="width">Matrix width.</param>
        /// <param name="tileHeight">Tile height.</param>
        /// <param name="tileWidth">Tile width.</param>
        public static void CreateTiledLayout(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            int height, int width,
            int tileHeight, int tileWidth)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(
                CreateTiledLayoutKernel);
            kernel(new Index1D(height * width), input, output, height, width, tileHeight, tileWidth);
        }

        /// <summary>
        /// Creates a Z-order (Morton order) layout for 2D spatial locality.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input data in row-major layout.</param>
        /// <param name="output">Output data in Z-order layout.</param>
        /// <param name="height">Matrix height (must be power of 2).</param>
        /// <param name="width">Matrix width (must be power of 2).</param>
        public static void CreateZOrderLayout(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            int height, int width)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                CreateZOrderLayoutKernel);
            kernel(new Index1D(height * width), input, output, height, width);
        }

        #endregion

        #region Memory Prefetching

        /// <summary>
        /// Performs software prefetching for AI workloads.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="data">Data to prefetch.</param>
        /// <param name="indices">Indices to prefetch.</param>
        /// <param name="prefetchDistance">Distance ahead to prefetch.</param>
        public static void PrefetchData(
            Accelerator accelerator,
            ArrayView<float> data,
            ArrayView<int> indices,
            int prefetchDistance)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, int>(
                PrefetchDataKernel);
            kernel(new Index1D(indices.IntLength), data, indices, prefetchDistance);
        }

        /// <summary>
        /// Performs streaming prefetch for sequential access patterns.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="data">Data array to stream.</param>
        /// <param name="output">Output processed data.</param>
        /// <param name="streamingDistance">Distance to stream ahead.</param>
        public static void StreamingPrefetch(
            Accelerator accelerator,
            ArrayView<float> data,
            ArrayView<float> output,
            int streamingDistance)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(
                StreamingPrefetchKernel);
            kernel(new Index1D(data.IntLength), data, output, streamingDistance);
        }

        #endregion

        #region Cache-Aware Algorithms

        /// <summary>
        /// Cache-aware matrix transpose optimized for AI workloads.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input matrix.</param>
        /// <param name="output">Transposed output matrix.</param>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="blockSize">Block size for cache optimization.</param>
        public static void CacheAwareTranspose(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            int rows, int cols,
            int blockSize)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(
                CacheAwareTransposeKernel);
            kernel(new Index1D(rows * cols), input, output, rows, cols, blockSize);
        }

        /// <summary>
        /// Cache-aware convolution Im2Col transformation.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="input">Input tensor.</param>
        /// <param name="output">Im2Col output matrix.</param>
        /// <param name="config">Memory configuration.</param>
        /// <param name="inputShape">Input tensor shape (N, C, H, W).</param>
        /// <param name="kernelShape">Kernel shape (H, W).</param>
        /// <param name="stride">Stride (H, W).</param>
        /// <param name="padding">Padding (H, W).</param>
        public static void CacheAwareIm2Col(
            Accelerator accelerator,
            ArrayView<float> input,
            ArrayView<float> output,
            AIMemoryConfig config,
            (int N, int C, int H, int W) inputShape,
            (int H, int W) kernelShape,
            (int H, int W) stride,
            (int H, int W) padding)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int,
                int, int, int, int, int, int, int>(
                CacheAwareIm2ColKernel);

            var totalElements = inputShape.N * inputShape.C * inputShape.H * inputShape.W;
            kernel(new Index1D(totalElements), input, output,
                   inputShape.N, inputShape.C, inputShape.H, inputShape.W,
                   kernelShape.H, kernelShape.W,
                   stride.H, stride.W,
                   padding.H, padding.W,
                   config.BlockSize);
        }

        #endregion

        #region Kernel Implementations

        /// <summary>
        /// Kernel for NCHW to NHWC transformation.
        /// </summary>
        private static void TransformNCHWToNHWCKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int n, int c, int h, int w)
        {
            if (index >= n * c * h * w) return;

            // Decompose linear index to NCHW coordinates
            var tempIndex = index;
            var wIdx = tempIndex % w;
            tempIndex /= w;
            var hIdx = tempIndex % h;
            tempIndex /= h;
            var cIdx = tempIndex % c;
            var nIdx = tempIndex / c;

            // Compute NHWC output index
            var outputIndex = nIdx * h * w * c + hIdx * w * c + wIdx * c + cIdx;
            output[outputIndex] = input[index];
        }

        /// <summary>
        /// Kernel for NHWC to NCHW transformation.
        /// </summary>
        private static void TransformNHWCToNCHWKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int n, int c, int h, int w)
        {
            if (index >= n * h * w * c) return;

            // Decompose linear index to NHWC coordinates
            var tempIndex = index;
            var cIdx = tempIndex % c;
            tempIndex /= c;
            var wIdx = tempIndex % w;
            tempIndex /= w;
            var hIdx = tempIndex % h;
            var nIdx = tempIndex / h;

            // Compute NCHW output index
            var outputIndex = nIdx * c * h * w + cIdx * h * w + hIdx * w + wIdx;
            output[outputIndex] = input[index];
        }

        /// <summary>
        /// Kernel for creating tiled memory layout.
        /// </summary>
        private static void CreateTiledLayoutKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int height, int width,
            int tileHeight, int tileWidth)
        {
            if (index >= height * width) return;

            var row = index / width;
            var col = index % width;

            // Calculate tile coordinates
            var tileRow = row / tileHeight;
            var tileCol = col / tileWidth;
            var inTileRow = row % tileHeight;
            var inTileCol = col % tileWidth;

            // Calculate number of tiles
            var tilesPerRow = (width + tileWidth - 1) / tileWidth;

            // Compute tiled output index
            var tileIndex = tileRow * tilesPerRow + tileCol;
            var outputIndex = tileIndex * tileHeight * tileWidth + inTileRow * tileWidth + inTileCol;

            output[outputIndex] = input[index];
        }

        /// <summary>
        /// Kernel for creating Z-order (Morton order) layout.
        /// </summary>
        private static void CreateZOrderLayoutKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int height, int width)
        {
            if (index >= height * width) return;

            var row = index / width;
            var col = index % width;

            // Compute Morton order index
            var mortonIndex = InterleaveZerosMorton(row) | (InterleaveZerosMorton(col) << 1);
            output[mortonIndex] = input[index];
        }

        /// <summary>
        /// Helper function to interleave zeros for Morton encoding.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int InterleaveZerosMorton(int x)
        {
            x = (x | (x << 8)) & 0x00FF00FF;
            x = (x | (x << 4)) & 0x0F0F0F0F;
            x = (x | (x << 2)) & 0x33333333;
            x = (x | (x << 1)) & 0x55555555;
            return x;
        }

        /// <summary>
        /// Kernel for data prefetching.
        /// </summary>
        private static void PrefetchDataKernel(
            Index1D index,
            ArrayView<float> data,
            ArrayView<int> indices,
            int prefetchDistance)
        {
            if (index >= indices.Length) return;

            var currentIndex = indices[index];
            
            // Prefetch future indices
            if (index + prefetchDistance < indices.Length)
            {
                var futureIndex = indices[index + prefetchDistance];
                if (futureIndex < data.Length)
                {
                    // Access future data to trigger prefetch
                    var _ = data[futureIndex];
                }
            }
        }

        /// <summary>
        /// Kernel for streaming prefetch.
        /// </summary>
        private static void StreamingPrefetchKernel(
            Index1D index,
            ArrayView<float> data,
            ArrayView<float> output,
            int streamingDistance)
        {
            if (index >= data.Length) return;

            // Prefetch future data
            if (index + streamingDistance < data.Length)
            {
                var _ = data[index + streamingDistance];
            }

            // Process current data
            output[index] = data[index] * 2.0f; // Example processing
        }

        /// <summary>
        /// Cache-aware matrix transpose kernel.
        /// </summary>
        private static void CacheAwareTransposeKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int rows, int cols,
            int blockSize)
        {
            if (index >= rows * cols) return;

            var row = index / cols;
            var col = index % cols;

            // Block-wise processing for cache efficiency
            var blockRow = row / blockSize;
            var blockCol = col / blockSize;
            var inBlockRow = row % blockSize;
            var inBlockCol = col % blockSize;

            // Transpose within blocks for better cache locality
            var outputRow = blockCol * blockSize + inBlockCol;
            var outputCol = blockRow * blockSize + inBlockRow;

            if (outputRow < cols && outputCol < rows)
            {
                var outputIndex = outputRow * rows + outputCol;
                output[outputIndex] = input[index];
            }
        }

        /// <summary>
        /// Cache-aware Im2Col kernel.
        /// </summary>
        private static void CacheAwareIm2ColKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int n, int c, int h, int w,
            int kernelH, int kernelW,
            int strideH, int strideW,
            int padH, int padW,
            int blockSize)
        {
            
            var outputH = (h + 2 * padH - kernelH) / strideH + 1;
            var outputW = (w + 2 * padW - kernelW) / strideW + 1;
            var totalOutputElements = n * c * kernelH * kernelW * outputH * outputW;
            
            if (index >= totalOutputElements) return;

            // Decompose index for cache-aware access
            var tempIndex = index;
            var owIdx = tempIndex % outputW;
            tempIndex /= outputW;
            var ohIdx = tempIndex % outputH;
            tempIndex /= outputH;
            var kwIdx = tempIndex % kernelW;
            tempIndex /= kernelW;
            var khIdx = tempIndex % kernelH;
            tempIndex /= kernelH;
            var cIdx = tempIndex % c;
            var nIdx = tempIndex / c;

            // Calculate input coordinates
            var inputH = ohIdx * strideH + khIdx - padH;
            var inputW = owIdx * strideW + kwIdx - padW;

            // Block-wise processing for cache efficiency
            var blockIdx = (nIdx * c + cIdx) / blockSize;
            var inBlockIdx = (nIdx * c + cIdx) % blockSize;

            float value = 0.0f;
            if (inputH >= 0 && inputH < h && inputW >= 0 && inputW < w)
            {
                var inputIndex = nIdx * c * h * w + cIdx * h * w + inputH * w + inputW;
                value = input[inputIndex];
            }

            output[index] = value;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Calculates optimal block size for the given data size and cache size.
        /// </summary>
        /// <param name="dataSize">Size of the data in bytes.</param>
        /// <param name="cacheSize">Cache size in bytes.</param>
        /// <returns>Optimal block size.</returns>
        public static int CalculateOptimalBlockSize(long dataSize, int cacheSize)
        {
            // Simple heuristic: use 1/3 of cache size to allow for multiple blocks
            var optimalSize = cacheSize / 3;
            
            // Ensure block size is a power of 2 for better alignment
            var blockSize = 1;
            while (blockSize < optimalSize && blockSize < 1024)
            {
                blockSize <<= 1;
            }
            
            return Math.Max(16, blockSize >> 1); // Minimum block size of 16
        }

        /// <summary>
        /// Estimates memory bandwidth utilization for the given access pattern.
        /// </summary>
        /// <param name="pattern">Access pattern.</param>
        /// <param name="dataSize">Size of data being accessed.</param>
        /// <param name="cacheLineSize">Cache line size.</param>
        /// <returns>Estimated bandwidth utilization (0.0 to 1.0).</returns>
        public static float EstimateBandwidthUtilization(
            AIAccessPattern pattern,
            long dataSize,
            int cacheLineSize)
        {
            return pattern switch
            {
                AIAccessPattern.Sequential => 0.95f, // Very high utilization
                AIAccessPattern.Strided => 0.7f,     // Good utilization
                AIAccessPattern.Blocked => 0.8f,     // Good utilization
                AIAccessPattern.Tiled => 0.85f,      // Very good utilization
                AIAccessPattern.Streaming => 0.9f,   // Excellent utilization
                AIAccessPattern.Random => 0.3f,      // Poor utilization
                AIAccessPattern.GatherScatter => 0.4f, // Poor utilization
                _ => 0.5f
            };
        }

        #endregion
    }
}