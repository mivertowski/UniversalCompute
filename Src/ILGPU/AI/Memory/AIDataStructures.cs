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
using System.Runtime.InteropServices;

namespace ILGPU.AI.Memory
{
    /// <summary>
    /// Tensor data layout formats for AI operations.
    /// </summary>
    public enum TensorFormat
    {
        /// <summary>
        /// NCHW format (batch, channels, height, width).
        /// </summary>
        NCHW,

        /// <summary>
        /// NHWC format (batch, height, width, channels).
        /// </summary>
        NHWC,

        /// <summary>
        /// NCL format (batch, channels, length) for 1D data.
        /// </summary>
        NCL,

        /// <summary>
        /// NLC format (batch, length, channels) for 1D data.
        /// </summary>
        NLC,

        /// <summary>
        /// CHW format (channels, height, width) for single batch.
        /// </summary>
        CHW,

        /// <summary>
        /// HWC format (height, width, channels) for single batch.
        /// </summary>
        HWC
    }

    /// <summary>
    /// Sparse tensor storage formats.
    /// </summary>
    public enum SparseFormat
    {
        /// <summary>
        /// Coordinate (COO) format.
        /// </summary>
        COO,

        /// <summary>
        /// Compressed Sparse Row (CSR) format.
        /// </summary>
        CSR,

        /// <summary>
        /// Compressed Sparse Column (CSC) format.
        /// </summary>
        CSC,

        /// <summary>
        /// Block Sparse Row (BSR) format.
        /// </summary>
        BSR,

        /// <summary>
        /// Ellpack (ELL) format.
        /// </summary>
        ELL
    }

    /// <summary>
    /// AI-optimized tensor descriptor.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AITensorDescriptor
    {
        /// <summary>
        /// Tensor format.
        /// </summary>
        public TensorFormat Format;

        /// <summary>
        /// Number of dimensions.
        /// </summary>
        public int Rank;

        /// <summary>
        /// Dimension sizes.
        /// </summary>
        public unsafe fixed int Dimensions[8]; // Support up to 8D tensors

        /// <summary>
        /// Stride information for each dimension.
        /// </summary>
        public unsafe fixed int Strides[8];

        /// <summary>
        /// Element size in bytes.
        /// </summary>
        public int ElementSize;

        /// <summary>
        /// Total number of elements.
        /// </summary>
        public long TotalElements;

        /// <summary>
        /// Memory alignment requirement.
        /// </summary>
        public int Alignment;

        /// <summary>
        /// Creates a tensor descriptor for the given shape and format.
        /// </summary>
        /// <param name="dimensions">Tensor dimensions.</param>
        /// <param name="format">Tensor format.</param>
        /// <param name="elementSize">Size of each element in bytes.</param>
        /// <returns>Tensor descriptor.</returns>
        public static unsafe AITensorDescriptor Create(ReadOnlySpan<int> dimensions, TensorFormat format, int elementSize)
        {
            var descriptor = new AITensorDescriptor
            {
                Format = format,
                Rank = dimensions.Length,
                ElementSize = elementSize,
                TotalElements = 1,
                Alignment = 32 // Default 32-byte alignment
            };

            // Copy dimensions and calculate strides
            for (int i = 0; i < dimensions.Length && i < 8; i++)
            {
                descriptor.Dimensions[i] = dimensions[i];
                descriptor.TotalElements *= dimensions[i];
            }

            CalculateStrides(ref descriptor, format);
            return descriptor;
        }

        /// <summary>
        /// Calculates strides for the given format.
        /// </summary>
        private static unsafe void CalculateStrides(ref AITensorDescriptor descriptor, TensorFormat format)
        {
            switch (format)
            {
                case TensorFormat.NCHW:
                    if (descriptor.Rank == 4)
                    {
                        descriptor.Strides[3] = 1; // W
                        descriptor.Strides[2] = descriptor.Dimensions[3]; // H
                        descriptor.Strides[1] = descriptor.Dimensions[2] * descriptor.Strides[2]; // C
                        descriptor.Strides[0] = descriptor.Dimensions[1] * descriptor.Strides[1]; // N
                    }
                    break;

                case TensorFormat.NHWC:
                    if (descriptor.Rank == 4)
                    {
                        descriptor.Strides[1] = 1; // C
                        descriptor.Strides[3] = descriptor.Dimensions[1]; // W
                        descriptor.Strides[2] = descriptor.Dimensions[3] * descriptor.Strides[3]; // H
                        descriptor.Strides[0] = descriptor.Dimensions[2] * descriptor.Strides[2]; // N
                    }
                    break;

                default:
                    // Row-major order (default)
                    descriptor.Strides[descriptor.Rank - 1] = 1;
                    for (int i = descriptor.Rank - 2; i >= 0; i--)
                    {
                        descriptor.Strides[i] = descriptor.Dimensions[i + 1] * descriptor.Strides[i + 1];
                    }
                    break;
            }
        }
    }

    /// <summary>
    /// Sparse tensor in COO (Coordinate) format.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    public struct SparseTensorCOO<T> where T : unmanaged
    {
        /// <summary>
        /// Non-zero values.
        /// </summary>
        public ArrayView<T> Values;

        /// <summary>
        /// Row indices of non-zero values.
        /// </summary>
        public ArrayView<int> RowIndices;

        /// <summary>
        /// Column indices of non-zero values.
        /// </summary>
        public ArrayView<int> ColIndices;

        /// <summary>
        /// Number of rows.
        /// </summary>
        public int Rows;

        /// <summary>
        /// Number of columns.
        /// </summary>
        public int Cols;

        /// <summary>
        /// Number of non-zero elements.
        /// </summary>
        public int NonZeroCount;

        /// <summary>
        /// Sparsity ratio (0.0 = dense, 1.0 = empty).
        /// </summary>
        public readonly float Sparsity => 1.0f - (float)NonZeroCount / (Rows * Cols);
    }

    /// <summary>
    /// Sparse tensor in CSR (Compressed Sparse Row) format.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    public struct SparseTensorCSR<T> where T : unmanaged
    {
        /// <summary>
        /// Non-zero values.
        /// </summary>
        public ArrayView<T> Values;

        /// <summary>
        /// Column indices of non-zero values.
        /// </summary>
        public ArrayView<int> ColIndices;

        /// <summary>
        /// Row pointer array.
        /// </summary>
        public ArrayView<int> RowPointers;

        /// <summary>
        /// Number of rows.
        /// </summary>
        public int Rows;

        /// <summary>
        /// Number of columns.
        /// </summary>
        public int Cols;

        /// <summary>
        /// Number of non-zero elements.
        /// </summary>
        public int NonZeroCount;
    }

    /// <summary>
    /// Memory pool for AI operations with optimized allocation patterns.
    /// </summary>
    public sealed class AIMemoryPool : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _blockSize;
        private readonly int _alignment;
        private readonly MemoryBuffer[] _blocks;
        private readonly bool[] _blockUsed;
        private readonly int _blockCount;
        private bool _disposed;

        /// <summary>
        /// Initializes a new AI memory pool.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="totalSize">Total pool size in bytes.</param>
        /// <param name="blockSize">Size of each block in bytes.</param>
        /// <param name="alignment">Memory alignment requirement.</param>
        public AIMemoryPool(Accelerator accelerator, long totalSize, int blockSize, int alignment = 32)
        {
            _accelerator = accelerator;
            _blockSize = blockSize;
            _alignment = alignment;
            _blockCount = (int)(totalSize / blockSize);
            _blocks = new MemoryBuffer[_blockCount];
            _blockUsed = new bool[_blockCount];

            // Pre-allocate all blocks
            for (int i = 0; i < _blockCount; i++)
            {
                _blocks[i] = _accelerator.Allocate1D<byte>(_blockSize);
            }
        }

        /// <summary>
        /// Allocates a memory block from the pool.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="elementCount">Number of elements.</param>
        /// <returns>Memory buffer view.</returns>
        public ArrayView<T> Allocate<T>(long elementCount) where T : unmanaged
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AIMemoryPool));

            var requiredBytes = elementCount * Unsafe.SizeOf<T>();
            var requiredBlocks = (int)((requiredBytes + _blockSize - 1) / _blockSize);

            // Find consecutive free blocks
            for (int i = 0; i <= _blockCount - requiredBlocks; i++)
            {
                bool canAllocate = true;
                for (int j = 0; j < requiredBlocks; j++)
                {
                    if (_blockUsed[i + j])
                    {
                        canAllocate = false;
                        break;
                    }
                }

                if (canAllocate)
                {
                    // Mark blocks as used
                    for (int j = 0; j < requiredBlocks; j++)
                    {
                        _blockUsed[i + j] = true;
                    }

                    // Return view of the first block (simplified - would need proper spanning)
                    return _blocks[i].AsRawArrayView().Cast<T>();
                }
            }

            throw new OutOfMemoryException("No available memory blocks in AI memory pool");
        }

        /// <summary>
        /// Releases a memory allocation back to the pool.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="memory">Memory to release.</param>
        public void Release<T>(ArrayView<T> memory) where T : unmanaged
        {
            // Find which blocks correspond to this memory and mark as free
            // This is a simplified implementation
            for (int i = 0; i < _blockCount; i++)
            {
                if (_blocks[i].NativePtr == memory.LoadEffectiveAddressAsPtr())
                {
                    _blockUsed[i] = false;
                    break;
                }
            }
        }

        /// <summary>
        /// Disposes the memory pool.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                for (int i = 0; i < _blockCount; i++)
                {
                    _blocks[i]?.Dispose();
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// AI-optimized data structures and memory management.
    /// </summary>
    public static class AIDataStructures
    {
        /// <summary>
        /// Creates an optimal tensor layout for the given operation type.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="shape">Tensor shape.</param>
        /// <param name="operationType">Type of operation.</param>
        /// <param name="elementSize">Element size in bytes.</param>
        /// <returns>Optimized tensor descriptor.</returns>
        public static AITensorDescriptor CreateOptimalLayout(
            Accelerator accelerator,
            ReadOnlySpan<int> shape,
            string operationType,
            int elementSize)
        {
            var format = operationType.ToUpperInvariant() switch
            {
                "convolution" => TensorFormat.NCHW, // Better for convolution
                "attention" => TensorFormat.NLC,    // Better for transformers
                "matmul" => TensorFormat.NCHW,      // Row-major for matrix ops
                _ => TensorFormat.NCHW              // Default
            };

            return AITensorDescriptor.Create(shape, format, elementSize);
        }

        /// <summary>
        /// Converts dense tensor to sparse COO format.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="dense">Dense tensor data.</param>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="threshold">Sparsity threshold (values below this are considered zero).</param>
        /// <returns>Sparse tensor in COO format.</returns>
        public static SparseTensorCOO<T> ConvertToSparseCOO<T>(
            Accelerator accelerator,
            ArrayView<T> dense,
            int rows, int cols,
            T threshold) where T : unmanaged, IComparable<T>
        {
            // Count non-zero elements first
            var nonZeroCount = accelerator.Allocate1D<int>(1);
            var countKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<T>, ArrayView<int>, T, int>(CountNonZeroKernel);
            countKernel(new Index1D(dense.IntLength), dense, nonZeroCount.View, threshold, dense.IntLength);

            accelerator.Synchronize();
            var count = nonZeroCount.GetAsArray1D()[0];

            // Allocate sparse arrays
            var values = accelerator.Allocate1D<T>(count);
            var rowIndices = accelerator.Allocate1D<int>(count);
            var colIndices = accelerator.Allocate1D<int>(count);

            // Extract sparse data
            var extractKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<T>, ArrayView<T>, ArrayView<int>, ArrayView<int>,
                T, int, int>(ExtractSparseKernel);
            extractKernel(new Index1D(dense.IntLength), dense, values.View, rowIndices.View, colIndices.View, threshold, rows, cols);

            nonZeroCount.Dispose();

            return new SparseTensorCOO<T>
            {
                Values = values.View,
                RowIndices = rowIndices.View,
                ColIndices = colIndices.View,
                Rows = rows,
                Cols = cols,
                NonZeroCount = count
            };
        }

        /// <summary>
        /// Converts sparse COO format to dense tensor.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="sparse">Sparse tensor in COO format.</param>
        /// <param name="dense">Output dense tensor.</param>
        public static void ConvertSparseCOOToDense<T>(
            Accelerator accelerator,
            SparseTensorCOO<T> sparse,
            ArrayView<T> dense) where T : unmanaged
        {
            // Clear dense array first
            var clearKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, T>(ClearArrayKernel);
            clearKernel(new Index1D(dense.IntLength), dense, default);

            // Scatter sparse values to dense array
            var scatterKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<T>, ArrayView<int>, ArrayView<int>, ArrayView<T>, int>(
                ScatterSparseKernel);
            scatterKernel(new Index1D(sparse.Values.IntLength), sparse.Values, sparse.RowIndices, sparse.ColIndices, dense, sparse.Cols);
        }

        /// <summary>
        /// Creates an attention-optimized memory layout for transformer operations.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="seqLength">Sequence length.</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <param name="headDim">Head dimension.</param>
        /// <returns>Memory layout configuration.</returns>
        public static AIMemoryConfig CreateAttentionLayout(
            Accelerator accelerator,
            int batchSize, int seqLength, int numHeads, int headDim) => new()
            {
                Layout = AIMemoryLayout.RowMajor, // Better for matrix operations
                AccessPattern = AIAccessPattern.Strided, // Attention has strided access
                TileSize = (Math.Min(seqLength, 64), Math.Min(headDim, 64)),
                BlockSize = 64,
                EnablePrefetching = true,
                AlignMemory = true,
                MemoryAlignment = 64 // Optimize for cache lines
            };

        #region Kernel Implementations

        /// <summary>
        /// Kernel to count non-zero elements.
        /// </summary>
        private static void CountNonZeroKernel<T>(
            Index1D index,
            ArrayView<T> data,
            ArrayView<int> count,
            T threshold,
            int length) where T : unmanaged, IComparable<T>
        {
            if (index >= length) return;

            if (data[index].CompareTo(threshold) != 0)
            {
                Atomic.Add(ref count[0], 1);
            }
        }

        /// <summary>
        /// Kernel to extract sparse data in COO format.
        /// </summary>
        private static void ExtractSparseKernel<T>(
            Index1D index,
            ArrayView<T> dense,
            ArrayView<T> values,
            ArrayView<int> rowIndices,
            ArrayView<int> colIndices,
            T threshold,
            int rows, int cols) where T : unmanaged, IComparable<T>
        {
            if (index >= dense.Length) return;

            var value = dense[index];
            if (value.CompareTo(threshold) != 0)
            {
                var row = index / cols;
                var col = index % cols;
                
                // This is simplified - would need atomic counter for proper indexing
                var sparseIndex = index; // Placeholder
                if (sparseIndex < values.Length)
                {
                    values[sparseIndex] = value;
                    rowIndices[sparseIndex] = row;
                    colIndices[sparseIndex] = col;
                }
            }
        }

        /// <summary>
        /// Kernel to scatter sparse values to dense array.
        /// </summary>
        private static void ScatterSparseKernel<T>(
            Index1D index,
            ArrayView<T> values,
            ArrayView<int> rowIndices,
            ArrayView<int> colIndices,
            ArrayView<T> dense,
            int cols) where T : unmanaged
        {
            if (index >= values.Length) return;

            var row = rowIndices[index];
            var col = colIndices[index];
            var denseIndex = row * cols + col;

            if (denseIndex < dense.Length)
            {
                dense[denseIndex] = values[index];
            }
        }

        /// <summary>
        /// Kernel to clear an array with a specific value.
        /// </summary>
        private static void ClearArrayKernel<T>(Index1D index, ArrayView<T> array, T value) where T : unmanaged
        {
            if (index < array.Length)
                array[index] = value;
        }

        #endregion
    }
}
