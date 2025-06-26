// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0using System;

namespace ILGPU.Numerics
{
    /// <summary>
    /// CPU-based tensor implementation.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    internal sealed class CpuTensor<T> : ITensor<T> where T : unmanaged
    {
        private readonly T[] _data;
        private readonly int[] _strides;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the CpuTensor class.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        public CpuTensor(TensorShape shape)
        {
            Shape = shape;
            Location = ComputeLocation.Cpu;
            Length = shape.Length;
            Rank = shape.Rank;

            _data = new T[Length];
            _strides = CalculateStrides(shape);
        }

        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// Gets the compute location.
        /// </summary>
        public ComputeLocation Location { get; }

        /// <summary>
        /// Gets the number of elements.
        /// </summary>
        public long Length { get; }

        /// <summary>
        /// Gets the rank.
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Gets or sets an element by indices.
        /// </summary>
        /// <param name="indices">The indices.</param>
        /// <returns>The element at the specified indices.</returns>
        public T this[params int[] indices]
        {
            get
            {
                ThrowIfDisposed();
                var index = CalculateLinearIndex(indices);
                return _data[index];
            }
            set
            {
                ThrowIfDisposed();
                var index = CalculateLinearIndex(indices);
                _data[index] = value;
            }
        }

        /// <summary>
        /// Gets a pointer to the tensor data.
        /// </summary>
        /// <returns>A pointer to the data.</returns>
        public unsafe nint GetDataPointer()
        {
            ThrowIfDisposed();
            fixed (T* ptr = _data)
            {
                return (nint)ptr;
            }
        }

        /// <summary>
        /// Copies data from another tensor.
        /// </summary>
        /// <param name="source">The source tensor.</param>
        public void CopyFrom(ITensor<T> source)
        {
            ThrowIfDisposed();
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (source.Length != Length)
                throw new ArgumentException("Tensor sizes must match for copying.");

            if (source is CpuTensor<T> cpuSource)
            {
                Array.Copy(cpuSource._data, _data, (int)Length);
            }
            else
            {
                unsafe
                {
                    var sourcePtr = source.GetDataPointer();
                    fixed (T* destPtr = _data)
                    {
                        Buffer.MemoryCopy((void*)sourcePtr, destPtr, 
                            Length * sizeof(T), Length * sizeof(T));
                    }
                }
            }
        }

        /// <summary>
        /// Copies data to another tensor.
        /// </summary>
        /// <param name="destination">The destination tensor.</param>
        public void CopyTo(ITensor<T> destination) => destination?.CopyFrom(this);

        /// <summary>
        /// Creates a reshaped view of the tensor.
        /// </summary>
        /// <param name="newShape">The new shape.</param>
        /// <returns>A reshaped tensor view.</returns>
        public ITensor<T> Reshape(TensorShape newShape)
        {
            ThrowIfDisposed();
            if (newShape.Length != Length)
                throw new ArgumentException("New shape must have the same number of elements.");

            var result = new CpuTensor<T>(newShape);
            result.CopyFrom(this);
            return result;
        }

        /// <summary>
        /// Creates a slice of the tensor.
        /// </summary>
        /// <param name="start">The starting indices.</param>
        /// <param name="length">The length in each dimension.</param>
        /// <returns>A tensor slice.</returns>
        public ITensor<T> Slice(int[] start, int[] length)
        {
            ThrowIfDisposed();
            if (start.Length != Rank || length.Length != Rank)
                throw new ArgumentException("Start and length arrays must match tensor rank.");

            var sliceShape = new TensorShape(length);
            var result = new CpuTensor<T>(sliceShape);

            // Simple implementation - copy elements
            // This could be optimized with views/strides
            CopySliceData(start, length, result);

            return result;
        }

        /// <summary>
        /// Disposes the tensor.
        /// </summary>
        public void Dispose() => _disposed = true;

        private static int[] CalculateStrides(TensorShape shape)
        {
            var strides = new int[shape.Rank];
            if (shape.Rank == 0) return strides;

            strides[shape.Rank - 1] = 1;
            for (int i = shape.Rank - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            return strides;
        }

        private int CalculateLinearIndex(int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}.");

            int index = 0;
            for (int i = 0; i < Rank; i++)
            {
                if (indices[i] < 0 || indices[i] >= Shape[i])
                    throw new IndexOutOfRangeException($"Index {indices[i]} is out of range for dimension {i} (size {Shape[i]}).");

                index += indices[i] * _strides[i];
            }

            return index;
        }

        private void CopySliceData(int[] start, int[] length, CpuTensor<T> destination) =>
            // Simplified slice copy - recursive implementation
            CopySliceRecursive(start, length, new int[Rank], 0, destination, new int[Rank]);

        private void CopySliceRecursive(int[] start, int[] length, int[] currentSourceIndex, 
            int dimension, CpuTensor<T> destination, int[] currentDestIndex)
        {
            if (dimension == Rank)
            {
                // Copy single element
                var sourceIndex = CalculateLinearIndex(currentSourceIndex);
                var destIndex = destination.CalculateLinearIndex(currentDestIndex);
                destination._data[destIndex] = _data[sourceIndex];
                return;
            }

            for (int i = 0; i < length[dimension]; i++)
            {
                currentSourceIndex[dimension] = start[dimension] + i;
                currentDestIndex[dimension] = i;
                CopySliceRecursive(start, length, currentSourceIndex, dimension + 1, destination, currentDestIndex);
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CpuTensor<T>));
        }
    }
}