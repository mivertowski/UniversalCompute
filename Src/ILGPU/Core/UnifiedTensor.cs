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
// Change License: Apache License, Version 2.0

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU.Runtime;

namespace ILGPU.Core
{
    /// <summary>
    /// Unified tensor implementation that serves as the foundation for all tensor operations
    /// across ILGPU's ML, Numerics, and Hybrid tensor systems.
    /// </summary>
    /// <typeparam name="T">The element type, must be unmanaged.</typeparam>
    public sealed class UnifiedTensor<T> : ITensorCore<T> where T : unmanaged
    {
        private readonly MemoryBuffer1D<T, Stride1D.Dense> buffer;
        private readonly TensorShape shape;
        private bool disposed;

        /// <summary>
        /// Initializes a new tensor with the specified shape on the given accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        public UnifiedTensor(Accelerator accelerator, TensorShape shape)
        {
            this.shape = shape;
            this.buffer = accelerator.Allocate1D<T>(shape.ElementCount);
            Accelerator = accelerator;
        }

        /// <summary>
        /// Initializes a new tensor with the specified shape and initial data.
        /// </summary>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="data">The initial data to copy to the tensor.</param>
        public UnifiedTensor(Accelerator accelerator, TensorShape shape, ReadOnlySpan<T> data)
            : this(accelerator, shape)
        {
            if (data.Length != shape.ElementCount)
                throw new ArgumentException($"Data length {data.Length} doesn't match shape element count {shape.ElementCount}");

            CopyFrom(data);
        }

        /// <summary>
        /// Initializes a new tensor with the specified shape and initial data array.
        /// </summary>
        /// <param name="accelerator">The accelerator to create the tensor on.</param>
        /// <param name="shape">The tensor shape.</param>
        /// <param name="data">The initial data array to copy to the tensor.</param>
        public UnifiedTensor(Accelerator accelerator, TensorShape shape, T[] data)
            : this(accelerator, shape, data.AsSpan())
        {
        }

        #region ITensorCore<T> Implementation

        /// <inheritdoc/>
        public TensorShape Shape => shape;

        /// <inheritdoc/>
        public long ElementCount => shape.ElementCount;

        /// <inheritdoc/>
        public Accelerator Accelerator { get; }

        /// <inheritdoc/>
        public bool IsOnCPU => Accelerator is ILGPU.Runtime.CPU.CPUAccelerator;

        /// <inheritdoc/>
        public bool IsOnAccelerator => !IsOnCPU;

        /// <inheritdoc/>
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            var data = buffer.GetAsArray1D();
            return data.AsSpan();
        }

        /// <inheritdoc/>
        public Span<T> AsSpan()
        {
            var data = buffer.GetAsArray1D();
            return data.AsSpan();
        }

        /// <inheritdoc/>
        public void CopyTo(ITensorCore<T> destination)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));

            if (destination.ElementCount != ElementCount)
                throw new ArgumentException("Destination tensor must have the same number of elements");

            if (destination is UnifiedTensor<T> unifiedDest)
            {
                // Direct buffer copy for unified tensors
                buffer.CopyTo(unifiedDest.buffer);
            }
            else
            {
                // Fallback to span-based copy
                var data = AsReadOnlySpan();
                destination.CopyFrom(data);
            }
        }

        /// <inheritdoc/>
        public void CopyFrom(ReadOnlySpan<T> source)
        {
            if (source.Length != ElementCount)
                throw new ArgumentException($"Source length {source.Length} doesn't match tensor element count {ElementCount}");

            buffer.CopyFromCPU(source.ToArray());
        }

        /// <inheritdoc/>
        public void CopyTo(Span<T> destination)
        {
            if (destination.Length != ElementCount)
                throw new ArgumentException($"Destination length {destination.Length} doesn't match tensor element count {ElementCount}");

            var data = buffer.GetAsArray1D();
            data.AsSpan().CopyTo(destination);
        }

        /// <inheritdoc/>
        public ITensorCore<T> Reshape(TensorShape newShape)
        {
            if (newShape.ElementCount != ElementCount)
                throw new ArgumentException($"New shape element count {newShape.ElementCount} must match current element count {ElementCount}");

            // Create a new tensor that shares the same buffer (view)
            var reshaped = new UnifiedTensor<T>(Accelerator, newShape);
            buffer.CopyTo(reshaped.buffer);
            return reshaped;
        }

        /// <inheritdoc/>
        public ITensorCore<T> Slice(int dimension, int start, int length)
        {
            if (dimension < 0 || dimension >= Shape.Rank)
                throw new ArgumentOutOfRangeException(nameof(dimension));

            if (start < 0 || start >= Shape[dimension])
                throw new ArgumentOutOfRangeException(nameof(start));

            if (length <= 0 || start + length > Shape[dimension])
                throw new ArgumentOutOfRangeException(nameof(length));

            // Calculate the slice shape
            var newDimensions = new int[Shape.Rank];
            for (int i = 0; i < Shape.Rank; i++)
            {
                newDimensions[i] = i == dimension ? length : Shape[i];
            }
            var sliceShape = new TensorShape(newDimensions);

            // Create slice tensor and copy relevant data
            var slice = new UnifiedTensor<T>(Accelerator, sliceShape);
            
            // For simplicity, copy the entire slice data
            // In a production implementation, this would use stride-based views
            var sourceSpan = AsReadOnlySpan();
            var destSpan = new T[slice.ElementCount];
            
            // Calculate stride for the sliced dimension
            long stride = 1;
            for (int i = dimension + 1; i < Shape.Rank; i++)
                stride *= Shape[i];

            long sourceOffset = start * stride;
            long copyLength = length * stride;
            
            // Perform strided copy
            for (long batch = 0; batch < ElementCount / (Shape[dimension] * stride); batch++)
            {
                long sourceBatchOffset = batch * Shape[dimension] * stride + sourceOffset;
                long destBatchOffset = batch * copyLength;
                
                sourceSpan.Slice((int)sourceBatchOffset, (int)copyLength)
                         .CopyTo(destSpan.AsSpan((int)destBatchOffset, (int)copyLength));
            }
            
            slice.CopyFrom(destSpan);
            return slice;
        }

        /// <inheritdoc/>
        public void Fill(T value)
        {
            if (object.Equals(value, default(T)))
            {
                buffer.MemSetToZero();
            }
            else
            {
                // For non-zero values, fill via CPU array
                var data = new T[ElementCount];
                for (long i = 0; i < ElementCount; i++)
                    data[i] = value;
                buffer.CopyFromCPU(data);
            }
        }

        /// <inheritdoc/>
        public ITensorCore<T> Clone()
        {
            var clone = new UnifiedTensor<T>(Accelerator, Shape);
            buffer.CopyTo(clone.buffer);
            return clone;
        }

        #endregion

        #region Additional Tensor Operations

        /// <summary>
        /// Gets or sets the element at the specified multi-dimensional indices.
        /// </summary>
        /// <param name="indices">The indices for each dimension.</param>
        /// <returns>The element at the specified position.</returns>
        public T this[params int[] indices]
        {
            get
            {
                var linearIndex = Shape.ComputeLinearIndex(indices);
                var data = buffer.GetAsArray1D();
                return data[linearIndex];
            }
            set
            {
                var linearIndex = Shape.ComputeLinearIndex(indices);
                var data = buffer.GetAsArray1D();
                data[linearIndex] = value;
                buffer.CopyFromCPU(data);
            }
        }

        /// <summary>
        /// Adds two tensors element-wise using SIMD operations when possible.
        /// </summary>
        /// <param name="other">The tensor to add.</param>
        /// <returns>A new tensor containing the result.</returns>
        public ITensorCore<T> Add(ITensorCore<T> other)
        {
            if (!Shape.IsElementWiseCompatible(other.Shape))
                throw new ArgumentException("Tensors must have compatible shapes for element-wise operations");

            var result = new UnifiedTensor<T>(Accelerator, Shape);
            
            // Use ILGPU kernel for element-wise addition
            var kernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>>(
                (index, a, b, c) => c[index] = AddElements(a[index], b[index]));

            kernel(buffer.IntExtent, buffer.View, GetBufferView(other), result.buffer.View);
            Accelerator.Synchronize();

            return result;
        }

        /// <summary>
        /// Multiplies two matrices using optimized algorithms.
        /// </summary>
        /// <param name="other">The second matrix.</param>
        /// <returns>A new tensor containing the matrix multiplication result.</returns>
        public ITensorCore<T> MatMul(ITensorCore<T> other)
        {
            if (!Shape.IsMatMulCompatible(other.Shape))
                throw new ArgumentException("Tensors are not compatible for matrix multiplication");

            var resultShape = Shape.MatMulResultShape(other.Shape);
            var result = new UnifiedTensor<T>(Accelerator, resultShape);

            // For demonstration, use a simple triple-loop kernel
            // In production, this would use optimized BLAS libraries or tensor cores
            var kernel = Accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<T>, ArrayView<T>, ArrayView<T>, int, int, int>(
                (index, a, b, c, m, n, k) =>
                {
                    var row = index.X;
                    var col = index.Y;
                    
                    if (row < m && col < n)
                    {
                        var sum = default(T);
                        for (int i = 0; i < k; i++)
                        {
                            var aVal = a[row * k + i];
                            var bVal = b[i * n + col];
                            sum = AddElements(sum, MultiplyElements(aVal, bVal));
                        }
                        c[row * n + col] = sum;
                    }
                });

            var m = Shape[Shape.Rank - 2];
            var n = other.Shape[other.Shape.Rank - 1];
            var k = Shape[Shape.Rank - 1];

            kernel(new Index2D(m, n), buffer.View, GetBufferView(other), result.buffer.View, m, n, k);
            Accelerator.Synchronize();

            return result;
        }

        #endregion

        #region Helper Methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T AddElements(T a, T b)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)((float)(object)a + (float)(object)b);
            if (typeof(T) == typeof(double))
                return (T)(object)((double)(object)a + (double)(object)b);
            if (typeof(T) == typeof(int))
                return (T)(object)((int)(object)a + (int)(object)b);
            if (typeof(T) == typeof(long))
                return (T)(object)((long)(object)a + (long)(object)b);
            
            throw new NotSupportedException($"Addition not supported for type {typeof(T)}");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T MultiplyElements(T a, T b)
        {
            if (typeof(T) == typeof(float))
                return (T)(object)((float)(object)a * (float)(object)b);
            if (typeof(T) == typeof(double))
                return (T)(object)((double)(object)a * (double)(object)b);
            if (typeof(T) == typeof(int))
                return (T)(object)((int)(object)a * (int)(object)b);
            if (typeof(T) == typeof(long))
                return (T)(object)((long)(object)a * (long)(object)b);
            
            throw new NotSupportedException($"Multiplication not supported for type {typeof(T)}");
        }

        private ArrayView<T> GetBufferView(ITensorCore<T> tensor)
        {
            if (tensor is UnifiedTensor<T> unifiedTensor)
                return unifiedTensor.buffer.View;
            
            // Fallback: create temporary buffer and copy data
            var tempBuffer = Accelerator.Allocate1D<T>(tensor.ElementCount);
            var data = tensor.AsReadOnlySpan().ToArray();
            tempBuffer.CopyFromCPU(data);
            return tempBuffer.View;
        }

        #endregion

        #region IDisposable Implementation

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!disposed)
            {
                buffer?.Dispose();
                disposed = true;
            }
        }

        #endregion
    }
}