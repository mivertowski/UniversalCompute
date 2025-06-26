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
using ILGPU.Runtime;

namespace ILGPU.Core
{
    /// <summary>
    /// Core tensor interface that provides the foundation for all tensor types in ILGPU.
    /// This interface serves as the universal base for ML, Numerics, and Hybrid tensor systems.
    /// </summary>
    /// <typeparam name="T">The element type, must be unmanaged.</typeparam>
    public interface ITensorCore<T> : IDisposable where T : unmanaged
    {
        /// <summary>
        /// Gets the shape of this tensor.
        /// </summary>
        TensorShape Shape { get; }

        /// <summary>
        /// Gets the total number of elements in this tensor.
        /// </summary>
        long ElementCount { get; }

        /// <summary>
        /// Gets the accelerator this tensor is associated with.
        /// </summary>
        Accelerator Accelerator { get; }

        /// <summary>
        /// Gets whether this tensor's data is currently on the CPU.
        /// </summary>
        bool IsOnCPU { get; }

        /// <summary>
        /// Gets whether this tensor's data is currently on a GPU/accelerator.
        /// </summary>
        bool IsOnAccelerator { get; }

        /// <summary>
        /// Gets a read-only span view of this tensor's data.
        /// Note: This may trigger a CPU copy if data is currently on an accelerator.
        /// </summary>
        /// <returns>A read-only span containing the tensor data.</returns>
        ReadOnlySpan<T> AsReadOnlySpan();

        /// <summary>
        /// Gets a mutable span view of this tensor's data.
        /// Note: This may trigger a CPU copy if data is currently on an accelerator.
        /// </summary>
        /// <returns>A mutable span containing the tensor data.</returns>
        Span<T> AsSpan();

        /// <summary>
        /// Copies this tensor's data to another tensor.
        /// </summary>
        /// <param name="destination">The destination tensor.</param>
        void CopyTo(ITensorCore<T> destination);

        /// <summary>
        /// Copies data from a CPU span to this tensor.
        /// </summary>
        /// <param name="source">The source span containing data to copy.</param>
        void CopyFrom(ReadOnlySpan<T> source);

        /// <summary>
        /// Copies data from this tensor to a CPU span.
        /// </summary>
        /// <param name="destination">The destination span.</param>
        void CopyTo(Span<T> destination);

        /// <summary>
        /// Creates a view of this tensor with a different shape.
        /// The total number of elements must remain the same.
        /// </summary>
        /// <param name="newShape">The new shape for the view.</param>
        /// <returns>A new tensor view with the specified shape.</returns>
        ITensorCore<T> Reshape(TensorShape newShape);

        /// <summary>
        /// Creates a slice of this tensor along the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension to slice along.</param>
        /// <param name="start">The start index (inclusive).</param>
        /// <param name="length">The length of the slice.</param>
        /// <returns>A new tensor view representing the slice.</returns>
        ITensorCore<T> Slice(int dimension, int start, int length);

        /// <summary>
        /// Fills this tensor with the specified value.
        /// </summary>
        /// <param name="value">The value to fill with.</param>
        void Fill(T value);

        /// <summary>
        /// Creates a copy of this tensor.
        /// </summary>
        /// <returns>A new tensor with the same data and shape.</returns>
        ITensorCore<T> Clone();
    }
}