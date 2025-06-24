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
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Numerics
{
    /// <summary>
    /// Represents a tensor with generic data type.
    /// </summary>
    /// <typeparam name="T">The element type of the tensor.</typeparam>
    public interface ITensor<T> : IDisposable where T : unmanaged
    {
        /// <summary>
        /// Gets the shape of the tensor.
        /// </summary>
        TensorShape Shape { get; }

        /// <summary>
        /// Gets the data location of the tensor.
        /// </summary>
        ComputeLocation Location { get; }

        /// <summary>
        /// Gets the number of elements in the tensor.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Gets the rank (number of dimensions) of the tensor.
        /// </summary>
        int Rank { get; }

        /// <summary>
        /// Gets a pointer to the tensor data.
        /// </summary>
        /// <returns>A pointer to the tensor data.</returns>
        unsafe nint GetDataPointer();

        /// <summary>
        /// Gets or sets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        T this[params int[] indices] { get; set; }

        /// <summary>
        /// Copies data from another tensor.
        /// </summary>
        /// <param name="source">The source tensor to copy from.</param>
        void CopyFrom(ITensor<T> source);

        /// <summary>
        /// Copies data to another tensor.
        /// </summary>
        /// <param name="destination">The destination tensor to copy to.</param>
        void CopyTo(ITensor<T> destination);

        /// <summary>
        /// Creates a view of this tensor with a different shape.
        /// </summary>
        /// <param name="newShape">The new shape for the view.</param>
        /// <returns>A new tensor view with the specified shape.</returns>
        ITensor<T> Reshape(TensorShape newShape);

        /// <summary>
        /// Creates a slice of this tensor.
        /// </summary>
        /// <param name="start">The starting indices for the slice.</param>
        /// <param name="length">The length of the slice in each dimension.</param>
        /// <returns>A new tensor slice.</returns>
        ITensor<T> Slice(int[] start, int[] length);
    }

    /// <summary>
    /// Represents the shape of a tensor.
    /// </summary>
    public readonly struct TensorShape : IEquatable<TensorShape>
    {
        private readonly int[] _dimensions;

        /// <summary>
        /// Initializes a new instance of the TensorShape struct.
        /// </summary>
        /// <param name="dimensions">The dimensions of the tensor.</param>
        public TensorShape(params int[] dimensions)
        {
            _dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
            Rank = _dimensions.Length;
            Length = CalculateLength(_dimensions);
        }

        /// <summary>
        /// Gets the rank (number of dimensions) of the tensor.
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Gets the total number of elements in the tensor.
        /// </summary>
        public long Length { get; }

        /// <summary>
        /// Gets the dimension at the specified index.
        /// </summary>
        /// <param name="index">The index of the dimension.</param>
        /// <returns>The size of the dimension.</returns>
        public int this[int index] => _dimensions[index];

        /// <summary>
        /// Gets the dimensions array.
        /// </summary>
        public ReadOnlySpan<int> Dimensions => _dimensions.AsSpan();

        private static long CalculateLength(int[] dimensions)
        {
            long length = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                length *= dimensions[i];
            }
            return length;
        }

        /// <summary>
        /// Computes the linear index from multi-dimensional indices.
        /// </summary>
        /// <param name="indices">The multi-dimensional indices.</param>
        /// <returns>The linear index.</returns>
        public long ComputeLinearIndex(params int[] indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Expected {Rank} indices but got {indices.Length}");

            long linearIndex = 0;
            long multiplier = 1;

            for (int i = Rank - 1; i >= 0; i--)
            {
                if (indices[i] < 0 || indices[i] >= _dimensions[i])
                    throw new ArgumentOutOfRangeException($"Index {indices[i]} is out of range for dimension {i} (size: {_dimensions[i]})");

                linearIndex += indices[i] * multiplier;
                multiplier *= _dimensions[i];
            }

            return linearIndex;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current TensorShape.
        /// </summary>
        /// <param name="other">The TensorShape to compare with.</param>
        /// <returns>True if the shapes are equal; otherwise, false.</returns>
        public bool Equals(TensorShape other)
        {
            if (Rank != other.Rank) return false;

            for (int i = 0; i < Rank; i++)
            {
                if (_dimensions[i] != other._dimensions[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current TensorShape.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the objects are equal; otherwise, false.</returns>
        public override bool Equals(object? obj) => obj is TensorShape shape && Equals(shape);

        /// <summary>
        /// Returns the hash code for this TensorShape.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(Rank);
            for (int i = 0; i < Rank; i++)
            {
                hash.Add(_dimensions[i]);
            }
            return hash.ToHashCode();
        }

        /// <summary>
        /// Determines whether two TensorShape instances are equal.
        /// </summary>
        /// <param name="left">The first TensorShape to compare.</param>
        /// <param name="right">The second TensorShape to compare.</param>
        /// <returns>True if the shapes are equal; otherwise, false.</returns>
        public static bool operator ==(TensorShape left, TensorShape right) => left.Equals(right);

        /// <summary>
        /// Determines whether two TensorShape instances are not equal.
        /// </summary>
        /// <param name="left">The first TensorShape to compare.</param>
        /// <param name="right">The second TensorShape to compare.</param>
        /// <returns>True if the shapes are not equal; otherwise, false.</returns>
        public static bool operator !=(TensorShape left, TensorShape right) => !left.Equals(right);

        /// <summary>
        /// Returns a string representation of the TensorShape.
        /// </summary>
        /// <returns>A string representation of the shape.</returns>
        public override string ToString()
        {
            return $"[{string.Join(", ", _dimensions)}]";
        }
    }

    /// <summary>
    /// Represents the compute location for tensor operations.
    /// </summary>
    public enum ComputeLocation
    {
        /// <summary>
        /// CPU computation.
        /// </summary>
        Cpu,

        /// <summary>
        /// CPU with SIMD optimizations.
        /// </summary>
        CpuSimd,

        /// <summary>
        /// GPU computation.
        /// </summary>
        Gpu,

        /// <summary>
        /// Neural processing unit.
        /// </summary>
        Npu,

        /// <summary>
        /// Unified memory accessible by both CPU and GPU.
        /// </summary>
        Unified
    }

    /// <summary>
    /// Factory for creating tensors.
    /// </summary>
    public static class TensorFactory
    {
        /// <summary>
        /// Creates a new tensor with the specified shape and location.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <returns>A new tensor instance.</returns>
        public static ITensor<T> Create<T>(TensorShape shape, ComputeLocation location) where T : unmanaged
        {
            return location switch
            {
                ComputeLocation.Cpu or ComputeLocation.CpuSimd => new CpuTensor<T>(shape),
                ComputeLocation.Unified => throw new NotSupportedException("Unified tensor creation requires accelerator"),
                _ => throw new NotSupportedException($"Compute location {location} not yet implemented")
            };
        }

        /// <summary>
        /// Creates a new tensor with the specified shape, location, and accelerator.
        /// </summary>
        /// <typeparam name="T">The element type of the tensor.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="location">The compute location for the tensor.</param>
        /// <param name="accelerator">The accelerator for GPU/unified tensors.</param>
        /// <returns>A new tensor instance.</returns>
        public static ITensor<T> Create<T>(TensorShape shape, ComputeLocation location, Runtime.Accelerator accelerator) where T : unmanaged, INumber<T>
        {
            return location switch
            {
                ComputeLocation.Cpu or ComputeLocation.CpuSimd => new CpuTensor<T>(shape),
                ComputeLocation.Gpu => new UnifiedTensor<T>(accelerator, shape, MemoryLayoutMode.GpuOptimized),
                ComputeLocation.Unified => new UnifiedTensor<T>(accelerator, shape, MemoryLayoutMode.Unified),
                _ => throw new NotSupportedException($"Compute location {location} not yet implemented")
            };
        }
    }
}