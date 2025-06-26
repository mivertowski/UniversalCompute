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

using System;

namespace ILGPU.Core
{
    /// <summary>
    /// Unified tensor shape representation that serves as the foundation for all tensor systems.
    /// This replaces the multiple incompatible TensorShape implementations.
    /// </summary>
    public readonly struct TensorShape : IEquatable<TensorShape>
    {
        private readonly int[] dimensions;

        /// <summary>
        /// Initializes a new tensor shape with the specified dimensions.
        /// </summary>
        /// <param name="dimensions">The dimensions of the tensor.</param>
        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
                throw new ArgumentException("Tensor must have at least one dimension", nameof(dimensions));

            this.dimensions = new int[dimensions.Length];
            Array.Copy(dimensions, this.dimensions, dimensions.Length);

            // Calculate total size and validate dimensions
            ElementCount = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] <= 0)
                    throw new ArgumentException($"Dimension {i} must be positive, got {dimensions[i]}", nameof(dimensions));
                
                ElementCount *= dimensions[i];
            }
        }

        /// <summary>
        /// Gets the number of dimensions (rank) of this tensor.
        /// </summary>
        public int Rank => dimensions?.Length ?? 0;

        /// <summary>
        /// Gets the total number of elements in this tensor.
        /// </summary>
        public long ElementCount { get; }

        /// <summary>
        /// Gets the dimension at the specified index.
        /// </summary>
        /// <param name="index">The dimension index.</param>
        /// <returns>The size of the dimension.</returns>
        public int this[int index] => dimensions[index];

        /// <summary>
        /// Gets all dimensions as a read-only span.
        /// </summary>
        public ReadOnlySpan<int> Dimensions => dimensions;

        /// <summary>
        /// Checks if this shape is compatible for matrix multiplication with another shape.
        /// </summary>
        /// <param name="other">The other tensor shape.</param>
        /// <returns>True if compatible for matrix multiplication.</returns>
        public bool IsMatMulCompatible(TensorShape other)
        {
            if (Rank < 2 || other.Rank < 2)
                return false;
                
            // For matrix multiplication: (..., A, B) * (..., B, C) -> (..., A, C)
            return this[Rank - 1] == other[other.Rank - 2];
        }

        /// <summary>
        /// Gets the resulting shape from matrix multiplication with another shape.
        /// </summary>
        /// <param name="other">The other tensor shape.</param>
        /// <returns>The result shape.</returns>
        public TensorShape MatMulResultShape(TensorShape other)
        {
            if (!IsMatMulCompatible(other))
                throw new ArgumentException("Shapes are not compatible for matrix multiplication", nameof(other));

            // Handle batched matrix multiplication
            var thisBatch = Rank > 2 ? dimensions[..^2] : Array.Empty<int>();
            var otherBatch = other.Rank > 2 ? other.dimensions[..^2] : Array.Empty<int>();
            
            // Batch dimensions must be broadcastable
            var resultBatch = BroadcastDimensions(thisBatch, otherBatch);
            
            // Result dimensions: [...batch, this[-2], other[-1]]
            var resultDims = new int[resultBatch.Length + 2];
            Array.Copy(resultBatch, resultDims, resultBatch.Length);
            resultDims[^2] = this[Rank - 2];
            resultDims[^1] = other[other.Rank - 1];
            
            return new TensorShape(resultDims);
        }

        /// <summary>
        /// Checks if this shape is compatible for element-wise operations with another shape.
        /// </summary>
        /// <param name="other">The other tensor shape.</param>
        /// <returns>True if compatible for element-wise operations.</returns>
        public bool IsElementWiseCompatible(TensorShape other)
        {
            // Same shape is always compatible
            if (Equals(other))
                return true;
                
            // Check if shapes are broadcastable
            return CanBroadcastTo(other) || other.CanBroadcastTo(this);
        }

        /// <summary>
        /// Checks if this shape can be broadcasted to another shape.
        /// </summary>
        /// <param name="target">The target shape.</param>
        /// <returns>True if this shape can be broadcasted to the target.</returns>
        public bool CanBroadcastTo(TensorShape target)
        {
            if (Rank > target.Rank)
                return false;

            // Check from the rightmost dimensions
            for (int i = 0; i < Rank; i++)
            {
                int thisIdx = Rank - 1 - i;
                int targetIdx = target.Rank - 1 - i;
                
                int thisDim = this[thisIdx];
                int targetDim = target[targetIdx];
                
                // Broadcasting rules: dimension must be 1 or equal
                if (thisDim != 1 && thisDim != targetDim)
                    return false;
            }
            
            return true;
        }

        /// <summary>
        /// Computes the linear index from multi-dimensional indices.
        /// </summary>
        /// <param name="indices">The multi-dimensional indices.</param>
        /// <returns>The linear index.</returns>
        public long ComputeLinearIndex(ReadOnlySpan<int> indices)
        {
            if (indices.Length != Rank)
                throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}", nameof(indices));

            long linearIndex = 0;
            long stride = 1;
            
            // Compute in row-major order (C-style)
            for (int i = Rank - 1; i >= 0; i--)
            {
                if (indices[i] < 0 || indices[i] >= dimensions[i])
                    throw new ArgumentOutOfRangeException(nameof(indices), 
                        $"Index {indices[i]} is out of range for dimension {i} (size {dimensions[i]})");
                        
                linearIndex += indices[i] * stride;
                stride *= dimensions[i];
            }
            
            return linearIndex;
        }

        /// <summary>
        /// Computes multi-dimensional indices from a linear index.
        /// </summary>
        /// <param name="linearIndex">The linear index.</param>
        /// <returns>The multi-dimensional indices.</returns>
        public int[] ComputeMultiDimensionalIndex(long linearIndex)
        {
            if (linearIndex < 0 || linearIndex >= ElementCount)
                throw new ArgumentOutOfRangeException(nameof(linearIndex), 
                    $"Linear index {linearIndex} is out of range (size {ElementCount})");

            var indices = new int[Rank];
            var remaining = linearIndex;
            
            for (int i = Rank - 1; i >= 0; i--)
            {
                indices[i] = (int)(remaining % dimensions[i]);
                remaining /= dimensions[i];
            }
            
            return indices;
        }

        /// <summary>
        /// Creates a new shape with the specified dimension changed.
        /// </summary>
        /// <param name="dimension">The dimension to change.</param>
        /// <param name="newSize">The new size for that dimension.</param>
        /// <returns>A new tensor shape with the modified dimension.</returns>
        public TensorShape WithDimension(int dimension, int newSize)
        {
            if (dimension < 0 || dimension >= Rank)
                throw new ArgumentOutOfRangeException(nameof(dimension));
                
            var newDimensions = new int[Rank];
            Array.Copy(dimensions, newDimensions, Rank);
            newDimensions[dimension] = newSize;
            
            return new TensorShape(newDimensions);
        }

        /// <summary>
        /// Creates a new shape with additional dimensions prepended.
        /// </summary>
        /// <param name="newDimensions">The dimensions to prepend.</param>
        /// <returns>A new tensor shape with the additional dimensions.</returns>
        public TensorShape Prepend(params int[] newDimensions)
        {
            var allDimensions = new int[newDimensions.Length + Rank];
            Array.Copy(newDimensions, allDimensions, newDimensions.Length);
            Array.Copy(dimensions, 0, allDimensions, newDimensions.Length, Rank);
            
            return new TensorShape(allDimensions);
        }

        /// <summary>
        /// Creates a new shape with additional dimensions appended.
        /// </summary>
        /// <param name="newDimensions">The dimensions to append.</param>
        /// <returns>A new tensor shape with the additional dimensions.</returns>
        public TensorShape Append(params int[] newDimensions)
        {
            var allDimensions = new int[Rank + newDimensions.Length];
            Array.Copy(dimensions, allDimensions, Rank);
            Array.Copy(newDimensions, 0, allDimensions, Rank, newDimensions.Length);
            
            return new TensorShape(allDimensions);
        }

        #region Equality and Hashing

        /// <summary>
        /// Determines whether this tensor shape is equal to another.
        /// </summary>
        /// <param name="other">The other tensor shape.</param>
        /// <returns>True if the shapes are equal.</returns>
        public bool Equals(TensorShape other)
        {
            if (Rank != other.Rank)
                return false;
                
            for (int i = 0; i < Rank; i++)
            {
                if (this[i] != other[i])
                    return false;
            }
            
            return true;
        }

        /// <summary>
        /// Determines whether this tensor shape is equal to another object.
        /// </summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True if the objects are equal.</returns>
        public override bool Equals(object? obj) => obj is TensorShape other && Equals(other);

        /// <summary>
        /// Gets the hash code for this tensor shape.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            for (int i = 0; i < Rank; i++)
                hash.Add(this[i]);
            return hash.ToHashCode();
        }

        /// <summary>
        /// Equality operator.
        /// </summary>
        public static bool operator ==(TensorShape left, TensorShape right) => left.Equals(right);

        /// <summary>
        /// Inequality operator.
        /// </summary>
        public static bool operator !=(TensorShape left, TensorShape right) => !left.Equals(right);

        #endregion

        /// <summary>
        /// Returns a string representation of this tensor shape.
        /// </summary>
        /// <returns>A string in the format [dim1, dim2, ...].</returns>
        public override string ToString()
        {
            if (dimensions == null || dimensions.Length == 0)
                return "[]";
                
            return $"[{string.Join(", ", dimensions)}]";
        }

        #region Helper Methods

        private static int[] BroadcastDimensions(int[] dims1, int[] dims2)
        {
            var maxRank = Math.Max(dims1.Length, dims2.Length);
            var result = new int[maxRank];
            
            for (int i = 0; i < maxRank; i++)
            {
                int idx1 = dims1.Length - 1 - i;
                int idx2 = dims2.Length - 1 - i;
                
                int dim1 = idx1 >= 0 ? dims1[idx1] : 1;
                int dim2 = idx2 >= 0 ? dims2[idx2] : 1;
                
                if (dim1 == 1)
                    result[maxRank - 1 - i] = dim2;
                else if (dim2 == 1)
                    result[maxRank - 1 - i] = dim1;
                else if (dim1 == dim2)
                    result[maxRank - 1 - i] = dim1;
                else
                    throw new ArgumentException($"Cannot broadcast dimensions {dim1} and {dim2}");
            }
            
            return result;
        }

        #endregion
    }
}