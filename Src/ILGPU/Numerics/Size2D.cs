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
    /// Represents a 2D size with width and height dimensions.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the Size2D struct.
    /// </remarks>
    /// <param name="width">The width dimension.</param>
    /// <param name="height">The height dimension.</param>
    public readonly struct Size2D(int width, int height) : IEquatable<Size2D>
    {
        /// <summary>
        /// The width dimension.
        /// </summary>
        public readonly int Width = width;

        /// <summary>
        /// The height dimension.
        /// </summary>
        public readonly int Height = height;

        /// <summary>
        /// Gets a Size2D with zero dimensions.
        /// </summary>
        public static Size2D Zero => new(0, 0);

        /// <summary>
        /// Gets a Size2D with unit dimensions.
        /// </summary>
        public static Size2D One => new(1, 1);

        /// <summary>
        /// Determines whether the specified object is equal to the current Size2D.
        /// </summary>
        /// <param name="other">The Size2D to compare with.</param>
        /// <returns>True if the sizes are equal; otherwise, false.</returns>
        public bool Equals(Size2D other) => Width == other.Width && Height == other.Height;

        /// <summary>
        /// Determines whether the specified object is equal to the current Size2D.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the objects are equal; otherwise, false.</returns>
        public override bool Equals(object obj) => obj is Size2D other && Equals(other);

        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode() => HashCode.Combine(Width, Height);

        /// <summary>
        /// Returns a string representation of the Size2D.
        /// </summary>
        /// <returns>A string representation of the size.</returns>
        public override string ToString() => $"({Width}, {Height})";

        /// <summary>
        /// Determines whether two Size2D instances are equal.
        /// </summary>
        /// <param name="left">The left Size2D.</param>
        /// <param name="right">The right Size2D.</param>
        /// <returns>True if the sizes are equal; otherwise, false.</returns>
        public static bool operator ==(Size2D left, Size2D right) => left.Equals(right);

        /// <summary>
        /// Determines whether two Size2D instances are not equal.
        /// </summary>
        /// <param name="left">The left Size2D.</param>
        /// <param name="right">The right Size2D.</param>
        /// <returns>True if the sizes are not equal; otherwise, false.</returns>
        public static bool operator !=(Size2D left, Size2D right) => !left.Equals(right);
    }
}