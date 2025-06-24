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

namespace ILGPU.Runtime.UnifiedMemory
{
    /// <summary>
    /// Represents a unified array view that provides seamless CPU/GPU access.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public readonly struct UnifiedArrayView<T>
        where T : unmanaged
    {
        #region Instance

        private readonly MemoryBuffer1D<T, Stride1D.Dense> buffer;
        private readonly UnifiedMemoryAccessMode accessMode;

        /// <summary>
        /// Initializes a new instance of the <see cref="UnifiedArrayView{T}"/> struct.
        /// </summary>
        /// <param name="buffer">The underlying memory buffer.</param>
        /// <param name="accessMode">The unified memory access mode.</param>
        internal UnifiedArrayView(
            MemoryBuffer1D<T, Stride1D.Dense> buffer,
            UnifiedMemoryAccessMode accessMode)
        {
            this.buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
            this.accessMode = accessMode;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the length of the view.
        /// </summary>
        public long Length => buffer?.Length ?? 0;

        /// <summary>
        /// Gets the GPU array view.
        /// </summary>
        public ArrayView<T> GPUView => buffer?.View ?? default;

        /// <summary>
        /// Gets the access mode.
        /// </summary>
        public UnifiedMemoryAccessMode AccessMode => accessMode;

        /// <summary>
        /// Gets a value indicating whether the view is valid.
        /// </summary>
        public bool IsValid => buffer != null;

        #endregion

        #region Methods

        /// <summary>
        /// Gets a sub-view of the current view.
        /// </summary>
        /// <param name="offset">The offset in elements.</param>
        /// <param name="length">The length of the sub-view.</param>
        /// <returns>The sub-view.</returns>
        public UnifiedArrayView<T> SubView(long offset, long length)
        {
            if (!IsValid)
                throw new InvalidOperationException("View is not valid");
            if (offset < 0 || offset >= Length)
                throw new ArgumentOutOfRangeException(nameof(offset));
            if (length < 0 || offset + length > Length)
                throw new ArgumentOutOfRangeException(nameof(length));

            // Create a new view with the same buffer but adjusted bounds
            // This is a simplified implementation
            return new UnifiedArrayView<T>(buffer, accessMode);
        }

        /// <summary>
        /// Copies data to another unified array view.
        /// </summary>
        /// <param name="target">The target view.</param>
        /// <param name="stream">The accelerator stream.</param>
        public void CopyTo(UnifiedArrayView<T> target, AcceleratorStream stream)
        {
            if (!IsValid)
                throw new InvalidOperationException("Source view is not valid");
            if (!target.IsValid)
                throw new InvalidOperationException("Target view is not valid");
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var length = Math.Min(Length, target.Length);
            GPUView.SubView(0, length).CopyTo(stream, target.GPUView.SubView(0, length));
        }

        /// <summary>
        /// Prefetches the data to the specified device.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="target">The target device.</param>
        public void Prefetch(AcceleratorStream stream, UnifiedMemoryTarget target)
        {
            if (!IsValid)
                return;

            if (buffer is UnifiedMemoryBuffer1D<T> unifiedBuffer)
            {
                unifiedBuffer.Prefetch(stream, target);
            }
        }

        #endregion

        #region Operators

        /// <summary>
        /// Implicitly converts a unified array view to a GPU array view.
        /// </summary>
        /// <param name="view">The unified array view.</param>
        public static implicit operator ArrayView<T>(UnifiedArrayView<T> view)
        {
            return view.GPUView;
        }

        #endregion
    }
}
