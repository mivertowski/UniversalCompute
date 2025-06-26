// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU.Runtime;
using ILGPU.Intel.IPP.Native;

namespace ILGPU.Intel.IPP
{
    /// <summary>
    /// Represents a memory buffer optimized for Intel IPP operations.
    /// Uses IPP-aligned memory allocation for optimal performance.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    internal sealed class IPPMemoryBuffer<T> : MemoryBuffer<T, Index1D>
        where T : unmanaged
    {
        #region Instance

        private IntPtr _nativePtr;
        private readonly long _lengthInBytes;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new IPP-optimized memory buffer.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="extent">The buffer extent (number of elements).</param>
        public IPPMemoryBuffer(Accelerator accelerator, Index1D extent)
            : base(accelerator, extent)
        {
            if (extent < 1)
                throw new ArgumentOutOfRangeException(nameof(extent), "Extent must be positive");

            _lengthInBytes = extent * Unsafe.SizeOf<T>();

            // Allocate aligned memory using IPP
            _nativePtr = AllocateIPPMemory<T>(extent);
            if (_nativePtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate IPP-aligned memory");
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the native pointer to the allocated memory.
        /// </summary>
        public override IntPtr NativePtr => _nativePtr;

        /// <summary>
        /// Gets the length of the buffer in bytes.
        /// </summary>
        protected override long LengthInBytes => _lengthInBytes;

        #endregion

        #region Methods

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        /// <param name="source">Source CPU memory.</param>
        /// <param name="sourceOffset">Source offset in elements.</param>
        /// <param name="targetOffset">Target offset in elements.</param>
        /// <param name="extent">Number of elements to copy.</param>
        protected override void CopyFromCPU(
            IntPtr source,
            Index1D sourceOffset,
            Index1D targetOffset,
            Index1D extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(source, sourceOffset * elementSize);
            var targetPtr = IntPtr.Add(_nativePtr, targetOffset * elementSize);
            var copySize = extent * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        /// <param name="target">Target CPU memory.</param>
        /// <param name="sourceOffset">Source offset in elements.</param>
        /// <param name="targetOffset">Target offset in elements.</param>
        /// <param name="extent">Number of elements to copy.</param>
        protected override void CopyToCPU(
            IntPtr target,
            Index1D sourceOffset,
            Index1D targetOffset,
            Index1D extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(_nativePtr, sourceOffset * elementSize);
            var targetPtr = IntPtr.Add(target, targetOffset * elementSize);
            var copySize = extent * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from another accelerator buffer to this buffer.
        /// </summary>
        /// <param name="source">Source buffer.</param>
        /// <param name="sourceOffset">Source offset in elements.</param>
        /// <param name="targetOffset">Target offset in elements.</param>
        /// <param name="extent">Number of elements to copy.</param>
        protected override void CopyFromAccelerator(
            MemoryBuffer source,
            Index1D sourceOffset,
            Index1D targetOffset,
            Index1D extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(source.NativePtr, sourceOffset * elementSize);
            var targetPtr = IntPtr.Add(_nativePtr, targetOffset * elementSize);
            var copySize = extent * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from this buffer to another accelerator buffer.
        /// </summary>
        /// <param name="target">Target buffer.</param>
        /// <param name="sourceOffset">Source offset in elements.</param>
        /// <param name="targetOffset">Target offset in elements.</param>
        /// <param name="extent">Number of elements to copy.</param>
        protected override void CopyToAccelerator(
            MemoryBuffer target,
            Index1D sourceOffset,
            Index1D targetOffset,
            Index1D extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(_nativePtr, sourceOffset * elementSize);
            var targetPtr = IntPtr.Add(target.NativePtr, targetOffset * elementSize);
            var copySize = extent * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Sets all elements in the buffer to zero.
        /// </summary>
        /// <param name="stream">Optional accelerator stream.</param>
        protected override void MemSetToZero(AcceleratorStream? stream = null)
        {
            unsafe
            {
                Unsafe.InitBlock(_nativePtr.ToPointer(), 0, (uint)_lengthInBytes);
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this memory buffer and frees the allocated memory.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed && _nativePtr != IntPtr.Zero)
            {
                IPPNative.ippsFree(_nativePtr);
                _nativePtr = IntPtr.Zero;
                _disposed = true;
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Allocates IPP-aligned memory for the specified type and count.
        /// </summary>
        /// <typeparam name="TElement">The element type.</typeparam>
        /// <param name="count">Number of elements.</param>
        /// <returns>Pointer to allocated memory.</returns>
        private static IntPtr AllocateIPPMemory<TElement>(long count)
            where TElement : unmanaged
        {
            // Use appropriate IPP allocation function based on type
            if (typeof(TElement) == typeof(float))
                return IPPNative.ippsMalloc_32f((int)count);
            else if (typeof(TElement) == typeof(System.Numerics.Complex) ||
                     typeof(TElement) == typeof(IPPNative.Ipp32fc))
                return IPPNative.ippsMalloc_32fc((int)count);
            else
            {
                // For other types, allocate float memory with appropriate size
                var elementSize = Unsafe.SizeOf<TElement>();
                var floatCount = (int)((count * elementSize + sizeof(float) - 1) / sizeof(float));
                return IPPNative.ippsMalloc_32f(floatCount);
            }
        }

        #endregion
    }

    /// <summary>
    /// Represents a 2D memory buffer optimized for Intel IPP operations.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <typeparam name="TIndex">The index type.</typeparam>
    internal sealed class IPPMemoryBuffer<T, TIndex> : MemoryBuffer<T, TIndex>
        where T : unmanaged
        where TIndex : struct, IIndex
    {
        #region Instance

        private IntPtr _nativePtr;
        private readonly long _lengthInBytes;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new 2D IPP-optimized memory buffer.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="extent">The buffer extent.</param>
        public IPPMemoryBuffer(Accelerator accelerator, TIndex extent)
            : base(accelerator, extent)
        {
            var linearLength = extent.Size;
            if (linearLength < 1)
                throw new ArgumentOutOfRangeException(nameof(extent), "Extent must be positive");

            _lengthInBytes = linearLength * Unsafe.SizeOf<T>();

            // Allocate aligned memory using IPP
            _nativePtr = AllocateIPPMemory<T>(linearLength);
            if (_nativePtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate IPP-aligned memory");
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the native pointer to the allocated memory.
        /// </summary>
        public override IntPtr NativePtr => _nativePtr;

        /// <summary>
        /// Gets the length of the buffer in bytes.
        /// </summary>
        protected override long LengthInBytes => _lengthInBytes;

        #endregion

        #region Methods

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        protected override void CopyFromCPU(
            IntPtr source,
            TIndex sourceOffset,
            TIndex targetOffset,
            TIndex extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(source, sourceOffset.LinearIndex * elementSize);
            var targetPtr = IntPtr.Add(_nativePtr, targetOffset.LinearIndex * elementSize);
            var copySize = extent.Size * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        protected override void CopyToCPU(
            IntPtr target,
            TIndex sourceOffset,
            TIndex targetOffset,
            TIndex extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(_nativePtr, sourceOffset.LinearIndex * elementSize);
            var targetPtr = IntPtr.Add(target, targetOffset.LinearIndex * elementSize);
            var copySize = extent.Size * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from another accelerator buffer to this buffer.
        /// </summary>
        protected override void CopyFromAccelerator(
            MemoryBuffer source,
            TIndex sourceOffset,
            TIndex targetOffset,
            TIndex extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(source.NativePtr, sourceOffset.LinearIndex * elementSize);
            var targetPtr = IntPtr.Add(_nativePtr, targetOffset.LinearIndex * elementSize);
            var copySize = extent.Size * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Copies data from this buffer to another accelerator buffer.
        /// </summary>
        protected override void CopyToAccelerator(
            MemoryBuffer target,
            TIndex sourceOffset,
            TIndex targetOffset,
            TIndex extent)
        {
            var elementSize = Unsafe.SizeOf<T>();
            var sourcePtr = IntPtr.Add(_nativePtr, sourceOffset.LinearIndex * elementSize);
            var targetPtr = IntPtr.Add(target.NativePtr, targetOffset.LinearIndex * elementSize);
            var copySize = extent.Size * elementSize;

            unsafe
            {
                Buffer.MemoryCopy(
                    sourcePtr.ToPointer(),
                    targetPtr.ToPointer(),
                    copySize,
                    copySize);
            }
        }

        /// <summary>
        /// Sets all elements in the buffer to zero.
        /// </summary>
        protected override void MemSetToZero(AcceleratorStream? stream = null)
        {
            unsafe
            {
                Unsafe.InitBlock(_nativePtr.ToPointer(), 0, (uint)_lengthInBytes);
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this memory buffer and frees the allocated memory.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed && _nativePtr != IntPtr.Zero)
            {
                IPPNative.ippsFree(_nativePtr);
                _nativePtr = IntPtr.Zero;
                _disposed = true;
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Allocates IPP-aligned memory for the specified type and count.
        /// </summary>
        private static IntPtr AllocateIPPMemory<TElement>(long count)
            where TElement : unmanaged
        {
            // Use appropriate IPP allocation function based on type
            if (typeof(TElement) == typeof(float))
                return IPPNative.ippsMalloc_32f((int)count);
            else if (typeof(TElement) == typeof(System.Numerics.Complex) ||
                     typeof(TElement) == typeof(IPPNative.Ipp32fc))
                return IPPNative.ippsMalloc_32fc((int)count);
            else
            {
                // For other types, allocate float memory with appropriate size
                var elementSize = Unsafe.SizeOf<TElement>();
                var floatCount = (int)((count * elementSize + sizeof(float) - 1) / sizeof(float));
                return IPPNative.ippsMalloc_32f(floatCount);
            }
        }

        #endregion
    }
}