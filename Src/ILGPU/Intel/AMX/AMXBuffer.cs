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

using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// AMX memory buffer implementation using CPU memory.
    /// </summary>
    public sealed class AMXBuffer : MemoryBuffer
    {
        private readonly IntPtr _nativePtr;
        private readonly long _lengthInBytes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AMXBuffer class.
        /// </summary>
        /// <param name="accelerator">The AMX accelerator.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        public AMXBuffer(AMXAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _lengthInBytes = length * elementSize;
            
            // Allocate aligned memory for optimal AMX performance
            _nativePtr = AllocateAlignedMemory(_lengthInBytes, 64); // 64-byte alignment for AMX
            if (_nativePtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate AMX buffer memory");
        }

        /// <summary>
        /// Gets a pointer to the buffer data.
        /// </summary>
        /// <returns>A pointer to the buffer data.</returns>
        public override unsafe void* NativePtr => (void*)_nativePtr;

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        /// <param name="source">The source CPU memory.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyFromCPU(
            IntPtr source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            var sourcePtr = (byte*)source + sourceOffset;
            var targetPtr = (byte*)_nativePtr + targetOffset;
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, _lengthInBytes - targetOffset, length);
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        /// <param name="target">The target CPU memory.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyToCPU(
            IntPtr target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            var sourcePtr = (byte*)_nativePtr + sourceOffset;
            var targetPtr = (byte*)target + targetOffset;
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, length, length);
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyFrom(
            MemoryBuffer source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (source is AMXBuffer amxSource)
            {
                // AMX-to-AMX copy
                var sourcePtr = (byte*)amxSource._nativePtr + sourceOffset;
                var targetPtr = (byte*)_nativePtr + targetOffset;
                
                Buffer.MemoryCopy(sourcePtr, targetPtr, _lengthInBytes - targetOffset, length);
            }
            else
            {
                // Cross-accelerator copy
                base.CopyFrom(source, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        /// <param name="target">The target buffer.</param>
        /// <param name="sourceOffset">The source offset in bytes.</param>
        /// <param name="targetOffset">The target offset in bytes.</param>
        /// <param name="length">The number of bytes to copy.</param>
        public override unsafe void CopyTo(
            MemoryBuffer target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (target is AMXBuffer amxTarget)
            {
                // AMX-to-AMX copy
                var sourcePtr = (byte*)_nativePtr + sourceOffset;
                var targetPtr = (byte*)amxTarget._nativePtr + targetOffset;
                
                Buffer.MemoryCopy(sourcePtr, targetPtr, length, length);
            }
            else
            {
                // Cross-accelerator copy
                base.CopyTo(target, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Disposes the AMX buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativePtr != IntPtr.Zero)
                {
                    FreeAlignedMemory(_nativePtr);
                }
                _disposed = true;
            }
        }

        private static IntPtr AllocateAlignedMemory(long size, int alignment)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return _aligned_malloc((UIntPtr)size, (UIntPtr)alignment);
            }
            else
            {
                // Use posix_memalign for Unix-like systems
                if (posix_memalign(out var ptr, (UIntPtr)alignment, (UIntPtr)size) == 0)
                    return ptr;
                return IntPtr.Zero;
            }
        }

        private static void FreeAlignedMemory(IntPtr ptr)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _aligned_free(ptr);
            }
            else
            {
                free(ptr);
            }
        }

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr _aligned_malloc(UIntPtr size, UIntPtr alignment);

        [DllImport("msvcrt.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void _aligned_free(IntPtr ptr);

        [DllImport("libc", EntryPoint = "posix_memalign")]
        private static extern int posix_memalign(out IntPtr ptr, UIntPtr alignment, UIntPtr size);

        [DllImport("libc", EntryPoint = "free")]
        private static extern void free(IntPtr ptr);
    }

    /// <summary>
    /// AMX kernel implementation.
    /// </summary>
    public sealed class AMXKernel : Kernel
    {
        private readonly AMXAccelerator _accelerator;
        private readonly CompiledKernel _compiledKernel;

        /// <summary>
        /// Initializes a new instance of the AMXKernel class.
        /// </summary>
        /// <param name="accelerator">The AMX accelerator.</param>
        /// <param name="compiledKernel">The compiled kernel.</param>
        public AMXKernel(AMXAccelerator accelerator, CompiledKernel compiledKernel)
            : base(accelerator, compiledKernel)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _compiledKernel = compiledKernel ?? throw new ArgumentNullException(nameof(compiledKernel));
        }

        /// <summary>
        /// Launches the kernel with the specified configuration.
        /// </summary>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="extent">The grid extent.</param>
        /// <param name="parameters">The kernel parameters.</param>
        protected override void LaunchInternal(
            AcceleratorStream stream,
            KernelConfig extent,
            RuntimeKernelConfig runtimeKernelConfig)
        {
            // AMX kernels are executed on the CPU with tile optimizations
            ExecuteAMXKernel(extent, runtimeKernelConfig);
        }

        private void ExecuteAMXKernel(KernelConfig extent, RuntimeKernelConfig runtimeConfig)
        {
            // For AMX, we would:
            // 1. Configure tiles optimally for the kernel
            // 2. Execute the kernel using tile-based operations
            // 3. Handle matrix operations with AMX intrinsics
            
            // This is a placeholder implementation
            // Real implementation would involve:
            // - Analyzing the kernel for matrix operations
            // - Automatically tiling large matrices
            // - Using AMX instructions for matrix multiplications
            
            throw new NotImplementedException("AMX kernel execution not fully implemented");
        }

        /// <summary>
        /// Disposes the AMX kernel.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No special cleanup needed for AMX kernels
        }
    }
}
