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

#if ENABLE_METAL_ACCELERATOR
namespace ILGPU.Backends.Metal
{
    /// <summary>
    /// Metal unified memory buffer implementation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class MetalUnifiedBuffer<T> : MemoryBuffer<T> where T : unmanaged
    {
        private readonly IntPtr _metalBuffer;
        private readonly MetalAccelerator _accelerator;

        /// <summary>
        /// Initializes a new instance of the MetalUnifiedBuffer class.
        /// </summary>
        /// <param name="accelerator">The Metal accelerator.</param>
        /// <param name="length">The number of elements.</param>
        public MetalUnifiedBuffer(MetalAccelerator accelerator, long length)
            : base(accelerator, length)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            var sizeInBytes = (nuint)(length * Interop.SizeOf<T>());
            _metalBuffer = MetalNative.MTLDeviceNewBuffer(
                accelerator.Device.NativeDevice, 
                sizeInBytes, 
                0); // MTLResourceStorageModeShared for unified memory

            if (_metalBuffer == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate Metal buffer");
        }

        /// <summary>
        /// Gets the native Metal buffer handle.
        /// </summary>
        public IntPtr NativeBuffer => _metalBuffer;

        /// <summary>
        /// Gets a pointer to the buffer data.
        /// </summary>
        /// <returns>A pointer to the buffer data.</returns>
        public override unsafe void* NativePtr => (void*)MetalNative.MTLBufferContents(_metalBuffer);

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        /// <param name="source">The source CPU memory.</param>
        /// <param name="sourceOffset">The source offset.</param>
        /// <param name="targetOffset">The target offset in this buffer.</param>
        /// <param name="extent">The number of elements to copy.</param>
        public override unsafe void CopyFromCPU(
            IntPtr source,
            long sourceOffset,
            long targetOffset,
            long extent)
        {
            var sourcePtr = (byte*)source + sourceOffset * Interop.SizeOf<T>();
            var targetPtr = (byte*)NativePtr + targetOffset * Interop.SizeOf<T>();
            var sizeInBytes = extent * Interop.SizeOf<T>();

            Buffer.MemoryCopy(sourcePtr, targetPtr, sizeInBytes, sizeInBytes);
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        /// <param name="target">The target CPU memory.</param>
        /// <param name="sourceOffset">The source offset in this buffer.</param>
        /// <param name="targetOffset">The target offset.</param>
        /// <param name="extent">The number of elements to copy.</param>
        public override unsafe void CopyToCPU(
            IntPtr target,
            long sourceOffset,
            long targetOffset,
            long extent)
        {
            var sourcePtr = (byte*)NativePtr + sourceOffset * Interop.SizeOf<T>();
            var targetPtr = (byte*)target + targetOffset * Interop.SizeOf<T>();
            var sizeInBytes = extent * Interop.SizeOf<T>();

            Buffer.MemoryCopy(sourcePtr, targetPtr, sizeInBytes, sizeInBytes);
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        /// <param name="source">The source buffer.</param>
        /// <param name="sourceOffset">The source offset.</param>
        /// <param name="targetOffset">The target offset in this buffer.</param>
        /// <param name="extent">The number of elements to copy.</param>
        public override unsafe void CopyFrom(
            ArrayView<T> source,
            long sourceOffset,
            long targetOffset,
            long extent)
        {
            if (source.GetAccelerator() is MetalAccelerator)
            {
                // Metal-to-Metal copy - use GPU copy command
                // This would use MTLBlitCommandEncoder for efficient GPU-side copy
                var sourcePtr = (byte*)source.LoadEffectiveAddress() + sourceOffset * Interop.SizeOf<T>();
                var targetPtr = (byte*)NativePtr + targetOffset * Interop.SizeOf<T>();
                var sizeInBytes = extent * Interop.SizeOf<T>();

                Buffer.MemoryCopy(sourcePtr, targetPtr, sizeInBytes, sizeInBytes);
            }
            else
            {
                // Cross-accelerator copy
                base.CopyFrom(source, sourceOffset, targetOffset, extent);
            }
        }

        /// <summary>
        /// Disposes the Metal buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (_metalBuffer != IntPtr.Zero)
            {
                MetalNative.CFRelease(_metalBuffer);
            }
        }
    }

    /// <summary>
    /// Metal 2D unified memory buffer implementation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class MetalUnifiedBuffer2D<T> : MemoryBuffer2D<T> where T : unmanaged
    {
        private readonly MetalUnifiedBuffer<T> _buffer;

        /// <summary>
        /// Initializes a new instance of the MetalUnifiedBuffer2D class.
        /// </summary>
        /// <param name="accelerator">The Metal accelerator.</param>
        /// <param name="width">The width dimension.</param>
        /// <param name="height">The height dimension.</param>
        public MetalUnifiedBuffer2D(MetalAccelerator accelerator, long width, long height)
            : base(accelerator, width, height)
        {
            _buffer = new MetalUnifiedBuffer<T>(accelerator, width * height);
        }

        /// <summary>
        /// Gets the underlying 1D buffer.
        /// </summary>
        public override MemoryBuffer<T> AsLinearBuffer() => _buffer;

        /// <summary>
        /// Gets a pointer to the buffer data.
        /// </summary>
        /// <returns>A pointer to the buffer data.</returns>
        public override unsafe void* NativePtr => _buffer.NativePtr;

        /// <summary>
        /// Disposes the 2D buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                _buffer?.Dispose();
            }
        }
    }

    /// <summary>
    /// Metal 3D unified memory buffer implementation.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class MetalUnifiedBuffer3D<T> : MemoryBuffer3D<T> where T : unmanaged
    {
        private readonly MetalUnifiedBuffer<T> _buffer;

        /// <summary>
        /// Initializes a new instance of the MetalUnifiedBuffer3D class.
        /// </summary>
        /// <param name="accelerator">The Metal accelerator.</param>
        /// <param name="width">The width dimension.</param>
        /// <param name="height">The height dimension.</param>
        /// <param name="depth">The depth dimension.</param>
        public MetalUnifiedBuffer3D(MetalAccelerator accelerator, long width, long height, long depth)
            : base(accelerator, width, height, depth)
        {
            _buffer = new MetalUnifiedBuffer<T>(accelerator, width * height * depth);
        }

        /// <summary>
        /// Gets the underlying 1D buffer.
        /// </summary>
        public override MemoryBuffer<T> AsLinearBuffer() => _buffer;

        /// <summary>
        /// Gets a pointer to the buffer data.
        /// </summary>
        /// <returns>A pointer to the buffer data.</returns>
        public override unsafe void* NativePtr => _buffer.NativePtr;

        /// <summary>
        /// Disposes the 3D buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing)
            {
                _buffer?.Dispose();
            }
        }
    }

    /// <summary>
    /// Metal command queue wrapper.
    /// </summary>
    public sealed class MetalCommandQueue : IDisposable
    {
        private readonly IntPtr _queue;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalCommandQueue class.
        /// </summary>
        /// <param name="queue">The native Metal command queue handle.</param>
        public MetalCommandQueue(IntPtr queue)
        {
            _queue = queue;
        }

        /// <summary>
        /// Gets the native command queue handle.
        /// </summary>
        public IntPtr NativeQueue => _queue;

        /// <summary>
        /// Creates a command buffer.
        /// </summary>
        /// <returns>A Metal command buffer.</returns>
        public MetalCommandBuffer CreateCommandBuffer()
        {
            var buffer = MetalNative.MTLCommandQueueCommandBuffer(_queue);
            if (buffer == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Metal command buffer");

            return new MetalCommandBuffer(buffer);
        }

        /// <summary>
        /// Disposes the command queue.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_queue != IntPtr.Zero)
                {
                    MetalNative.CFRelease(_queue);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Metal command buffer wrapper.
    /// </summary>
    public sealed class MetalCommandBuffer : IDisposable
    {
        private readonly IntPtr _buffer;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalCommandBuffer class.
        /// </summary>
        /// <param name="buffer">The native Metal command buffer handle.</param>
        public MetalCommandBuffer(IntPtr buffer)
        {
            _buffer = buffer;
        }

        /// <summary>
        /// Gets the native command buffer handle.
        /// </summary>
        public IntPtr NativeBuffer => _buffer;

        /// <summary>
        /// Commits the command buffer for execution.
        /// </summary>
        public void Commit()
        {
            MetalNative.MTLCommandBufferCommit(_buffer);
        }

        /// <summary>
        /// Waits for the command buffer to complete execution.
        /// </summary>
        public void WaitUntilCompleted()
        {
            MetalNative.MTLCommandBufferWaitUntilCompleted(_buffer);
        }

        /// <summary>
        /// Disposes the command buffer.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_buffer != IntPtr.Zero)
                {
                    MetalNative.CFRelease(_buffer);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Metal library wrapper.
    /// </summary>
    public sealed class MetalLibrary : IDisposable
    {
        private readonly IntPtr _library;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalLibrary class.
        /// </summary>
        /// <param name="library">The native Metal library handle.</param>
        public MetalLibrary(IntPtr library)
        {
            _library = library;
        }

        /// <summary>
        /// Gets the native library handle.
        /// </summary>
        public IntPtr NativeLibrary => _library;

        /// <summary>
        /// Creates a function from the library.
        /// </summary>
        /// <param name="functionName">The name of the function.</param>
        /// <returns>A Metal function.</returns>
        public MetalFunction CreateFunction(string functionName)
        {
            var nameHandle = System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(functionName);
            try
            {
                var function = MetalNative.MTLLibraryNewFunctionWithName(_library, nameHandle);
                if (function == IntPtr.Zero)
                    throw new InvalidOperationException($"Function '{functionName}' not found in Metal library");

                return new MetalFunction(function);
            }
            finally
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(nameHandle);
            }
        }

        /// <summary>
        /// Disposes the library.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_library != IntPtr.Zero)
                {
                    MetalNative.CFRelease(_library);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Metal function wrapper.
    /// </summary>
    public sealed class MetalFunction : IDisposable
    {
        private readonly IntPtr _function;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the MetalFunction class.
        /// </summary>
        /// <param name="function">The native Metal function handle.</param>
        public MetalFunction(IntPtr function)
        {
            _function = function;
        }

        /// <summary>
        /// Gets the native function handle.
        /// </summary>
        public IntPtr NativeFunction => _function;

        /// <summary>
        /// Disposes the function.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_function != IntPtr.Zero)
                {
                    MetalNative.CFRelease(_function);
                }
                _disposed = true;
            }
        }
    }
}
#endif