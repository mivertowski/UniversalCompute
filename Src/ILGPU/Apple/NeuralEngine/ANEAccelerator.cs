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

using ILGPU.Apple.NeuralEngine.Native;
using ILGPU.Backends;
using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Apple Neural Engine accelerator for AI/ML inference on Apple Silicon.
    /// </summary>
    public sealed class ANEAccelerator : Accelerator
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ANEAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The ANE device.</param>
        public ANEAccelerator(Context context, Device device)
            : base(context, device)
        {
            if (!ANENative.IsNeuralEngineAvailable())
                throw new NotSupportedException("Apple Neural Engine not available on this device");

            ContextHandle = ANENative.CreateContext();
            if (ContextHandle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Apple Neural Engine context");

            Capabilities = ANECapabilities.Query();

            // Initialize accelerator properties
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the ANE capabilities.
        /// </summary>
        public new ANECapabilities Capabilities { get; }

        /// <summary>
        /// Gets the ANE context handle.
        /// </summary>
        internal IntPtr ContextHandle { get; }

        #region AI/ML Operations

        /// <summary>
        /// Performs convolution operation optimized for ANE.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="weights">Convolution weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Convolution configuration.</param>
        public unsafe void Convolution2D(
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> output,
            ANEConvolutionConfig config)
        {
            ThrowIfDisposed();
            
            // TODO: Implement proper ANE data transfer
            // ArrayView doesn't support AsSpan - need proper GPU-to-ANE data copy
            throw new NotSupportedException("ANE convolution not fully implemented");
        }

        /// <summary>
        /// Performs matrix multiplication optimized for ANE.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="m">Number of rows in A.</param>
        /// <param name="n">Number of columns in B.</param>
        /// <param name="k">Number of columns in A / rows in B.</param>
        public unsafe void MatrixMultiply(
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            int m, int n, int k)
        {
            ThrowIfDisposed();
            
            // TODO: Implement proper ANE matrix multiplication
            throw new NotSupportedException("ANE matrix multiplication not fully implemented");
        }

        /// <summary>
        /// Performs transformer attention mechanism on ANE.
        /// </summary>
        /// <param name="queries">Query tensor.</param>
        /// <param name="keys">Key tensor.</param>
        /// <param name="values">Value tensor.</param>
        /// <param name="output">Output attention tensor.</param>
        /// <param name="config">Attention configuration.</param>
        public unsafe void Attention(
            ArrayView<float> queries,
            ArrayView<float> keys,
            ArrayView<float> values,
            ArrayView<float> output,
            ANEAttentionConfig config)
        {
            ThrowIfDisposed();
            
            // TODO: Implement proper ANE attention mechanism
            throw new NotSupportedException("ANE attention not fully implemented");
        }

        /// <summary>
        /// Executes a Core ML model on the Neural Engine.
        /// </summary>
        /// <param name="model">The Core ML model.</param>
        /// <param name="input">Input data.</param>
        /// <param name="output">Output data.</param>
        public unsafe void ExecuteCoreMLModel(
            ANECoreMLModel model,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            ThrowIfDisposed();
            
            // TODO: Implement proper ArrayView to pointer conversion - AsSpan not available
            throw new NotSupportedException("ANE CoreML inference not implemented - requires proper memory access");
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal() => new ANEStream(this);

        protected override void SynchronizeInternal()
        {
            // ANE operations are synchronous by default
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new ANEBuffer(this, length, elementSize);

        protected override Kernel LoadKernelInternal(CompiledKernel kernel) =>
            // ANE uses specialized operations rather than general kernels
            throw new NotSupportedException(
                "Apple Neural Engine does not support general kernel loading. " +
                "Use specialized ANE operations instead.");

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel kernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = null;
            return LoadKernelInternal(kernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel kernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = null;
            return LoadKernelInternal(kernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes) =>
            // ANE has different architecture, return a conservative estimate
            Capabilities.MaxConcurrentOperations;

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, Capabilities.OptimalWorkGroupSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, Capabilities.OptimalWorkGroupSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) =>
            // ANE typically shares memory with CPU
            otherAccelerator.AcceleratorType == AcceleratorType.CPU;

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // ANE peer access is managed by the OS
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // ANE peer access is managed by the OS
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements) =>
            // ANE doesn't support page locking, return a no-op implementation
            null!; // Return null to indicate no page locking support

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by ANE accelerator");

        protected override void OnBind()
        {
            // ANE binding is handled by the context
        }

        protected override void OnUnbind()
        {
            // ANE unbinding is handled by the context
        }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing && ContextHandle != IntPtr.Zero)
                {
                    ANENative.ReleaseContext(ContextHandle);
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Gets the ANE accelerator name.
        /// </summary>
        public new string Name => $"Apple Neural Engine ({Capabilities.ChipGeneration})";

        /// <summary>
        /// Gets the maximum grid size for ANE operations.
        /// </summary>
        public new Index3D MaxGridSize => new(Capabilities.MaxTensorWidth, Capabilities.MaxTensorHeight, 1);

        /// <summary>
        /// Gets the maximum group size for ANE operations.
        /// </summary>
        public new Index3D MaxGroupSize => new(Capabilities.OptimalWorkGroupSize, 1, 1);

        /// <summary>
        /// Gets the ANE warp size.
        /// </summary>
        public new int WarpSize => Capabilities.OptimalWorkGroupSize;

        /// <summary>
        /// Gets the number of compute units (multiprocessors).
        /// </summary>
        public new int NumMultiprocessors => Capabilities.NumComputeUnits;

        /// <summary>
        /// Gets the maximum shared memory per multiprocessor.
        /// </summary>
        public new int MaxSharedMemoryPerGroup => Capabilities.MaxSharedMemoryPerUnit;

        /// <summary>
        /// Gets the maximum constant memory.
        /// </summary>
        public new int MaxConstantMemory => Capabilities.MaxConstantMemory;

        /// <summary>
        /// Gets the memory bandwidth.
        /// </summary>
        public long MemoryBandwidth => Capabilities.MemoryBandwidth;

        private static void InitializeAcceleratorProperties()
        {
            // Properties are now computed via overrides
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ANEAccelerator));
        }

        #endregion

        #region Static Methods

        /// <summary>
        /// Checks if Apple Neural Engine is available on this system.
        /// </summary>
        /// <returns>True if ANE is available; otherwise, false.</returns>
        public static bool IsAvailable()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                return RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && 
                       ANENative.IsNeuralEngineAvailable();
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Creates an ANE accelerator if available.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>ANE accelerator or null if not available.</returns>
        public static ANEAccelerator? CreateIfAvailable(Context context)
        {
            if (!IsAvailable()) return null;
            
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var device = AppleNeuralEngineDevice.Default;
                return new ANEAccelerator(context, device);
            }
            catch
            {
                return null;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion
    }

    /// <summary>
    /// Apple Neural Engine stream implementation.
    /// </summary>
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix
    public sealed class ANEStream : AcceleratorStream
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
    {
        /// <summary>
        /// Initializes a new instance of the ANEStream class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        public ANEStream(ANEAccelerator accelerator) : base(accelerator)
        {
        }

        /// <summary>
        /// Synchronizes the ANE stream (no-op as ANE operations are synchronous).
        /// </summary>
        public override void Synchronize()
        {
            // ANE operations are inherently synchronous
        }

        /// <summary>
        /// Adds a profiling marker to the stream.
        /// </summary>
        protected override ProfilingMarker AddProfilingMarkerInternal() =>
            // ANE doesn't support detailed profiling markers, return null
            null!;

        /// <summary>
        /// Disposes the ANE stream.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No specific cleanup needed for ANE streams
        }
    }

    /// <summary>
    /// Apple Neural Engine memory buffer implementation.
    /// </summary>
    public sealed class ANEBuffer : MemoryBuffer
    {
        private readonly IntPtr _nativePtr;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ANEBuffer class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element.</param>
        public ANEBuffer(ANEAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            var sizeInBytes = length * elementSize;
#pragma warning disable CA2020 // Prevent behavioral change
            _nativePtr = Marshal.AllocHGlobal((IntPtr)sizeInBytes);
#pragma warning restore CA2020 // Prevent behavioral change
            
            if (_nativePtr == IntPtr.Zero)
                throw new GpuMemoryException("Failed to allocate ANE buffer memory");
                
            NativePtr = _nativePtr;
        }

        /// <summary>
        /// Gets the native pointer to the buffer data as void*.
        /// </summary>
        public unsafe void* RawPtr => (void*)_nativePtr;

        /// <summary>
        /// Sets the contents of this buffer to the given byte value.
        /// </summary>
        protected internal override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
            unsafe
            {
                var ptr = (byte*)_nativePtr + targetView.Index;
                var length = targetView.Length;
                for (long i = 0; i < length; i++)
                    ptr[i] = value;
            }
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        protected internal override void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView) =>
            // TODO: Implement proper buffer access for ANE
            throw new NotSupportedException("ANE buffer access not implemented");

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        protected internal override void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView) =>
            // TODO: Implement proper buffer access for ANE
            throw new NotSupportedException("ANE buffer access not implemented");

        /// <summary>
        /// Disposes the ANE buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativePtr != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_nativePtr);
                }
                _disposed = true;
            }
        }
    }
}
