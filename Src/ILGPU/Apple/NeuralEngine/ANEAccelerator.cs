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
        private readonly IntPtr _aneContext;
        private readonly ANECapabilities _capabilities;
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

            _aneContext = ANENative.CreateContext();
            if (_aneContext == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Apple Neural Engine context");

            _capabilities = ANECapabilities.Query();
            
            // Initialize accelerator properties
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the ANE capabilities.
        /// </summary>
        public new ANECapabilities Capabilities => _capabilities;

        /// <summary>
        /// Gets the ANE context handle.
        /// </summary>
        internal IntPtr ContextHandle => _aneContext;

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
            
            fixed (float* inputPtr = input.SubView(0, (int)input.Length).AsSpan())
            fixed (float* outputPtr = output.SubView(0, (int)output.Length).AsSpan())
            {
                ANENative.ExecuteCoreMLInference(
                    inputPtr, outputPtr, 
                    input.Length, output.Length,
                    model.ModelHandle, _aneContext);
            }
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal()
        {
            return new ANEStream(this);
        }

        protected override void SynchronizeInternal()
        {
            // ANE operations are synchronous by default
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new ANEBuffer(this, length, elementSize);
        }

        protected override Kernel LoadKernelInternal(CompiledKernel compiledKernel)
        {
            // ANE uses specialized operations rather than general kernels
            throw new NotSupportedException(
                "Apple Neural Engine does not support general kernel loading. " +
                "Use specialized ANE operations instead.");
        }

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel compiledKernel,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = null;
            return LoadKernelInternal(compiledKernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel compiledKernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            kernelInfo = null;
            return LoadKernelInternal(compiledKernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            // ANE has different architecture, return a conservative estimate
            return _capabilities.MaxConcurrentOperations;
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, _capabilities.OptimalWorkGroupSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, _capabilities.OptimalWorkGroupSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            // ANE typically shares memory with CPU
            return otherAccelerator.AcceleratorType == AcceleratorType.CPU;
        }

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // ANE peer access is managed by the OS
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // ANE peer access is managed by the OS
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements)
        {
            // ANE doesn't support page locking, return a no-op implementation
            return null!; // Return null to indicate no page locking support
        }

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider)
        {
            throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by ANE accelerator");
        }

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
                if (disposing && _aneContext != IntPtr.Zero)
                {
                    ANENative.ReleaseContext(_aneContext);
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Gets the ANE accelerator name.
        /// </summary>
        public new string Name => $"Apple Neural Engine ({_capabilities.ChipGeneration})";

        /// <summary>
        /// Gets the maximum grid size for ANE operations.
        /// </summary>
        public new Index3D MaxGridSize => new Index3D(_capabilities.MaxTensorWidth, _capabilities.MaxTensorHeight, 1);

        /// <summary>
        /// Gets the maximum group size for ANE operations.
        /// </summary>
        public new Index3D MaxGroupSize => new Index3D(_capabilities.OptimalWorkGroupSize, 1, 1);

        /// <summary>
        /// Gets the ANE warp size.
        /// </summary>
        public new int WarpSize => _capabilities.OptimalWorkGroupSize;

        /// <summary>
        /// Gets the number of compute units (multiprocessors).
        /// </summary>
        public new int NumMultiprocessors => _capabilities.NumComputeUnits;

        /// <summary>
        /// Gets the maximum shared memory per multiprocessor.
        /// </summary>
        public new int MaxSharedMemoryPerGroup => _capabilities.MaxSharedMemoryPerUnit;

        /// <summary>
        /// Gets the maximum constant memory.
        /// </summary>
        public new int MaxConstantMemory => _capabilities.MaxConstantMemory;

        /// <summary>
        /// Gets the memory bandwidth.
        /// </summary>
        public long MemoryBandwidth => _capabilities.MemoryBandwidth;

        private void InitializeAcceleratorProperties()
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
            try
            {
                return RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && 
                       ANENative.IsNeuralEngineAvailable();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Creates an ANE accelerator if available.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>ANE accelerator or null if not available.</returns>
        public static ANEAccelerator? CreateIfAvailable(Context context)
        {
            if (!IsAvailable()) return null;
            
            try
            {
                var device = AppleNeuralEngineDevice.Default;
                return new ANEAccelerator(context, device);
            }
            catch
            {
                return null;
            }
        }

        #endregion
    }

    /// <summary>
    /// Apple Neural Engine stream implementation.
    /// </summary>
    public sealed class ANEStream : AcceleratorStream
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
        protected override ProfilingMarker AddProfilingMarkerInternal()
        {
            // ANE doesn't support detailed profiling markers, return null
            return null!;
        }

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
            _nativePtr = Marshal.AllocHGlobal((IntPtr)sizeInBytes);
            
            if (_nativePtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate ANE buffer memory");
                
            NativePtr = _nativePtr;
        }

        /// <summary>
        /// Gets the native pointer to the buffer data as void*.
        /// </summary>
        public unsafe void* GetNativePtr() => (void*)_nativePtr;

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
            in ArrayView<byte> targetView)
        {
            // TODO: Implement proper buffer access for ANE
            throw new NotSupportedException("ANE buffer access not implemented");
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        protected internal override void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            // TODO: Implement proper buffer access for ANE
            throw new NotSupportedException("ANE buffer access not implemented");
        }

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