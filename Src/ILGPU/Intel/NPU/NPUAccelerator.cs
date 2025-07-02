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

using ILGPU.Backends;
using ILGPU.Intel.NPU.Native;
using ILGPU.Runtime;
using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Intel Neural Processing Unit (NPU) accelerator for AI inference on Intel Arc and Core processors.
    /// </summary>
    public sealed class NPUAccelerator : Accelerator
    {
        private readonly IntPtr _npuContext;
        private readonly NPUCapabilities _capabilities;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the NPUAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The NPU device.</param>
        public NPUAccelerator(Context context, Device device)
            : base(context, device)
        {
            if (!NPUNative.IsNPUAvailable())
                throw new NotSupportedException("Intel NPU not available on this device");

            _npuContext = NPUNative.CreateContext();
            if (_npuContext == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create Intel NPU context");

            _capabilities = NPUCapabilities.Query();
            
            // Initialize accelerator properties
            InitializeAcceleratorProperties();
        }

        /// <summary>
        /// Gets the NPU capabilities.
        /// </summary>
        public NPUCapabilities Capabilities => _capabilities;

        /// <summary>
        /// Gets the NPU context handle.
        /// </summary>
        internal IntPtr ContextHandle => _npuContext;

        #region AI/ML Operations

        /// <summary>
        /// Performs convolution operation optimized for NPU.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="weights">Convolution weights.</param>
        /// <param name="output">Output tensor.</param>
        /// <param name="config">Convolution configuration.</param>
        public unsafe void Convolution2D(
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> output,
            NPUConvolutionConfig config)
        {
            ThrowIfDisposed();
            
            fixed (float* inputPtr = input.GetSubView(0, (int)input.Length).AsSpan())
            fixed (float* outputPtr = output.GetSubView(0, (int)output.Length).AsSpan())
            {
                NPUOperations.ExecuteConvolution(
                    inputPtr, weights, outputPtr, 
                    config, _npuContext);
            }
        }

        /// <summary>
        /// Performs matrix multiplication optimized for NPU.
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
            
            fixed (float* aPtr = a.GetSubView(0, (int)a.Length).AsSpan())
            fixed (float* bPtr = b.GetSubView(0, (int)b.Length).AsSpan())
            fixed (float* cPtr = c.GetSubView(0, (int)c.Length).AsSpan())
            {
                NPUOperations.ExecuteMatrixMultiply(
                    aPtr, bPtr, cPtr, m, n, k, _npuContext);
            }
        }

        /// <summary>
        /// Executes an OpenVINO model on the NPU.
        /// </summary>
        /// <param name="model">The OpenVINO model.</param>
        /// <param name="input">Input data.</param>
        /// <param name="output">Output data.</param>
        public unsafe void ExecuteOpenVINOModel(
            NPUOpenVINOModel model,
            ArrayView<float> input,
            ArrayView<float> output)
        {
            ThrowIfDisposed();
            
            fixed (float* inputPtr = input.GetSubView(0, (int)input.Length).AsSpan())
            fixed (float* outputPtr = output.GetSubView(0, (int)output.Length).AsSpan())
            {
                NPUNative.ExecuteOpenVINOInference(
                    inputPtr, outputPtr, 
                    input.Length, output.Length,
                    model.ModelHandle, _npuContext);
            }
        }

        /// <summary>
        /// Performs quantized inference using INT8 precision.
        /// </summary>
        /// <param name="input">Input data in INT8 format.</param>
        /// <param name="weights">Quantized weights.</param>
        /// <param name="output">Output data.</param>
        /// <param name="config">Quantization configuration.</param>
        public unsafe void QuantizedInference(
            ArrayView<sbyte> input,
            ArrayView<sbyte> weights,
            ArrayView<float> output,
            NPUQuantizationConfig config)
        {
            ThrowIfDisposed();
            
            fixed (sbyte* inputPtr = input.GetSubView(0, (int)input.Length).AsSpan())
            fixed (sbyte* weightsPtr = weights.GetSubView(0, (int)weights.Length).AsSpan())
            fixed (float* outputPtr = output.GetSubView(0, (int)output.Length).AsSpan())
            {
                NPUOperations.ExecuteQuantizedInference(
                    inputPtr, weightsPtr, outputPtr, config, _npuContext);
            }
        }

        #endregion

        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal()
        {
            return new NPUStream(this);
        }

        protected override void SynchronizeInternal()
        {
            NPUNative.Synchronize(_npuContext);
        }

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize)
        {
            return new NPUBuffer(this, length, elementSize);
        }

        protected override Kernel LoadKernelInternal(CompiledKernel compiledKernel)
        {
            // NPU uses specialized operations rather than general kernels
            throw new NotSupportedException(
                "Intel NPU does not support general kernel loading. " +
                "Use specialized NPU operations instead.");
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
            // NPU has different architecture, return a conservative estimate
            return _capabilities.MaxConcurrentInferences;
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, _capabilities.OptimalBatchSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, _capabilities.OptimalBatchSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator)
        {
            // NPU typically shares memory with CPU and GPU through shared system memory
            return otherAccelerator.AcceleratorType == AcceleratorType.CPU ||
                   otherAccelerator.AcceleratorType == AcceleratorType.OpenCL;
        }

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // NPU peer access is managed by the driver
        }

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator)
        {
            // NPU peer access is managed by the driver
        }

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements)
        {
            return new PageLockScope<T>(this, pinned, numElements);
        }

        protected override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider)
        {
            throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by NPU accelerator");
        }

        protected override void OnBind()
        {
            // NPU binding is handled by the context
        }

        protected override void OnUnbind()
        {
            // NPU unbinding is handled by the context
        }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing && _npuContext != IntPtr.Zero)
                {
                    NPUNative.ReleaseContext(_npuContext);
                }
                _disposed = true;
            }
        }

        #endregion

        #region Private Methods

        private void InitializeAcceleratorProperties()
        {
            // Set NPU-specific properties
            Name = $"Intel NPU ({_capabilities.DeviceName})";
            MaxGridSize = new Index3D(_capabilities.MaxInputWidth, _capabilities.MaxInputHeight, 1);
            MaxGroupSize = new Index3D(_capabilities.OptimalBatchSize, 1, 1);
            WarpSize = _capabilities.OptimalBatchSize;
            NumMultiprocessors = _capabilities.NumComputeUnits;
            MaxSharedMemoryPerMultiprocessor = _capabilities.MaxSharedMemoryPerUnit;
            MaxConstantMemory = _capabilities.MaxConstantMemory;
            MaxMemoryBandwidth = _capabilities.MemoryBandwidth;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(NPUAccelerator));
        }

        #endregion

        #region Static Methods

        /// <summary>
        /// Checks if Intel NPU is available on this system.
        /// </summary>
        /// <returns>True if NPU is available; otherwise, false.</returns>
        public static bool IsAvailable()
        {
            try
            {
                return RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && 
                       NPUNative.IsNPUAvailable();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Creates an NPU accelerator if available.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>NPU accelerator or null if not available.</returns>
        public static NPUAccelerator? CreateIfAvailable(Context context)
        {
            if (!IsAvailable()) return null;
            
            try
            {
                var device = new Device(
                    "Intel NPU",
                    0,
                    AcceleratorType.CPU); // NPU is CPU-adjacent
                    
                return new NPUAccelerator(context, device);
            }
            catch
            {
                return null;
            }
        }

        #endregion
    }

    /// <summary>
    /// Intel NPU stream implementation.
    /// </summary>
    public sealed class NPUStream : AcceleratorStream
    {
        /// <summary>
        /// Initializes a new instance of the NPUStream class.
        /// </summary>
        /// <param name="accelerator">The NPU accelerator.</param>
        public NPUStream(NPUAccelerator accelerator) : base(accelerator)
        {
        }

        /// <summary>
        /// Synchronizes the NPU stream.
        /// </summary>
        public override void Synchronize()
        {
            ((NPUAccelerator)Accelerator).ContextHandle.Let(handle =>
                NPUNative.Synchronize(handle));
        }

        /// <summary>
        /// Synchronizes the NPU stream asynchronously.
        /// </summary>
        public override Task SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            return Task.Run(() => Synchronize(), cancellationToken);
        }

        /// <summary>
        /// Disposes the NPU stream.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No specific cleanup needed for NPU streams
        }
    }

    /// <summary>
    /// Intel NPU memory buffer implementation.
    /// </summary>
    public sealed class NPUBuffer : MemoryBuffer
    {
        private readonly IntPtr _nativePtr;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the NPUBuffer class.
        /// </summary>
        /// <param name="accelerator">The NPU accelerator.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element.</param>
        public NPUBuffer(NPUAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            var sizeInBytes = length * elementSize;
            _nativePtr = NPUNative.AllocateMemory((ulong)sizeInBytes);
            
            if (_nativePtr == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate NPU buffer memory");
        }

        /// <summary>
        /// Gets the native pointer to the buffer data.
        /// </summary>
        public override unsafe void* NativePtr => (void*)_nativePtr;

        /// <summary>
        /// Copies data from CPU memory to this buffer.
        /// </summary>
        public override unsafe void CopyFromCPU(
            IntPtr source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            var src = (byte*)source + sourceOffset;
            var dst = (byte*)_nativePtr + targetOffset;
            Buffer.MemoryCopy(src, dst, length, length);
        }

        /// <summary>
        /// Copies data from this buffer to CPU memory.
        /// </summary>
        public override unsafe void CopyToCPU(
            IntPtr target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            var src = (byte*)_nativePtr + sourceOffset;
            var dst = (byte*)target + targetOffset;
            Buffer.MemoryCopy(src, dst, length, length);
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        public override unsafe void CopyFrom(
            MemoryBuffer source,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (source is NPUBuffer npuSource)
            {
                var src = (byte*)npuSource._nativePtr + sourceOffset;
                var dst = (byte*)_nativePtr + targetOffset;
                Buffer.MemoryCopy(src, dst, length, length);
            }
            else
            {
                base.CopyFrom(source, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        public override unsafe void CopyTo(
            MemoryBuffer target,
            long sourceOffset,
            long targetOffset,
            long length)
        {
            if (target is NPUBuffer npuTarget)
            {
                var src = (byte*)_nativePtr + sourceOffset;
                var dst = (byte*)npuTarget._nativePtr + targetOffset;
                Buffer.MemoryCopy(src, dst, length, length);
            }
            else
            {
                base.CopyTo(target, sourceOffset, targetOffset, length);
            }
        }

        /// <summary>
        /// Disposes the NPU buffer.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativePtr != IntPtr.Zero)
                {
                    NPUNative.FreeMemory(_nativePtr);
                }
                _disposed = true;
            }
        }
    }
}