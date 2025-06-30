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
using ILGPU.Runtime;
using ILGPU.IR.Analyses;
using System;
using System.Collections.Immutable;
using System.Threading.Tasks;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Apple Neural Engine accelerator implementation for ILGPU.
    /// </summary>
    public sealed class AppleNeuralEngineAccelerator : Accelerator
    {
        private readonly AppleNeuralEngine _neuralEngine;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AppleNeuralEngineAccelerator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="device">The ANE device.</param>
        public AppleNeuralEngineAccelerator(Context context, AppleNeuralEngineDevice device)
            : base(context, device)
        {
            // Create the ANE instance with a dummy MetalDevice
            // In a real implementation, this would properly interface with Metal
            _neuralEngine = new AppleNeuralEngine(null);
            ANECapabilities = _neuralEngine.Capabilities;
            
            // Properties are inherited from the device parameter
        }

        /// <summary>
        /// Gets the Neural Engine capabilities.
        /// </summary>
        public ANECapabilities ANECapabilities { get; }


        #region Accelerator Implementation

        protected override AcceleratorStream CreateStreamInternal() => new ANEExecutionContext(this);

        protected override void SynchronizeInternal() =>
            // ANE operations are asynchronous but we can sync here
            System.Threading.Thread.MemoryBarrier();

        protected override MemoryBuffer AllocateRawInternal(long length, int elementSize) => new ANEBuffer(this, length, elementSize);

        protected override Kernel LoadKernelInternal(CompiledKernel compiledKernel) => new ANEKernel(this, compiledKernel);

        protected override Kernel LoadAutoGroupedKernelInternal(
            CompiledKernel compiledKernel,
            out KernelInfo? kernelInfo)
        {
            var allocaInfo = default(AllocaKindInformation);
            kernelInfo = new KernelInfo(0, 0, in allocaInfo, ImmutableArray<CompiledKernel.FunctionInfo>.Empty);
            return LoadKernelInternal(compiledKernel);
        }

        protected override Kernel LoadImplicitlyGroupedKernelInternal(
            CompiledKernel compiledKernel,
            int customGroupSize,
            out KernelInfo? kernelInfo)
        {
            var allocaInfo = default(AllocaKindInformation);
            kernelInfo = new KernelInfo(0, 0, in allocaInfo, ImmutableArray<CompiledKernel.FunctionInfo>.Empty);
            return LoadKernelInternal(compiledKernel);
        }

        protected override int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes) => Math.Max(1, NumMultiprocessors / groupSize);

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, WarpSize);
        }

        protected override int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            minGridSize = 1;
            return Math.Min(maxGroupSize, WarpSize);
        }

        protected override bool CanAccessPeerInternal(Accelerator otherAccelerator) => false;

        protected override void EnablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("ANE does not support peer access");

        protected override void DisablePeerAccessInternal(Accelerator otherAccelerator) => throw new NotSupportedException("ANE does not support peer access");

        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(IntPtr pinned, long numElements) => new NullPageLockScope<T>(this, pinned, numElements);

        public override TExtension CreateExtension<TExtension, TExtensionProvider>(TExtensionProvider provider) => throw new NotSupportedException($"Extension {typeof(TExtension)} not supported by ANE accelerator");

        protected override void OnBind()
        {
            // Initialize ANE when bound
        }

        protected override void OnUnbind()
        {
            // Cleanup ANE when unbound
        }

        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _neuralEngine?.Dispose();
                }
                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// ANE execution context implementation.
    /// </summary>
    public sealed class ANEExecutionContext : AcceleratorStream
    {
        private readonly AppleNeuralEngineAccelerator _accelerator;

        /// <summary>
        /// Initializes a new instance of the ANEExecutionContext class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        public ANEExecutionContext(AppleNeuralEngineAccelerator accelerator)
            : base(accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        /// <summary>
        /// Synchronizes the stream.
        /// </summary>
        public override void Synchronize() => System.Threading.Thread.MemoryBarrier();


        /// <summary>
        /// Adds a profiling marker for ANE operations.
        /// </summary>
        /// <returns>A profiling marker for timing measurements.</returns>
        protected override ProfilingMarker AddProfilingMarkerInternal() => new ANEProfilingMarker(_accelerator);

        /// <summary>
        /// Disposes the ANE stream.
        /// </summary>
        /// <param name="disposing">Whether disposing is in progress.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose for ANE streams
        }
    }

    /// <summary>
    /// ANE buffer implementation.
    /// </summary>
    public sealed class ANEBuffer : MemoryBuffer
    {
        private readonly IntPtr _nativePtr;
        private readonly long _lengthInBytes;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ANEBuffer class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        /// <param name="length">The number of elements.</param>
        /// <param name="elementSize">The size of each element in bytes.</param>
        internal ANEBuffer(AppleNeuralEngineAccelerator accelerator, long length, int elementSize)
            : base(accelerator, length, elementSize)
        {
            _lengthInBytes = length * elementSize;
            _nativePtr = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)_lengthInBytes);
            NativePtr = _nativePtr;
        }


        /// <summary>
        /// Sets memory to a specific value.
        /// </summary>
        protected internal override void MemSet(
            AcceleratorStream stream,
            byte value,
            in ArrayView<byte> targetView)
        {
            unsafe
            {
                var targetPtr = (byte*)_nativePtr + targetView.Index;
                var length = targetView.Length;
                
                for (long i = 0; i < length; i++)
                    targetPtr[i] = value;
            }
        }

        /// <summary>
        /// Copies data from another buffer to this buffer.
        /// </summary>
        protected internal override unsafe void CopyFrom(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            var sourcePtr = (byte*)sourceView.LoadEffectiveAddress();
            var targetPtr = (byte*)_nativePtr + targetView.Index;
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, _lengthInBytes - targetView.Index, targetView.Length);
        }

        /// <summary>
        /// Copies data from this buffer to another buffer.
        /// </summary>
        protected internal override unsafe void CopyTo(
            AcceleratorStream stream,
            in ArrayView<byte> sourceView,
            in ArrayView<byte> targetView)
        {
            var sourcePtr = (byte*)_nativePtr + sourceView.Index;
            var targetPtr = (byte*)targetView.LoadEffectiveAddress();
            
            Buffer.MemoryCopy(sourcePtr, targetPtr, targetView.Length, sourceView.Length);
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
                    System.Runtime.InteropServices.Marshal.FreeHGlobal(_nativePtr);
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// ANE kernel implementation.
    /// </summary>
    public sealed class ANEKernel : Kernel
    {
        private readonly AppleNeuralEngineAccelerator _accelerator;
        private readonly CompiledKernel _compiledKernel;

        /// <summary>
        /// Initializes a new instance of the ANEKernel class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        /// <param name="compiledKernel">The compiled kernel.</param>
        public ANEKernel(AppleNeuralEngineAccelerator accelerator, CompiledKernel compiledKernel)
            : base(accelerator, compiledKernel, null)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _compiledKernel = compiledKernel ?? throw new ArgumentNullException(nameof(compiledKernel));
        }


        /// <summary>
        /// Disposes the ANE kernel.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No special cleanup needed for ANE kernels
        }
    }

    /// <summary>
    /// ANE profiling marker implementation.
    /// </summary>
    internal sealed class ANEProfilingMarker : ProfilingMarker
    {
        private readonly DateTime _timestamp;

        /// <summary>
        /// Initializes a new instance of the ANEProfilingMarker class.
        /// </summary>
        /// <param name="accelerator">The ANE accelerator.</param>
        internal ANEProfilingMarker(Accelerator accelerator) : base(accelerator)
        {
            _timestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Synchronizes the profiling marker.
        /// </summary>
        public override void Synchronize()
        {
            // ANE operations are typically synchronous, no action needed
        }

        /// <summary>
        /// Measures elapsed time from another marker.
        /// </summary>
        /// <param name="marker">The starting marker.</param>
        /// <returns>The elapsed time.</returns>
        public override TimeSpan MeasureFrom(ProfilingMarker marker)
        {
            if (marker is ANEProfilingMarker aneMarker)
                return _timestamp - aneMarker._timestamp;
            throw new ArgumentException("Marker must be an ANE profiling marker", nameof(marker));
        }

        /// <summary>
        /// Disposes the profiling marker.
        /// </summary>
        /// <param name="disposing">Whether disposing is in progress.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose
        }
    }
}
