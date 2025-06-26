// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Runtime.CompilerServices;
using ILGPU.Runtime;
using ILGPU.Intel.IPP.Native;

namespace ILGPU.Intel.IPP
{
    /// <summary>
    /// Intel Integrated Performance Primitives (IPP) accelerator for high-performance CPU-based computations.
    /// Specializes in signal processing, image processing, and mathematical operations including FFT.
    /// </summary>
    public sealed class IPPAccelerator : Accelerator
    {
        #region Instance

        private readonly IPPInfo _capabilities;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new IPP accelerator instance.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        internal IPPAccelerator(Context context)
            : base(context, AcceleratorType.CPU)
        {
            if (!IPPCapabilities.DetectIPP())
                throw new NotSupportedException("Intel IPP is not available on this system");

            _capabilities = IPPCapabilities.Query();
            
            if (!_capabilities.IsAvailable)
                throw new InvalidOperationException("Intel IPP is not functional");

            // Initialize IPP library
            var status = IPPNative.ippInit();
            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Failed to initialize Intel IPP: {status}");

            Init();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the accelerator type (CPU).
        /// </summary>
        public override AcceleratorType AcceleratorType => AcceleratorType.CPU;

        /// <summary>
        /// Returns the name of this accelerator.
        /// </summary>
        public override string Name => "Intel IPP CPU Accelerator";

        /// <summary>
        /// Returns the memory size (system RAM).
        /// </summary>
        public override long MemorySize => GC.GetTotalMemory(false);

        /// <summary>
        /// Returns the maximum grid size for kernel launches.
        /// </summary>
        public override Index3D MaxGridSize => new Index3D(int.MaxValue, int.MaxValue, int.MaxValue);

        /// <summary>
        /// Returns the maximum group size for kernel launches.
        /// </summary>
        public override Index3D MaxGroupSize => new Index3D(1024, 1024, 1024);

        /// <summary>
        /// Returns the warp size (always 1 for CPU).
        /// </summary>
        public override int WarpSize => 1;

        /// <summary>
        /// Returns the number of multiprocessors (logical CPU cores).
        /// </summary>
        public override int NumMultiprocessors => Environment.ProcessorCount;

        /// <summary>
        /// Returns the maximum number of threads per multiprocessor.
        /// </summary>
        public override int MaxNumThreadsPerMultiprocessor => IPPCapabilities.GetRecommendedThreadCount();

        /// <summary>
        /// Returns the maximum shared memory per group in bytes.
        /// </summary>
        public override long MaxSharedMemoryPerGroup => 1024 * 1024; // 1MB - reasonable limit for CPU

        /// <summary>
        /// Returns the maximum constant memory in bytes.
        /// </summary>
        public override long MaxConstantMemory => long.MaxValue; // No practical limit on CPU

        /// <summary>
        /// Gets the IPP capabilities of this accelerator.
        /// </summary>
        public IPPInfo Capabilities => _capabilities;

        #endregion

        #region Methods

        /// <summary>
        /// Waits for all operations to complete (no-op for CPU).
        /// </summary>
        public override void Synchronize()
        {
            // CPU operations are inherently synchronous
        }

        /// <summary>
        /// Creates a new accelerator stream for asynchronous operations.
        /// </summary>
        /// <returns>A new accelerator stream.</returns>
        protected override AcceleratorStream CreateStreamInternal()
        {
            return new IPPAcceleratorStream(this);
        }

        /// <summary>
        /// Allocates memory on this accelerator.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="extent">The extent (number of elements).</param>
        /// <returns>An allocated memory buffer.</returns>
        protected override MemoryBuffer<T, Index1D> AllocateInternal<T>(Index1D extent)
        {
            return new IPPMemoryBuffer<T>(this, extent);
        }

        /// <summary>
        /// Allocates 2D memory on this accelerator.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="extent">The 2D extent.</param>
        /// <returns>An allocated 2D memory buffer.</returns>
        protected override MemoryBuffer<T, Index2D> AllocateInternal<T>(Index2D extent)
        {
            return new IPPMemoryBuffer<T, Index2D>(this, extent);
        }

        /// <summary>
        /// Allocates 3D memory on this accelerator.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="extent">The 3D extent.</param>
        /// <returns>An allocated 3D memory buffer.</returns>
        protected override MemoryBuffer<T, Index3D> AllocateInternal<T>(Index3D extent)
        {
            return new IPPMemoryBuffer<T, Index3D>(this, extent);
        }

        /// <summary>
        /// Creates a page-locked memory scope for efficient data transfers.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="hostPtr">Host memory pointer.</param>
        /// <param name="numElements">Number of elements.</param>
        /// <returns>A page-locked memory scope.</returns>
        protected override PageLockScope<T> CreatePageLockFromPinnedInternal<T>(
            IntPtr hostPtr, Index1D numElements)
        {
            // CPU memory is already accessible, return a null scope
            return new NullPageLockScope<T>(this, hostPtr, numElements);
        }

        /// <summary>
        /// Gets compute capability information.
        /// </summary>
        /// <param name="instruction">The instruction to query.</param>
        /// <returns>Compute capability information.</returns>
        protected override ComputeCapability GetComputeCapabilityInternal(ComputeCapabilityInstruction instruction)
        {
            // Return CPU compute capability based on IPP features
            var major = _capabilities.SupportsAVX512 ? 8 : _capabilities.SupportsAVX2 ? 6 : _capabilities.SupportsSSE42 ? 4 : 2;
            var minor = _capabilities.SupportsAVX512 ? 0 : _capabilities.SupportsAVX2 ? 2 : _capabilities.SupportsSSE42 ? 2 : 0;
            
            return new ComputeCapability(major, minor);
        }

        #endregion

        #region FFT Operations

        /// <summary>
        /// Performs a 1D complex-to-complex FFT using Intel IPP.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void FFT1D(
            MemoryBuffer<System.Numerics.Complex> input,
            MemoryBuffer<System.Numerics.Complex> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output buffers must have the same length");

            var length = input.Length;
            var order = (int)Math.Log2(length);
            
            if ((1 << order) != length)
                throw new ArgumentException("FFT length must be a power of 2");

            var fft = new IPPFFT1D(order, forward);
            try
            {
                fft.Execute(input, output);
            }
            finally
            {
                fft.Dispose();
            }
        }

        /// <summary>
        /// Performs a 1D real-to-complex FFT using Intel IPP.
        /// </summary>
        /// <param name="input">Input real data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void FFT1DReal(
            MemoryBuffer<float> input,
            MemoryBuffer<System.Numerics.Complex> output,
            AcceleratorStream? stream = null)
        {
            var length = input.Length;
            var order = (int)Math.Log2(length);
            
            if ((1 << order) != length)
                throw new ArgumentException("FFT length must be a power of 2");

            if (output.Length < length / 2 + 1)
                throw new ArgumentException("Output buffer too small for real FFT");

            var fft = new IPPFFTReal(order);
            try
            {
                fft.ExecuteForward(input, output);
            }
            finally
            {
                fft.Dispose();
            }
        }

        /// <summary>
        /// Performs a 2D complex-to-complex FFT using Intel IPP.
        /// </summary>
        /// <param name="input">Input 2D complex data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void FFT2D(
            MemoryBuffer<System.Numerics.Complex, Index2D> input,
            MemoryBuffer<System.Numerics.Complex, Index2D> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (input.Extent != output.Extent)
                throw new ArgumentException("Input and output buffers must have the same dimensions");

            var extent = input.Extent;
            var fft = new IPPFFT2D(extent, forward);
            try
            {
                fft.Execute(input, output);
            }
            finally
            {
                fft.Dispose();
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this accelerator and frees associated resources.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected override void DisposeAccelerator_SyncRoot(bool disposing)
        {
            if (!_disposed && disposing)
            {
                // IPP cleanup is handled by individual FFT objects
                _disposed = true;
            }
            base.DisposeAccelerator_SyncRoot(disposing);
        }

        #endregion
    }

    /// <summary>
    /// IPP accelerator stream for managing asynchronous operations.
    /// </summary>
    internal sealed class IPPAcceleratorStream : AcceleratorStream
    {
        /// <summary>
        /// Constructs a new IPP accelerator stream.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        public IPPAcceleratorStream(Accelerator accelerator)
            : base(accelerator)
        {
        }

        /// <summary>
        /// Synchronizes the stream (no-op for CPU operations).
        /// </summary>
        public override void Synchronize()
        {
            // CPU operations are inherently synchronous
        }

        /// <summary>
        /// Disposes the stream.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No specific cleanup required for IPP streams
        }
    }
}