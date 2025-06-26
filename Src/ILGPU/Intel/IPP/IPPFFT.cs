// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Numerics;
using System.Runtime.InteropServices;
using ILGPU.Runtime;
using ILGPU.Intel.IPP.Native;

namespace ILGPU.Intel.IPP
{
    /// <summary>
    /// High-performance 1D FFT implementation using Intel IPP.
    /// Supports complex-to-complex transforms with optimized performance.
    /// </summary>
    public sealed class IPPFFT1D : IDisposable
    {
        #region Instance

        private IntPtr _fftSpec = IntPtr.Zero;
        private IntPtr _workBuffer = IntPtr.Zero;
        private readonly int _order;
        private readonly int _length;
        private readonly bool _forward;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new 1D FFT instance.
        /// </summary>
        /// <param name="order">FFT order (log2 of length).</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        public IPPFFT1D(int order, bool forward = true)
        {
            if (order < 1 || order > 30)
                throw new ArgumentOutOfRangeException(nameof(order), "Order must be between 1 and 30");

            _order = order;
            _length = 1 << order;
            _forward = forward;

            Initialize();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the FFT length (number of points).
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Gets the FFT order (log2 of length).
        /// </summary>
        public int Order => _order;

        /// <summary>
        /// Gets whether this is a forward transform.
        /// </summary>
        public bool IsForward => _forward;

        #endregion

        #region Methods

        /// <summary>
        /// Executes the FFT transform.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        public void Execute(MemoryBuffer<Complex> input, MemoryBuffer<Complex> output)
        {
            if (input.Length != _length)
                throw new ArgumentException($"Input buffer length {input.Length} does not match FFT length {_length}");
            
            if (output.Length != _length)
                throw new ArgumentException($"Output buffer length {output.Length} does not match FFT length {_length}");

            IPPNative.IppStatus status;

            if (_forward)
            {
                status = IPPNative.ippsFFTFwd_CToC_32fc(
                    input.NativePtr,
                    output.NativePtr,
                    _fftSpec,
                    _workBuffer);
            }
            else
            {
                status = IPPNative.ippsFFTInv_CToC_32fc(
                    input.NativePtr,
                    output.NativePtr,
                    _fftSpec,
                    _workBuffer);
            }

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"FFT execution failed with status: {status}");
        }

        /// <summary>
        /// Executes the FFT transform in-place.
        /// </summary>
        /// <param name="data">Input/output complex data buffer.</param>
        public void ExecuteInPlace(MemoryBuffer<Complex> data)
        {
            Execute(data, data);
        }

        private void Initialize()
        {
            var hint = IPPCapabilities.GetOptimalHint();
            var flag = 8; // IPP_FFT_NODIV_BY_ANY

            // Get required buffer sizes
            var status = IPPNative.ippsFFTGetSize_C_32fc(
                _order,
                flag,
                hint,
                out int specSize,
                out int specBufferSize,
                out int workBufferSize);

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Failed to get FFT sizes: {status}");

            // Allocate specification memory
            var specMem = IPPNative.ippsMalloc_32f(specSize / sizeof(float));
            if (specMem == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate FFT specification memory");

            IntPtr specBuffer = IntPtr.Zero;
            if (specBufferSize > 0)
            {
                specBuffer = IPPNative.ippsMalloc_32f(specBufferSize / sizeof(float));
                if (specBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate FFT specification buffer");
                }
            }

            // Initialize FFT specification
            status = IPPNative.ippsFFTInit_C_32fc(
                out _fftSpec,
                _order,
                flag,
                hint,
                specMem,
                specBuffer);

            // Free temporary buffers
            if (specBuffer != IntPtr.Zero)
                IPPNative.ippsFree(specBuffer);

            if (status != IPPNative.IppStatus.ippStsNoErr)
            {
                IPPNative.ippsFree(specMem);
                throw new InvalidOperationException($"Failed to initialize FFT: {status}");
            }

            // Allocate work buffer
            if (workBufferSize > 0)
            {
                _workBuffer = IPPNative.ippsMalloc_32f(workBufferSize / sizeof(float));
                if (_workBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate FFT work buffer");
                }
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes the FFT instance and frees associated resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_workBuffer != IntPtr.Zero)
                {
                    IPPNative.ippsFree(_workBuffer);
                    _workBuffer = IntPtr.Zero;
                }

                if (_fftSpec != IntPtr.Zero)
                {
                    // Note: _fftSpec points to the beginning of the allocated spec memory
                    // It will be freed when the spec memory is freed
                    _fftSpec = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// High-performance 1D real FFT implementation using Intel IPP.
    /// Supports real-to-complex and complex-to-real transforms.
    /// </summary>
    public sealed class IPPFFTReal : IDisposable
    {
        #region Instance

        private IntPtr _fftSpec = IntPtr.Zero;
        private IntPtr _workBuffer = IntPtr.Zero;
        private readonly int _order;
        private readonly int _length;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new 1D real FFT instance.
        /// </summary>
        /// <param name="order">FFT order (log2 of length).</param>
        public IPPFFTReal(int order)
        {
            if (order < 1 || order > 30)
                throw new ArgumentOutOfRangeException(nameof(order), "Order must be between 1 and 30");

            _order = order;
            _length = 1 << order;

            Initialize();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the FFT length (number of real points).
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Gets the FFT order (log2 of length).
        /// </summary>
        public int Order => _order;

        #endregion

        #region Methods

        /// <summary>
        /// Executes a forward real-to-complex FFT.
        /// </summary>
        /// <param name="input">Input real data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        public void ExecuteForward(MemoryBuffer<float> input, MemoryBuffer<Complex> output)
        {
            if (input.Length != _length)
                throw new ArgumentException($"Input buffer length {input.Length} does not match FFT length {_length}");
            
            if (output.Length < _length / 2 + 1)
                throw new ArgumentException($"Output buffer too small: {output.Length} < {_length / 2 + 1}");

            var status = IPPNative.ippsFFTFwd_RToPack_32f(
                input.NativePtr,
                output.NativePtr,
                _fftSpec,
                _workBuffer);

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Real FFT execution failed with status: {status}");
        }

        /// <summary>
        /// Executes an inverse complex-to-real FFT.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output real data buffer.</param>
        public void ExecuteInverse(MemoryBuffer<Complex> input, MemoryBuffer<float> output)
        {
            if (input.Length < _length / 2 + 1)
                throw new ArgumentException($"Input buffer too small: {input.Length} < {_length / 2 + 1}");
            
            if (output.Length != _length)
                throw new ArgumentException($"Output buffer length {output.Length} does not match FFT length {_length}");

            var status = IPPNative.ippsFFTInv_PackToR_32f(
                input.NativePtr,
                output.NativePtr,
                _fftSpec,
                _workBuffer);

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Inverse real FFT execution failed with status: {status}");
        }

        private void Initialize()
        {
            var hint = IPPCapabilities.GetOptimalHint();
            var flag = 8; // IPP_FFT_NODIV_BY_ANY

            // Get required buffer sizes
            var status = IPPNative.ippsFFTGetSize_R_32f(
                _order,
                flag,
                hint,
                out int specSize,
                out int specBufferSize,
                out int workBufferSize);

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Failed to get real FFT sizes: {status}");

            // Allocate specification memory
            var specMem = IPPNative.ippsMalloc_32f(specSize / sizeof(float));
            if (specMem == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate real FFT specification memory");

            IntPtr specBuffer = IntPtr.Zero;
            if (specBufferSize > 0)
            {
                specBuffer = IPPNative.ippsMalloc_32f(specBufferSize / sizeof(float));
                if (specBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate real FFT specification buffer");
                }
            }

            // Initialize FFT specification
            status = IPPNative.ippsFFTInit_R_32f(
                out _fftSpec,
                _order,
                flag,
                hint,
                specMem,
                specBuffer);

            // Free temporary buffers
            if (specBuffer != IntPtr.Zero)
                IPPNative.ippsFree(specBuffer);

            if (status != IPPNative.IppStatus.ippStsNoErr)
            {
                IPPNative.ippsFree(specMem);
                throw new InvalidOperationException($"Failed to initialize real FFT: {status}");
            }

            // Allocate work buffer
            if (workBufferSize > 0)
            {
                _workBuffer = IPPNative.ippsMalloc_32f(workBufferSize / sizeof(float));
                if (_workBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate real FFT work buffer");
                }
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes the real FFT instance and frees associated resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_workBuffer != IntPtr.Zero)
                {
                    IPPNative.ippsFree(_workBuffer);
                    _workBuffer = IntPtr.Zero;
                }

                if (_fftSpec != IntPtr.Zero)
                {
                    _fftSpec = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        #endregion
    }

    /// <summary>
    /// High-performance 2D FFT implementation using Intel IPP.
    /// Supports complex-to-complex 2D transforms.
    /// </summary>
    public sealed class IPPFFT2D : IDisposable
    {
        #region Instance

        private IntPtr _fftSpec = IntPtr.Zero;
        private IntPtr _workBuffer = IntPtr.Zero;
        private readonly Index2D _size;
        private readonly bool _forward;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new 2D FFT instance.
        /// </summary>
        /// <param name="size">2D size of the FFT.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        public IPPFFT2D(Index2D size, bool forward = true)
        {
            if (size.X < 1 || size.Y < 1)
                throw new ArgumentOutOfRangeException(nameof(size), "Size dimensions must be positive");

            _size = size;
            _forward = forward;

            Initialize();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the 2D size of the FFT.
        /// </summary>
        public Index2D Size => _size;

        /// <summary>
        /// Gets whether this is a forward transform.
        /// </summary>
        public bool IsForward => _forward;

        #endregion

        #region Methods

        /// <summary>
        /// Executes the 2D FFT transform.
        /// </summary>
        /// <param name="input">Input 2D complex data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        public void Execute(
            MemoryBuffer<Complex, Index2D> input,
            MemoryBuffer<Complex, Index2D> output)
        {
            if (input.Extent != _size)
                throw new ArgumentException($"Input buffer size {input.Extent} does not match FFT size {_size}");
            
            if (output.Extent != _size)
                throw new ArgumentException($"Output buffer size {output.Extent} does not match FFT size {_size}");

            var srcStep = _size.X * Marshal.SizeOf<Complex>();
            var dstStep = _size.X * Marshal.SizeOf<Complex>();

            IPPNative.IppStatus status;

            if (_forward)
            {
                status = IPPNative.ippiFFTFwd_CToC_32fc_C1R(
                    input.NativePtr,
                    srcStep,
                    output.NativePtr,
                    dstStep,
                    _fftSpec,
                    _workBuffer);
            }
            else
            {
                status = IPPNative.ippiFFTInv_CToC_32fc_C1R(
                    input.NativePtr,
                    srcStep,
                    output.NativePtr,
                    dstStep,
                    _fftSpec,
                    _workBuffer);
            }

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"2D FFT execution failed with status: {status}");
        }

        /// <summary>
        /// Executes the 2D FFT transform in-place.
        /// </summary>
        /// <param name="data">Input/output 2D complex data buffer.</param>
        public void ExecuteInPlace(MemoryBuffer<Complex, Index2D> data)
        {
            Execute(data, data);
        }

        private void Initialize()
        {
            var hint = IPPCapabilities.GetOptimalHint();
            var flag = 8; // IPP_FFT_NODIV_BY_ANY
            var roiSize = new IPPNative.IppiSize(_size.X, _size.Y);

            // Get required buffer sizes
            var status = IPPNative.ippiFFTGetSize_C_32fc(
                roiSize,
                flag,
                hint,
                out int specSize,
                out int initBufferSize,
                out int workBufferSize);

            if (status != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"Failed to get 2D FFT sizes: {status}");

            // Allocate specification memory
            var specMem = IPPNative.ippsMalloc_32f(specSize / sizeof(float));
            if (specMem == IntPtr.Zero)
                throw new OutOfMemoryException("Failed to allocate 2D FFT specification memory");

            IntPtr initBuffer = IntPtr.Zero;
            if (initBufferSize > 0)
            {
                initBuffer = IPPNative.ippsMalloc_32f(initBufferSize / sizeof(float));
                if (initBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate 2D FFT initialization buffer");
                }
            }

            // Initialize FFT specification
            status = IPPNative.ippiFFTInit_C_32fc(
                out _fftSpec,
                roiSize,
                flag,
                hint,
                specMem,
                initBuffer);

            // Free temporary buffers
            if (initBuffer != IntPtr.Zero)
                IPPNative.ippsFree(initBuffer);

            if (status != IPPNative.IppStatus.ippStsNoErr)
            {
                IPPNative.ippsFree(specMem);
                throw new InvalidOperationException($"Failed to initialize 2D FFT: {status}");
            }

            // Allocate work buffer
            if (workBufferSize > 0)
            {
                _workBuffer = IPPNative.ippsMalloc_32f(workBufferSize / sizeof(float));
                if (_workBuffer == IntPtr.Zero)
                {
                    IPPNative.ippsFree(specMem);
                    throw new OutOfMemoryException("Failed to allocate 2D FFT work buffer");
                }
            }
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes the 2D FFT instance and frees associated resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_workBuffer != IntPtr.Zero)
                {
                    IPPNative.ippsFree(_workBuffer);
                    _workBuffer = IntPtr.Zero;
                }

                if (_fftSpec != IntPtr.Zero)
                {
                    _fftSpec = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        #endregion
    }
}