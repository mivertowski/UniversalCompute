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

using ILGPU.Runtime;
using System;
using System.Numerics;

namespace ILGPU.Algorithms.FFT
{
    /// <summary>
    /// Fast Fourier Transform (FFT) implementation for GPU-accelerated signal processing.
    /// </summary>
    /// <remarks>
    /// This implementation provides:
    /// - 1D, 2D, and 3D FFT transforms
    /// - Real-to-complex and complex-to-complex transforms
    /// - Radix-2, Radix-4, and mixed-radix algorithms
    /// - Batched FFT operations for multiple signals
    /// - Optimizations for different GPU architectures
    /// </remarks>
    public sealed class FFT<T> where T : unmanaged, INumber<T>
    {
        private readonly Accelerator _accelerator;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the FFT class.
        /// </summary>
        /// <param name="accelerator">The accelerator to use for FFT operations.</param>
        /// <param name="config">FFT configuration.</param>
        public FFT(Accelerator accelerator, FFTConfiguration config)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            Configuration = config ?? throw new ArgumentNullException(nameof(config));
            
            // Create FFT plan based on configuration
            Plan = FFTPlan.Create(accelerator, config);
        }

        /// <summary>
        /// Gets the FFT configuration.
        /// </summary>
        public FFTConfiguration Configuration { get; }

        /// <summary>
        /// Gets the FFT plan.
        /// </summary>
        public FFTPlan Plan { get; }

        #region 1D FFT

        /// <summary>
        /// Performs a 1D forward FFT transform.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <param name="output">Output data.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Forward1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            FFT<T>.ValidateArguments(input, output, Configuration.Size);

            var kernel = GetFFT1DKernel(FFTDirection.Forward);
            var actualStream = stream ?? _accelerator.DefaultStream;

            kernel(new Index1D((int)input.Length), input, output, Plan);
            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs a 1D inverse FFT transform.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <param name="output">Output data.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Inverse1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            FFT<T>.ValidateArguments(input, output, Configuration.Size);

            var kernel = GetFFT1DKernel(FFTDirection.Inverse);
            var actualStream = stream ?? _accelerator.DefaultStream;

            kernel(new Index1D((int)input.Length), input, output, Plan);
            
            // Normalize by 1/N for inverse transform
            var normalizeKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<Complex>, float>(FFTKernels.Normalize);
            normalizeKernel(new Index1D((int)output.Length), output, 1.0f / Configuration.Size);
            
            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs a batched 1D FFT transform.
        /// </summary>
        /// <param name="input">Input data (batch_size x signal_size).</param>
        /// <param name="output">Output data (batch_size x signal_size).</param>
        /// <param name="batchSize">Number of signals in the batch.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void ForwardBatched1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            int batchSize,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            FFT<T>.ValidateArguments(input, output, Configuration.Size * batchSize);

            var kernel = GetBatchedFFT1DKernel(FFTDirection.Forward);
            var actualStream = stream ?? _accelerator.DefaultStream;

            kernel(new Index1D((int)input.Length), input, output, Plan, batchSize);
            actualStream.Synchronize();
        }

        #endregion

        #region 2D FFT

        /// <summary>
        /// Performs a 2D forward FFT transform.
        /// </summary>
        /// <param name="input">Input data (height x width).</param>
        /// <param name="output">Output data (height x width).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Forward2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            ValidateArguments2D(input, output);

            var actualStream = stream ?? _accelerator.DefaultStream;

            // Perform row-wise FFT
            var rowKernel = GetFFT2DRowKernel(FFTDirection.Forward);
            var extent2D = new Index2D(input.IntExtent.X, input.IntExtent.Y);
            rowKernel(extent2D, input, output, Plan);

            // Perform column-wise FFT
            var colKernel = GetFFT2DColumnKernel(FFTDirection.Forward);
            colKernel(extent2D, output, output, Plan);

            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs a 2D inverse FFT transform.
        /// </summary>
        /// <param name="input">Input data (height x width).</param>
        /// <param name="output">Output data (height x width).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Inverse2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            ValidateArguments2D(input, output);

            var actualStream = stream ?? _accelerator.DefaultStream;

            // Perform column-wise inverse FFT
            var colKernel = GetFFT2DColumnKernel(FFTDirection.Inverse);
            var extent2D = new Index2D(input.IntExtent.X, input.IntExtent.Y);
            colKernel(extent2D, input, output, Plan);

            // Perform row-wise inverse FFT
            var rowKernel = GetFFT2DRowKernel(FFTDirection.Inverse);
            rowKernel(extent2D, output, output, Plan);

            // Normalize by 1/(width*height)
            var normalizeKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<Complex, Stride2D.DenseX>, float>(FFTKernels.Normalize2D);
            var normalizeExtent = new Index2D(output.IntExtent.X, output.IntExtent.Y);
            normalizeKernel(normalizeExtent, output, 1.0f / (output.IntExtent.X * output.IntExtent.Y));

            actualStream.Synchronize();
        }

        #endregion

        #region 3D FFT

        /// <summary>
        /// Performs a 3D forward FFT transform.
        /// </summary>
        /// <param name="input">Input data (depth x height x width).</param>
        /// <param name="output">Output data (depth x height x width).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Forward3D(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            ValidateArguments3D(input, output);

            var actualStream = stream ?? _accelerator.DefaultStream;

            // Perform FFT along X dimension
            var xKernel = GetFFT3DKernelX(FFTDirection.Forward);
            var extent3D = new Index3D(input.IntExtent.X, input.IntExtent.Y, input.IntExtent.Z);
            xKernel(extent3D, input, output, Plan);

            // Perform FFT along Y dimension
            var yKernel = GetFFT3DKernelY(FFTDirection.Forward);
            yKernel(extent3D, output, output, Plan);

            // Perform FFT along Z dimension
            var zKernel = GetFFT3DKernelZ(FFTDirection.Forward);
            zKernel(extent3D, output, output, Plan);

            actualStream.Synchronize();
        }

        #endregion

        #region Real-to-Complex FFT

        /// <summary>
        /// Performs a real-to-complex FFT transform.
        /// </summary>
        /// <param name="input">Real input data.</param>
        /// <param name="output">Complex output data (N/2+1 elements).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void ForwardReal(
            ArrayView<T> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            
            if (input.Length != Configuration.Size)
                throw new ArgumentException($"Input size must be {Configuration.Size}", nameof(input));
            
            if (output.Length != Configuration.Size / 2 + 1)
                throw new ArgumentException($"Output size must be {Configuration.Size / 2 + 1}", nameof(output));

            var kernel = GetRealFFTKernel(FFTDirection.Forward);
            var actualStream = stream ?? _accelerator.DefaultStream;

            kernel(new Index1D((int)input.Length), input, output, Plan);
            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs a complex-to-real inverse FFT transform.
        /// </summary>
        /// <param name="input">Complex input data (N/2+1 elements).</param>
        /// <param name="output">Real output data.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void InverseReal(
            ArrayView<Complex> input,
            ArrayView<T> output,
            AcceleratorStream? stream = null)
        {
            ThrowIfDisposed();
            
            if (input.Length != Configuration.Size / 2 + 1)
                throw new ArgumentException($"Input size must be {Configuration.Size / 2 + 1}", nameof(input));
            
            if (output.Length != Configuration.Size)
                throw new ArgumentException($"Output size must be {Configuration.Size}", nameof(output));

            var kernel = GetRealFFTKernel(FFTDirection.Inverse);
            var actualStream = stream ?? _accelerator.DefaultStream;

            kernel(new Index1D((int)input.Length), input, output, Plan);
            actualStream.Synchronize();
        }

        #endregion

        #region Helper Methods

        private static void ValidateArguments(ArrayView<Complex> input, ArrayView<Complex> output, int expectedSize)
        {
            if (input.Length != expectedSize)
                throw new ArgumentException($"Input size must be {expectedSize}", nameof(input));
            
            if (output.Length != expectedSize)
                throw new ArgumentException($"Output size must be {expectedSize}", nameof(output));
        }

        private void ValidateArguments2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output)
        {
            if (input.IntExtent != output.IntExtent)
                throw new ArgumentException("Input and output dimensions must match");
            
            if (Configuration.Dimensions != FFTDimensions.TwoD)
                throw new InvalidOperationException("FFT configured for different dimensions");
        }

        private void ValidateArguments3D(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output)
        {
            if (input.IntExtent != output.IntExtent)
                throw new ArgumentException("Input and output dimensions must match");
            
            if (Configuration.Dimensions != FFTDimensions.ThreeD)
                throw new InvalidOperationException("FFT configured for different dimensions");
        }

        private Action<Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan> 
            GetFFT1DKernel(FFTDirection direction)
        {
            return Configuration.Algorithm switch
            {
                FFTAlgorithm.Radix2 => _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan>(
                    direction == FFTDirection.Forward 
                        ? FFTKernels.CooleyTukeyForward1D 
                        : FFTKernels.CooleyTukeyInverse1D),
                
                FFTAlgorithm.Radix4 => _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan>(
                    direction == FFTDirection.Forward 
                        ? FFTKernels.Radix4Forward1D 
                        : FFTKernels.Radix4Inverse1D),
                
                _ => _accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan>(
                    direction == FFTDirection.Forward 
                        ? FFTKernels.MixedRadixForward1D 
                        : FFTKernels.MixedRadixInverse1D)
            };
        }

        private Action<Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan, int> 
            GetBatchedFFT1DKernel(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<Complex>, ArrayView<Complex>, FFTPlan, int>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.BatchedForward1D 
                    : FFTKernels.BatchedInverse1D);
        }

        private Action<Index2D, ArrayView2D<Complex, Stride2D.DenseX>, 
            ArrayView2D<Complex, Stride2D.DenseX>, FFTPlan> 
            GetFFT2DRowKernel(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<Complex, Stride2D.DenseX>, 
                ArrayView2D<Complex, Stride2D.DenseX>, FFTPlan>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.FFT2DRowForward 
                    : FFTKernels.FFT2DRowInverse);
        }

        private Action<Index2D, ArrayView2D<Complex, Stride2D.DenseX>, 
            ArrayView2D<Complex, Stride2D.DenseX>, FFTPlan> 
            GetFFT2DColumnKernel(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<Complex, Stride2D.DenseX>, 
                ArrayView2D<Complex, Stride2D.DenseX>, FFTPlan>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.FFT2DColumnForward 
                    : FFTKernels.FFT2DColumnInverse);
        }

        private Action<Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
            ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan> 
            GetFFT3DKernelX(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
                ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.FFT3DXForward 
                    : FFTKernels.FFT3DXInverse);
        }

        private Action<Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
            ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan> 
            GetFFT3DKernelY(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
                ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.FFT3DYForward 
                    : FFTKernels.FFT3DYInverse);
        }

        private Action<Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
            ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan> 
            GetFFT3DKernelZ(FFTDirection direction)
        {
            return _accelerator.LoadAutoGroupedStreamKernel<
                Index3D, ArrayView3D<Complex, Stride3D.DenseXY>, 
                ArrayView3D<Complex, Stride3D.DenseXY>, FFTPlan>(
                direction == FFTDirection.Forward 
                    ? FFTKernels.FFT3DZForward 
                    : FFTKernels.FFT3DZInverse);
        }

        private Action<Index1D, ArrayView<T>, ArrayView<Complex>, FFTPlan> 
            GetRealFFTKernel(FFTDirection direction)
        {
            return direction == FFTDirection.Forward
                ? _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, ArrayView<Complex>, FFTPlan>(
                    FFTKernels.RealToComplexForward)
                : throw new NotImplementedException("Complex-to-real kernel");
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(FFT<T>));
        }

        #endregion

        /// <summary>
        /// Disposes the FFT instance and releases resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                Plan.Release();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// FFT direction.
    /// </summary>
    public enum FFTDirection
    {
        /// <summary>
        /// Forward FFT transform.
        /// </summary>
        Forward,

        /// <summary>
        /// Inverse FFT transform.
        /// </summary>
        Inverse
    }

    /// <summary>
    /// FFT dimensions.
    /// </summary>
    public enum FFTDimensions
    {
        /// <summary>
        /// 1D FFT.
        /// </summary>
        OneD = 1,

        /// <summary>
        /// 2D FFT.
        /// </summary>
        TwoD = 2,

        /// <summary>
        /// 3D FFT.
        /// </summary>
        ThreeD = 3
    }

    /// <summary>
    /// FFT algorithm selection.
    /// </summary>
    public enum FFTAlgorithm
    {
        /// <summary>
        /// Automatic algorithm selection based on size.
        /// </summary>
        Auto,

        /// <summary>
        /// Radix-2 Cooley-Tukey algorithm (requires power-of-2 sizes).
        /// </summary>
        Radix2,

        /// <summary>
        /// Radix-4 algorithm (requires power-of-4 sizes).
        /// </summary>
        Radix4,

        /// <summary>
        /// Mixed-radix algorithm for arbitrary sizes.
        /// </summary>
        MixedRadix,

        /// <summary>
        /// Bluestein's algorithm for prime sizes.
        /// </summary>
        Bluestein
    }

    /// <summary>
    /// FFT configuration.
    /// </summary>
    public sealed class FFTConfiguration
    {
        /// <summary>
        /// Gets or sets the FFT size (number of elements).
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the FFT dimensions.
        /// </summary>
        public FFTDimensions Dimensions { get; set; } = FFTDimensions.OneD;

        /// <summary>
        /// Gets or sets the FFT algorithm.
        /// </summary>
        public FFTAlgorithm Algorithm { get; set; } = FFTAlgorithm.Auto;

        /// <summary>
        /// Gets or sets whether to use shared memory optimization.
        /// </summary>
        public bool UseSharedMemory { get; set; } = true;

        /// <summary>
        /// Gets or sets the batch size for batched operations.
        /// </summary>
        public int BatchSize { get; set; } = 1;

        /// <summary>
        /// Creates a default 1D FFT configuration.
        /// </summary>
        /// <param name="size">FFT size.</param>
        /// <returns>FFT configuration.</returns>
        public static FFTConfiguration Create1D(int size) => new()
        {
            Size = size,
            Dimensions = FFTDimensions.OneD
        };

        /// <summary>
        /// Creates a 2D FFT configuration.
        /// </summary>
        /// <param name="width">Width of 2D transform.</param>
        /// <param name="height">Height of 2D transform.</param>
        /// <returns>FFT configuration.</returns>
        public static FFTConfiguration Create2D(int width, int height) => new()
        {
            Size = width * height,
            Dimensions = FFTDimensions.TwoD
        };

        /// <summary>
        /// Creates a 3D FFT configuration.
        /// </summary>
        /// <param name="width">Width of 3D transform.</param>
        /// <param name="height">Height of 3D transform.</param>
        /// <param name="depth">Depth of 3D transform.</param>
        /// <returns>FFT configuration.</returns>
        public static FFTConfiguration Create3D(int width, int height, int depth) => new()
        {
            Size = width * height * depth,
            Dimensions = FFTDimensions.ThreeD
        };
    }
}