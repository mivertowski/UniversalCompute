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

// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Numerics;
using ILGPU.Runtime;

namespace ILGPU.FFT
{
    /// <summary>
    /// Abstract base class for FFT accelerators that provide high-performance FFT operations.
    /// </summary>
    /// <remarks>
    /// Constructs a new FFT accelerator.
    /// </remarks>
    /// <param name="parentAccelerator">The parent ILGPU accelerator.</param>
    public abstract class FFTAccelerator(Accelerator parentAccelerator) : IDisposable
    {
        #region Properties

        /// <summary>
        /// Gets the parent ILGPU accelerator.
        /// </summary>
        public Accelerator ParentAccelerator { get; } = parentAccelerator ?? throw new ArgumentNullException(nameof(parentAccelerator));

        /// <summary>
        /// Gets the FFT accelerator type.
        /// </summary>
        public abstract FFTAcceleratorType AcceleratorType { get; }

        /// <summary>
        /// Gets a human-readable name for this FFT accelerator.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Gets whether this FFT accelerator is available and functional.
        /// </summary>
        public abstract bool IsAvailable { get; }

        /// <summary>
        /// Gets the performance characteristics of this FFT accelerator.
        /// </summary>
        public abstract FFTPerformanceInfo PerformanceInfo { get; }

        #endregion
        #region Constructor

        #endregion

        #region 1D FFT Methods

        /// <summary>
        /// Performs a 1D complex-to-complex FFT.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public abstract void FFT1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            bool forward = true,
            AcceleratorStream? stream = null);

        /// <summary>
        /// Performs a 1D real-to-complex FFT.
        /// </summary>
        /// <param name="input">Input real data buffer.</param>
        /// <param name="output">Output complex data buffer (size N/2+1).</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public abstract void FFT1DReal(
            ArrayView<float> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null);

        /// <summary>
        /// Performs a 1D complex-to-real inverse FFT.
        /// </summary>
        /// <param name="input">Input complex data buffer (size N/2+1).</param>
        /// <param name="output">Output real data buffer.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public abstract void IFFT1DReal(
            ArrayView<Complex> input,
            ArrayView<float> output,
            AcceleratorStream? stream = null);

        #endregion

        #region 2D FFT Methods

        /// <summary>
        /// Performs a 2D complex-to-complex FFT.
        /// </summary>
        /// <param name="input">Input 2D complex data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public abstract void FFT2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            bool forward = true,
            AcceleratorStream? stream = null);

        /// <summary>
        /// Performs a 2D real-to-complex FFT.
        /// </summary>
        /// <param name="input">Input 2D real data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public virtual void FFT2DReal(
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            AcceleratorStream? stream = null) => throw new NotSupportedException("2D real FFT is not supported by this accelerator");

        #endregion

        #region 3D FFT Methods

        /// <summary>
        /// Performs a 3D complex-to-complex FFT.
        /// </summary>
        /// <param name="input">Input 3D complex data buffer.</param>
        /// <param name="output">Output 3D complex data buffer.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public virtual void FFT3D(
            ArrayView3D<Complex, Stride3D.DenseXY> input,
            ArrayView3D<Complex, Stride3D.DenseXY> output,
            bool forward = true,
            AcceleratorStream? stream = null) => throw new NotSupportedException("3D FFT is not supported by this accelerator");

        #endregion

        #region Batch FFT Methods

        /// <summary>
        /// Performs multiple 1D FFTs in parallel (batch processing).
        /// </summary>
        /// <param name="inputs">Array of input complex data buffers.</param>
        /// <param name="outputs">Array of output complex data buffers.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream for asynchronous execution.</param>
        public virtual void BatchFFT1D(
            ArrayView<Complex>[] inputs,
            ArrayView<Complex>[] outputs,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (inputs.Length != outputs.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            for (int i = 0; i < inputs.Length; i++)
            {
                FFT1D(inputs[i], outputs[i], forward, stream);
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Gets the optimal FFT size for the given length.
        /// </summary>
        /// <param name="length">Desired length.</param>
        /// <returns>Optimal FFT size (typically next power of 2).</returns>
        public virtual int GetOptimalFFTSize(int length) => NextPowerOf2(length);

        /// <summary>
        /// Checks if the given FFT size is supported.
        /// </summary>
        /// <param name="length">FFT length to check.</param>
        /// <returns>True if the size is supported.</returns>
        public virtual bool IsSizeSupported(int length) => IsPowerOf2(length) && length >= 2 && length <= 1 << 30;

        /// <summary>
        /// Estimates the performance for a given FFT size.
        /// </summary>
        /// <param name="length">FFT length.</param>
        /// <param name="is2D">True for 2D FFT, false for 1D.</param>
        /// <returns>Estimated performance information.</returns>
        public abstract FFTPerformanceEstimate EstimatePerformance(int length, bool is2D = false);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Calculates the next power of 2 greater than or equal to the given value.
        /// </summary>
        /// <param name="value">Input value.</param>
        /// <returns>Next power of 2.</returns>
        protected static int NextPowerOf2(int value)
        {
            if (value <= 0) return 1;
            if (IsPowerOf2(value)) return value;
            
            int power = 1;
            while (power < value)
                power <<= 1;
            
            return power;
        }

        /// <summary>
        /// Checks if a value is a power of 2.
        /// </summary>
        /// <param name="value">Value to check.</param>
        /// <returns>True if the value is a power of 2.</returns>
        protected static bool IsPowerOf2(int value) => value > 0 && (value & (value - 1)) == 0;

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this FFT accelerator and frees associated resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes managed and unmanaged resources.
        /// </summary>
        /// <param name="disposing">True to dispose managed resources.</param>
        protected abstract void Dispose(bool disposing);

        #endregion
    }

    /// <summary>
    /// Types of FFT accelerators available.
    /// </summary>
    public enum FFTAcceleratorType
    {
        /// <summary>
        /// CPU-based FFT using standard algorithms.
        /// </summary>
        CPU,

        /// <summary>
        /// Intel IPP optimized CPU FFT.
        /// </summary>
        IntelIPP,

        /// <summary>
        /// NVIDIA cuFFT GPU acceleration.
        /// </summary>
        CUDA,

        /// <summary>
        /// OpenCL-based GPU FFT.
        /// </summary>
        OpenCL,

        /// <summary>
        /// Apple Accelerate framework.
        /// </summary>
        AppleAccelerate,

        /// <summary>
        /// Generic GPU compute using ILGPU kernels.
        /// </summary>
        GPU
    }

    /// <summary>
    /// Performance characteristics of an FFT accelerator.
    /// </summary>
    public struct FFTPerformanceInfo
    {
        /// <summary>
        /// Relative performance compared to reference (1.0 = reference, >1.0 = faster).
        /// </summary>
        public double RelativePerformance { get; set; }

        /// <summary>
        /// Estimated GFLOPS for typical FFT operations.
        /// </summary>
        public double EstimatedGFLOPS { get; set; }

        /// <summary>
        /// Memory bandwidth utilization efficiency (0.0 to 1.0).
        /// </summary>
        public double MemoryEfficiency { get; set; }

        /// <summary>
        /// Preferred minimum FFT size for good performance.
        /// </summary>
        public int MinimumEfficientSize { get; set; }

        /// <summary>
        /// Maximum FFT size supported.
        /// </summary>
        public int MaximumSize { get; set; }

        /// <summary>
        /// Whether this accelerator supports in-place transforms.
        /// </summary>
        public bool SupportsInPlace { get; set; }

        /// <summary>
        /// Whether this accelerator supports batched operations.
        /// </summary>
        public bool SupportsBatch { get; set; }
    }

    /// <summary>
    /// Performance estimate for a specific FFT operation.
    /// </summary>
    public struct FFTPerformanceEstimate
    {
        /// <summary>
        /// Estimated execution time in milliseconds.
        /// </summary>
        public double EstimatedTimeMs { get; set; }

        /// <summary>
        /// Estimated memory usage in bytes.
        /// </summary>
        public long EstimatedMemoryBytes { get; set; }

        /// <summary>
        /// Estimated GFLOPS for this specific operation.
        /// </summary>
        public double EstimatedGFLOPS { get; set; }

        /// <summary>
        /// Confidence level of the estimate (0.0 to 1.0).
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Whether this size is optimal for the accelerator.
        /// </summary>
        public bool IsOptimalSize { get; set; }
    }
}