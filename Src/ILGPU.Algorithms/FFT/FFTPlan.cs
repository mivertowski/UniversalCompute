// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Numerics;
using ILGPU.Runtime;
using ILGPU.FFT;

namespace ILGPU.Algorithms.FFT
{
    /// <summary>
    /// Base class for FFT plans that optimize repeated FFT operations.
    /// </summary>
    public abstract class FFTPlan : IDisposable
    {
        #region Properties

        /// <summary>
        /// Gets the ILGPU context.
        /// </summary>
        public Context Context { get; }

        /// <summary>
        /// Gets the FFT manager for this plan.
        /// </summary>
        protected FFTManager FFTManager { get; }

        /// <summary>
        /// Gets the selected FFT accelerator.
        /// </summary>
        public FFTAccelerator? SelectedAccelerator { get; protected set; }

        /// <summary>
        /// Gets whether this plan is valid and ready for use.
        /// </summary>
        public bool IsValid => SelectedAccelerator?.IsAvailable == true;

        #endregion

        #region Constructor

        /// <summary>
        /// Constructs a new FFT plan.
        /// </summary>
        /// <param name="context">ILGPU context.</param>
        protected FFTPlan(Context context)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
            FFTManager = new FFTManager(context);
        }

        #endregion

        #region Abstract Methods

        /// <summary>
        /// Gets the estimated performance for this plan.
        /// </summary>
        /// <returns>Performance estimate.</returns>
        public abstract FFTPerformanceEstimate GetPerformanceEstimate();

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this FFT plan and associated resources.
        /// </summary>
        public virtual void Dispose()
        {
            FFTManager?.Dispose();
        }

        #endregion
    }

    /// <summary>
    /// Optimized plan for 1D FFT operations.
    /// </summary>
    public sealed class FFTPlan1D : FFTPlan
    {
        #region Properties

        /// <summary>
        /// Gets the FFT length.
        /// </summary>
        public int Length { get; }

        /// <summary>
        /// Gets whether this is a real-to-complex FFT.
        /// </summary>
        public bool IsReal { get; }

        /// <summary>
        /// Gets the optimal FFT length (may be larger than requested for performance).
        /// </summary>
        public int OptimalLength { get; }

        #endregion

        #region Constructor

        /// <summary>
        /// Constructs a new 1D FFT plan.
        /// </summary>
        /// <param name="context">ILGPU context.</param>
        /// <param name="length">FFT length.</param>
        /// <param name="isReal">True for real-to-complex FFTs.</param>
        public FFTPlan1D(Context context, int length, bool isReal = false)
            : base(context)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive");

            Length = length;
            IsReal = isReal;

            // Find the best accelerator for this size
            SelectedAccelerator = FFTManager.GetBestAccelerator(length, false, true);
            
            if (SelectedAccelerator != null)
            {
                OptimalLength = SelectedAccelerator.GetOptimalFFTSize(length);
            }
            else
            {
                OptimalLength = FFTAlgorithms.NextPowerOf2(length);
            }
        }

        #endregion

        #region Execute Methods

        /// <summary>
        /// Executes a forward 1D FFT using this plan.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void Forward(ArrayView<Complex> input, ArrayView<Complex> output, AcceleratorStream? stream = null)
        {
            ValidatePlan();
            ValidateBuffers1D(input.Length, output.Length, false);
            SelectedAccelerator!.FFT1D(input, output, true, stream);
        }

        /// <summary>
        /// Executes an inverse 1D FFT using this plan.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void Inverse(ArrayView<Complex> input, ArrayView<Complex> output, AcceleratorStream? stream = null)
        {
            ValidatePlan();
            ValidateBuffers1D(input.Length, output.Length, false);
            SelectedAccelerator!.FFT1D(input, output, false, stream);
        }

        /// <summary>
        /// Executes a forward real-to-complex 1D FFT using this plan.
        /// </summary>
        /// <param name="input">Input real data buffer.</param>
        /// <param name="output">Output complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void ForwardReal(ArrayView<float> input, ArrayView<Complex> output, AcceleratorStream? stream = null)
        {
            if (!IsReal)
                throw new InvalidOperationException("This plan is not configured for real FFTs");

            ValidatePlan();
            ValidateBuffers1D(input.Length, output.Length, true);
            SelectedAccelerator!.FFT1DReal(input, output, stream);
        }

        /// <summary>
        /// Executes an inverse complex-to-real 1D FFT using this plan.
        /// </summary>
        /// <param name="input">Input complex data buffer.</param>
        /// <param name="output">Output real data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void InverseReal(ArrayView<Complex> input, ArrayView<float> output, AcceleratorStream? stream = null)
        {
            if (!IsReal)
                throw new InvalidOperationException("This plan is not configured for real FFTs");

            ValidatePlan();
            ValidateBuffers1D(input.Length, output.Length, true);
            SelectedAccelerator!.IFFT1DReal(input, output, stream);
        }

        #endregion

        #region Performance

        /// <summary>
        /// Gets the estimated performance for this 1D FFT plan.
        /// </summary>
        public override FFTPerformanceEstimate GetPerformanceEstimate()
        {
            if (SelectedAccelerator == null)
            {
                return new FFTPerformanceEstimate { Confidence = 0.0 };
            }

            return SelectedAccelerator.EstimatePerformance(Length, false);
        }

        #endregion

        #region Validation

        private void ValidatePlan()
        {
            if (!IsValid)
                throw new InvalidOperationException("FFT plan is not valid or no suitable accelerator available");
        }

        private void ValidateBuffers1D(long inputLength, long outputLength, bool isReal)
        {
            if (isReal)
            {
                if (inputLength != Length)
                    throw new ArgumentException($"Input buffer length {inputLength} does not match plan length {Length}");
                
                var expectedOutputLength = Length / 2 + 1;
                if (outputLength < expectedOutputLength)
                    throw new ArgumentException($"Output buffer too small: {outputLength} < {expectedOutputLength}");
            }
            else
            {
                if (inputLength != Length || outputLength != Length)
                    throw new ArgumentException($"Buffer lengths {inputLength}, {outputLength} do not match plan length {Length}");
            }
        }

        #endregion
    }

    /// <summary>
    /// Optimized plan for 2D FFT operations.
    /// </summary>
    public sealed class FFTPlan2D : FFTPlan
    {
        #region Properties

        /// <summary>
        /// Gets the FFT width.
        /// </summary>
        public int Width { get; }

        /// <summary>
        /// Gets the FFT height.
        /// </summary>
        public int Height { get; }

        /// <summary>
        /// Gets the 2D extent of this FFT.
        /// </summary>
        public Index2D Extent => new Index2D(Width, Height);

        /// <summary>
        /// Gets the optimal 2D extent (may be larger than requested for performance).
        /// </summary>
        public Index2D OptimalExtent { get; }

        #endregion

        #region Constructor

        /// <summary>
        /// Constructs a new 2D FFT plan.
        /// </summary>
        /// <param name="context">ILGPU context.</param>
        /// <param name="width">FFT width.</param>
        /// <param name="height">FFT height.</param>
        public FFTPlan2D(Context context, int width, int height)
            : base(context)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException("Width and height must be positive");

            Width = width;
            Height = height;

            // Find the best accelerator for this size
            var maxDimension = Math.Max(width, height);
            SelectedAccelerator = FFTManager.GetBestAccelerator(maxDimension, true, true);
            
            if (SelectedAccelerator != null)
            {
                var optimalWidth = SelectedAccelerator.GetOptimalFFTSize(width);
                var optimalHeight = SelectedAccelerator.GetOptimalFFTSize(height);
                OptimalExtent = new Index2D(optimalWidth, optimalHeight);
            }
            else
            {
                OptimalExtent = new Index2D(
                    FFTAlgorithms.NextPowerOf2(width),
                    FFTAlgorithms.NextPowerOf2(height));
            }
        }

        #endregion

        #region Execute Methods

        /// <summary>
        /// Executes a forward 2D FFT using this plan.
        /// </summary>
        /// <param name="input">Input 2D complex data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void Forward(ArrayView2D<Complex, Stride2D.DenseX> input, ArrayView2D<Complex, Stride2D.DenseX> output, AcceleratorStream? stream = null)
        {
            ValidatePlan();
            ValidateBuffers2D(input.Extent, output.Extent);
            SelectedAccelerator!.FFT2D(input, output, true, stream);
        }

        /// <summary>
        /// Executes an inverse 2D FFT using this plan.
        /// </summary>
        /// <param name="input">Input 2D complex data buffer.</param>
        /// <param name="output">Output 2D complex data buffer.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        public void Inverse(ArrayView2D<Complex, Stride2D.DenseX> input, ArrayView2D<Complex, Stride2D.DenseX> output, AcceleratorStream? stream = null)
        {
            ValidatePlan();
            ValidateBuffers2D(input.Extent, output.Extent);
            SelectedAccelerator!.FFT2D(input, output, false, stream);
        }

        #endregion

        #region Performance

        /// <summary>
        /// Gets the estimated performance for this 2D FFT plan.
        /// </summary>
        public override FFTPerformanceEstimate GetPerformanceEstimate()
        {
            if (SelectedAccelerator == null)
            {
                return new FFTPerformanceEstimate { Confidence = 0.0 };
            }

            var maxDimension = Math.Max(Width, Height);
            return SelectedAccelerator.EstimatePerformance(maxDimension, true);
        }

        #endregion

        #region Validation

        private void ValidatePlan()
        {
            if (!IsValid)
                throw new InvalidOperationException("FFT plan is not valid or no suitable accelerator available");
        }

        private void ValidateBuffers2D(Index2D inputExtent, Index2D outputExtent)
        {
            if (inputExtent != Extent || outputExtent != Extent)
                throw new ArgumentException($"Buffer extents {inputExtent}, {outputExtent} do not match plan extent {Extent}");
        }

        #endregion
    }
}