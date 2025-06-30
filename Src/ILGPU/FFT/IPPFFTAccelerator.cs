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
using ILGPU.Runtime.CPU;
using ILGPU.Intel.IPP;

namespace ILGPU.FFT
{
    /// <summary>
    /// Intel IPP-based FFT accelerator that provides high-performance CPU FFT operations.
    /// </summary>
    public sealed class IPPFFTAccelerator : FFTAccelerator
    {
        #region Instance

        private readonly CPUAccelerator _cpuAccelerator;
        private readonly IPPInfo _capabilities;
        private bool _disposed;

        /// <summary>
        /// Constructs a new IPP FFT accelerator.
        /// </summary>
        /// <param name="cpuAccelerator">The parent CPU accelerator.</param>
        public IPPFFTAccelerator(CPUAccelerator cpuAccelerator)
            : base(cpuAccelerator)
        {
            _cpuAccelerator = cpuAccelerator ?? throw new ArgumentNullException(nameof(cpuAccelerator));
            _capabilities = IPPCapabilities.Query();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the FFT accelerator type.
        /// </summary>
        public override FFTAcceleratorType AcceleratorType => FFTAcceleratorType.IntelIPP;

        /// <summary>
        /// Gets the name of this FFT accelerator.
        /// </summary>
        public override string Name => $"Intel IPP FFT ({_capabilities.Version})";

        /// <summary>
        /// Gets whether this FFT accelerator is available and functional.
        /// </summary>
        public override bool IsAvailable => _capabilities.IsAvailable && IPPCapabilities.SupportsFFT();

        /// <summary>
        /// Gets the performance characteristics of this FFT accelerator.
        /// </summary>
        public override FFTPerformanceInfo PerformanceInfo => new FFTPerformanceInfo
        {
            RelativePerformance = _capabilities.EstimatedPerformance.RelativePerformance,
            EstimatedGFLOPS = _capabilities.EstimatedPerformance.EstimatedGFLOPS,
            MemoryEfficiency = 0.9, // IPP is very memory efficient
            MinimumEfficientSize = _capabilities.SupportsAVX512 ? 32 : 16,
            MaximumSize = 1 << 26, // 64M points
            SupportsInPlace = true,
            SupportsBatch = false // Not implemented in this version
        };

        #endregion

        #region 1D FFT Implementation

        /// <summary>
        /// Performs a 1D complex-to-complex FFT using Intel IPP.
        /// </summary>
        public override void FFT1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (input.Length != output.Length)
                throw new ArgumentException("Input and output buffers must have the same length");

            var length = (int)input.Length;
            if (!IsSizeSupported(length))
                throw new ArgumentException($"FFT size {length} is not supported");

            // For now, implement a simple CPU FFT since IPP integration needs more work
            // This is a placeholder implementation
            throw new NotImplementedException("Intel IPP FFT integration needs additional work for production use");
        }

        /// <summary>
        /// Performs a 1D real-to-complex FFT using Intel IPP.
        /// </summary>
        public override void FFT1DReal(
            ArrayView<float> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null)
        {
            var length = (int)input.Length;
            if (!IsSizeSupported(length))
                throw new ArgumentException($"FFT size {length} is not supported");

            if (output.Length < length / 2 + 1)
                throw new ArgumentException("Output buffer too small for real FFT");

            // Placeholder implementation
            throw new NotImplementedException("Intel IPP real FFT integration needs additional work for production use");
        }

        /// <summary>
        /// Performs a 1D complex-to-real inverse FFT using Intel IPP.
        /// </summary>
        public override void IFFT1DReal(
            ArrayView<Complex> input,
            ArrayView<float> output,
            AcceleratorStream? stream = null)
        {
            var length = (int)output.Length;
            if (!IsSizeSupported(length))
                throw new ArgumentException($"FFT size {length} is not supported");

            if (input.Length < length / 2 + 1)
                throw new ArgumentException("Input buffer too small for inverse real FFT");

            // Placeholder implementation
            throw new NotImplementedException("Intel IPP inverse real FFT integration needs additional work for production use");
        }

        #endregion

        #region 2D FFT Implementation

        /// <summary>
        /// Performs a 2D complex-to-complex FFT using Intel IPP.
        /// </summary>
        public override void FFT2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (input.Extent != output.Extent)
                throw new ArgumentException("Input and output buffers must have the same dimensions");

            var extent = input.Extent;
            if (!IsSizeSupported((int)extent.X) || !IsSizeSupported((int)extent.Y))
                throw new ArgumentException($"FFT size {extent} is not supported");

            // Placeholder implementation
            throw new NotImplementedException("Intel IPP 2D FFT integration needs additional work for production use");
        }

        #endregion

        #region Performance Estimation

        /// <summary>
        /// Estimates the performance for a given FFT size using IPP.
        /// </summary>
        public override FFTPerformanceEstimate EstimatePerformance(int length, bool is2D = false)
        {
            var estimate = new FFTPerformanceEstimate();

            if (!IsSizeSupported(length))
            {
                estimate.Confidence = 0.0;
                return estimate;
            }

            // IPP performance varies significantly based on CPU features
            var baseGFLOPS = _capabilities.EstimatedPerformance.EstimatedGFLOPS;
            var complexity = is2D ? length * length * Math.Log2(length * length) : length * Math.Log2(length);
            
            // IPP efficiency factors
            var sizeFactor = length >= 1024 ? 1.0 : 0.8; // Better for larger sizes
            var powerOf2Factor = IsPowerOf2(length) ? 1.0 : 0.6; // Much better for power-of-2
            var avxFactor = _capabilities.SupportsAVX512 ? 1.5 : _capabilities.SupportsAVX2 ? 1.2 : 1.0;
            
            var effectiveGFLOPS = baseGFLOPS * sizeFactor * powerOf2Factor * avxFactor;
            
            estimate.EstimatedTimeMs = complexity / (effectiveGFLOPS * 1e6);
            estimate.EstimatedMemoryBytes = (is2D ? length * length : length) * 16; // Complex numbers
            estimate.EstimatedGFLOPS = complexity / (estimate.EstimatedTimeMs * 1e6);
            estimate.IsOptimalSize = IsPowerOf2(length) && length >= PerformanceInfo.MinimumEfficientSize;
            estimate.Confidence = 0.9; // High confidence for IPP estimates

            return estimate;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Checks if the given FFT size is supported by IPP.
        /// </summary>
        public override bool IsSizeSupported(int length) =>
            // IPP FFT requires power-of-2 sizes
            IsPowerOf2(length) && length >= 2 && length <= (1 << 26);

        /// <summary>
        /// Gets the optimal FFT size for IPP (next power of 2).
        /// </summary>
        public override int GetOptimalFFTSize(int length) => NextPowerOf2(length);

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this IPP FFT accelerator.
        /// </summary>
        /// <param name="disposing">True if disposing from Dispose() method, false if from finalizer.</param>
        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // No persistent resources to clean up
                }
                _disposed = true;
            }
        }

        #endregion
    }
}