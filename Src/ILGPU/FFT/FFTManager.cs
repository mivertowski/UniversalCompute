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
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Intel.IPP;

namespace ILGPU.FFT
{
    /// <summary>
    /// Central manager for FFT operations that automatically selects the best available FFT accelerator.
    /// Provides a unified interface for all FFT operations across different hardware types.
    /// </summary>
    public sealed class FFTManager : IDisposable
    {
        #region Instance

        private readonly Context _context;
        private readonly List<FFTAccelerator> _accelerators;
        private readonly Dictionary<AcceleratorType, FFTAccelerator> _acceleratorMap;
        private bool _disposed;

        /// <summary>
        /// Constructs a new FFT manager.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        public FFTManager(Context context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _accelerators = [];
            _acceleratorMap = [];
            
            InitializeAccelerators();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets all available FFT accelerators.
        /// </summary>
        public IReadOnlyList<FFTAccelerator> AvailableAccelerators => _accelerators.AsReadOnly();

        /// <summary>
        /// Gets the default FFT accelerator (highest performance available).
        /// </summary>
        public FFTAccelerator? DefaultAccelerator => GetBestAccelerator();

        /// <summary>
        /// Gets whether any FFT accelerator is available.
        /// </summary>
        public bool HasAccelerators => _accelerators.Count > 0;

        #endregion

        #region Accelerator Selection

        /// <summary>
        /// Gets the best FFT accelerator for the given requirements.
        /// </summary>
        /// <param name="length">FFT length.</param>
        /// <param name="is2D">Whether this is a 2D FFT.</param>
        /// <param name="prioritizeThroughput">True to prioritize throughput over latency.</param>
        /// <returns>The best FFT accelerator, or null if none available.</returns>
        public FFTAccelerator? GetBestAccelerator(int length = 1024, bool is2D = false, bool prioritizeThroughput = true)
        {
            if (_accelerators.Count == 0)
                return null;

            // Score each accelerator based on performance for this use case
            var scored = _accelerators
                .Where(acc => acc.IsAvailable && acc.IsSizeSupported(length))
                .Select(acc => new
                {
                    Accelerator = acc,
                    Score = CalculateScore(acc, length, is2D, prioritizeThroughput)
                })
                .OrderByDescending(x => x.Score)
                .ToList();

            return scored.FirstOrDefault()?.Accelerator;
        }

        /// <summary>
        /// Gets an FFT accelerator of the specified type.
        /// </summary>
        /// <param name="type">The accelerator type.</param>
        /// <returns>The FFT accelerator, or null if not available.</returns>
        public FFTAccelerator? GetAccelerator(FFTAcceleratorType type) => _accelerators.FirstOrDefault(acc => acc.AcceleratorType == type && acc.IsAvailable);

        /// <summary>
        /// Gets an FFT accelerator for the specified ILGPU accelerator.
        /// </summary>
        /// <param name="accelerator">The ILGPU accelerator.</param>
        /// <returns>The FFT accelerator, or null if not available.</returns>
        public FFTAccelerator? GetAccelerator(Accelerator accelerator)
        {
            if (_acceleratorMap.TryGetValue(accelerator.AcceleratorType, out var fftAccelerator))
            {
                if (fftAccelerator.ParentAccelerator == accelerator && fftAccelerator.IsAvailable)
                    return fftAccelerator;
            }

            // Search through all accelerators
            return _accelerators.FirstOrDefault(acc => 
                acc.ParentAccelerator == accelerator && acc.IsAvailable);
        }

        #endregion

        #region High-Level FFT Operations

        /// <summary>
        /// Performs a 1D FFT using the best available accelerator.
        /// </summary>
        /// <param name="input">Input complex data.</param>
        /// <param name="output">Output complex data.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        /// <returns>The accelerator used for the operation.</returns>
        public FFTAccelerator? FFT1D(
            ArrayView<Complex> input,
            ArrayView<Complex> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            var accelerator = GetBestAccelerator((int)input.Length, false, true) ?? throw new InvalidOperationException("No suitable FFT accelerator available");
            accelerator.FFT1D(input, output, forward, stream);
            return accelerator;
        }

        /// <summary>
        /// Performs a 1D real FFT using the best available accelerator.
        /// </summary>
        /// <param name="input">Input real data.</param>
        /// <param name="output">Output complex data.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        /// <returns>The accelerator used for the operation.</returns>
        public FFTAccelerator? FFT1DReal(
            ArrayView<float> input,
            ArrayView<Complex> output,
            AcceleratorStream? stream = null)
        {
            var accelerator = GetBestAccelerator((int)input.Length, false, true) ?? throw new InvalidOperationException("No suitable FFT accelerator available");
            accelerator.FFT1DReal(input, output, stream);
            return accelerator;
        }

        /// <summary>
        /// Performs a 2D FFT using the best available accelerator.
        /// </summary>
        /// <param name="input">Input 2D complex data.</param>
        /// <param name="output">Output 2D complex data.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <param name="stream">Optional accelerator stream.</param>
        /// <returns>The accelerator used for the operation.</returns>
        public FFTAccelerator? FFT2D(
            ArrayView2D<Complex, Stride2D.DenseX> input,
            ArrayView2D<Complex, Stride2D.DenseX> output,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            var size = Math.Max((int)input.Extent.X, (int)input.Extent.Y);
            var accelerator = GetBestAccelerator(size, true, true) ?? throw new InvalidOperationException("No suitable FFT accelerator available");
            accelerator.FFT2D(input, output, forward, stream);
            return accelerator;
        }

        #endregion

        #region Performance Analysis

        /// <summary>
        /// Estimates performance for all available accelerators.
        /// </summary>
        /// <param name="length">FFT length.</param>
        /// <param name="is2D">Whether this is a 2D FFT.</param>
        /// <returns>Performance estimates for each accelerator.</returns>
        public Dictionary<FFTAccelerator, FFTPerformanceEstimate> EstimatePerformance(int length, bool is2D = false)
        {
            var estimates = new Dictionary<FFTAccelerator, FFTPerformanceEstimate>();

            foreach (var accelerator in _accelerators.Where(acc => acc.IsAvailable))
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    var estimate = accelerator.EstimatePerformance(length, is2D);
                    estimates[accelerator] = estimate;
                }
                catch
                {
                    // Ignore accelerators that can't provide estimates
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }

            return estimates;
        }

        /// <summary>
        /// Gets a performance report for all available accelerators.
        /// </summary>
        /// <returns>Formatted performance report.</returns>
        public string GetPerformanceReport()
        {
            if (!HasAccelerators)
                return "No FFT accelerators available.";

            var report = "Available FFT Accelerators:\n";
            report += "============================\n\n";

            foreach (var accelerator in _accelerators.Where(acc => acc.IsAvailable))
            {
                var perf = accelerator.PerformanceInfo;
                report += $"{accelerator.Name}:\n";
                report += $"  Type: {accelerator.AcceleratorType}\n";
                report += $"  Relative Performance: {perf.RelativePerformance:F1}x\n";
                report += $"  Estimated GFLOPS: {perf.EstimatedGFLOPS:F1}\n";
                report += $"  Memory Efficiency: {perf.MemoryEfficiency:P1}\n";
                report += $"  Size Range: {perf.MinimumEfficientSize} - {perf.MaximumSize:N0}\n";
                report += $"  Features: In-place={perf.SupportsInPlace}, Batch={perf.SupportsBatch}\n\n";
            }

            var defaultAccel = DefaultAccelerator;
            if (defaultAccel != null)
            {
                report += $"Default Accelerator: {defaultAccel.Name}\n";
            }

            return report;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Initializes all available FFT accelerators.
        /// </summary>
        private void InitializeAccelerators()
        {
            // Try to create IPP FFT accelerator with CPU accelerator
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                if (IPPCapabilities.DetectIPP())
                {
                    using var cpuAccelerator = _context.CreateCPUAccelerator(0, CPUAcceleratorMode.Auto);
                    var ippFFT = new IPPFFTAccelerator(cpuAccelerator);
                    if (ippFFT.IsAvailable)
                    {
                        _accelerators.Add(ippFFT);
                        _acceleratorMap[AcceleratorType.CPU] = ippFFT;
                    }
                    else
                    {
                        ippFFT.Dispose();
                    }
                }
            }
            catch
            {
                // IPP not available
            }
#pragma warning restore CA1031 // Do not catch general exception types

            // Try to create CUDA FFT accelerators
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                foreach (var device in _context.GetCudaDevices())
                {
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        using var cudaAccelerator = device.CreateCudaAccelerator(_context);
                        var cudaFFT = new CudaFFTAccelerator(cudaAccelerator);
                        if (cudaFFT.IsAvailable)
                        {
                            _accelerators.Add(cudaFFT);
                            if (!_acceleratorMap.ContainsKey(AcceleratorType.Cuda))
                                _acceleratorMap[AcceleratorType.Cuda] = cudaFFT;
                        }
                        else
                        {
                            cudaFFT.Dispose();
                        }
                    }
                    catch
                    {
                        // This CUDA device not available for FFT
                    }
#pragma warning restore CA1031 // Do not catch general exception types
                }
            }
            catch
            {
                // CUDA not available
            }
#pragma warning restore CA1031 // Do not catch general exception types

            // Additional accelerators can be added here (OpenCL, Apple Accelerate, etc.)
        }

        /// <summary>
        /// Calculates a score for an accelerator based on the given requirements.
        /// </summary>
        private static double CalculateScore(FFTAccelerator accelerator, int length, bool is2D, bool prioritizeThroughput)
        {
            var perf = accelerator.PerformanceInfo;
            var estimate = accelerator.EstimatePerformance(length, is2D);

            double score = 0.0;

            // Base performance score
            score += perf.RelativePerformance * 100;

            // GFLOPS contribution
            score += Math.Log10(perf.EstimatedGFLOPS) * 50;

            // Memory efficiency bonus
            score += perf.MemoryEfficiency * 30;

            // Size optimization bonus
            if (estimate.IsOptimalSize)
                score += 20;

            // Confidence penalty
            score *= estimate.Confidence;

            // Throughput vs latency preference
            if (prioritizeThroughput)
            {
                // Prefer GPU accelerators for throughput
                if (accelerator.AcceleratorType == FFTAcceleratorType.CUDA)
                    score *= 1.5;
            }
            else
            {
                // Prefer CPU accelerators for latency
                if (accelerator.AcceleratorType == FFTAcceleratorType.IntelIPP)
                    score *= 1.3;
            }

            return score;
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes the FFT manager and all accelerators.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                foreach (var accelerator in _accelerators)
                {
#pragma warning disable CA1031 // Do not catch general exception types
                    try
                    {
                        accelerator.Dispose();
                    }
                    catch
                    {
                        // Ignore disposal errors
                    }
#pragma warning restore CA1031 // Do not catch general exception types
                }

                _accelerators.Clear();
                _acceleratorMap.Clear();
                _disposed = true;
            }
        }

        #endregion
    }
}
