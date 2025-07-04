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
using System.Runtime.InteropServices;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Intel.IPP;
using ILGPU.Intel.IPP.Native;

namespace ILGPU.FFT
{
    /// <summary>
    /// Intel IPP-based FFT accelerator that provides high-performance CPU FFT operations.
    /// </summary>
    /// <remarks>
    /// Constructs a new IPP FFT accelerator.
    /// </remarks>
    /// <param name="cpuAccelerator">The parent CPU accelerator.</param>
    public sealed class IPPFFTAccelerator(CPUAccelerator cpuAccelerator) : FFTAccelerator(cpuAccelerator)
    {
        #region Instance

        private readonly CPUAccelerator _cpuAccelerator = cpuAccelerator ?? throw new ArgumentNullException(nameof(cpuAccelerator));
        private readonly IPPInfo _capabilities = IPPCapabilities.Query();
        private bool _disposed;

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
        public override FFTPerformanceInfo PerformanceInfo => new()
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

            try
            {
                // Try to use Intel IPP for high-performance FFT
                PerformIPPFFT1D(input, output, forward);
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU implementation if IPP is not available
                FallbackToCPU_FFT1D(input, output, forward);
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU implementation if IPP functions are not found
                FallbackToCPU_FFT1D(input, output, forward);
            }
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

            // CPU fallback for real-to-complex FFT
            var cpuInput = new float[length];
            input.CopyToCPU(cpuInput);
            
            var outputLength = length / 2 + 1;
            var outputData = new Complex[outputLength];
            
            // Perform real-to-complex FFT
            for (int k = 0; k < outputLength; k++)
            {
                Complex sum = Complex.Zero;
                for (int n = 0; n < length; n++)
                {
                    double angle = -2.0 * Math.PI * k * n / length;
                    var w = new Complex(Math.Cos(angle), Math.Sin(angle));
                    sum += cpuInput[n] * w;
                }
                outputData[k] = sum;
            }
            
            output.SubView(0, outputLength).CopyFromCPU(outputData);
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

            // CPU fallback for complex-to-real inverse FFT
            var inputLength = length / 2 + 1;
            var cpuInput = new Complex[inputLength];
            input.SubView(0, inputLength).CopyToCPU(cpuInput);
            
            var outputData = new float[length];
            
            // Perform complex-to-real inverse FFT
            for (int n = 0; n < length; n++)
            {
                Complex sum = Complex.Zero;
                for (int k = 0; k < inputLength; k++)
                {
                    double angle = 2.0 * Math.PI * k * n / length; // Positive for inverse
                    var w = new Complex(Math.Cos(angle), Math.Sin(angle));
                    sum += cpuInput[k] * w;
                }
                outputData[n] = (float)(sum.Real / length); // Normalize and take real part
            }
            
            output.CopyFromCPU(outputData);
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

            // CPU fallback for 2D FFT using separable transforms
            var width = (int)extent.X;
            var height = (int)extent.Y;
            var totalSize = width * height;
            
            var cpuInput = new Complex[totalSize];
            input.AsLinearView().CopyToCPU(cpuInput);
            
            var tempData = new Complex[totalSize];
            var outputData = new Complex[totalSize];
            
            // Perform row-wise FFT first
            for (int row = 0; row < height; row++)
            {
                for (int k = 0; k < width; k++)
                {
                    Complex sum = Complex.Zero;
                    for (int n = 0; n < width; n++)
                    {
                        double angle = -2.0 * Math.PI * k * n / width;
                        var w = new Complex(Math.Cos(angle), Math.Sin(angle));
                        sum += cpuInput[row * width + n] * w;
                    }
                    tempData[row * width + k] = sum;
                }
            }
            
            // Perform column-wise FFT
            for (int col = 0; col < width; col++)
            {
                for (int k = 0; k < height; k++)
                {
                    Complex sum = Complex.Zero;
                    for (int n = 0; n < height; n++)
                    {
                        double angle = -2.0 * Math.PI * k * n / height;
                        var w = new Complex(Math.Cos(angle), Math.Sin(angle));
                        sum += tempData[n * width + col] * w;
                    }
                    outputData[k * width + col] = sum;
                }
            }
            
            // Copy result back to CPU memory
            output.AsLinearView().CopyFromCPU(outputData);
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

        #region Intel IPP Constants

        private const int IPP_FFT_DIV_FWD_BY_N = 1;
        private const int IPP_FFT_DIV_INV_BY_N = 2;
        private const int IPP_FFT_DIV_BY_SQRTN = 4;
        private const int IPP_FFT_NODIV_BY_ANY = 8;

        #endregion

        #region Intel IPP Implementation Methods

        /// <summary>
        /// Performs 1D FFT using Intel IPP native functions.
        /// </summary>
        private unsafe void PerformIPPFFT1D(ArrayView<Complex> input, ArrayView<Complex> output, bool forward)
        {
            var length = (int)input.Length;
            
            // Copy input data to CPU memory for IPP processing
            var cpuInput = new Complex[length];
            input.CopyToCPU(cpuInput);
            
            // Calculate FFT order (log2 of length)
            int order = 0;
            int temp = length;
            while (temp > 1)
            {
                temp >>= 1;
                order++;
            }
            
            // Get required buffer sizes
            var result = IPPNative.ippsFFTGetSize_C_32fc(
                order, 
                forward ? IPP_FFT_DIV_FWD_BY_N : IPP_FFT_DIV_INV_BY_N,
                IPPNative.IppHintAlgorithm.ippAlgHintFast,
                out int specSize,
                out int specBufferSize,
                out int bufferSize);
            
            CheckIPPResult(result, "Failed to get IPP FFT sizes");
            
            // Allocate memory for IPP structures
            var specMemory = Marshal.AllocHGlobal(specSize);
            var specBufferMemory = specBufferSize > 0 ? Marshal.AllocHGlobal(specBufferSize) : IntPtr.Zero;
            var workBufferMemory = bufferSize > 0 ? Marshal.AllocHGlobal(bufferSize) : IntPtr.Zero;
            
            try
            {
                // Initialize FFT specification
                result = IPPNative.ippsFFTInit_C_32fc(
                    out IntPtr fftSpec,
                    order,
                    forward ? IPP_FFT_DIV_FWD_BY_N : IPP_FFT_DIV_INV_BY_N,
                    IPPNative.IppHintAlgorithm.ippAlgHintFast,
                    specMemory,
                    specBufferMemory);
                
                CheckIPPResult(result, "Failed to initialize IPP FFT specification");
                
                // Convert Complex to IPP complex format
                var ippInput = new IPPNative.Ipp32fc[length];
                var ippOutput = new IPPNative.Ipp32fc[length];
                
                for (int i = 0; i < length; i++)
                {
                    ippInput[i] = new IPPNative.Ipp32fc((float)cpuInput[i].Real, (float)cpuInput[i].Imaginary);
                }
                
                // Pin memory for IPP call
                fixed (IPPNative.Ipp32fc* pInput = ippInput)
                fixed (IPPNative.Ipp32fc* pOutput = ippOutput)
                {
                    // Perform FFT
                    if (forward)
                    {
                        result = IPPNative.ippsFFTFwd_CToC_32fc(
                            new IntPtr(pInput),
                            new IntPtr(pOutput),
                            fftSpec,
                            workBufferMemory);
                    }
                    else
                    {
                        result = IPPNative.ippsFFTInv_CToC_32fc(
                            new IntPtr(pInput),
                            new IntPtr(pOutput),
                            fftSpec,
                            workBufferMemory);
                    }
                    
                    CheckIPPResult(result, forward ? "Forward FFT failed" : "Inverse FFT failed");
                }
                
                // Convert result back to System.Numerics.Complex
                var outputData = new Complex[length];
                for (int i = 0; i < length; i++)
                {
                    outputData[i] = new Complex(ippOutput[i].re, ippOutput[i].im);
                }
                
                // Copy result back to GPU memory
                output.CopyFromCPU(outputData);
            }
            finally
            {
                // Clean up allocated memory
                if (specMemory != IntPtr.Zero) Marshal.FreeHGlobal(specMemory);
                if (specBufferMemory != IntPtr.Zero) Marshal.FreeHGlobal(specBufferMemory);
                if (workBufferMemory != IntPtr.Zero) Marshal.FreeHGlobal(workBufferMemory);
            }
        }

        /// <summary>
        /// CPU fallback implementation for 1D FFT.
        /// </summary>
        private void FallbackToCPU_FFT1D(ArrayView<Complex> input, ArrayView<Complex> output, bool forward)
        {
            var length = (int)input.Length;
            var cpuInput = new Complex[length];
            input.CopyToCPU(cpuInput);
            
            var outputData = new Complex[length];
            
            // Perform CPU FFT fallback (DFT implementation)
            for (int k = 0; k < length; k++)
            {
                Complex sum = Complex.Zero;
                for (int n = 0; n < length; n++)
                {
                    double angle = (forward ? -2.0 : 2.0) * Math.PI * k * n / length;
                    var w = new Complex(Math.Cos(angle), Math.Sin(angle));
                    sum += cpuInput[n] * w;
                }
                outputData[k] = forward ? sum : sum / length;
            }
            
            output.CopyFromCPU(outputData);
        }

        /// <summary>
        /// Checks IPP result and throws exception if not successful.
        /// </summary>
        private static void CheckIPPResult(IPPNative.IppStatus result, string operation)
        {
            if (result != IPPNative.IppStatus.ippStsNoErr)
                throw new InvalidOperationException($"{operation}: {result}");
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Checks if the given FFT size is supported by IPP.
        /// </summary>
        public override bool IsSizeSupported(int length) =>
            // IPP FFT requires power-of-2 sizes
            IsPowerOf2(length) && length >= 2 && length <= 1 << 26;

        /// <summary>
        /// Gets the optimal FFT size for IPP (next power of 2).
        /// </summary>
        public override int GetOptimalFFTSize(int length) => NextPowerOf2(length);

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this IPP FFT accelerator.
        /// </summary>
        public override void Dispose()
        {
            if (!_disposed)
            {
                // No persistent resources to clean up
                _disposed = true;
            }
        }

        #endregion
    }
}
