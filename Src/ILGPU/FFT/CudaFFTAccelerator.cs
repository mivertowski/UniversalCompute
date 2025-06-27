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
using ILGPU.Runtime.Cuda;

namespace ILGPU.FFT
{
    /// <summary>
    /// CUDA-based FFT accelerator using NVIDIA cuFFT library.
    /// Provides high-performance GPU FFT operations.
    /// </summary>
    public sealed class CudaFFTAccelerator : FFTAccelerator
    {
        #region Native cuFFT Bindings

        private const string CuFFTLibrary = "cufft";

        /// <summary>
        /// cuFFT result codes.
        /// </summary>
        public enum CufftResult : int
        {
            Success = 0,
            InvalidPlan = 1,
            AllocFailed = 2,
            InvalidType = 3,
            InvalidValue = 4,
            InternalError = 5,
            ExecFailed = 6,
            SetupFailed = 7,
            InvalidSize = 8,
            UnalignedData = 9,
            IncompleteParameterList = 10,
            InvalidDevice = 11,
            ParseError = 12,
            NoWorkspace = 13,
            NotImplemented = 14,
            LicenseError = 15,
            NotSupported = 16
        }

        /// <summary>
        /// cuFFT transform types.
        /// </summary>
        public enum CufftType : int
        {
            R2C = 0x2a,     // Real to Complex (interleaved)
            C2R = 0x2c,     // Complex (interleaved) to Real
            C2C = 0x29      // Complex to Complex
        }

        /// <summary>
        /// cuFFT transform directions.
        /// </summary>
        public enum CufftDirection : int
        {
            Forward = -1,
            Inverse = 1
        }

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftPlan1d(out IntPtr plan, int nx, CufftType type, int batch);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftPlan2d(out IntPtr plan, int nx, int ny, CufftType type);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftDestroy(IntPtr plan);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftExecC2C(IntPtr plan, IntPtr idata, IntPtr odata, CufftDirection direction);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftExecR2C(IntPtr plan, IntPtr idata, IntPtr odata);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftExecC2R(IntPtr plan, IntPtr idata, IntPtr odata);

        [DllImport(CuFFTLibrary, CallingConvention = CallingConvention.Cdecl)]
        private static extern CufftResult cufftSetStream(IntPtr plan, IntPtr stream);

        #endregion

        #region Instance

        private readonly CudaAccelerator _cudaAccelerator;
        private bool _disposed;

        /// <summary>
        /// Constructs a new CUDA FFT accelerator.
        /// </summary>
        /// <param name="cudaAccelerator">The parent CUDA accelerator.</param>
        public CudaFFTAccelerator(CudaAccelerator cudaAccelerator)
            : base(cudaAccelerator)
        {
            _cudaAccelerator = cudaAccelerator ?? throw new ArgumentNullException(nameof(cudaAccelerator));
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the FFT accelerator type.
        /// </summary>
        public override FFTAcceleratorType AcceleratorType => FFTAcceleratorType.CUDA;

        /// <summary>
        /// Gets the name of this FFT accelerator.
        /// </summary>
        public override string Name => $"NVIDIA cuFFT ({_cudaAccelerator.Name})";

        /// <summary>
        /// Gets whether this FFT accelerator is available and functional.
        /// </summary>
        public override bool IsAvailable => CheckCuFFTAvailability();

        /// <summary>
        /// Gets the performance characteristics of this FFT accelerator.
        /// </summary>
        public override FFTPerformanceInfo PerformanceInfo => new FFTPerformanceInfo
        {
            RelativePerformance = 5.0, // High performance
            EstimatedGFLOPS = _cudaAccelerator.NumMultiprocessors * 100.0, // Rough estimate
            MemoryEfficiency = 0.8,
            MinimumEfficientSize = 64,
            MaximumSize = 1 << 27, // 128M points
            SupportsInPlace = true,
            SupportsBatch = true
        };

        #endregion

        #region 1D FFT Implementation

        /// <summary>
        /// Performs a 1D complex-to-complex FFT using cuFFT.
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

            // For now, provide a placeholder implementation
            // Full cuFFT integration would require more complex memory management
            throw new NotImplementedException("CUDA cuFFT integration needs additional work for production use");
        }

        /// <summary>
        /// Performs a 1D real-to-complex FFT using cuFFT.
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
            throw new NotImplementedException("CUDA cuFFT real FFT integration needs additional work for production use");
        }

        /// <summary>
        /// Performs a 1D complex-to-real inverse FFT using cuFFT.
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
            throw new NotImplementedException("CUDA cuFFT inverse real FFT integration needs additional work for production use");
        }

        #endregion

        #region 2D FFT Implementation

        /// <summary>
        /// Performs a 2D complex-to-complex FFT using cuFFT.
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
            throw new NotImplementedException("CUDA cuFFT 2D FFT integration needs additional work for production use");
        }

        #endregion

        #region Batch FFT Implementation

        /// <summary>
        /// Performs multiple 1D FFTs in parallel using cuFFT batching.
        /// </summary>
        public override void BatchFFT1D(
            ArrayView<Complex>[] inputs,
            ArrayView<Complex>[] outputs,
            bool forward = true,
            AcceleratorStream? stream = null)
        {
            if (inputs.Length != outputs.Length)
                throw new ArgumentException("Input and output arrays must have the same length");

            if (inputs.Length == 0)
                return;

            // All inputs must have the same length
            var length = (int)inputs[0].Length;
            for (int i = 1; i < inputs.Length; i++)
            {
                if (inputs[i].Length != length || outputs[i].Length != length)
                    throw new ArgumentException("All FFTs in batch must have the same length");
            }

            if (!IsSizeSupported(length))
                throw new ArgumentException($"FFT size {length} is not supported");

            // Placeholder implementation
            throw new NotImplementedException("CUDA cuFFT batch FFT integration needs additional work for production use");
        }

        #endregion

        #region Performance Estimation

        /// <summary>
        /// Estimates the performance for a given FFT size.
        /// </summary>
        public override FFTPerformanceEstimate EstimatePerformance(int length, bool is2D = false)
        {
            var estimate = new FFTPerformanceEstimate();

            if (!IsSizeSupported(length))
            {
                estimate.Confidence = 0.0;
                return estimate;
            }

            // Estimate based on GPU specifications
            var complexity = is2D ? length * length * Math.Log2(length * length) : length * Math.Log2(length);
            var gflops = _cudaAccelerator.NumMultiprocessors * 50.0; // Conservative estimate
            
            estimate.EstimatedTimeMs = complexity / (gflops * 1e6);
            estimate.EstimatedMemoryBytes = (is2D ? length * length : length) * 16; // Complex numbers are 16 bytes
            estimate.EstimatedGFLOPS = complexity / (estimate.EstimatedTimeMs * 1e6);
            estimate.IsOptimalSize = IsPowerOf2(length);
            estimate.Confidence = 0.8;

            return estimate;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Checks if cuFFT is available on this system.
        /// </summary>
        private static bool CheckCuFFTAvailability()
        {
            try
            {
                // Try to create a simple plan to test availability
                var result = cufftPlan1d(out IntPtr plan, 64, CufftType.C2C, 1);
                if (result == CufftResult.Success)
                {
                    cufftDestroy(plan);
                    return true;
                }
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch
            {
                // Other errors might indicate the library is present but not functional
                return false;
            }

            return false;
        }

        /// <summary>
        /// Checks cuFFT result and throws an exception if not successful.
        /// </summary>
        private static void CheckCuFFTResult(CufftResult result, string message)
        {
            if (result != CufftResult.Success)
                throw new InvalidOperationException($"{message}: {result}");
        }

        #endregion

        #region Dispose

        /// <summary>
        /// Disposes this CUDA FFT accelerator.
        /// </summary>
        public override void Dispose()
        {
            if (!_disposed)
            {
                // No persistent resources to clean up in this implementation
                _disposed = true;
            }
        }

        #endregion
    }
}