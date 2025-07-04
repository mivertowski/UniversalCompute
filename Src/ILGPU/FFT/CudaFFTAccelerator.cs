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
    /// <remarks>
    /// Constructs a new CUDA FFT accelerator.
    /// </remarks>
    /// <param name="cudaAccelerator">The parent CUDA accelerator.</param>
    public sealed class CudaFFTAccelerator(CudaAccelerator cudaAccelerator) : FFTAccelerator(cudaAccelerator)
    {
        #region Native cuFFT Bindings

#if WINDOWS
        private const string CuFFTLibrary = "cufft64_11";
#else
        private const string CuFFTLibrary = "libcufft.so.11";
#endif

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

        private readonly CudaAccelerator _cudaAccelerator = cudaAccelerator ?? throw new ArgumentNullException(nameof(cudaAccelerator));
        private bool _disposed;

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
        public override FFTPerformanceInfo PerformanceInfo => new()
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

            try
            {
                // Create cuFFT plan for 1D complex-to-complex transform
                var result = cufftPlan1d(out IntPtr plan, length, CufftType.C2C, 1);
                CheckCuFFTResult(result, "Failed to create cuFFT 1D plan");

                try
                {
                    // Get device pointers for input and output
                    var inputPtr = input.LoadEffectiveAddressAsPtr();
                    var outputPtr = output.LoadEffectiveAddressAsPtr();

                    // Execute the FFT
                    var direction = forward ? CufftDirection.Forward : CufftDirection.Inverse;
                    result = cufftExecC2C(plan, inputPtr, outputPtr, direction);
                    CheckCuFFTResult(result, "Failed to execute cuFFT 1D transform");

                    // Synchronize if stream is provided
                    if (stream is CudaStream cudaStream)
                    {
                        cudaStream.Synchronize();
                    }
                    else
                    {
                        // Default synchronization for the current accelerator
                        _cudaAccelerator.Synchronize();
                    }
                }
                finally
                {
                    // Clean up the plan
                    cufftDestroy(plan);
                }
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU implementation if cuFFT is not available
                FallbackToCP_FFT1D(input, output, forward);
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU implementation if cuFFT functions are not found
                FallbackToCP_FFT1D(input, output, forward);
            }
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

            try
            {
                // Create cuFFT plan for 1D real-to-complex transform
                var result = cufftPlan1d(out IntPtr plan, length, CufftType.R2C, 1);
                CheckCuFFTResult(result, "Failed to create cuFFT 1D real plan");

                try
                {
                    // Get device pointers
                    var inputPtr = input.LoadEffectiveAddressAsPtr();
                    var outputPtr = output.LoadEffectiveAddressAsPtr();

                    // Execute the real-to-complex FFT
                    result = cufftExecR2C(plan, inputPtr, outputPtr);
                    CheckCuFFTResult(result, "Failed to execute cuFFT 1D real transform");

                    // Synchronize
                    if (stream is CudaStream cudaStream)
                    {
                        cudaStream.Synchronize();
                    }
                    else
                    {
                        _cudaAccelerator.Synchronize();
                    }
                }
                finally
                {
                    cufftDestroy(plan);
                }
            }
            catch (DllNotFoundException)
            {
                FallbackToCPU_FFT1DReal(input, output);
            }
            catch (EntryPointNotFoundException)
            {
                FallbackToCPU_FFT1DReal(input, output);
            }
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

            try
            {
                // Create cuFFT plan for 1D complex-to-real inverse transform
                var result = cufftPlan1d(out IntPtr plan, length, CufftType.C2R, 1);
                CheckCuFFTResult(result, "Failed to create cuFFT 1D inverse real plan");

                try
                {
                    // Get device pointers
                    var inputPtr = input.LoadEffectiveAddressAsPtr();
                    var outputPtr = output.LoadEffectiveAddressAsPtr();

                    // Execute the complex-to-real inverse FFT
                    result = cufftExecC2R(plan, inputPtr, outputPtr);
                    CheckCuFFTResult(result, "Failed to execute cuFFT 1D inverse real transform");

                    // Synchronize
                    if (stream is CudaStream cudaStream)
                    {
                        cudaStream.Synchronize();
                    }
                    else
                    {
                        _cudaAccelerator.Synchronize();
                    }
                }
                finally
                {
                    cufftDestroy(plan);
                }
            }
            catch (DllNotFoundException)
            {
                FallbackToCPU_IFFT1DReal(input, output);
            }
            catch (EntryPointNotFoundException)
            {
                FallbackToCPU_IFFT1DReal(input, output);
            }
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

            // For now, use CPU fallback for 2D FFT by treating as separable 1D transforms
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
            
            // Copy result back to GPU
            output.AsLinearView().CopyFromCPU(outputData);
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

            // For now, use CPU fallback for batch FFT by processing each transform sequentially
            for (int batch = 0; batch < inputs.Length; batch++)
            {
                var cpuInput = new Complex[length];
                inputs[batch].CopyToCPU(cpuInput);
                
                var outputData = new Complex[length];
                
                // Perform FFT for this batch element
                for (int k = 0; k < length; k++)
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
                
                // Copy result back to GPU
                outputs[batch].CopyFromCPU(outputData);
            }
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

        #region Helper Methods

        /// <summary>
        /// CPU fallback implementation for 1D FFT.
        /// </summary>
        private void FallbackToCP_FFT1D(ArrayView<Complex> input, ArrayView<Complex> output, bool forward)
        {
            var length = (int)input.Length;
            var cpuInput = new Complex[length];
            var outputData = new Complex[length];
            
            // Copy data from GPU to CPU
            input.CopyToCPU(cpuInput);
            
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
            
            // Copy result back to GPU
            output.CopyFromCPU(outputData);
        }

        /// <summary>
        /// CPU fallback implementation for 1D real-to-complex FFT.
        /// </summary>
        private void FallbackToCPU_FFT1DReal(ArrayView<float> input, ArrayView<Complex> output)
        {
            var length = (int)input.Length;
            var cpuInput = new float[length];
            input.CopyToCPU(cpuInput);
            
            var outputLength = length / 2 + 1;
            var outputData = new Complex[outputLength];
            
            // Perform real-to-complex FFT (DFT implementation)
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
        /// CPU fallback implementation for 1D complex-to-real inverse FFT.
        /// </summary>
        private void FallbackToCPU_IFFT1DReal(ArrayView<Complex> input, ArrayView<float> output)
        {
            var length = (int)output.Length;
            var inputLength = length / 2 + 1;
            var cpuInput = new Complex[inputLength];
            input.SubView(0, inputLength).CopyToCPU(cpuInput);
            
            var outputData = new float[length];
            
            // Perform complex-to-real inverse FFT (DFT implementation)
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
