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
    /// High-level FFT algorithms that automatically select the best implementation
    /// based on available hardware and data characteristics.
    /// </summary>
    public static class FFTAlgorithms
    {
        #region FFT Plans

        /// <summary>
        /// Creates an optimized FFT plan for 1D transforms.
        /// </summary>
        /// <param name="context">ILGPU context.</param>
        /// <param name="length">FFT length.</param>
        /// <param name="isReal">True for real-to-complex FFTs.</param>
        /// <returns>FFT plan for efficient repeated transforms.</returns>
        public static FFTPlan1D CreatePlan1D(Context context, int length, bool isReal = false)
        {
            return new FFTPlan1D(context, length, isReal);
        }

        /// <summary>
        /// Creates an optimized FFT plan for 2D transforms.
        /// </summary>
        /// <param name="context">ILGPU context.</param>
        /// <param name="width">FFT width.</param>
        /// <param name="height">FFT height.</param>
        /// <returns>FFT plan for efficient repeated transforms.</returns>
        public static FFTPlan2D CreatePlan2D(Context context, int width, int height)
        {
            return new FFTPlan2D(context, width, height);
        }

        #endregion

        #region Convenience Methods

        /// <summary>
        /// Performs a 1D FFT with automatic hardware selection and memory management.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="input">Input data array.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <returns>Output FFT data.</returns>
        public static Complex[] FFT1D(Accelerator accelerator, Complex[] input, bool forward = true)
        {
            if (input == null || input.Length == 0)
                throw new ArgumentException("Input array cannot be null or empty");

            using var fftManager = new FFTManager(accelerator.Context);
            using var inputBuffer = accelerator.Allocate1D(input);
            using var outputBuffer = accelerator.Allocate1D<Complex>(input.Length);

            fftManager.FFT1D(inputBuffer, outputBuffer, forward);
            return outputBuffer.GetAsArray1D();
        }

        /// <summary>
        /// Performs a 1D real FFT with automatic hardware selection and memory management.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="input">Input real data array.</param>
        /// <returns>Output complex FFT data.</returns>
        public static Complex[] FFT1DReal(Accelerator accelerator, float[] input)
        {
            if (input == null || input.Length == 0)
                throw new ArgumentException("Input array cannot be null or empty");

            var outputLength = input.Length / 2 + 1;

            using var fftManager = new FFTManager(accelerator.Context);
            using var inputBuffer = accelerator.Allocate1D(input);
            using var outputBuffer = accelerator.Allocate1D<Complex>(outputLength);

            fftManager.FFT1DReal(inputBuffer, outputBuffer);
            return outputBuffer.GetAsArray1D();
        }

        /// <summary>
        /// Performs a 2D FFT with automatic hardware selection and memory management.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="input">Input 2D data array.</param>
        /// <param name="forward">True for forward transform, false for inverse.</param>
        /// <returns>Output 2D FFT data.</returns>
        public static Complex[,] FFT2D(Accelerator accelerator, Complex[,] input, bool forward = true)
        {
            if (input == null)
                throw new ArgumentException("Input array cannot be null");

            var extent = new Index2D(input.GetLength(1), input.GetLength(0)); // Width, Height

            using var fftManager = new FFTManager(accelerator.Context);
            using var inputBuffer = accelerator.Allocate2DDenseXY(input);
            using var outputBuffer = accelerator.Allocate2DDenseXY<Complex>(extent);

            fftManager.FFT2D(inputBuffer, outputBuffer, forward);
            return outputBuffer.GetAs2DArray();
        }

        #endregion

        #region Signal Processing Algorithms

        /// <summary>
        /// Performs convolution using FFT for improved performance on large signals.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="signal">Input signal.</param>
        /// <param name="kernel">Convolution kernel.</param>
        /// <returns>Convolved signal.</returns>
        public static float[] Convolve(Accelerator accelerator, float[] signal, float[] kernel)
        {
            if (signal == null || kernel == null)
                throw new ArgumentException("Signal and kernel cannot be null");

            var resultLength = signal.Length + kernel.Length - 1;
            var fftLength = NextPowerOf2(resultLength);

            // Zero-pad inputs to FFT length
            var paddedSignal = new float[fftLength];
            var paddedKernel = new float[fftLength];
            
            Array.Copy(signal, paddedSignal, signal.Length);
            Array.Copy(kernel, paddedKernel, kernel.Length);

            // Perform FFT-based convolution
            var signalFFT = FFT1DReal(accelerator, paddedSignal);
            var kernelFFT = FFT1DReal(accelerator, paddedKernel);

            // Element-wise multiplication in frequency domain
            var resultFFT = new Complex[signalFFT.Length];
            for (int i = 0; i < signalFFT.Length; i++)
            {
                resultFFT[i] = signalFFT[i] * kernelFFT[i];
            }

            // Inverse FFT to get result
            using var fftManager = new FFTManager(accelerator.Context);
            using var resultFFTBuffer = accelerator.Allocate1D(resultFFT);
            using var resultBuffer = accelerator.Allocate1D<float>(fftLength);

            var fftAccelerator = fftManager.GetBestAccelerator(fftLength);
            if (fftAccelerator != null)
            {
                fftAccelerator.IFFT1DReal(resultFFTBuffer, resultBuffer);
            }

            var result = resultBuffer.GetAsArray1D();

            // Extract the valid portion and normalize
            var output = new float[resultLength];
            for (int i = 0; i < resultLength; i++)
            {
                output[i] = result[i] / fftLength; // Normalize
            }

            return output;
        }

        /// <summary>
        /// Computes the power spectral density of a signal.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="signal">Input signal.</param>
        /// <param name="windowFunction">Optional windowing function.</param>
        /// <returns>Power spectral density.</returns>
        public static float[] PowerSpectralDensity(Accelerator accelerator, float[] signal, WindowFunction windowFunction = WindowFunction.Hann)
        {
            if (signal == null || signal.Length == 0)
                throw new ArgumentException("Signal cannot be null or empty");

            // Apply window function
            var windowed = ApplyWindow(signal, windowFunction);

            // Compute FFT
            var fft = FFT1DReal(accelerator, windowed);

            // Compute power spectral density
            var psd = new float[fft.Length];
            for (int i = 0; i < fft.Length; i++)
            {
                var magnitude = fft[i].Magnitude;
                psd[i] = magnitude * magnitude;
            }

            return psd;
        }

        /// <summary>
        /// Performs spectral filtering using FFT.
        /// </summary>
        /// <param name="accelerator">Target accelerator.</param>
        /// <param name="signal">Input signal.</param>
        /// <param name="filter">Frequency domain filter (same length as FFT output).</param>
        /// <returns>Filtered signal.</returns>
        public static float[] SpectralFilter(Accelerator accelerator, float[] signal, Complex[] filter)
        {
            if (signal == null || filter == null)
                throw new ArgumentException("Signal and filter cannot be null");

            // Perform forward FFT
            var signalFFT = FFT1DReal(accelerator, signal);

            if (signalFFT.Length != filter.Length)
                throw new ArgumentException("Filter length must match FFT output length");

            // Apply filter in frequency domain
            var filteredFFT = new Complex[signalFFT.Length];
            for (int i = 0; i < signalFFT.Length; i++)
            {
                filteredFFT[i] = signalFFT[i] * filter[i];
            }

            // Inverse FFT to get filtered signal
            using var fftManager = new FFTManager(accelerator.Context);
            using var filteredFFTBuffer = accelerator.Allocate1D(filteredFFT);
            using var resultBuffer = accelerator.Allocate1D<float>(signal.Length);

            var fftAccelerator = fftManager.GetBestAccelerator(signal.Length);
            if (fftAccelerator != null)
            {
                fftAccelerator.IFFT1DReal(filteredFFTBuffer, resultBuffer);
            }

            var result = resultBuffer.GetAsArray1D();

            // Normalize result
            for (int i = 0; i < result.Length; i++)
            {
                result[i] /= signal.Length;
            }

            return result;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Calculates the next power of 2 greater than or equal to the given value.
        /// </summary>
        /// <param name="value">Input value.</param>
        /// <returns>Next power of 2.</returns>
        public static int NextPowerOf2(int value)
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
        public static bool IsPowerOf2(int value)
        {
            return value > 0 && (value & (value - 1)) == 0;
        }

        /// <summary>
        /// Applies a window function to a signal.
        /// </summary>
        /// <param name="signal">Input signal.</param>
        /// <param name="windowFunction">Window function type.</param>
        /// <returns>Windowed signal.</returns>
        public static float[] ApplyWindow(float[] signal, WindowFunction windowFunction)
        {
            if (signal == null || signal.Length == 0)
                return signal;

            var windowed = new float[signal.Length];
            var n = signal.Length;

            for (int i = 0; i < n; i++)
            {
                float window = windowFunction switch
                {
                    WindowFunction.None => 1.0f,
                    WindowFunction.Hann => 0.5f * (1.0f - (float)Math.Cos(2.0 * Math.PI * i / (n - 1))),
                    WindowFunction.Hamming => 0.54f - 0.46f * (float)Math.Cos(2.0 * Math.PI * i / (n - 1)),
                    WindowFunction.Blackman => 0.42f - 0.5f * (float)Math.Cos(2.0 * Math.PI * i / (n - 1)) + 
                                              0.08f * (float)Math.Cos(4.0 * Math.PI * i / (n - 1)),
                    _ => 1.0f
                };

                windowed[i] = signal[i] * window;
            }

            return windowed;
        }

        #endregion
    }

    /// <summary>
    /// Available window functions for signal processing.
    /// </summary>
    public enum WindowFunction
    {
        /// <summary>
        /// No windowing (rectangular window).
        /// </summary>
        None,

        /// <summary>
        /// Hann window (raised cosine).
        /// </summary>
        Hann,

        /// <summary>
        /// Hamming window.
        /// </summary>
        Hamming,

        /// <summary>
        /// Blackman window.
        /// </summary>
        Blackman
    }
}