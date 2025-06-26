// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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
using ILGPU;
using ILGPU.Runtime;
using ILGPU.FFT;
using ILGPU.Algorithms.FFT;
using ILGPU.Examples.Common;

namespace ILGPU.Examples.FFTOperations
{
    /// <summary>
    /// Demonstrates the ILGPU FFT framework architecture and capabilities.
    /// Shows the complete FFT infrastructure with placeholder implementations.
    /// </summary>
    class FFTFrameworkDemo
    {
        static void Main(string[] args)
        {
            Console.WriteLine("🔄 ILGPU FFT Framework Demonstration");
            Console.WriteLine("====================================\n");

            try
            {
                // Detect available hardware
                var hardware = HardwareDetection.DetectAvailableHardware();
                HardwareDetection.PrintHardwareReport(hardware);

                using var context = Context.CreateDefault();
                
                // Demonstrate FFT framework
                DemonstrateFFTFramework(context);
                DemonstrateFFTAlgorithms(context);
                DemonstrateFFTPlans(context);
                ProvideImplementationGuidance();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }

            Console.WriteLine("\n✅ FFT framework demonstration completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Demonstrates the FFT framework structure and capabilities.
        /// </summary>
        static void DemonstrateFFTFramework(Context context)
        {
            Console.WriteLine("🔧 FFT Framework Architecture");
            Console.WriteLine("=============================\n");

            Console.WriteLine("Core Components:");
            Console.WriteLine("• FFTAccelerator - Abstract base for all FFT implementations");
            Console.WriteLine("• IPPFFTAccelerator - Intel IPP CPU acceleration");
            Console.WriteLine("• CudaFFTAccelerator - NVIDIA cuFFT GPU acceleration");
            Console.WriteLine("• FFTManager - Automatic hardware selection and optimization");
            Console.WriteLine("• FFTAlgorithms - High-level signal processing operations");
            Console.WriteLine("• FFTPlan - Optimized plans for repeated operations");
            Console.WriteLine();

            Console.WriteLine("Supported Operations:");
            Console.WriteLine("• 1D Complex FFT (forward/inverse)");
            Console.WriteLine("• 1D Real FFT (real-to-complex)");
            Console.WriteLine("• 2D Complex FFT (forward/inverse)");
            Console.WriteLine("• Batch FFT processing");
            Console.WriteLine("• Signal processing algorithms (convolution, filtering)");
            Console.WriteLine();

            // Demonstrate FFT Manager
            try
            {
                using var fftManager = new FFTManager(context);
                Console.WriteLine($"FFT Manager initialized with {fftManager.AvailableAccelerators.Count} accelerators");
                
                if (fftManager.DefaultAccelerator != null)
                {
                    Console.WriteLine($"Default accelerator: {fftManager.DefaultAccelerator.Name}");
                    Console.WriteLine($"Performance info: {fftManager.DefaultAccelerator.PerformanceInfo.RelativePerformance:F1}x relative performance");
                }
                else
                {
                    Console.WriteLine("No FFT accelerators available (expected - implementations are placeholders)");
                }
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("FFT accelerators use placeholder implementations");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FFT Manager: {ex.Message}");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Demonstrates FFT algorithms and signal processing.
        /// </summary>
        static void DemonstrateFFTAlgorithms(Context context)
        {
            Console.WriteLine("📊 FFT Algorithms and Signal Processing");
            Console.WriteLine("=======================================\n");

            using var accelerator = context.CreateCPUAccelerator();

            // Test data
            var signalLength = 1024;
            var testSignal = GenerateTestSignal(signalLength, 100.0, 44100.0);
            var testKernel = GenerateImpulseResponse(64);

            Console.WriteLine("High-Level Algorithm Examples:");
            Console.WriteLine($"• Signal length: {signalLength} samples");
            Console.WriteLine($"• Kernel length: {testKernel.Length} samples");
            Console.WriteLine();

            try
            {
                // Demonstrate FFT algorithms (these use placeholder implementations)
                Console.WriteLine("1D FFT Analysis:");
                var complexInput = new Complex[signalLength];
                for (int i = 0; i < signalLength; i++)
                {
                    complexInput[i] = new Complex(testSignal[i], 0);
                }
                
                var fftResult = FFTAlgorithms.FFT1D(accelerator, complexInput);
                Console.WriteLine($"   ✓ Complex FFT completed: {fftResult.Length} output samples");

                var realFFTResult = FFTAlgorithms.FFT1DReal(accelerator, testSignal);
                Console.WriteLine($"   ✓ Real FFT completed: {realFFTResult.Length} output samples");

                // Signal processing operations
                Console.WriteLine("\nSignal Processing Operations:");
                
                var convResult = FFTAlgorithms.Convolve(accelerator, testSignal, testKernel);
                Console.WriteLine($"   ✓ Convolution completed: {convResult.Length} output samples");

                var psd = FFTAlgorithms.PowerSpectralDensity(accelerator, testSignal);
                Console.WriteLine($"   ✓ Power spectral density: {psd.Length} frequency bins");

                // Window functions
                var windowedSignal = FFTAlgorithms.ApplyWindow(testSignal, WindowFunction.Hann);
                Console.WriteLine($"   ✓ Hann window applied: {windowedSignal.Length} samples");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   Note: {ex.Message}");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Demonstrates FFT plans for optimized repeated operations.
        /// </summary>
        static void DemonstrateFFTPlans(Context context)
        {
            Console.WriteLine("⚡ FFT Plans for Optimized Operations");
            Console.WriteLine("====================================\n");

            try
            {
                // Create FFT plans
                var plan1D = FFTAlgorithms.CreatePlan1D(context, 1024, false);
                Console.WriteLine($"1D Complex FFT Plan:");
                Console.WriteLine($"   ✓ Created for length {plan1D.Length}");
                Console.WriteLine($"   ✓ Optimal length: {plan1D.OptimalLength}");
                Console.WriteLine($"   ✓ Plan valid: {plan1D.IsValid}");

                var realPlan = FFTAlgorithms.CreatePlan1D(context, 1024, true);
                Console.WriteLine($"\n1D Real FFT Plan:");
                Console.WriteLine($"   ✓ Created for length {realPlan.Length}");
                Console.WriteLine($"   ✓ Real FFT: {realPlan.IsReal}");

                var plan2D = FFTAlgorithms.CreatePlan2D(context, 512, 512);
                Console.WriteLine($"\n2D Complex FFT Plan:");
                Console.WriteLine($"   ✓ Created for size {plan2D.Width}x{plan2D.Height}");
                Console.WriteLine($"   ✓ Optimal size: {plan2D.OptimalExtent.X}x{plan2D.OptimalExtent.Y}");

                // Plans would be used for repeated operations in production
                Console.WriteLine("\nPlan Usage Benefits:");
                Console.WriteLine("• Pre-optimized for specific sizes");
                Console.WriteLine("• Reusable for multiple operations");
                Console.WriteLine("• Hardware-specific optimizations");
                Console.WriteLine("• Reduced setup overhead");

                // Cleanup
                plan1D.Dispose();
                realPlan.Dispose();
                plan2D.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Plan demonstration: {ex.Message}");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Provides guidance for implementing production FFT operations.
        /// </summary>
        static void ProvideImplementationGuidance()
        {
            Console.WriteLine("🚀 Implementation Guidance");
            Console.WriteLine("==========================\n");

            Console.WriteLine("Current Status:");
            Console.WriteLine("• ✅ Complete FFT framework architecture");
            Console.WriteLine("• ✅ Native library bindings (Intel IPP, cuFFT)");
            Console.WriteLine("• ✅ Automatic hardware detection and selection");
            Console.WriteLine("• ✅ Performance estimation and optimization");
            Console.WriteLine("• ⏳ Full FFT implementations (require additional integration)");
            Console.WriteLine();

            Console.WriteLine("Next Steps for Production:");
            Console.WriteLine("1. Complete Intel IPP memory buffer integration");
            Console.WriteLine("2. Implement cuFFT ArrayView marshalling");
            Console.WriteLine("3. Add real FFT kernel implementations");
            Console.WriteLine("4. Integrate with ILGPU memory management");
            Console.WriteLine("5. Add comprehensive testing and validation");
            Console.WriteLine();

            Console.WriteLine("Performance Optimization Tips:");
            Console.WriteLine("• Use power-of-2 sizes when possible (2x faster)");
            Console.WriteLine("• Batch multiple FFTs together for better throughput");
            Console.WriteLine("• Reuse FFT plans for repeated operations");
            Console.WriteLine("• Consider real FFTs for real-valued signals");
            Console.WriteLine("• Align data to accelerator requirements");
            Console.WriteLine();

            Console.WriteLine("Hardware Selection Guidelines:");
            Console.WriteLine("• Intel IPP: High-precision CPU computations, real-time audio");
            Console.WriteLine("• CUDA cuFFT: Large datasets, batch processing, deep learning");
            Console.WriteLine("• Automatic selection based on workload characteristics");
            Console.WriteLine("• Fallback to CPU implementations when needed");
            Console.WriteLine();
        }

        #region Helper Methods

        static float[] GenerateTestSignal(int length, double frequency, double sampleRate)
        {
            var signal = new float[length];
            var dt = 1.0 / sampleRate;
            
            for (int i = 0; i < length; i++)
            {
                var t = i * dt;
                signal[i] = (float)Math.Sin(2.0 * Math.PI * frequency * t);
            }
            
            return signal;
        }

        static float[] GenerateImpulseResponse(int length)
        {
            var impulse = new float[length];
            
            // Simple low-pass filter impulse response
            for (int i = 0; i < length; i++)
            {
                var t = (i - length / 2.0) / (length / 2.0);
                impulse[i] = (float)(Math.Exp(-t * t * 5.0) * Math.Cos(Math.PI * t));
            }
            
            return impulse;
        }

        #endregion
    }
}