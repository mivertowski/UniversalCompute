// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.FFT;
using ILGPU.Algorithms.FFT;
using ILGPU.Examples.Common;

namespace ILGPU.Examples.FFTOperations
{
    /// <summary>
    /// Comprehensive FFT performance comparison and benchmarking example.
    /// Demonstrates automatic hardware selection and performance optimization.
    /// </summary>
    class FFTPerformanceComparison
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🔄 ILGPU FFT Performance Comparison");
            Console.WriteLine("===================================\n");

            try
            {
                // Detect available hardware
                var hardware = HardwareDetection.DetectAvailableHardware();
                HardwareDetection.PrintHardwareReport(hardware);

                using var context = Context.CreateDefault();
                
                // Run comprehensive FFT benchmarks
                await RunFFTAcceleratorComparison(context);
                await RunFFTSizeScaling(context);
                await RunSignalProcessingBenchmarks(context);
                await RunBatchProcessingBenchmarks(context);
                await RunRealWorldApplications(context);

                // Provide optimization recommendations
                ProvideOptimizationRecommendations(context);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }

            Console.WriteLine("\n✅ FFT comparison completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Compares FFT performance across different accelerator types.
        /// </summary>
        static async Task RunFFTAcceleratorComparison(Context context)
        {
            Console.WriteLine("⚡ FFT Accelerator Performance Comparison");
            Console.WriteLine("=========================================\n");

            using var fftManager = new FFTManager(context);
            
            if (!fftManager.HasAccelerators)
            {
                Console.WriteLine("No FFT accelerators available for comparison.\n");
                return;
            }

            var testSizes = new[] { 1024, 4096, 16384, 65536 };
            var results = new Dictionary<FFTAccelerator, List<BenchmarkResult>>();

            Console.WriteLine($"{"Accelerator",-25} {"Size",-8} {"1D FFT",-12} {"Real FFT",-12} {"2D FFT",-12} {"GFLOPS",-10}");
            Console.WriteLine($"{new string('-', 80)}");

            foreach (var accelerator in fftManager.AvailableAccelerators.Where(a => a.IsAvailable))
            {
                results[accelerator] = new List<BenchmarkResult>();

                foreach (var size in testSizes)
                {
                    if (!accelerator.IsSizeSupported(size))
                        continue;

                    var result = await BenchmarkFFTAccelerator(accelerator, size);
                    results[accelerator].Add(result);

                    Console.WriteLine($"{accelerator.Name,-25} {size,-8} " +
                                    $"{result.FFT1DTime,-12:F2} ms " +
                                    $"{result.RealFFTTime,-12:F2} ms " +
                                    $"{result.FFT2DTime,-12:F2} ms " +
                                    $"{result.GFLOPS,-10:F1}");
                }
                Console.WriteLine();
            }

            // Analysis
            Console.WriteLine("📊 Performance Analysis:");
            AnalyzeFFTResults(results);
            Console.WriteLine();
        }

        /// <summary>
        /// Tests FFT performance scaling with different sizes.
        /// </summary>
        static async Task RunFFTSizeScaling(Context context)
        {
            Console.WriteLine("📈 FFT Size Scaling Analysis");
            Console.WriteLine("============================\n");

            using var fftManager = new FFTManager(context);
            var bestAccelerator = fftManager.DefaultAccelerator;

            if (bestAccelerator == null)
            {
                Console.WriteLine("No FFT accelerator available for scaling analysis.\n");
                return;
            }

            Console.WriteLine($"Using accelerator: {bestAccelerator.Name}\n");

            var sizes = new[] { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
            var scalingResults = new List<ScalingResult>();

            Console.WriteLine($"{"Size",-8} {"Time (ms)",-12} {"GFLOPS",-10} {"Efficiency",-12} {"Memory (MB)",-12}");
            Console.WriteLine($"{new string('-', 60)}");

            foreach (var size in sizes)
            {
                if (!bestAccelerator.IsSizeSupported(size))
                    continue;

                var result = await BenchmarkFFTScaling(bestAccelerator, size);
                scalingResults.Add(result);

                Console.WriteLine($"{size,-8} {result.Time,-12:F2} {result.GFLOPS,-10:F1} " +
                                $"{result.Efficiency,-12:F1}% {result.MemoryMB,-12:F1}");
            }

            Console.WriteLine();
            AnalyzeScalingResults(scalingResults);
            Console.WriteLine();
        }

        /// <summary>
        /// Benchmarks signal processing operations using FFT.
        /// </summary>
        static async Task RunSignalProcessingBenchmarks(Context context)
        {
            Console.WriteLine("🎵 Signal Processing Benchmarks");
            Console.WriteLine("===============================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var signalLength = 8192;
            var kernelLength = 256;

            // Generate test signals
            var signal = GenerateTestSignal(signalLength, 1000.0, 44100.0); // 1kHz signal at 44.1kHz sample rate
            var noiseSignal = AddNoise(signal, 0.1);
            var impulseResponse = GenerateImpulseResponse(kernelLength);

            Console.WriteLine($"Signal length: {signalLength} samples");
            Console.WriteLine($"Kernel length: {kernelLength} samples");
            Console.WriteLine($"Sample rate: 44.1 kHz\n");

            var operations = new[]
            {
                new { Name = "FFT Analysis", Operation = (Func<Task<double>>)(() => BenchmarkFFTAnalysis(accelerator, signal)) },
                new { Name = "Convolution", Operation = (Func<Task<double>>)(() => BenchmarkConvolution(accelerator, signal, impulseResponse)) },
                new { Name = "Spectral Filtering", Operation = (Func<Task<double>>)(() => BenchmarkSpectralFiltering(accelerator, noiseSignal)) },
                new { Name = "Power Spectrum", Operation = (Func<Task<double>>)(() => BenchmarkPowerSpectrum(accelerator, signal)) }
            };

            Console.WriteLine($"{"Operation",-20} {"Time (ms)",-12} {"Throughput",-15}");
            Console.WriteLine($"{new string('-', 50)}");

            foreach (var op in operations)
            {
                var time = await op.Operation();
                var throughput = signalLength / (time * 1000.0); // MSamples/sec
                
                Console.WriteLine($"{op.Name,-20} {time,-12:F2} {throughput,-15:F1} MSamp/s");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Benchmarks batch FFT processing capabilities.
        /// </summary>
        static async Task RunBatchProcessingBenchmarks(Context context)
        {
            Console.WriteLine("🔄 Batch Processing Benchmarks");
            Console.WriteLine("==============================\n");

            using var fftManager = new FFTManager(context);
            var bestAccelerator = fftManager.DefaultAccelerator;

            if (bestAccelerator == null)
            {
                Console.WriteLine("No FFT accelerator available for batch processing.\n");
                return;
            }

            var batchSizes = new[] { 1, 4, 16, 64 };
            var fftSize = 1024;

            Console.WriteLine($"FFT Size: {fftSize}\n");
            Console.WriteLine($"{"Batch Size",-12} {"Total Time",-12} {"Per FFT",-12} {"Speedup",-10} {"Efficiency",-12}");
            Console.WriteLine($"{new string('-', 65)}");

            double baselineTime = 0;

            foreach (var batchSize in batchSizes)
            {
                var result = await BenchmarkBatchFFT(bestAccelerator, fftSize, batchSize);
                var perFFTTime = result.TotalTime / batchSize;
                var speedup = baselineTime > 0 ? baselineTime / perFFTTime : 1.0;
                var efficiency = speedup / batchSize * 100.0;

                if (baselineTime == 0) baselineTime = perFFTTime;

                Console.WriteLine($"{batchSize,-12} {result.TotalTime,-12:F2} ms " +
                                $"{perFFTTime,-12:F2} ms {speedup,-10:F1}x {efficiency,-12:F1}%");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Demonstrates real-world FFT applications.
        /// </summary>
        static async Task RunRealWorldApplications(Context context)
        {
            Console.WriteLine("🌍 Real-World FFT Applications");
            Console.WriteLine("==============================\n");

            using var accelerator = context.CreateCPUAccelerator();

            var applications = new[]
            {
                new { Name = "Audio Processing", Size = 2048, Description = "Real-time audio frame processing" },
                new { Name = "Image Processing", Size = 512, Description = "2D image filtering and enhancement" },
                new { Name = "Radar Signal Processing", Size = 4096, Description = "Range-Doppler processing" },
                new { Name = "Spectral Analysis", Size = 8192, Description = "High-resolution spectrum analysis" },
                new { Name = "Communication Systems", Size = 1024, Description = "OFDM symbol processing" }
            };

            Console.WriteLine($"{"Application",-25} {"Size",-8} {"Latency",-12} {"Throughput",-12} {"Real-time",-12}");
            Console.WriteLine($"{new string('-', 75)}");

            foreach (var app in applications)
            {
                var result = await BenchmarkApplication(accelerator, app.Size, app.Name);
                var isRealTime = result.Latency < GetMaxLatency(app.Name);

                Console.WriteLine($"{app.Name,-25} {app.Size,-8} " +
                                $"{result.Latency,-12:F2} ms " +
                                $"{result.Throughput,-12:F1} fps " +
                                $"{(isRealTime ? "✅ Yes" : "❌ No"),-12}");
            }

            Console.WriteLine();
            Console.WriteLine("Real-time Requirements:");
            Console.WriteLine("• Audio Processing: < 23 ms (48kHz, 1024 samples)");
            Console.WriteLine("• Image Processing: < 33 ms (30 fps)");
            Console.WriteLine("• Radar Processing: < 10 ms (radar sweep)");
            Console.WriteLine("• Spectral Analysis: < 100 ms (interactive)");
            Console.WriteLine("• Communication: < 1 ms (LTE frame)");
            Console.WriteLine();
        }

        /// <summary>
        /// Provides optimization recommendations based on benchmarks.
        /// </summary>
        static void ProvideOptimizationRecommendations(Context context)
        {
            Console.WriteLine("💡 FFT Optimization Recommendations");
            Console.WriteLine("===================================\n");

            using var fftManager = new FFTManager(context);

            Console.WriteLine("🎯 Hardware Selection Guidelines:");
            
            if (fftManager.HasAccelerators)
            {
                var accelerators = fftManager.AvailableAccelerators.Where(a => a.IsAvailable).ToList();
                
                Console.WriteLine($"Available accelerators: {accelerators.Count}");
                foreach (var accelerator in accelerators)
                {
                    var perf = accelerator.PerformanceInfo;
                    Console.WriteLine($"• {accelerator.Name}:");
                    Console.WriteLine($"  └─ Best for: {GetUseCaseRecommendation(accelerator)}");
                    Console.WriteLine($"  └─ Optimal range: {perf.MinimumEfficientSize} - {perf.MaximumSize:N0} points");
                }
            }
            else
            {
                Console.WriteLine("• No hardware accelerators detected");
                Console.WriteLine("• Consider installing Intel IPP for CPU optimization");
                Console.WriteLine("• CUDA GPU would provide significant acceleration");
            }

            Console.WriteLine();
            Console.WriteLine("⚡ Performance Optimization Tips:");
            Console.WriteLine("• Use power-of-2 sizes when possible (2x faster)");
            Console.WriteLine("• Batch multiple FFTs together for better throughput");
            Console.WriteLine("• Reuse FFT plans for repeated operations");
            Console.WriteLine("• Consider real FFTs for real-valued signals");
            Console.WriteLine("• Use appropriate precision (FP32 vs FP64)");
            Console.WriteLine("• Align data to accelerator requirements");
            Console.WriteLine();

            Console.WriteLine("🔧 Implementation Best Practices:");
            Console.WriteLine("• Profile your specific use case and data sizes");
            Console.WriteLine("• Consider memory bandwidth limitations");
            Console.WriteLine("• Use asynchronous operations when possible");
            Console.WriteLine("• Minimize memory allocations in hot paths");
            Console.WriteLine("• Validate numerical accuracy for your application");
            Console.WriteLine();
        }

        #region Benchmark Implementation Methods

        static async Task<BenchmarkResult> BenchmarkFFTAccelerator(FFTAccelerator accelerator, int size)
        {
            const int iterations = 10;
            const int warmupIterations = 3;

            var result = new BenchmarkResult { Size = size };

            // Create test data
            var complexData = GenerateComplexTestData(size);
            var realData = GenerateRealTestData(size);

            using var complexInput = accelerator.ParentAccelerator.Allocate1D(complexData);
            using var complexOutput = accelerator.ParentAccelerator.Allocate1D<Complex>(size);
            using var realInput = accelerator.ParentAccelerator.Allocate1D(realData);
            using var realOutput = accelerator.ParentAccelerator.Allocate1D<Complex>(size / 2 + 1);

            // Benchmark 1D complex FFT
            for (int i = 0; i < warmupIterations; i++)
            {
                accelerator.FFT1D(complexInput, complexOutput, true);
                accelerator.ParentAccelerator.Synchronize();
            }

            var times = new List<double>();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.ParentAccelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                accelerator.FFT1D(complexInput, complexOutput, true);
                accelerator.ParentAccelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            result.FFT1DTime = times.Average();

            // Benchmark real FFT
            times.Clear();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.ParentAccelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                accelerator.FFT1DReal(realInput, realOutput);
                accelerator.ParentAccelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }
            result.RealFFTTime = times.Average();

            // Benchmark 2D FFT (if supported)
            try
            {
                var extent2D = new Index2D(32, size / 32); // Balanced 2D size
                using var input2D = accelerator.ParentAccelerator.Allocate2DDenseXY<Complex>(extent2D);
                using var output2D = accelerator.ParentAccelerator.Allocate2DDenseXY<Complex>(extent2D);

                times.Clear();
                for (int i = 0; i < iterations; i++)
                {
                    accelerator.ParentAccelerator.Synchronize();
                    var stopwatch = Stopwatch.StartNew();
                    accelerator.FFT2D(input2D, output2D, true);
                    accelerator.ParentAccelerator.Synchronize();
                    stopwatch.Stop();
                    times.Add(stopwatch.ElapsedMilliseconds);
                }
                result.FFT2DTime = times.Average();
            }
            catch
            {
                result.FFT2DTime = double.NaN; // Not supported
            }

            // Calculate GFLOPS (5 * N * log2(N) operations for complex FFT)
            var operations = 5.0 * size * Math.Log2(size);
            result.GFLOPS = operations / (result.FFT1DTime * 1e6);

            return result;
        }

        static async Task<ScalingResult> BenchmarkFFTScaling(FFTAccelerator accelerator, int size)
        {
            const int iterations = 5;
            
            var complexData = GenerateComplexTestData(size);
            using var input = accelerator.ParentAccelerator.Allocate1D(complexData);
            using var output = accelerator.ParentAccelerator.Allocate1D<Complex>(size);

            var times = new List<double>();
            for (int i = 0; i < iterations; i++)
            {
                accelerator.ParentAccelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                accelerator.FFT1D(input, output, true);
                accelerator.ParentAccelerator.Synchronize();
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
            }

            var avgTime = times.Average();
            var operations = 5.0 * size * Math.Log2(size);
            var gflops = operations / (avgTime * 1e6);
            var memoryMB = size * 16.0 / (1024 * 1024); // Complex numbers are 16 bytes
            var perf = accelerator.PerformanceInfo;
            var efficiency = gflops / perf.EstimatedGFLOPS * 100.0;

            return new ScalingResult
            {
                Size = size,
                Time = avgTime,
                GFLOPS = gflops,
                Efficiency = efficiency,
                MemoryMB = memoryMB
            };
        }

        static async Task<double> BenchmarkFFTAnalysis(Accelerator accelerator, float[] signal)
        {
            const int iterations = 100;
            
            var stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var fft = FFTAlgorithms.FFT1DReal(accelerator, signal);
            }
            stopwatch.Stop();

            return stopwatch.ElapsedMilliseconds / (double)iterations;
        }

        static async Task<double> BenchmarkConvolution(Accelerator accelerator, float[] signal, float[] kernel)
        {
            const int iterations = 50;
            
            var stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var result = FFTAlgorithms.Convolve(accelerator, signal, kernel);
            }
            stopwatch.Stop();

            return stopwatch.ElapsedMilliseconds / (double)iterations;
        }

        static async Task<double> BenchmarkSpectralFiltering(Accelerator accelerator, float[] signal)
        {
            const int iterations = 50;
            
            // Create a simple low-pass filter
            var fft = FFTAlgorithms.FFT1DReal(accelerator, signal);
            var filter = new Complex[fft.Length];
            for (int i = 0; i < filter.Length; i++)
            {
                // Low-pass filter (cutoff at 1/4 of Nyquist frequency)
                filter[i] = i < filter.Length / 4 ? Complex.One : Complex.Zero;
            }
            
            var stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var result = FFTAlgorithms.SpectralFilter(accelerator, signal, filter);
            }
            stopwatch.Stop();

            return stopwatch.ElapsedMilliseconds / (double)iterations;
        }

        static async Task<double> BenchmarkPowerSpectrum(Accelerator accelerator, float[] signal)
        {
            const int iterations = 100;
            
            var stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var psd = FFTAlgorithms.PowerSpectralDensity(accelerator, signal);
            }
            stopwatch.Stop();

            return stopwatch.ElapsedMilliseconds / (double)iterations;
        }

        static async Task<BatchResult> BenchmarkBatchFFT(FFTAccelerator accelerator, int fftSize, int batchSize)
        {
            var inputs = new MemoryBuffer<Complex>[batchSize];
            var outputs = new MemoryBuffer<Complex>[batchSize];

            try
            {
                // Create batch data
                for (int i = 0; i < batchSize; i++)
                {
                    var data = GenerateComplexTestData(fftSize);
                    inputs[i] = accelerator.ParentAccelerator.Allocate1D(data);
                    outputs[i] = accelerator.ParentAccelerator.Allocate1D<Complex>(fftSize);
                }

                accelerator.ParentAccelerator.Synchronize();
                var stopwatch = Stopwatch.StartNew();
                accelerator.BatchFFT1D(inputs, outputs, true);
                accelerator.ParentAccelerator.Synchronize();
                stopwatch.Stop();

                return new BatchResult
                {
                    BatchSize = batchSize,
                    TotalTime = stopwatch.ElapsedMilliseconds
                };
            }
            finally
            {
                foreach (var buffer in inputs) buffer?.Dispose();
                foreach (var buffer in outputs) buffer?.Dispose();
            }
        }

        static async Task<ApplicationResult> BenchmarkApplication(Accelerator accelerator, int size, string appName)
        {
            const int iterations = 100;
            
            var signal = GenerateTestSignal(size, 1000.0, 44100.0);
            
            var stopwatch = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var fft = FFTAlgorithms.FFT1DReal(accelerator, signal);
                // Simulate some processing
                var processed = new float[signal.Length];
                Array.Copy(signal, processed, signal.Length);
            }
            stopwatch.Stop();

            var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
            var throughput = 1000.0 / avgTime; // Operations per second

            return new ApplicationResult
            {
                Application = appName,
                Latency = avgTime,
                Throughput = throughput
            };
        }

        #endregion

        #region Helper Methods

        static Complex[] GenerateComplexTestData(int size)
        {
            var data = new Complex[size];
            var random = new Random(42);
            
            for (int i = 0; i < size; i++)
            {
                data[i] = new Complex(
                    (random.NextDouble() - 0.5) * 2.0,
                    (random.NextDouble() - 0.5) * 2.0);
            }
            
            return data;
        }

        static float[] GenerateRealTestData(int size)
        {
            var data = new float[size];
            var random = new Random(42);
            
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)((random.NextDouble() - 0.5) * 2.0);
            }
            
            return data;
        }

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

        static float[] AddNoise(float[] signal, double noiseLevel)
        {
            var noisy = new float[signal.Length];
            var random = new Random(123);
            
            for (int i = 0; i < signal.Length; i++)
            {
                var noise = (random.NextDouble() - 0.5) * 2.0 * noiseLevel;
                noisy[i] = signal[i] + (float)noise;
            }
            
            return noisy;
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

        static void AnalyzeFFTResults(Dictionary<FFTAccelerator, List<BenchmarkResult>> results)
        {
            if (results.Count == 0) return;

            var best = results
                .SelectMany(kvp => kvp.Value.Select(r => new { Accelerator = kvp.Key, Result = r }))
                .Where(x => !double.IsNaN(x.Result.GFLOPS))
                .OrderByDescending(x => x.Result.GFLOPS)
                .FirstOrDefault();

            if (best != null)
            {
                Console.WriteLine($"• Best performer: {best.Accelerator.Name} ({best.Result.GFLOPS:F1} GFLOPS)");
            }

            var consistentBest = results
                .Where(kvp => kvp.Value.Count > 0)
                .OrderByDescending(kvp => kvp.Value.Average(r => r.GFLOPS))
                .FirstOrDefault();

            if (consistentBest.Key != null)
            {
                Console.WriteLine($"• Most consistent: {consistentBest.Key.Name}");
            }
        }

        static void AnalyzeScalingResults(List<ScalingResult> results)
        {
            if (results.Count < 2) return;

            var bestEfficiency = results.OrderByDescending(r => r.Efficiency).First();
            var optimalSize = results.Where(r => r.Efficiency > 80).OrderBy(r => r.Size).FirstOrDefault();

            Console.WriteLine($"Analysis:");
            Console.WriteLine($"• Best efficiency: {bestEfficiency.Efficiency:F1}% at size {bestEfficiency.Size}");
            
            if (optimalSize != null)
            {
                Console.WriteLine($"• Minimum efficient size: {optimalSize.Size}");
            }

            // Check for power-of-2 performance advantage
            var powerOf2Avg = results.Where(r => FFTAlgorithms.IsPowerOf2(r.Size)).Average(r => r.GFLOPS);
            var nonPowerOf2Avg = results.Where(r => !FFTAlgorithms.IsPowerOf2(r.Size)).Average(r => r.GFLOPS);
            
            if (!double.IsNaN(powerOf2Avg) && !double.IsNaN(nonPowerOf2Avg))
            {
                var advantage = powerOf2Avg / nonPowerOf2Avg;
                Console.WriteLine($"• Power-of-2 advantage: {advantage:F1}x");
            }
        }

        static string GetUseCaseRecommendation(FFTAccelerator accelerator)
        {
            return accelerator.AcceleratorType switch
            {
                FFTAcceleratorType.IntelIPP => "High-precision CPU computations, real-time audio",
                FFTAcceleratorType.CUDA => "Large datasets, batch processing, deep learning",
                FFTAcceleratorType.OpenCL => "Cross-platform applications, mixed workloads",
                _ => "General purpose FFT operations"
            };
        }

        static double GetMaxLatency(string application)
        {
            return application switch
            {
                "Audio Processing" => 23.0,
                "Image Processing" => 33.0,
                "Radar Signal Processing" => 10.0,
                "Spectral Analysis" => 100.0,
                "Communication Systems" => 1.0,
                _ => 50.0
            };
        }

        #endregion

        #region Data Structures

        public class BenchmarkResult
        {
            public int Size { get; set; }
            public double FFT1DTime { get; set; }
            public double RealFFTTime { get; set; }
            public double FFT2DTime { get; set; }
            public double GFLOPS { get; set; }
        }

        public class ScalingResult
        {
            public int Size { get; set; }
            public double Time { get; set; }
            public double GFLOPS { get; set; }
            public double Efficiency { get; set; }
            public double MemoryMB { get; set; }
        }

        public class BatchResult
        {
            public int BatchSize { get; set; }
            public double TotalTime { get; set; }
        }

        public class ApplicationResult
        {
            public string Application { get; set; } = "";
            public double Latency { get; set; }
            public double Throughput { get; set; }
        }

        #endregion
    }
}