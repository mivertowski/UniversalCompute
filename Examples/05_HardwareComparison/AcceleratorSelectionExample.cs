// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.NPU;
using ILGPU.Intel.AMX;

namespace ILGPU.Examples.HardwareComparison
{
    /// <summary>
    /// Demonstrates how to automatically select the best hardware accelerator
    /// for different types of workloads based on performance characteristics.
    /// </summary>
    class AcceleratorSelectionExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🎯 Hardware Accelerator Selection Guide");
            Console.WriteLine("======================================\n");

            try
            {
                using var context = Context.CreateDefault();
                
                // Detect available accelerators
                var availableAccelerators = await DetectAvailableAccelerators(context);
                
                if (availableAccelerators.Count == 0)
                {
                    Console.WriteLine("❌ No hardware accelerators detected.");
                    return;
                }

                Console.WriteLine($"🔍 Found {availableAccelerators.Count} accelerator(s):\n");
                foreach (var acc in availableAccelerators)
                {
                    Console.WriteLine($"   • {acc.Name}: {acc.Type}");
                    Console.WriteLine($"     └─ Peak Performance: {acc.PeakTOPS:F1} TOPS");
                    Console.WriteLine($"     └─ Power Efficiency: {acc.PowerEfficiency:F1} TOPS/W");
                    Console.WriteLine($"     └─ Best For: {string.Join(", ", acc.OptimalWorkloads)}");
                    Console.WriteLine();
                }

                // Run workload-specific recommendations
                await RunWorkloadAnalysis(availableAccelerators);
                
                // Run performance comparison
                await RunPerformanceComparison(context, availableAccelerators);
                
                // Provide selection guidance
                ProvideSelectionGuidance(availableAccelerators);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }

            Console.WriteLine("\n✅ Example completed. Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// Detects all available hardware accelerators and their capabilities.
        /// </summary>
        static async Task<List<AcceleratorInfo>> DetectAvailableAccelerators(Context context)
        {
            var accelerators = new List<AcceleratorInfo>();

            // Check Apple Neural Engine
            if (ANECapabilities.DetectNeuralEngine())
            {
                var capabilities = ANECapabilities.Query();
                accelerators.Add(new AcceleratorInfo
                {
                    Name = $"Apple Neural Engine {capabilities.Generation}",
                    Type = AcceleratorType.AppleNeuralEngine,
                    PeakTOPS = capabilities.MaxTOPS,
                    PowerEfficiency = capabilities.GetPowerEfficiency(),
                    MemoryBandwidth = 200.0, // Unified memory estimate
                    OptimalWorkloads = new[] { "Neural Networks", "Computer Vision", "NLP", "Real-time Inference" },
                    Strengths = new[] { "Ultra-low power", "Optimized for iOS/macOS", "Hardware acceleration for Core ML" },
                    Limitations = new[] { "Apple Silicon only", "Limited model formats", "Fixed-function units" }
                });
            }

            // Check Intel NPU
            if (NPUCapabilities.DetectNPU())
            {
                var capabilities = NPUCapabilities.Query();
                accelerators.Add(new AcceleratorInfo
                {
                    Name = $"Intel NPU {capabilities.Generation}",
                    Type = AcceleratorType.IntelNPU,
                    PeakTOPS = capabilities.MaxTOPS,
                    PowerEfficiency = capabilities.GetEstimatedPower(100) > 0 ? capabilities.MaxTOPS / capabilities.GetEstimatedPower(100) : 0,
                    MemoryBandwidth = capabilities.MemoryBandwidth,
                    OptimalWorkloads = new[] { "ONNX Models", "Edge AI", "Batch Inference", "Quantized Networks" },
                    Strengths = new[] { "ONNX runtime integration", "INT8/BF16 support", "OpenVINO ecosystem" },
                    Limitations = new[] { "Intel platforms only", "Limited by memory bandwidth", "Best for specific model types" }
                });
            }

            // Check Intel AMX
            if (AMXCapabilities.CheckAMXSupport())
            {
                var capabilities = AMXCapabilities.QueryCapabilities();
                accelerators.Add(new AcceleratorInfo
                {
                    Name = "Intel AMX",
                    Type = AcceleratorType.IntelAMX,
                    PeakTOPS = EstimateAMXTOPS(capabilities),
                    PowerEfficiency = EstimateAMXTOPS(capabilities) / 75.0, // Rough power estimate
                    MemoryBandwidth = capabilities.EstimatedBandwidthGBps,
                    OptimalWorkloads = new[] { "Dense Linear Algebra", "GEMM Operations", "Training Acceleration", "HPC Workloads" },
                    Strengths = new[] { "Massive matrix throughput", "Multiple precisions", "CPU integration" },
                    Limitations = new[] { "Limited tile size", "Xeon/Core processors only", "Programming complexity" }
                });
            }

            // Check standard CUDA GPUs
            try
            {
                foreach (var device in context.GetCudaDevices())
                {
                    accelerators.Add(new AcceleratorInfo
                    {
                        Name = device.Name,
                        Type = AcceleratorType.Cuda,
                        PeakTOPS = EstimateGPUTOPS(device),
                        PowerEfficiency = EstimateGPUTOPS(device) / EstimateGPUPower(device),
                        MemoryBandwidth = EstimateGPUMemoryBandwidth(device),
                        OptimalWorkloads = new[] { "Parallel Computing", "Deep Learning Training", "Scientific Computing", "Rendering" },
                        Strengths = new[] { "Massive parallelism", "CUDA ecosystem", "High memory bandwidth" },
                        Limitations = new[] { "High power consumption", "Complex programming model", "Memory limited" }
                    });
                }
            }
            catch { /* CUDA not available */ }

            // Add CPU as baseline
            accelerators.Add(new AcceleratorInfo
            {
                Name = "CPU (Baseline)",
                Type = AcceleratorType.Cpu,
                PeakTOPS = 0.5, // Typical CPU SIMD performance
                PowerEfficiency = 0.01, // Very low efficiency for AI workloads
                MemoryBandwidth = 100.0, // Typical DDR bandwidth
                OptimalWorkloads = new[] { "Control Logic", "I/O Operations", "Single-threaded Tasks", "General Computing" },
                Strengths = new[] { "Universal compatibility", "Flexible programming", "Large memory capacity" },
                Limitations = new[] { "Low parallel throughput", "Poor AI efficiency", "High power per operation" }
            });

            return accelerators;
        }

        /// <summary>
        /// Analyzes different workload types and recommends optimal accelerators.
        /// </summary>
        static async Task RunWorkloadAnalysis(List<AcceleratorInfo> accelerators)
        {
            Console.WriteLine("📊 Workload-Specific Recommendations");
            Console.WriteLine("===================================\n");

            var workloads = new[]
            {
                new WorkloadScenario
                {
                    Name = "Real-time Object Detection",
                    Description = "Mobile app detecting objects in camera feed",
                    Requirements = new[] { "Low latency (<16ms)", "Low power", "Edge deployment" },
                    OptimalTypes = new[] { AcceleratorType.AppleNeuralEngine, AcceleratorType.IntelNPU }
                },
                new WorkloadScenario
                {
                    Name = "Large Language Model Inference",
                    Description = "Running GPT-style models for text generation",
                    Requirements = new[] { "High memory bandwidth", "FP16/BF16 support", "Batch processing" },
                    OptimalTypes = new[] { AcceleratorType.Cuda, AcceleratorType.IntelAMX }
                },
                new WorkloadScenario
                {
                    Name = "Scientific Matrix Computation",
                    Description = "Dense linear algebra for simulation",
                    Requirements = new[] { "High FLOPS", "Double precision", "Large matrices" },
                    OptimalTypes = new[] { AcceleratorType.IntelAMX, AcceleratorType.Cuda }
                },
                new WorkloadScenario
                {
                    Name = "Edge AI Inference",
                    Description = "Running lightweight models on edge devices",
                    Requirements = new[] { "Ultra-low power", "Small models", "Fast startup" },
                    OptimalTypes = new[] { AcceleratorType.AppleNeuralEngine, AcceleratorType.IntelNPU }
                },
                new WorkloadScenario
                {
                    Name = "Deep Learning Training",
                    Description = "Training neural networks with backpropagation",
                    Requirements = new[] { "Mixed precision", "Large memory", "High throughput" },
                    OptimalTypes = new[] { AcceleratorType.Cuda, AcceleratorType.IntelAMX }
                }
            };

            foreach (var workload in workloads)
            {
                Console.WriteLine($"🎯 {workload.Name}");
                Console.WriteLine($"   {workload.Description}");
                Console.WriteLine($"   Requirements: {string.Join(", ", workload.Requirements)}");
                
                var recommendations = GetRecommendations(accelerators, workload);
                Console.WriteLine($"   Recommended Accelerators:");
                
                foreach (var (accelerator, score) in recommendations.Take(3))
                {
                    var ranking = recommendations.ToList().IndexOf((accelerator, score)) switch
                    {
                        0 => "🥇 Best",
                        1 => "🥈 Good",
                        2 => "🥉 Alternative",
                        _ => "⭐ Option"
                    };
                    
                    Console.WriteLine($"      {ranking}: {accelerator.Name} (Score: {score:F1})");
                    Console.WriteLine($"         └─ {accelerator.PeakTOPS:F1} TOPS, {accelerator.PowerEfficiency:F1} TOPS/W");
                }
                
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Runs performance comparison across different accelerators.
        /// </summary>
        static async Task RunPerformanceComparison(Context context, List<AcceleratorInfo> accelerators)
        {
            Console.WriteLine("⚡ Performance Comparison Matrix");
            Console.WriteLine("==============================\n");

            var testCases = new[]
            {
                new { Name = "Conv2D (224x224)", Type = "Computer Vision", Complexity = 1.0 },
                new { Name = "GEMM (1024x1024)", Type = "Linear Algebra", Complexity = 1.5 },
                new { Name = "Attention (512 seq)", Type = "NLP", Complexity = 2.0 },
                new { Name = "Batch Inference (8x)", Type = "Throughput", Complexity = 0.8 }
            };

            Console.WriteLine($"{"Accelerator",-25} {"Conv2D",-12} {"GEMM",-12} {"Attention",-12} {"Batch",-12} {"Power",-10}");
            Console.WriteLine($"{new string('-', 85)}");

            foreach (var accelerator in accelerators)
            {
                var results = new List<string>();
                
                foreach (var testCase in testCases)
                {
                    var performance = EstimatePerformance(accelerator, testCase.Type, testCase.Complexity);
                    results.Add($"{performance:F1}");
                }
                
                var powerRating = GetPowerRating(accelerator.PowerEfficiency);
                var resultsStr = string.Join("      ", results.Select(r => r.PadLeft(8)));
                
                Console.WriteLine($"{accelerator.Name,-25} {resultsStr} {powerRating,-10}");
            }

            Console.WriteLine();
            Console.WriteLine("📝 Legend:");
            Console.WriteLine("   Numbers show relative performance (higher is better)");
            Console.WriteLine("   Power: ⭐⭐⭐⭐⭐ = Excellent, ⭐ = Poor");
            Console.WriteLine();
        }

        /// <summary>
        /// Provides comprehensive selection guidance based on use cases.
        /// </summary>
        static void ProvideSelectionGuidance(List<AcceleratorInfo> accelerators)
        {
            Console.WriteLine("🎯 Accelerator Selection Guide");
            Console.WriteLine("=============================\n");

            // Group accelerators by primary use case
            var guidelines = new[]
            {
                new SelectionGuideline
                {
                    Title = "🧠 AI/ML Inference (Production)",
                    Criteria = "Low latency, high efficiency, reliable",
                    Recommendations = accelerators
                        .Where(a => a.Type == AcceleratorType.AppleNeuralEngine || a.Type == AcceleratorType.IntelNPU)
                        .OrderByDescending(a => a.PowerEfficiency)
                        .Take(2)
                        .ToArray()
                },
                new SelectionGuideline
                {
                    Title = "⚡ High-Performance Computing",
                    Criteria = "Maximum throughput, precision flexibility",
                    Recommendations = accelerators
                        .Where(a => a.Type == AcceleratorType.IntelAMX || a.Type == AcceleratorType.Cuda)
                        .OrderByDescending(a => a.PeakTOPS)
                        .Take(2)
                        .ToArray()
                },
                new SelectionGuideline
                {
                    Title = "📱 Mobile/Edge Applications",
                    Criteria = "Ultra-low power, small footprint",
                    Recommendations = accelerators
                        .Where(a => a.PowerEfficiency > 10) // High efficiency threshold
                        .OrderByDescending(a => a.PowerEfficiency)
                        .Take(2)
                        .ToArray()
                },
                new SelectionGuideline
                {
                    Title = "🔬 Research & Development",
                    Criteria = "Flexibility, debugging capabilities",
                    Recommendations = accelerators
                        .Where(a => a.Type == AcceleratorType.Cuda || a.Type == AcceleratorType.Cpu)
                        .OrderByDescending(a => a.PeakTOPS)
                        .Take(2)
                        .ToArray()
                }
            };

            foreach (var guideline in guidelines)
            {
                Console.WriteLine($"{guideline.Title}");
                Console.WriteLine($"   Criteria: {guideline.Criteria}");
                
                if (guideline.Recommendations.Any())
                {
                    Console.WriteLine("   Recommended:");
                    foreach (var acc in guideline.Recommendations)
                    {
                        Console.WriteLine($"      • {acc.Name}");
                        Console.WriteLine($"        └─ {acc.PeakTOPS:F1} TOPS, {acc.PowerEfficiency:F1} TOPS/W");
                        Console.WriteLine($"        └─ Best for: {string.Join(", ", acc.OptimalWorkloads.Take(2))}");
                    }
                }
                else
                {
                    Console.WriteLine("   No suitable accelerators found for this use case");
                }
                Console.WriteLine();
            }

            // Decision tree
            Console.WriteLine("🌳 Decision Tree:");
            Console.WriteLine("================");
            Console.WriteLine();
            Console.WriteLine("1. Are you building a mobile/edge application?");
            Console.WriteLine("   └─ YES: Apple Neural Engine (iOS) or Intel NPU (x86)");
            Console.WriteLine("   └─ NO: Continue to question 2");
            Console.WriteLine();
            Console.WriteLine("2. Do you need maximum computational throughput?");
            Console.WriteLine("   └─ YES: CUDA GPU or Intel AMX (for matrix operations)");
            Console.WriteLine("   └─ NO: Continue to question 3");
            Console.WriteLine();
            Console.WriteLine("3. Is power efficiency a primary concern?");
            Console.WriteLine("   └─ YES: Neural Engine > NPU > AMX > GPU > CPU");
            Console.WriteLine("   └─ NO: GPU > AMX > NPU > Neural Engine > CPU");
            Console.WriteLine();
            Console.WriteLine("4. What type of workload?");
            Console.WriteLine("   └─ Neural Networks: ANE, NPU, GPU");
            Console.WriteLine("   └─ Matrix Operations: AMX, GPU, CPU");
            Console.WriteLine("   └─ General Compute: GPU, CPU");
            Console.WriteLine("   └─ Control Logic: CPU");
            Console.WriteLine();

            // Best practices
            Console.WriteLine("💡 Best Practices:");
            Console.WriteLine("==================");
            Console.WriteLine("• Profile your specific workload on available hardware");
            Console.WriteLine("• Consider total cost of ownership (hardware + power + development)");
            Console.WriteLine("• Test across different batch sizes and model complexities");
            Console.WriteLine("• Evaluate precision requirements (FP32 vs BF16 vs INT8)");
            Console.WriteLine("• Consider deployment constraints (driver dependencies, OS support)");
            Console.WriteLine("• Plan for graceful fallbacks when optimal hardware isn't available");
            Console.WriteLine();
        }

        #region Helper Methods and Classes

        /// <summary>
        /// Gets recommendations for a specific workload scenario.
        /// </summary>
        static IOrderedEnumerable<(AcceleratorInfo, double)> GetRecommendations(
            List<AcceleratorInfo> accelerators, WorkloadScenario workload)
        {
            return accelerators
                .Select(acc => (acc, CalculateWorkloadScore(acc, workload)))
                .OrderByDescending(x => x.Item2);
        }

        /// <summary>
        /// Calculates a score for how well an accelerator fits a workload.
        /// </summary>
        static double CalculateWorkloadScore(AcceleratorInfo accelerator, WorkloadScenario workload)
        {
            double score = 0;

            // Base performance score
            score += accelerator.PeakTOPS * 10;

            // Power efficiency bonus
            score += accelerator.PowerEfficiency * 5;

            // Type matching bonus
            if (workload.OptimalTypes.Contains(accelerator.Type))
            {
                score += 50;
            }

            // Specific bonuses based on workload requirements
            foreach (var requirement in workload.Requirements)
            {
                if (requirement.Contains("Low power") && accelerator.PowerEfficiency > 10)
                    score += 30;
                if (requirement.Contains("High memory bandwidth") && accelerator.MemoryBandwidth > 300)
                    score += 20;
                if (requirement.Contains("Low latency") && (accelerator.Type == AcceleratorType.AppleNeuralEngine || accelerator.Type == AcceleratorType.IntelNPU))
                    score += 25;
            }

            return score;
        }

        /// <summary>
        /// Estimates performance for a specific workload type.
        /// </summary>
        static double EstimatePerformance(AcceleratorInfo accelerator, string workloadType, double complexity)
        {
            var basePerformance = accelerator.PeakTOPS;

            // Apply workload-specific multipliers
            var multiplier = (accelerator.Type, workloadType) switch
            {
                (AcceleratorType.AppleNeuralEngine, "Computer Vision") => 1.2,
                (AcceleratorType.AppleNeuralEngine, "NLP") => 1.1,
                (AcceleratorType.IntelNPU, "Computer Vision") => 1.0,
                (AcceleratorType.IntelNPU, "NLP") => 1.1,
                (AcceleratorType.IntelAMX, "Linear Algebra") => 1.3,
                (AcceleratorType.IntelAMX, "Throughput") => 0.8,
                (AcceleratorType.Cuda, "Computer Vision") => 1.1,
                (AcceleratorType.Cuda, "Linear Algebra") => 1.2,
                (AcceleratorType.Cuda, "Throughput") => 1.4,
                _ => 0.7
            };

            return basePerformance * multiplier / complexity;
        }

        /// <summary>
        /// Gets a power efficiency rating as stars.
        /// </summary>
        static string GetPowerRating(double efficiency)
        {
            return efficiency switch
            {
                > 15 => "⭐⭐⭐⭐⭐",
                > 10 => "⭐⭐⭐⭐",
                > 5 => "⭐⭐⭐",
                > 1 => "⭐⭐",
                _ => "⭐"
            };
        }

        /// <summary>
        /// Estimates AMX TOPS performance.
        /// </summary>
        static double EstimateAMXTOPS(AMXNativeCapabilities capabilities)
        {
            // AMX can achieve high throughput for specific matrix operations
            return capabilities.EstimatedBandwidthGBps / 100.0; // Rough estimate
        }

        /// <summary>
        /// Estimates GPU TOPS performance.
        /// </summary>
        static double EstimateGPUTOPS(Device device)
        {
            // Very rough estimate based on memory size as proxy for GPU class
            var memoryGB = device.MemorySize / (1024.0 * 1024.0 * 1024.0);
            return memoryGB * 10; // Rough approximation
        }

        /// <summary>
        /// Estimates GPU power consumption.
        /// </summary>
        static double EstimateGPUPower(Device device)
        {
            var memoryGB = device.MemorySize / (1024.0 * 1024.0 * 1024.0);
            return Math.Max(50, memoryGB * 30); // Rough power estimate
        }

        /// <summary>
        /// Estimates GPU memory bandwidth.
        /// </summary>
        static double EstimateGPUMemoryBandwidth(Device device)
        {
            var memoryGB = device.MemorySize / (1024.0 * 1024.0 * 1024.0);
            return memoryGB * 50; // Rough bandwidth estimate
        }

        #endregion

        #region Data Structures

        /// <summary>
        /// Information about an available accelerator.
        /// </summary>
        public class AcceleratorInfo
        {
            public string Name { get; set; } = "";
            public AcceleratorType Type { get; set; }
            public double PeakTOPS { get; set; }
            public double PowerEfficiency { get; set; }
            public double MemoryBandwidth { get; set; }
            public string[] OptimalWorkloads { get; set; } = Array.Empty<string>();
            public string[] Strengths { get; set; } = Array.Empty<string>();
            public string[] Limitations { get; set; } = Array.Empty<string>();
        }

        /// <summary>
        /// Workload scenario for testing recommendations.
        /// </summary>
        public class WorkloadScenario
        {
            public string Name { get; set; } = "";
            public string Description { get; set; } = "";
            public string[] Requirements { get; set; } = Array.Empty<string>();
            public AcceleratorType[] OptimalTypes { get; set; } = Array.Empty<AcceleratorType>();
        }

        /// <summary>
        /// Selection guideline for specific use cases.
        /// </summary>
        public class SelectionGuideline
        {
            public string Title { get; set; } = "";
            public string Criteria { get; set; } = "";
            public AcceleratorInfo[] Recommendations { get; set; } = Array.Empty<AcceleratorInfo>();
        }

        /// <summary>
        /// Types of accelerators supported.
        /// </summary>
        public enum AcceleratorType
        {
            Cpu,
            Cuda,
            OpenCL,
            AppleNeuralEngine,
            IntelNPU,
            IntelAMX
        }

        #endregion
    }
}