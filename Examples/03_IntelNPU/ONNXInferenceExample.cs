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
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Intel.NPU;
using ILGPU.Core;

namespace ILGPU.Examples.IntelNPU
{
    /// <summary>
    /// Demonstrates ONNX model inference on Intel NPU.
    /// This example shows how to run neural network inference using NPU optimization.
    /// </summary>
    class ONNXInferenceExample
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🔧 Intel NPU - ONNX Inference Example");
            Console.WriteLine("=====================================\n");

            if (!NPUCapabilities.DetectNPU())
            {
                Console.WriteLine("❌ Intel NPU not available on this system.");
                Console.WriteLine("This example requires Intel Core Ultra processors (Meteor Lake or newer).");
                Console.WriteLine("Ensure OpenVINO runtime is installed and NPU drivers are up to date.");
                return;
            }

            try
            {
                using var context = Context.CreateDefault();
                
                // Get NPU capabilities
                var capabilities = NPUCapabilities.Query();
                Console.WriteLine($"🔍 NPU Generation: {capabilities.Generation}");
                Console.WriteLine($"🔍 Max TOPS: {capabilities.MaxTOPS:F1}");
                Console.WriteLine($"🔍 Compute Units: {capabilities.ComputeUnits}");
                Console.WriteLine($"🔍 Memory Bandwidth: {capabilities.MemoryBandwidth:F1} GB/s");
                Console.WriteLine($"🔍 Supports BF16: {capabilities.SupportsBF16}");
                Console.WriteLine($"🔍 Supports INT8: {capabilities.SupportsInt8}");
                Console.WriteLine();

                // Create NPU accelerator
                using var device = IntelNPUDevice.Default;
                using var accelerator = new IntelNPUAccelerator(context, device);
                
                Console.WriteLine($"🎯 Using: {accelerator.Name}\n");

                // Run inference examples
                await RunImageClassification(accelerator, capabilities);
                await RunObjectDetection(accelerator, capabilities);
                await RunQuantizedInference(accelerator, capabilities);
                await RunBatchInference(accelerator, capabilities);
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
        /// Demonstrates image classification inference on NPU.
        /// </summary>
        static async Task RunImageClassification(IntelNPUAccelerator accelerator, NPUCapabilities capabilities)
        {
            Console.WriteLine("🖼️  Image Classification (ResNet-50)");
            Console.WriteLine("-----------------------------------");

            const int batchSize = 1;
            const int channels = 3;
            const int height = 224;
            const int width = 224;
            const int numClasses = 1000;

            var inputShape = new TensorShape(batchSize, channels, height, width);
            var outputShape = new TensorShape(batchSize, numClasses);

            Console.WriteLine($"   └─ Model: ResNet-50");
            Console.WriteLine($"   └─ Input: {inputShape} (FP32)");
            Console.WriteLine($"   └─ Output: {outputShape}");
            Console.WriteLine($"   └─ Parameters: ~25.6M");

            // Create sample input (simulated preprocessed image)
            var input = CreateSampleImageBatch(inputShape);

            try
            {
                // Configure inference parameters
                var inferenceParams = new NPUInferenceParameters
                {
                    OptimizationLevel = NPUOptimizationLevel.High,
                    PrecisionMode = capabilities.SupportsBF16 ? NPUPrecisionMode.BF16 : NPUPrecisionMode.FP32,
                    BatchSize = batchSize,
                    EnableDynamicShapes = false
                };

                Console.WriteLine($"   └─ Precision: {inferenceParams.PrecisionMode}");

                // Run inference
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var output = await SimulateNPUInference(input, "ResNet50", inferenceParams);
                stopwatch.Stop();

                // Calculate metrics
                var inferenceTime = stopwatch.ElapsedMilliseconds;
                var throughput = 1000.0 / inferenceTime; // Images per second
                var efficiency = CalculatePowerEfficiency(capabilities, inferenceTime);

                Console.WriteLine($"   └─ Inference Time: {inferenceTime:F2} ms");
                Console.WriteLine($"   └─ Throughput: {throughput:F1} images/sec");
                Console.WriteLine($"   └─ Power Efficiency: {efficiency:F1} TOPS/W");

                // Show top predictions (simulated)
                var predictions = GetTopPredictions(output, 5);
                Console.WriteLine("   └─ Top 5 Predictions:");
                foreach (var (className, confidence) in predictions)
                {
                    Console.WriteLine($"      • {className}: {confidence:P1}");
                }

                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Image classification failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Demonstrates object detection inference on NPU.
        /// </summary>
        static async Task RunObjectDetection(IntelNPUAccelerator accelerator, NPUCapabilities capabilities)
        {
            Console.WriteLine("🎯 Object Detection (YOLOv8n)");
            Console.WriteLine("-----------------------------");

            const int batchSize = 1;
            const int channels = 3;
            const int height = 640;
            const int width = 640;

            var inputShape = new TensorShape(batchSize, channels, height, width);

            Console.WriteLine($"   └─ Model: YOLOv8n");
            Console.WriteLine($"   └─ Input: {inputShape}");
            Console.WriteLine($"   └─ Parameters: ~3.2M");

            // Create sample input
            var input = CreateSampleImageBatch(inputShape);

            try
            {
                var inferenceParams = new NPUInferenceParameters
                {
                    OptimizationLevel = NPUOptimizationLevel.Balanced,
                    PrecisionMode = capabilities.SupportsInt8 ? NPUPrecisionMode.INT8 : NPUPrecisionMode.FP32,
                    BatchSize = batchSize,
                    EnableDynamicShapes = true
                };

                Console.WriteLine($"   └─ Precision: {inferenceParams.PrecisionMode}");

                // Run inference
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var output = await SimulateNPUInference(input, "YOLOv8n", inferenceParams);
                stopwatch.Stop();

                var inferenceTime = stopwatch.ElapsedMilliseconds;
                var fps = 1000.0 / inferenceTime;

                Console.WriteLine($"   └─ Inference Time: {inferenceTime:F2} ms");
                Console.WriteLine($"   └─ FPS: {fps:F1}");

                // Simulate detection results
                var detections = SimulateDetections();
                Console.WriteLine($"   └─ Detections: {detections.Length} objects found");
                foreach (var detection in detections)
                {
                    Console.WriteLine($"      • {detection.Class}: {detection.Confidence:P1} at ({detection.X:F0},{detection.Y:F0})");
                }

                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Object detection failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Demonstrates quantized model inference for better NPU utilization.
        /// </summary>
        static async Task RunQuantizedInference(IntelNPUAccelerator accelerator, NPUCapabilities capabilities)
        {
            Console.WriteLine("⚡ Quantized Model Inference");
            Console.WriteLine("---------------------------");

            if (!capabilities.SupportsInt8)
            {
                Console.WriteLine("   ❌ INT8 quantization not supported on this NPU generation");
                Console.WriteLine();
                return;
            }

            const int batchSize = 1;
            const int inputSize = 224;

            Console.WriteLine($"   └─ Model: MobileNetV2 (INT8 Quantized)");
            Console.WriteLine($"   └─ Quantization: INT8 with BF16 accumulation");

            var inputShape = new TensorShape(batchSize, 3, inputSize, inputSize);
            var input = CreateSampleImageBatch(inputShape);

            try
            {
                // Compare FP32 vs INT8 performance
                var fp32Params = new NPUInferenceParameters
                {
                    PrecisionMode = NPUPrecisionMode.FP32,
                    OptimizationLevel = NPUOptimizationLevel.High,
                    BatchSize = batchSize
                };

                var int8Params = new NPUInferenceParameters
                {
                    PrecisionMode = NPUPrecisionMode.INT8,
                    OptimizationLevel = NPUOptimizationLevel.High,
                    BatchSize = batchSize
                };

                // FP32 inference
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                var fp32Output = await SimulateNPUInference(input, "MobileNetV2_FP32", fp32Params);
                var fp32Time = stopwatch.ElapsedMilliseconds;

                // INT8 inference
                stopwatch.Restart();
                var int8Output = await SimulateNPUInference(input, "MobileNetV2_INT8", int8Params);
                var int8Time = stopwatch.ElapsedMilliseconds;

                var speedup = (double)fp32Time / int8Time;
                var powerSavings = CalculatePowerSavings(fp32Time, int8Time);

                Console.WriteLine($"   └─ FP32 Time: {fp32Time:F2} ms");
                Console.WriteLine($"   └─ INT8 Time: {int8Time:F2} ms");
                Console.WriteLine($"   └─ Speedup: {speedup:F1}x");
                Console.WriteLine($"   └─ Power Savings: {powerSavings:P0}");
                Console.WriteLine($"   └─ Accuracy Loss: ~1-2% (typical for INT8)");

                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Quantized inference failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Demonstrates batch inference for improved throughput.
        /// </summary>
        static async Task RunBatchInference(IntelNPUAccelerator accelerator, NPUCapabilities capabilities)
        {
            Console.WriteLine("📦 Batch Inference Optimization");
            Console.WriteLine("------------------------------");

            const int inputSize = 224;
            const int channels = 3;
            var batchSizes = new[] { 1, 2, 4, 8 };

            Console.WriteLine("   └─ Model: EfficientNet-B0");
            Console.WriteLine("   └─ Testing optimal batch size for NPU");

            Console.WriteLine($"\n   {"Batch Size",-12} {"Time (ms)",-12} {"Throughput",-15} {"Efficiency",-12}");
            Console.WriteLine($"   {new string('-', 55)}");

            foreach (var batchSize in batchSizes)
            {
                try
                {
                    var optimalBatch = capabilities.GetOptimalBatchSize(20.0); // 20MB model
                    if (batchSize > optimalBatch * 2)
                    {
                        Console.WriteLine($"   {batchSize,-12} {"Skipped",-12} {"(too large)",-15} {"N/A",-12}");
                        continue;
                    }

                    var inputShape = new TensorShape(batchSize, channels, inputSize, inputSize);
                    var input = CreateSampleImageBatch(inputShape);

                    var inferenceParams = new NPUInferenceParameters
                    {
                        PrecisionMode = capabilities.SupportsBF16 ? NPUPrecisionMode.BF16 : NPUPrecisionMode.FP32,
                        OptimizationLevel = NPUOptimizationLevel.High,
                        BatchSize = batchSize
                    };

                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    await SimulateNPUInference(input, "EfficientNet-B0", inferenceParams);
                    stopwatch.Stop();

                    var totalTime = stopwatch.ElapsedMilliseconds;
                    var throughput = batchSize * 1000.0 / totalTime;
                    var efficiency = throughput / capabilities.MaxTOPS;

                    Console.WriteLine($"   {batchSize,-12} {totalTime,-12:F2} {throughput,-15:F1} img/s {efficiency,-12:F3}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   {batchSize,-12} {"Failed",-12} {ex.Message,-15} {"N/A",-12}");
                }
            }

            Console.WriteLine($"\n   📝 Optimal batch size for this NPU: {capabilities.GetOptimalBatchSize(20.0)}");
            Console.WriteLine();
        }

        #region Helper Methods

        /// <summary>
        /// Creates a sample image batch for testing.
        /// </summary>
        static ITensor<float> CreateSampleImageBatch(TensorShape shape)
        {
            var data = new float[shape.ElementCount];
            var random = new Random(42);
            
            // Generate normalized image data (0-1 range)
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble();
            }
            
            return new CpuTensor<float>(shape, data);
        }

        /// <summary>
        /// Simulates NPU inference operation.
        /// </summary>
        static async Task<ITensor<float>> SimulateNPUInference(
            ITensor<float> input,
            string modelName,
            NPUInferenceParameters parameters)
        {
            // Simulate model-specific processing times
            var processingTime = modelName switch
            {
                "ResNet50" => 45,
                "YOLOv8n" => 28,
                "MobileNetV2_FP32" => 15,
                "MobileNetV2_INT8" => 8,
                "EfficientNet-B0" => 12,
                _ => 20
            };

            // Adjust for precision mode
            processingTime = parameters.PrecisionMode switch
            {
                NPUPrecisionMode.INT8 => (int)(processingTime * 0.6),
                NPUPrecisionMode.BF16 => (int)(processingTime * 0.8),
                _ => processingTime
            };

            // Adjust for batch size
            processingTime = (int)(processingTime * Math.Sqrt(parameters.BatchSize));

            await Task.Delay(Math.Max(1, processingTime / 10)); // Simulate processing

            // Create dummy output
            var outputSize = modelName switch
            {
                "ResNet50" => 1000,
                "YOLOv8n" => 8400 * 84, // YOLO output format
                "MobileNetV2_FP32" or "MobileNetV2_INT8" => 1000,
                "EfficientNet-B0" => 1000,
                _ => 1000
            };

            var outputShape = new TensorShape(parameters.BatchSize, outputSize);
            var outputData = new float[outputShape.ElementCount];
            var random = new Random(123);

            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i] = (float)random.NextDouble();
            }

            return new CpuTensor<float>(outputShape, outputData);
        }

        /// <summary>
        /// Gets top predictions from classification output.
        /// </summary>
        static (string ClassName, float Confidence)[] GetTopPredictions(ITensor<float> output, int topK)
        {
            var classes = new[] { "Egyptian cat", "Tiger", "Dog", "Airplane", "Car" };
            var random = new Random(456);
            
            var predictions = new (string, float)[topK];
            for (int i = 0; i < topK; i++)
            {
                predictions[i] = (classes[i % classes.Length], (float)(random.NextDouble() * 0.8 + 0.1));
            }
            
            return predictions;
        }

        /// <summary>
        /// Simulates object detection results.
        /// </summary>
        static Detection[] SimulateDetections()
        {
            return new[]
            {
                new Detection { Class = "person", Confidence = 0.89f, X = 150, Y = 200 },
                new Detection { Class = "car", Confidence = 0.76f, X = 300, Y = 400 },
                new Detection { Class = "dog", Confidence = 0.65f, X = 100, Y = 350 }
            };
        }

        /// <summary>
        /// Calculates power efficiency based on NPU capabilities.
        /// </summary>
        static double CalculatePowerEfficiency(NPUCapabilities capabilities, double timeMs)
        {
            var utilizationPercent = Math.Min(100.0, 1000.0 / timeMs * 10); // Rough estimate
            var estimatedPower = capabilities.GetEstimatedPower(utilizationPercent);
            return capabilities.MaxTOPS / Math.Max(estimatedPower, 0.1);
        }

        /// <summary>
        /// Calculates power savings from quantization.
        /// </summary>
        static double CalculatePowerSavings(double fp32Time, double int8Time)
        {
            // INT8 typically saves 30-50% power on NPU
            var powerRatio = int8Time / fp32Time;
            return 1.0 - (powerRatio * 0.7); // Account for reduced precision power savings
        }

        #endregion
    }

    /// <summary>
    /// NPU inference parameters.
    /// </summary>
    public struct NPUInferenceParameters
    {
        public NPUOptimizationLevel OptimizationLevel;
        public NPUPrecisionMode PrecisionMode;
        public int BatchSize;
        public bool EnableDynamicShapes;
    }

    /// <summary>
    /// NPU optimization levels.
    /// </summary>
    public enum NPUOptimizationLevel
    {
        Fast,
        Balanced,
        High,
        Maximum
    }

    /// <summary>
    /// NPU precision modes.
    /// </summary>
    public enum NPUPrecisionMode
    {
        FP32,
        BF16,
        INT8,
        Mixed
    }

    /// <summary>
    /// Object detection result.
    /// </summary>
    public struct Detection
    {
        public string Class;
        public float Confidence;
        public float X, Y;
    }
}