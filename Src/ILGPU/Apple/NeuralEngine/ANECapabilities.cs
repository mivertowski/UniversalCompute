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

using ILGPU.Apple.NeuralEngine.Native;
using System;

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Represents the capabilities of the Apple Neural Engine.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ANECapabilities struct.
    /// </remarks>
    /// <param name="isAvailable">Whether ANE is available.</param>
    /// <param name="generation">The ANE generation.</param>
    /// <param name="maxTOPS">Maximum TOPS performance.</param>
    /// <param name="supportsFloat16">Whether Float16 is supported.</param>
    /// <param name="supportsInt8">Whether INT8 is supported.</param>
    /// <param name="supportsConvolution">Whether convolution operations are supported.</param>
    /// <param name="supportsAttention">Whether attention operations are supported.</param>
    /// <param name="supportsTransformer">Whether transformer models are supported.</param>
    /// <param name="supportsCoreML">Whether Core ML integration is supported.</param>
    /// <param name="maxBatchSize">Maximum batch size.</param>
    public readonly struct ANECapabilities(
        bool isAvailable,
        ANEGeneration generation,
        double maxTOPS,
        bool supportsFloat16,
        bool supportsInt8,
        bool supportsConvolution,
        bool supportsAttention,
        bool supportsTransformer,
        bool supportsCoreML,
        int maxBatchSize)
    {

        /// <summary>
        /// Gets whether the Apple Neural Engine is available.
        /// </summary>
        public bool IsAvailable { get; } = isAvailable;

        /// <summary>
        /// Gets the Neural Engine generation.
        /// </summary>
        public ANEGeneration Generation { get; } = generation;

        /// <summary>
        /// Gets the maximum TOPS (Tera Operations Per Second) performance.
        /// </summary>
        public double MaxTOPS { get; } = maxTOPS;

        /// <summary>
        /// Gets whether Float16 operations are supported.
        /// </summary>
        public bool SupportsFloat16 { get; } = supportsFloat16;

        /// <summary>
        /// Gets whether INT8 quantization is supported.
        /// </summary>
        public bool SupportsInt8 { get; } = supportsInt8;

        /// <summary>
        /// Gets whether convolution operations are accelerated.
        /// </summary>
        public bool SupportsConvolution { get; } = supportsConvolution;

        /// <summary>
        /// Gets whether attention mechanisms are accelerated.
        /// </summary>
        public bool SupportsAttention { get; } = supportsAttention;

        /// <summary>
        /// Gets whether transformer models are optimized.
        /// </summary>
        public bool SupportsTransformer { get; } = supportsTransformer;

        /// <summary>
        /// Gets whether Core ML integration is supported.
        /// </summary>
        public bool SupportsCoreML { get; } = supportsCoreML;

        /// <summary>
        /// Gets the maximum supported batch size.
        /// </summary>
        public int MaxBatchSize { get; } = maxBatchSize;

        /// <summary>
        /// Gets the chip generation string.
        /// </summary>
        public string ChipGeneration => Generation.ToString();

        /// <summary>
        /// Gets the maximum tensor width.
        /// </summary>
        public int MaxTensorWidth => Generation switch
        {
            ANEGeneration.ANE1 => 16384,
            ANEGeneration.ANE2 => 32768,
            ANEGeneration.ANE3 => 65536,
            ANEGeneration.ANE4 => 131072,
            _ => 8192
        };

        /// <summary>
        /// Gets the maximum tensor height.
        /// </summary>
        public int MaxTensorHeight => MaxTensorWidth;

        /// <summary>
        /// Gets the optimal work group size.
        /// </summary>
        public int OptimalWorkGroupSize => Generation switch
        {
            ANEGeneration.ANE1 => 32,
            ANEGeneration.ANE2 => 64,
            ANEGeneration.ANE3 => 128,
            ANEGeneration.ANE4 => 256,
            _ => 16
        };

        /// <summary>
        /// Gets the number of compute units.
        /// </summary>
        public int NumComputeUnits => Generation switch
        {
            ANEGeneration.ANE1 => 8,
            ANEGeneration.ANE2 => 16,
            ANEGeneration.ANE3 => 24,
            ANEGeneration.ANE4 => 32,
            _ => 4
        };

        /// <summary>
        /// Gets the maximum shared memory per unit.
        /// </summary>
        public int MaxSharedMemoryPerUnit => Generation switch
        {
            ANEGeneration.ANE1 => 32768,    // 32KB
            ANEGeneration.ANE2 => 65536,    // 64KB
            ANEGeneration.ANE3 => 131072,   // 128KB
            ANEGeneration.ANE4 => 262144,   // 256KB
            _ => 16384                      // 16KB
        };

        /// <summary>
        /// Gets the maximum constant memory.
        /// </summary>
        public int MaxConstantMemory => Generation switch
        {
            ANEGeneration.ANE1 => 1048576,   // 1MB
            ANEGeneration.ANE2 => 2097152,   // 2MB
            ANEGeneration.ANE3 => 4194304,   // 4MB
            ANEGeneration.ANE4 => 8388608,   // 8MB
            _ => 524288                      // 512KB
        };

        /// <summary>
        /// Gets the memory bandwidth in bytes per second.
        /// </summary>
        public long MemoryBandwidth => Generation switch
        {
            ANEGeneration.ANE1 => 34359738368L,    // 32 GB/s
            ANEGeneration.ANE2 => 68719476736L,    // 64 GB/s
            ANEGeneration.ANE3 => 137438953472L,   // 128 GB/s
            ANEGeneration.ANE4 => 274877906944L,   // 256 GB/s
            _ => 17179869184L                      // 16 GB/s
        };

        /// <summary>
        /// Gets the maximum concurrent operations.
        /// </summary>
        public int MaxConcurrentOperations => Generation switch
        {
            ANEGeneration.ANE1 => 16,
            ANEGeneration.ANE2 => 32,
            ANEGeneration.ANE3 => 64,
            ANEGeneration.ANE4 => 128,
            _ => 8
        };

        /// <summary>
        /// Queries the Neural Engine capabilities of the current system.
        /// </summary>
        /// <returns>ANE capabilities structure.</returns>
        public static ANECapabilities Query()
        {
            if (!DetectNeuralEngine())
                return new ANECapabilities();

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var nativeCapabilities = ANENative.QueryCapabilities();
                return MapFromNative(nativeCapabilities);
            }
            catch
            {
                return new ANECapabilities();
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Detects whether Apple Neural Engine is available on the current system.
        /// </summary>
        /// <returns>True if ANE is available; otherwise, false.</returns>
        public static bool DetectNeuralEngine()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Check if we're on macOS with Apple Silicon
                if (!OperatingSystem.IsMacOS())
                    return false;

                // Check for Neural Engine availability
                return ANENative.IsNeuralEngineAvailable();
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Gets the optimal batch size for the given model complexity.
        /// </summary>
        /// <param name="modelComplexity">The model complexity (number of parameters).</param>
        /// <returns>The recommended batch size.</returns>
        public int GetOptimalBatchSize(long modelComplexity) =>
            // ANE is optimized for low-latency inference, typically batch size 1
            Generation switch
            {
                ANEGeneration.ANE1 => 1, // A11, A12 - optimized for single inference
                ANEGeneration.ANE2 => modelComplexity < 1000000 ? 2 : 1, // A13, A14
                ANEGeneration.ANE3 => modelComplexity < 1000000 ? 4 : 2, // A15, A16, M1, M2
                ANEGeneration.ANE4 => modelComplexity < 1000000 ? 8 : 4, // Future generations
                _ => 1
            };

        /// <summary>
        /// Gets the estimated power consumption for the given utilization.
        /// </summary>
        /// <param name="utilizationPercent">The ANE utilization percentage.</param>
        /// <returns>The estimated power consumption in watts.</returns>
        public double GetEstimatedPower(double utilizationPercent)
        {
            var basePower = Generation switch
            {
                ANEGeneration.ANE1 => 0.5, // A11, A12
                ANEGeneration.ANE2 => 0.8, // A13, A14
                ANEGeneration.ANE3 => 1.2, // A15, A16, M1, M2
                ANEGeneration.ANE4 => 1.5, // Future generations
                _ => 0.3
            };

            return basePower * (utilizationPercent / 100.0);
        }

        /// <summary>
        /// Gets the power efficiency in TOPS/Watt.
        /// </summary>
        /// <returns>The power efficiency.</returns>
        public double GetPowerEfficiency() => Generation switch
        {
            ANEGeneration.ANE1 => 11.0, // ~5.5 TOPS / 0.5W
            ANEGeneration.ANE2 => 13.75, // ~11 TOPS / 0.8W
            ANEGeneration.ANE3 => 13.33, // ~16 TOPS / 1.2W
            ANEGeneration.ANE4 => 20.0, // Estimated future efficiency
            _ => 10.0
        };

        /// <summary>
        /// Checks if the specified model type is optimally supported.
        /// </summary>
        /// <param name="modelType">The model type to check.</param>
        /// <returns>True if optimally supported; otherwise, false.</returns>
        public bool IsModelTypeOptimal(ANEModelType modelType) => modelType switch
        {
            ANEModelType.ConvolutionalNeuralNetwork => SupportsConvolution,
            ANEModelType.RecurrentNeuralNetwork => Generation >= ANEGeneration.ANE2,
            ANEModelType.Transformer => SupportsTransformer && SupportsAttention,
            ANEModelType.ObjectDetection => SupportsConvolution && Generation >= ANEGeneration.ANE2,
            ANEModelType.NaturalLanguageProcessing => SupportsAttention && Generation >= ANEGeneration.ANE3,
            ANEModelType.ComputerVision => SupportsConvolution,
            _ => false
        };

        private static ANECapabilities MapFromNative(ANENativeCapabilities native)
        {
            var generation = (ANEGeneration)native.Generation;
            
            return new ANECapabilities(
                native.IsAvailable != 0,
                generation,
                native.MaxTOPS,
                native.SupportsFloat16 != 0,
                native.SupportsInt8 != 0,
                native.SupportsConvolution != 0,
                native.SupportsAttention != 0,
                native.SupportsTransformer != 0,
                native.SupportsCoreML != 0,
                native.MaxBatchSize
            );
        }

        /// <summary>
        /// Returns a string representation of the ANE capabilities.
        /// </summary>
        /// <returns>A string describing the ANE capabilities.</returns>
        public override string ToString() => $"Apple Neural Engine {Generation}: {MaxTOPS:F1} TOPS, " +
                   $"FP16={SupportsFloat16}, INT8={SupportsInt8}, " +
                   $"Conv={SupportsConvolution}, Attn={SupportsAttention}, " +
                   $"Efficiency={GetPowerEfficiency():F1} TOPS/W";
    }

    /// <summary>
    /// Apple Neural Engine generations.
    /// </summary>
    public enum ANEGeneration
    {
        /// <summary>
        /// No Neural Engine available.
        /// </summary>
        None = 0,

        /// <summary>
        /// Not supported on this platform.
        /// </summary>
        NotSupported = -1,

        /// <summary>
        /// Unknown Neural Engine generation.
        /// </summary>
        Unknown = -2,

        /// <summary>
        /// First generation ANE (A11, A12).
        /// </summary>
        ANE1 = 1,

        /// <summary>
        /// Second generation ANE (A13, A14).
        /// </summary>
        ANE2 = 2,

        /// <summary>
        /// Third generation ANE (A15, A16, M1, M2).
        /// </summary>
        ANE3 = 3,

        /// <summary>
        /// Fourth generation ANE (future chips).
        /// </summary>
        ANE4 = 4
    }

    /// <summary>
    /// Model types optimized for Apple Neural Engine.
    /// </summary>
    public enum ANEModelType
    {
        /// <summary>
        /// Convolutional Neural Network.
        /// </summary>
        ConvolutionalNeuralNetwork,

        /// <summary>
        /// Recurrent Neural Network.
        /// </summary>
        RecurrentNeuralNetwork,

        /// <summary>
        /// Transformer model.
        /// </summary>
        Transformer,

        /// <summary>
        /// Object detection model.
        /// </summary>
        ObjectDetection,

        /// <summary>
        /// Natural language processing model.
        /// </summary>
        NaturalLanguageProcessing,

        /// <summary>
        /// Computer vision model.
        /// </summary>
        ComputerVision
    }

    /// <summary>
    /// Performance metrics for Apple Neural Engine.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ANEPerformanceMetrics struct.
    /// </remarks>
    /// <param name="utilizationPercent">Current ANE utilization percentage.</param>
    /// <param name="throughputTOPS">Current throughput in TOPS.</param>
    /// <param name="powerConsumption">Current power consumption in watts.</param>
    /// <param name="inferenceLatency">Average inference latency in milliseconds.</param>
    /// <param name="modelCacheHitRate">Model cache hit rate percentage.</param>
    public readonly struct ANEPerformanceMetrics(
        double utilizationPercent,
        double throughputTOPS,
        double powerConsumption,
        double inferenceLatency,
        double modelCacheHitRate)
    {

        /// <summary>
        /// Gets the current ANE utilization percentage.
        /// </summary>
        public double UtilizationPercent { get; } = utilizationPercent;

        /// <summary>
        /// Gets the current throughput in TOPS.
        /// </summary>
        public double ThroughputTOPS { get; } = throughputTOPS;

        /// <summary>
        /// Gets the current power consumption in watts.
        /// </summary>
        public double PowerConsumption { get; } = powerConsumption;

        /// <summary>
        /// Gets the average inference latency in milliseconds.
        /// </summary>
        public double InferenceLatency { get; } = inferenceLatency;

        /// <summary>
        /// Gets the model cache hit rate percentage.
        /// </summary>
        public double ModelCacheHitRate { get; } = modelCacheHitRate;
    }

    /// <summary>
    /// Power information for Apple Neural Engine.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ANEPowerInfo struct.
    /// </remarks>
    /// <param name="currentPower">Current power consumption in watts.</param>
    /// <param name="thermalState">Current thermal state.</param>
    /// <param name="powerEfficiency">Current power efficiency in TOPS/Watt.</param>
    /// <param name="batteryImpact">Battery impact level.</param>
    public readonly struct ANEPowerInfo(
        double currentPower,
        ANEThermalState thermalState,
        double powerEfficiency,
        ANEBatteryImpact batteryImpact)
    {

        /// <summary>
        /// Gets the current power consumption in watts.
        /// </summary>
        public double CurrentPower { get; } = currentPower;

        /// <summary>
        /// Gets the current thermal state.
        /// </summary>
        public ANEThermalState ThermalState { get; } = thermalState;

        /// <summary>
        /// Gets the current power efficiency in TOPS/Watt.
        /// </summary>
        public double PowerEfficiency { get; } = powerEfficiency;

        /// <summary>
        /// Gets the battery impact level.
        /// </summary>
        public ANEBatteryImpact BatteryImpact { get; } = batteryImpact;
    }

    /// <summary>
    /// Apple Neural Engine thermal states.
    /// </summary>
    public enum ANEThermalState
    {
        /// <summary>
        /// Normal operation.
        /// </summary>
        Normal,

        /// <summary>
        /// Fair thermal state.
        /// </summary>
        Fair,

        /// <summary>
        /// Serious thermal state.
        /// </summary>
        Serious,

        /// <summary>
        /// Critical thermal state.
        /// </summary>
        Critical
    }

    /// <summary>
    /// Battery impact levels for ANE operations.
    /// </summary>
    public enum ANEBatteryImpact
    {
        /// <summary>
        /// Minimal battery impact.
        /// </summary>
        Minimal,

        /// <summary>
        /// Low battery impact.
        /// </summary>
        Low,

        /// <summary>
        /// Medium battery impact.
        /// </summary>
        Medium,

        /// <summary>
        /// High battery impact.
        /// </summary>
        High
    }
}