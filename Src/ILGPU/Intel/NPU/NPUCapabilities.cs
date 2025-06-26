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
// Change License: Apache License, Version 2.0using ILGPU.Intel.NPU.Native;

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Represents the capabilities of an Intel NPU device.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the NPUCapabilities struct.
    /// </remarks>
    /// <param name="generation">The NPU generation.</param>
    /// <param name="computeUnits">The number of compute units.</param>
    /// <param name="maxTops">Maximum TOPS (Tera Operations Per Second).</param>
    /// <param name="memoryBandwidth">Memory bandwidth in GB/s.</param>
    /// <param name="supportsBF16">Whether BFloat16 is supported.</param>
    /// <param name="supportsInt8">Whether INT8 quantization is supported.</param>
    /// <param name="supportsConvolution">Whether convolution operations are supported.</param>
    /// <param name="supportsMatMul">Whether matrix multiplication is supported.</param>
    /// <param name="supportsAttention">Whether attention operations are supported.</param>
    /// <param name="supportsSparsity">Whether sparse operations are supported.</param>
    public readonly struct NPUCapabilities(
        NPUGeneration generation,
        int computeUnits,
        double maxTops,
        double memoryBandwidth,
        bool supportsBF16,
        bool supportsInt8,
        bool supportsConvolution,
        bool supportsMatMul,
        bool supportsAttention,
        bool supportsSparsity)
    {

        /// <summary>
        /// Gets the NPU generation.
        /// </summary>
        public NPUGeneration Generation { get; } = generation;

        /// <summary>
        /// Gets the number of compute units in the NPU.
        /// </summary>
        public int ComputeUnits { get; } = computeUnits;

        /// <summary>
        /// Gets the maximum TOPS (Tera Operations Per Second) performance.
        /// </summary>
        public double MaxTOPS { get; } = maxTops;

        /// <summary>
        /// Gets the memory bandwidth in GB/s.
        /// </summary>
        public double MemoryBandwidth { get; } = memoryBandwidth;

        /// <summary>
        /// Gets whether BFloat16 operations are supported.
        /// </summary>
        public bool SupportsBF16 { get; } = supportsBF16;

        /// <summary>
        /// Gets whether INT8 quantization is supported.
        /// </summary>
        public bool SupportsInt8 { get; } = supportsInt8;

        /// <summary>
        /// Gets whether convolution operations are accelerated.
        /// </summary>
        public bool SupportsConvolution { get; } = supportsConvolution;

        /// <summary>
        /// Gets whether matrix multiplication is accelerated.
        /// </summary>
        public bool SupportsMatMul { get; } = supportsMatMul;

        /// <summary>
        /// Gets whether transformer attention operations are accelerated.
        /// </summary>
        public bool SupportsAttention { get; } = supportsAttention;

        /// <summary>
        /// Gets whether sparse operations are supported.
        /// </summary>
        public bool SupportsSparsity { get; } = supportsSparsity;

        /// <summary>
        /// Detects whether Intel NPU is available on the current system.
        /// </summary>
        /// <returns>True if NPU is available; otherwise, false.</returns>
        public static bool DetectNPU()
        {
            try
            {
                return NPUNative.IsNPUSupported();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Queries the NPU capabilities of the current system.
        /// </summary>
        /// <returns>NPU capabilities structure.</returns>
        public static NPUCapabilities Query()
        {
            if (!DetectNPU())
                return new NPUCapabilities();

            var nativeCapabilities = NPUNative.QueryCapabilities();
            return MapFromNative(nativeCapabilities);
        }

        /// <summary>
        /// Gets whether this NPU supports the specified model format.
        /// </summary>
        /// <param name="format">The model format to check.</param>
        /// <returns>True if the format is supported; otherwise, false.</returns>
        public bool SupportsModelFormat(ModelFormat format) => format switch
        {
            ModelFormat.ONNX => true, // All NPU generations support ONNX
            ModelFormat.OpenVINO => true, // Native Intel format
            ModelFormat.TensorFlow => Generation >= NPUGeneration.NPU3,
            ModelFormat.PyTorch => Generation >= NPUGeneration.NPU3,
            _ => false
        };

        /// <summary>
        /// Gets the optimal batch size for the given model size.
        /// </summary>
        /// <param name="modelSizeMB">The model size in megabytes.</param>
        /// <returns>The recommended batch size.</returns>
        public int GetOptimalBatchSize(double modelSizeMB) =>
            // Heuristic based on NPU generation and model size
            Generation switch
            {
                NPUGeneration.NPU2 => modelSizeMB < 100 ? 4 : 2,
                NPUGeneration.NPU3 => modelSizeMB < 100 ? 8 : 4,
                NPUGeneration.NPU4 => modelSizeMB < 100 ? 16 : 8,
                _ => 1
            };

        /// <summary>
        /// Gets the estimated power consumption for the given workload.
        /// </summary>
        /// <param name="utilizationPercent">The NPU utilization percentage.</param>
        /// <returns>The estimated power consumption in watts.</returns>
        public double GetEstimatedPower(double utilizationPercent)
        {
            var basePower = Generation switch
            {
                NPUGeneration.NPU2 => 3.5, // Meteor Lake baseline
                NPUGeneration.NPU3 => 4.0, // Lunar Lake baseline
                NPUGeneration.NPU4 => 5.0, // Arrow Lake and future
                _ => 1.0
            };

            return basePower * (utilizationPercent / 100.0);
        }

        private static NPUCapabilities MapFromNative(NPUNativeCapabilities native)
        {
            var generation = (NPUGeneration)native.Generation;
            
            return new NPUCapabilities(
                generation,
                native.ComputeUnits,
                native.MaxTOPS,
                native.MemoryBandwidth,
                native.SupportsBF16 != 0,
                native.SupportsInt8 != 0,
                native.SupportsConvolution != 0,
                native.SupportsMatMul != 0,
                native.SupportsAttention != 0,
                native.SupportsSparsity != 0
            );
        }

        /// <summary>
        /// Returns a string representation of the NPU capabilities.
        /// </summary>
        /// <returns>A string describing the NPU capabilities.</returns>
        public override string ToString() => $"Intel NPU {Generation}: {MaxTOPS:F1} TOPS, {ComputeUnits} CUs, " +
                   $"{MemoryBandwidth:F1} GB/s, BF16={SupportsBF16}, INT8={SupportsInt8}";
    }

    /// <summary>
    /// Performance metrics for Intel NPU.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the NPUPerformanceMetrics struct.
    /// </remarks>
    /// <param name="utilizationPercent">Current NPU utilization percentage.</param>
    /// <param name="throughputTOPS">Current throughput in TOPS.</param>
    /// <param name="powerConsumption">Current power consumption in watts.</param>
    /// <param name="temperatureCelsius">Current temperature in Celsius.</param>
    /// <param name="memoryUsage">Memory usage percentage.</param>
    public readonly struct NPUPerformanceMetrics(
        double utilizationPercent,
        double throughputTOPS,
        double powerConsumption,
        double temperatureCelsius,
        double memoryUsage)
    {

        /// <summary>
        /// Gets the current NPU utilization percentage.
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
        /// Gets the current temperature in Celsius.
        /// </summary>
        public double TemperatureCelsius { get; } = temperatureCelsius;

        /// <summary>
        /// Gets the memory usage percentage.
        /// </summary>
        public double MemoryUsage { get; } = memoryUsage;
    }

    /// <summary>
    /// Power information for Intel NPU.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the NPUPowerInfo struct.
    /// </remarks>
    /// <param name="currentPower">Current power consumption in watts.</param>
    /// <param name="maxPower">Maximum power limit in watts.</param>
    /// <param name="thermalThrottling">Whether thermal throttling is active.</param>
    /// <param name="powerEfficiency">Power efficiency in TOPS/Watt.</param>
    public readonly struct NPUPowerInfo(
        double currentPower,
        double maxPower,
        bool thermalThrottling,
        double powerEfficiency)
    {

        /// <summary>
        /// Gets the current power consumption in watts.
        /// </summary>
        public double CurrentPower { get; } = currentPower;

        /// <summary>
        /// Gets the maximum power limit in watts.
        /// </summary>
        public double MaxPower { get; } = maxPower;

        /// <summary>
        /// Gets whether thermal throttling is currently active.
        /// </summary>
        public bool ThermalThrottling { get; } = thermalThrottling;

        /// <summary>
        /// Gets the current power efficiency in TOPS/Watt.
        /// </summary>
        public double PowerEfficiency { get; } = powerEfficiency;
    }

    /// <summary>
    /// Intel NPU generations.
    /// </summary>
    public enum NPUGeneration
    {
        /// <summary>
        /// No NPU available.
        /// </summary>
        None = 0,

        /// <summary>
        /// Not supported on this platform.
        /// </summary>
        NotSupported = -1,

        /// <summary>
        /// Unknown NPU generation.
        /// </summary>
        Unknown = -2,

        /// <summary>
        /// Second generation NPU (Meteor Lake - Core Ultra).
        /// </summary>
        NPU2 = 2,

        /// <summary>
        /// Third generation NPU (Lunar Lake).
        /// </summary>
        NPU3 = 3,

        /// <summary>
        /// Fourth generation NPU (Arrow Lake and future).
        /// </summary>
        NPU4 = 4
    }

    /// <summary>
    /// Model formats supported by Intel NPU.
    /// </summary>
    public enum ModelFormat
    {
        /// <summary>
        /// ONNX model format.
        /// </summary>
        ONNX,

        /// <summary>
        /// Intel OpenVINO model format.
        /// </summary>
        OpenVINO,

        /// <summary>
        /// TensorFlow model format.
        /// </summary>
        TensorFlow,

        /// <summary>
        /// PyTorch model format.
        /// </summary>
        PyTorch
    }
}