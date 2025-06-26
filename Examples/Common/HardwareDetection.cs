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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.NPU;
using ILGPU.Intel.AMX;

namespace ILGPU.Examples.Common
{
    /// <summary>
    /// Utility class for detecting available hardware accelerators.
    /// </summary>
    public static class HardwareDetection
    {
        /// <summary>
        /// Detects all available hardware accelerators on the current system.
        /// </summary>
        public static HardwareInfo DetectAvailableHardware()
        {
            var info = new HardwareInfo
            {
                Platform = GetPlatformInfo(),
                AppleNeuralEngine = DetectAppleNeuralEngine(),
                IntelNPU = DetectIntelNPU(),
                IntelAMX = DetectIntelAMX(),
                CudaDevices = DetectCudaDevices(),
                OpenCLDevices = DetectOpenCLDevices()
            };

            return info;
        }

        /// <summary>
        /// Gets platform information.
        /// </summary>
        public static PlatformInfo GetPlatformInfo()
        {
            return new PlatformInfo
            {
                OperatingSystem = Environment.OSVersion.ToString(),
                Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
                IsAppleSilicon = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && 
                                RuntimeInformation.ProcessArchitecture == Architecture.Arm64,
                DotNetVersion = Environment.Version.ToString()
            };
        }

        /// <summary>
        /// Detects Apple Neural Engine availability.
        /// </summary>
        public static ANEInfo? DetectAppleNeuralEngine()
        {
            try
            {
                if (!ANECapabilities.DetectNeuralEngine())
                    return null;

                var capabilities = ANECapabilities.Query();
                return new ANEInfo
                {
                    IsAvailable = capabilities.IsAvailable,
                    Generation = capabilities.Generation.ToString(),
                    MaxTOPS = capabilities.MaxTOPS,
                    SupportsFloat16 = capabilities.SupportsFloat16,
                    SupportsInt8 = capabilities.SupportsInt8,
                    SupportsConvolution = capabilities.SupportsConvolution,
                    SupportsAttention = capabilities.SupportsAttention,
                    MaxBatchSize = capabilities.MaxBatchSize,
                    PowerEfficiency = capabilities.GetPowerEfficiency()
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Detects Intel NPU availability.
        /// </summary>
        public static NPUInfo? DetectIntelNPU()
        {
            try
            {
                if (!NPUCapabilities.DetectNPU())
                    return null;

                var capabilities = NPUCapabilities.Query();
                return new NPUInfo
                {
                    IsAvailable = true,
                    Generation = capabilities.Generation.ToString(),
                    MaxTOPS = capabilities.MaxTOPS,
                    ComputeUnits = capabilities.ComputeUnits,
                    MemoryBandwidth = capabilities.MemoryBandwidth,
                    SupportsBF16 = capabilities.SupportsBF16,
                    SupportsInt8 = capabilities.SupportsInt8,
                    SupportsConvolution = capabilities.SupportsConvolution,
                    SupportsMatMul = capabilities.SupportsMatMul,
                    OptimalBatchSize = capabilities.GetOptimalBatchSize(20.0) // 20MB model estimate
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Detects Intel AMX availability.
        /// </summary>
        public static AMXInfo? DetectIntelAMX()
        {
            try
            {
                if (!AMXCapabilities.CheckAMXSupport())
                    return null;

                var capabilities = AMXCapabilities.QueryCapabilities();
                return new AMXInfo
                {
                    IsSupported = capabilities.IsSupported != 0,
                    MaxTiles = capabilities.MaxTiles,
                    MaxTileRows = capabilities.MaxTileRows,
                    MaxTileColumns = capabilities.MaxTileColumns,
                    MaxTileBytes = capabilities.MaxTileBytes,
                    SupportsBF16 = capabilities.SupportsBF16 != 0,
                    SupportsInt8 = capabilities.SupportsInt8 != 0,
                    SupportsFloat32 = capabilities.SupportsFloat32 != 0,
                    EstimatedBandwidth = capabilities.EstimatedBandwidthGBps
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Detects CUDA devices.
        /// </summary>
        public static List<CudaDeviceInfo> DetectCudaDevices()
        {
            var devices = new List<CudaDeviceInfo>();
            
            try
            {
                using var context = Context.CreateDefault();
                foreach (var device in context.GetCudaDevices())
                {
                    devices.Add(new CudaDeviceInfo
                    {
                        DeviceId = device.DeviceId.ToString(),
                        Name = device.Name,
                        ComputeCapability = device.ComputeCapability.ToString(),
                        MemorySize = device.MemorySize,
                        MaxGridSize = device.MaxGridSize.ToString(),
                        MaxGroupSize = device.MaxGroupSize.ToString(),
                        WarpSize = device.WarpSize,
                        NumMultiprocessors = device.NumMultiprocessors,
                        MaxThreadsPerMultiprocessor = device.MaxNumThreadsPerMultiprocessor
                    });
                }
            }
            catch
            {
                // CUDA not available
            }

            return devices;
        }

        /// <summary>
        /// Detects OpenCL devices.
        /// </summary>
        public static List<OpenCLDeviceInfo> DetectOpenCLDevices()
        {
            var devices = new List<OpenCLDeviceInfo>();
            
            try
            {
                using var context = Context.CreateDefault();
                foreach (var device in context.GetCLDevices())
                {
                    devices.Add(new OpenCLDeviceInfo
                    {
                        DeviceId = device.DeviceId.ToString(),
                        Name = device.Name,
                        MemorySize = device.MemorySize,
                        MaxGridSize = device.MaxGridSize.ToString(),
                        MaxGroupSize = device.MaxGroupSize.ToString(),
                        WarpSize = device.WarpSize
                    });
                }
            }
            catch
            {
                // OpenCL not available
            }

            return devices;
        }

        /// <summary>
        /// Provides recommendations based on detected hardware.
        /// </summary>
        public static string GetRecommendations(HardwareInfo hardware)
        {
            var recommendations = new List<string>();

            if (hardware.AppleNeuralEngine?.IsAvailable == true)
            {
                recommendations.Add("🧠 Apple Neural Engine detected - Excellent for iOS/macOS AI applications");
                recommendations.Add("   └─ Best for: Core ML models, real-time inference, power-efficient AI");
            }

            if (hardware.IntelNPU?.IsAvailable == true)
            {
                recommendations.Add("🔧 Intel NPU detected - Great for edge AI and ONNX models");
                recommendations.Add("   └─ Best for: Quantized inference, batch processing, OpenVINO integration");
            }

            if (hardware.IntelAMX?.IsSupported == true)
            {
                recommendations.Add("⚡ Intel AMX detected - Optimal for matrix computations");
                recommendations.Add("   └─ Best for: GEMM operations, neural network training, HPC workloads");
            }

            if (hardware.CudaDevices.Count > 0)
            {
                recommendations.Add($"🎮 {hardware.CudaDevices.Count} CUDA device(s) detected - Versatile for all GPU computing");
                recommendations.Add("   └─ Best for: Deep learning training, parallel computing, graphics");
            }

            if (hardware.OpenCLDevices.Count > 0)
            {
                recommendations.Add($"🔄 {hardware.OpenCLDevices.Count} OpenCL device(s) detected - Cross-platform compute");
                recommendations.Add("   └─ Best for: Portable applications, heterogeneous computing");
            }

            if (recommendations.Count == 0)
            {
                recommendations.Add("💻 CPU-only system detected");
                recommendations.Add("   └─ Consider upgrading hardware for better AI/compute performance");
            }

            return string.Join("\n", recommendations);
        }

        /// <summary>
        /// Prints a comprehensive hardware report.
        /// </summary>
        public static void PrintHardwareReport(HardwareInfo hardware)
        {
            Console.WriteLine("🖥️  Hardware Detection Report");
            Console.WriteLine("============================\n");

            // Platform information
            Console.WriteLine($"Platform: {hardware.Platform.OperatingSystem}");
            Console.WriteLine($"Architecture: {hardware.Platform.Architecture}");
            Console.WriteLine($".NET Version: {hardware.Platform.DotNetVersion}");
            Console.WriteLine($"Apple Silicon: {(hardware.Platform.IsAppleSilicon ? "Yes" : "No")}");
            Console.WriteLine();

            // Apple Neural Engine
            if (hardware.AppleNeuralEngine != null)
            {
                var ane = hardware.AppleNeuralEngine;
                Console.WriteLine($"🧠 Apple Neural Engine: {(ane.IsAvailable ? "Available" : "Not Available")}");
                if (ane.IsAvailable)
                {
                    Console.WriteLine($"   └─ Generation: {ane.Generation}");
                    Console.WriteLine($"   └─ Max TOPS: {ane.MaxTOPS:F1}");
                    Console.WriteLine($"   └─ Power Efficiency: {ane.PowerEfficiency:F1} TOPS/W");
                    Console.WriteLine($"   └─ Supports: FP16={ane.SupportsFloat16}, INT8={ane.SupportsInt8}");
                }
                Console.WriteLine();
            }

            // Intel NPU
            if (hardware.IntelNPU != null)
            {
                var npu = hardware.IntelNPU;
                Console.WriteLine($"🔧 Intel NPU: Available");
                Console.WriteLine($"   └─ Generation: {npu.Generation}");
                Console.WriteLine($"   └─ Max TOPS: {npu.MaxTOPS:F1}");
                Console.WriteLine($"   └─ Compute Units: {npu.ComputeUnits}");
                Console.WriteLine($"   └─ Memory Bandwidth: {npu.MemoryBandwidth:F1} GB/s");
                Console.WriteLine($"   └─ Optimal Batch Size: {npu.OptimalBatchSize}");
                Console.WriteLine();
            }

            // Intel AMX
            if (hardware.IntelAMX != null)
            {
                var amx = hardware.IntelAMX;
                Console.WriteLine($"⚡ Intel AMX: Supported");
                Console.WriteLine($"   └─ Max Tiles: {amx.MaxTiles}");
                Console.WriteLine($"   └─ Tile Dimensions: {amx.MaxTileRows}x{amx.MaxTileColumns}");
                Console.WriteLine($"   └─ Estimated Bandwidth: {amx.EstimatedBandwidth:F1} GB/s");
                Console.WriteLine($"   └─ Data Types: BF16={amx.SupportsBF16}, INT8={amx.SupportsInt8}, FP32={amx.SupportsFloat32}");
                Console.WriteLine();
            }

            // CUDA devices
            if (hardware.CudaDevices.Count > 0)
            {
                Console.WriteLine($"🎮 CUDA Devices ({hardware.CudaDevices.Count}):");
                foreach (var device in hardware.CudaDevices)
                {
                    Console.WriteLine($"   └─ {device.Name}");
                    Console.WriteLine($"      └─ Compute Capability: {device.ComputeCapability}");
                    Console.WriteLine($"      └─ Memory: {device.MemorySize / (1024L * 1024 * 1024):F1} GB");
                    Console.WriteLine($"      └─ Multiprocessors: {device.NumMultiprocessors}");
                }
                Console.WriteLine();
            }

            // OpenCL devices
            if (hardware.OpenCLDevices.Count > 0)
            {
                Console.WriteLine($"🔄 OpenCL Devices ({hardware.OpenCLDevices.Count}):");
                foreach (var device in hardware.OpenCLDevices)
                {
                    Console.WriteLine($"   └─ {device.Name}");
                    Console.WriteLine($"      └─ Memory: {device.MemorySize / (1024L * 1024 * 1024):F1} GB");
                }
                Console.WriteLine();
            }

            // Recommendations
            Console.WriteLine("💡 Recommendations:");
            Console.WriteLine("==================");
            Console.WriteLine(GetRecommendations(hardware));
        }
    }

    #region Data Structures

    /// <summary>
    /// Complete hardware information.
    /// </summary>
    public class HardwareInfo
    {
        public PlatformInfo Platform { get; set; } = new();
        public ANEInfo? AppleNeuralEngine { get; set; }
        public NPUInfo? IntelNPU { get; set; }
        public AMXInfo? IntelAMX { get; set; }
        public List<CudaDeviceInfo> CudaDevices { get; set; } = new();
        public List<OpenCLDeviceInfo> OpenCLDevices { get; set; } = new();
    }

    /// <summary>
    /// Platform information.
    /// </summary>
    public class PlatformInfo
    {
        public string OperatingSystem { get; set; } = "";
        public string Architecture { get; set; } = "";
        public bool IsAppleSilicon { get; set; }
        public string DotNetVersion { get; set; } = "";
    }

    /// <summary>
    /// Apple Neural Engine information.
    /// </summary>
    public class ANEInfo
    {
        public bool IsAvailable { get; set; }
        public string Generation { get; set; } = "";
        public double MaxTOPS { get; set; }
        public bool SupportsFloat16 { get; set; }
        public bool SupportsInt8 { get; set; }
        public bool SupportsConvolution { get; set; }
        public bool SupportsAttention { get; set; }
        public int MaxBatchSize { get; set; }
        public double PowerEfficiency { get; set; }
    }

    /// <summary>
    /// Intel NPU information.
    /// </summary>
    public class NPUInfo
    {
        public bool IsAvailable { get; set; }
        public string Generation { get; set; } = "";
        public double MaxTOPS { get; set; }
        public int ComputeUnits { get; set; }
        public double MemoryBandwidth { get; set; }
        public bool SupportsBF16 { get; set; }
        public bool SupportsInt8 { get; set; }
        public bool SupportsConvolution { get; set; }
        public bool SupportsMatMul { get; set; }
        public int OptimalBatchSize { get; set; }
    }

    /// <summary>
    /// Intel AMX information.
    /// </summary>
    public class AMXInfo
    {
        public bool IsSupported { get; set; }
        public int MaxTiles { get; set; }
        public int MaxTileRows { get; set; }
        public int MaxTileColumns { get; set; }
        public int MaxTileBytes { get; set; }
        public bool SupportsBF16 { get; set; }
        public bool SupportsInt8 { get; set; }
        public bool SupportsFloat32 { get; set; }
        public double EstimatedBandwidth { get; set; }
    }

    /// <summary>
    /// CUDA device information.
    /// </summary>
    public class CudaDeviceInfo
    {
        public string DeviceId { get; set; } = "";
        public string Name { get; set; } = "";
        public string ComputeCapability { get; set; } = "";
        public long MemorySize { get; set; }
        public string MaxGridSize { get; set; } = "";
        public string MaxGroupSize { get; set; } = "";
        public int WarpSize { get; set; }
        public int NumMultiprocessors { get; set; }
        public int MaxThreadsPerMultiprocessor { get; set; }
    }

    /// <summary>
    /// OpenCL device information.
    /// </summary>
    public class OpenCLDeviceInfo
    {
        public string DeviceId { get; set; } = "";
        public string Name { get; set; } = "";
        public long MemorySize { get; set; }
        public string MaxGridSize { get; set; } = "";
        public string MaxGroupSize { get; set; } = "";
        public int WarpSize { get; set; }
    }

    #endregion
}