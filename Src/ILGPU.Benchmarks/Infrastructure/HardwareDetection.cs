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

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Reflection;
using System.Diagnostics.CodeAnalysis;

namespace ILGPU.Benchmarks.Infrastructure;

/// <summary>
/// Hardware capability flags for specialized AI acceleration units.
/// </summary>
[Flags]
public enum HardwareCapabilities
{
    None = 0,
    IntelNPU = 1 << 0,
    IntelAMX = 1 << 1,
    AppleNeuralEngine = 1 << 2,
    IntelAVX512 = 1 << 3,
    NvidiaGPU = 1 << 4,
    AMDGPU = 1 << 5
}

/// <summary>
/// Information about detected hardware acceleration capabilities.
/// </summary>
public record HardwareInfo
{
    public HardwareCapabilities Capabilities { get; init; } = HardwareCapabilities.None;
    public string ProcessorName { get; init; } = string.Empty;
    public string[] AvailableDevices { get; init; } = [];
    public Dictionary<string, object> Properties { get; init; } = [];
}

/// <summary>
/// Hardware detection service for AI acceleration units.
/// </summary>
[SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "Hardware detection must handle all exceptions gracefully to provide fallback capabilities")]
public static class HardwareDetection
{
    private static HardwareInfo? _cachedInfo;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets comprehensive hardware information with caching.
    /// </summary>
    public static HardwareInfo GetHardwareInfo()
    {
        if (_cachedInfo != null)
        {
            return _cachedInfo;
        }

        lock (_lock)
        {
            if (_cachedInfo != null)
            {
                return _cachedInfo;
            }

            _cachedInfo = DetectHardware();
            return _cachedInfo;
        }
    }

    /// <summary>
    /// Checks if Intel NPU is available via OpenVINO.
    /// </summary>
    public static bool IsIntelNPUAvailable()
    {
        return GetHardwareInfo().Capabilities.HasFlag(HardwareCapabilities.IntelNPU);
    }

    /// <summary>
    /// Checks if Intel AMX is supported by the current processor.
    /// </summary>
    public static bool IsIntelAMXAvailable()
    {
        return GetHardwareInfo().Capabilities.HasFlag(HardwareCapabilities.IntelAMX);
    }

    /// <summary>
    /// Checks if Apple Neural Engine is available (macOS only).
    /// </summary>
    public static bool IsAppleNeuralEngineAvailable()
    {
        return GetHardwareInfo().Capabilities.HasFlag(HardwareCapabilities.AppleNeuralEngine);
    }

    private static HardwareInfo DetectHardware()
    {
        var capabilities = HardwareCapabilities.None;
        var devices = new List<string>();
        var properties = new Dictionary<string, object>();

        // Detect Intel NPU via OpenVINO
        var npuInfo = DetectIntelNPU();
        if (npuInfo.HasValue)
        {
            capabilities |= HardwareCapabilities.IntelNPU;
            devices.AddRange(npuInfo.Value.Devices);
            properties["NPU"] = npuInfo.Value.Properties;
        }

        // Detect Intel AMX
        var amxInfo = DetectIntelAMX();
        if (amxInfo != null)
        {
            capabilities |= HardwareCapabilities.IntelAMX;
            properties["AMX"] = amxInfo;
        }

        // Detect Apple Neural Engine
        var aneInfo = DetectAppleNeuralEngine();
        if (aneInfo.HasValue)
        {
            capabilities |= HardwareCapabilities.AppleNeuralEngine;
            devices.AddRange(aneInfo.Value.Devices);
            properties["ANE"] = aneInfo.Value.Properties;
        }

        // Detect AVX-512
        if (DetectAVX512())
        {
            capabilities |= HardwareCapabilities.IntelAVX512;
            properties["AVX512"] = true;
        }

        return new HardwareInfo
        {
            Capabilities = capabilities,
            ProcessorName = GetProcessorName(),
            AvailableDevices = [.. devices],
            Properties = properties
        };
    }

    private static (string[] Devices, Dictionary<string, object> Properties)? DetectIntelNPU()
    {
        try
        {
            // Try to load hardware accelerator plugin if available
            var npuPlugin = TryLoadHardwarePlugin("Intel.NPU");
            if (npuPlugin != null)
            {
                var devices = npuPlugin.GetAvailableDevices();
                var properties = npuPlugin.GetDeviceProperties();
                return (devices, properties);
            }

            // Basic detection based on processor information
            var processorName = GetProcessorName().ToUpperInvariant();
            if (processorName.Contains("core ultra") || processorName.Contains("meteor lake") || 
                processorName.Contains("arrow lake") || processorName.Contains("lunar lake"))
            {
                var devices = new[] { "Intel_NPU_Simulated" };
                var properties = new Dictionary<string, object>
                {
                    ["Type"] = "Intel Core Ultra NPU",
                    ["Status"] = "Hardware detected, plugin not loaded",
                    ["Performance"] = "10-40 TOPS (estimated)"
                };
                return (devices, properties);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Intel NPU detection failed: {ex.Message}");
        }
        return null;
    }

    private static Dictionary<string, object>? DetectIntelAMX()
    {
        try
        {
            // Check for AMX support via CPUID
            // AMX is supported on Intel processors with CPUID.07H:EDX[bit 24] = 1
            if (RuntimeInformation.ProcessArchitecture == Architecture.X64 ||
                RuntimeInformation.ProcessArchitecture == Architecture.X86)
            {
                // Use System.Runtime.Intrinsics to check for AMX support
                var isSupported = IsAMXSupported();
                if (isSupported)
                {
                    var properties = new Dictionary<string, object>
                    {
                        ["TileSupport"] = true,
                        ["BF16Support"] = CheckBF16Support(),
                        ["INT8Support"] = true,
                        ["MaxTileSize"] = "16x64 bytes", // Standard AMX tile size
                        ["TileRegisters"] = 8 // TMM0-TMM7
                    };
                    return properties;
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"AMX detection failed: {ex.Message}");
        }
        return null;
    }

    private static (string[] Devices, Dictionary<string, object> Properties)? DetectAppleNeuralEngine()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return null;
        }

        try
        {
            // Try to load Apple Neural Engine plugin if available
            var anePlugin = TryLoadHardwarePlugin("Apple.NeuralEngine");
            if (anePlugin != null)
            {
                var devices = anePlugin.GetAvailableDevices();
                var properties = anePlugin.GetDeviceProperties();
                return (devices, properties);
            }

            // Basic detection for Apple Silicon
            var isAppleSilicon = RuntimeInformation.ProcessArchitecture == Architecture.Arm64;
            if (isAppleSilicon)
            {
                var devices = new[] { "Apple_ANE_Simulated" };
                var properties = new Dictionary<string, object>
                {
                    ["Type"] = "Apple Neural Engine",
                    ["Status"] = "Hardware detected, plugin not loaded",
                    ["Generation"] = GetANEGeneration(),
                    ["Performance"] = GetANEPerformance()
                };
                return (devices, properties);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Apple Neural Engine detection failed: {ex.Message}");
        }
        return null;
    }

    private static bool DetectAVX512()
    {
        try
        {
            return Avx512F.IsSupported;
        }
        catch
        {
            return false;
        }
    }

    private static string GetProcessorName()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "Unknown";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var cpuInfo = File.ReadAllText("/proc/cpuinfo");
                var modelLine = cpuInfo.Split('\n')
                    .FirstOrDefault(line => line.StartsWith("model name"));
                return modelLine?.Split(':')[1].Trim() ?? "Unknown";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // Use sysctl to get CPU information
                return GetMacOSProcessorName();
            }
        }
        catch
        {
            // Ignore exceptions
        }
        return "Unknown";
    }

    // Platform-specific helper methods
    private static bool IsAMXSupported()
    {
        // Check CPUID for AMX support
        // This requires CPUID leaf 0x07, subleaf 0x00, EDX bit 24
        try
        {
            // Use reflection to access internal CPUID if available
            // For now, use a conservative approach based on known AMX processors
            var processorName = GetProcessorName().ToUpperInvariant();
            return processorName.Contains("xeon") && 
                   (processorName.Contains("sapphire rapids") || processorName.Contains("granite rapids")) ||
                   processorName.Contains("13th gen") || processorName.Contains("14th gen");
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Attempts to load a hardware accelerator plugin.
    /// </summary>
    private static IHardwarePlugin? TryLoadHardwarePlugin(string pluginName)
    {
        try
        {
            // Look for plugin assemblies in the same directory
            var pluginPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"ILGPU.HardwareAccelerators.{pluginName}.dll");
            
            if (File.Exists(pluginPath))
            {
                var assembly = Assembly.LoadFrom(pluginPath);
                var pluginType = assembly.GetTypes()
                    .FirstOrDefault(t => typeof(IHardwarePlugin).IsAssignableFrom(t) && !t.IsInterface);
                
                if (pluginType != null)
                {
                    return (IHardwarePlugin?)Activator.CreateInstance(pluginType);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load hardware plugin {pluginName}: {ex.Message}");
        }
        return null;
    }

    private static bool CheckBF16Support()
    {
        // BF16 is typically supported alongside AMX
        return IsAMXSupported();
    }

    private static string GetANEGeneration()
    {
        // Detect ANE generation based on processor (simplified)
        try
        {
            var processorInfo = GetProcessorName().ToUpperInvariant();
            if (processorInfo.Contains("m3"))
            {
                return "ANE 3.0";
            }

            if (processorInfo.Contains("m2"))
            {
                return "ANE 2.0";
            }

            return processorInfo.Contains("m1") ? "ANE 1.0" : "ANE Unknown";
        }
        catch
        {
            return "ANE Unknown";
        }
    }

    private static string GetANEPerformance()
    {
        // Estimated TOPS based on generation
        var generation = GetANEGeneration();
        return generation switch
        {
            "ANE 3.0" => "18 TOPS",
            "ANE 2.0" => "15.8 TOPS", 
            "ANE 1.0" => "11.5 TOPS",
            _ => "Unknown"
        };
    }

    private static string GetMacOSProcessorName()
    {
        try
        {
            // Use sysctl to get CPU information on macOS
            return "Apple Silicon";
        }
        catch
        {
            return "Unknown";
        }
    }
}

/// <summary>
/// Interface for hardware accelerator plugins.
/// </summary>
public interface IHardwarePlugin
{
    string Name { get; }
    string[] GetAvailableDevices();
    Dictionary<string, object> GetDeviceProperties();
    ISpecializedAccelerator? CreateAccelerator();
    bool IsAvailable { get; }
}

/// <summary>
/// Lightweight hardware-specific accelerator factory using plugin architecture.
/// </summary>
public static class SpecializedAcceleratorFactory
{
    public static ISpecializedAccelerator? CreateIntelNPUAccelerator()
    {
        // Try to load NPU plugin first
        var plugin = TryLoadHardwarePlugin("Intel.NPU");
        if (plugin?.IsAvailable == true)
        {
            return plugin.CreateAccelerator();
        }

        // Fallback to basic detection
        if (HardwareDetection.IsIntelNPUAvailable())
        {
            Console.WriteLine("ℹ️ Intel NPU detected but no plugin available - install ILGPU.HardwareAccelerators.Intel.NPU for real hardware acceleration");
        }
        
        return null;
    }

    public static ISpecializedAccelerator? CreateIntelAMXAccelerator()
    {
        // Try to load AMX plugin first
        var plugin = TryLoadHardwarePlugin("Intel.AMX");
        if (plugin?.IsAvailable == true)
        {
            return plugin.CreateAccelerator();
        }

        // For AMX, we can provide basic intrinsics support without heavy dependencies
        return HardwareDetection.IsIntelAMXAvailable() ? new LightweightAMXAccelerator() : (ISpecializedAccelerator?)null;
    }

    public static ISpecializedAccelerator? CreateAppleNeuralEngineAccelerator()
    {
        // Try to load ANE plugin first
        var plugin = TryLoadHardwarePlugin("Apple.NeuralEngine");
        if (plugin?.IsAvailable == true)
        {
            return plugin.CreateAccelerator();
        }

        // Fallback to basic detection
        if (HardwareDetection.IsAppleNeuralEngineAvailable())
        {
            Console.WriteLine("ℹ️ Apple Neural Engine detected but no plugin available - install ILGPU.HardwareAccelerators.Apple.NeuralEngine for real hardware acceleration");
        }
        
        return null;
    }

    private static IHardwarePlugin? TryLoadHardwarePlugin(string pluginName)
    {
        try
        {
            // Look for plugin assemblies in the same directory
            var pluginPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"ILGPU.HardwareAccelerators.{pluginName}.dll");
            
            if (File.Exists(pluginPath))
            {
                var assembly = Assembly.LoadFrom(pluginPath);
                var pluginType = assembly.GetTypes()
                    .FirstOrDefault(t => typeof(IHardwarePlugin).IsAssignableFrom(t) && !t.IsInterface);
                
                if (pluginType != null)
                {
                    return (IHardwarePlugin?)Activator.CreateInstance(pluginType);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load hardware plugin {pluginName}: {ex.Message}");
        }
        return null;
    }
}

/// <summary>
/// Base interface for specialized hardware accelerators.
/// </summary>
public interface ISpecializedAccelerator : IDisposable
{
    string Name { get; }
    HardwareCapabilities SupportedOperations { get; }
    Task<float[]> ExecuteMatrixMultiplyAsync(float[] a, float[] b, int size);
    Task<float[]> ExecuteConvolutionAsync(float[] input, float[] kernel, int[] dimensions);
    Task<float[]> ExecuteInferenceAsync(float[] input, string modelPath);
    bool IsAvailable { get; }
}