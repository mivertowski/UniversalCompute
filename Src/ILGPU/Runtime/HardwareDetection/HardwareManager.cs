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

using ILGPU.Apple.NeuralEngine;
using ILGPU.Runtime.AMX;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.OneAPI;
using ILGPU.Runtime.ROCm;
using ILGPU.Runtime.Velocity;
using ILGPU.Runtime.Vulkan;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace ILGPU.Runtime.HardwareDetection
{
    /// <summary>
    /// Centralized hardware detection and management system.
    /// </summary>
    public static class HardwareManager
    {
        #region Properties

        /// <summary>
        /// Gets the detected hardware capabilities.
        /// </summary>
        public static HardwareCapabilities Capabilities { get; private set; }

        /// <summary>
        /// Gets whether hardware detection has been performed.
        /// </summary>
        public static bool IsInitialized { get; private set; }

        #endregion

        #region Initialization

        /// <summary>
        /// Initializes hardware detection and discovery.
        /// </summary>
        public static void Initialize()
        {
            if (IsInitialized)
                return;

            try
            {
                Capabilities = DetectHardwareCapabilities();
                IsInitialized = true;
            }
            catch (Exception ex)
            {
                // Create fallback capabilities on any error
                Capabilities = CreateFallbackCapabilities();
                IsInitialized = true;
                System.Diagnostics.Debug.WriteLine($"Hardware detection failed, using fallback: {ex.Message}");
            }
        }

        /// <summary>
        /// Forces re-detection of hardware capabilities.
        /// </summary>
        public static void Refresh()
        {
            IsInitialized = false;
            Initialize();
        }

        #endregion

        #region Hardware Detection

        /// <summary>
        /// Detects all available hardware capabilities.
        /// </summary>
        /// <returns>Hardware capabilities.</returns>
        private static HardwareCapabilities DetectHardwareCapabilities()
        {
            var capabilities = new HardwareCapabilities();

            // Detect NVIDIA CUDA support
            capabilities.CUDA = DetectCUDACapabilities();

            // Detect AMD ROCm/HIP support
            capabilities.ROCm = DetectROCmCapabilities();

            // Detect Intel OneAPI/SYCL support
            capabilities.OneAPI = DetectOneAPICapabilities();

            // Detect Intel AMX support
            capabilities.AMX = DetectAMXCapabilities();

            // Detect Apple hardware support
            capabilities.Apple = DetectAppleCapabilities();

            // Detect OpenCL support
            capabilities.OpenCL = DetectOpenCLCapabilities();

            // Detect Vulkan support
            capabilities.Vulkan = DetectVulkanCapabilities();

            // Detect Velocity (SIMD) support
            capabilities.Velocity = DetectVelocityCapabilities();

            return capabilities;
        }

        /// <summary>
        /// Detects NVIDIA CUDA capabilities.
        /// </summary>
        private static CUDACapabilities DetectCUDACapabilities()
        {
            try
            {
                var devices = CudaDevice.GetDevices(_ => true);
                return new CUDACapabilities
                {
                    IsSupported = devices.Length > 0,
                    DeviceCount = devices.Length,
                    MaxComputeCapability = devices.Length > 0 ? 
                        devices.Max(d => d.ComputeCapability.Major * 10 + d.ComputeCapability.Minor) : 0,
                    TotalMemory = devices.Length > 0 ? devices.Max(d => d.MemorySize) : 0,
                    SupportsCooperativeKernels = devices.Any(d => d.SupportsCooperativeKernels),
                    SupportsCUBLAS = true, // Assume available if CUDA is present
                    SupportsCUFFT = true,
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new CUDACapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects AMD ROCm/HIP capabilities.
        /// </summary>
        private static ROCmCapabilities DetectROCmCapabilities()
        {
            try
            {
                // First check if ROCm runtime is available
                if (!ROCm.Native.ROCmNative.IsROCmSupported())
                {
                    return new ROCmCapabilities
                    {
                        IsSupported = false,
                        ErrorMessage = "ROCm runtime not available"
                    };
                }

                // Initialize ROCm and get device count
                var deviceCount = ROCm.Native.ROCmNative.GetDeviceCountSafe();
                if (deviceCount == 0)
                {
                    return new ROCmCapabilities
                    {
                        IsSupported = false,
                        ErrorMessage = "No ROCm devices found"
                    };
                }

                // Get device properties for capability detection
                long maxMemory = 0;
                int maxComputeCapability = 0;
                bool rocBlasAvailable = false;
                bool rocFFTAvailable = false;

                for (int i = 0; i < deviceCount; i++)
                {
                    var props = ROCm.Native.ROCmNative.GetDevicePropertiesSafe(i);
                    if (props.HasValue)
                    {
                        var deviceProps = props.Value;
                        maxMemory = Math.Max(maxMemory, (long)deviceProps.TotalGlobalMem);
                        
                        // Calculate compute capability from architecture
                        var computeCapability = DetermineComputeCapability(deviceProps);
                        maxComputeCapability = Math.Max(maxComputeCapability, computeCapability);
                    }
                }

                // Check for ROCm library availability
                rocBlasAvailable = CheckROCBlasAvailability();
                rocFFTAvailable = CheckROCFFTAvailability();

                return new ROCmCapabilities
                {
                    IsSupported = true,
                    DeviceCount = deviceCount,
                    MaxComputeCapability = maxComputeCapability,
                    TotalMemory = maxMemory,
                    SupportsROCBLAS = rocBlasAvailable,
                    SupportsROCFFT = rocFFTAvailable,
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new ROCmCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Determines compute capability from HIP device properties.
        /// </summary>
        private static int DetermineComputeCapability(ROCm.Native.HipDeviceProperties props)
        {
            // Map architecture information to compute capability
            // This is based on the device name and architecture features
            var deviceName = props.Name?.ToLowerInvariant() ?? "";
            
            // RDNA3 (gfx11xx)
            if (deviceName.Contains("7900", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("7800", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("7700", StringComparison.OrdinalIgnoreCase))
                return 113; // RDNA3 generation

            // RDNA2 (gfx10xx)  
            if (deviceName.Contains("6900", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("6800", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("6700", StringComparison.OrdinalIgnoreCase) || 
                deviceName.Contains("6600", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("6500", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("6400", StringComparison.OrdinalIgnoreCase))
                return 103; // RDNA2 generation

            // RDNA1 (gfx10xx)
            if (deviceName.Contains("5700", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("5600", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("5500", StringComparison.OrdinalIgnoreCase))
                return 101; // RDNA1 generation

            // CDNA2 (MI200 series)
            if (deviceName.Contains("mi250", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("mi210", StringComparison.OrdinalIgnoreCase))
                return 90; // CDNA2 (gfx90a)

            // CDNA1 (MI100)
            if (deviceName.Contains("mi100", StringComparison.OrdinalIgnoreCase))
                return 90; // CDNA1 (gfx908)

            // GCN5 (Vega)
            if (deviceName.Contains("vega", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("radeon vii", StringComparison.OrdinalIgnoreCase))
                return 90; // GCN5 (gfx906)

            // GCN4 (Polaris, older Vega)
            if (deviceName.Contains("polaris", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("rx 480", StringComparison.OrdinalIgnoreCase) || deviceName.Contains("rx 580", StringComparison.OrdinalIgnoreCase))
                return 80; // GCN4 

            // Conservative fallback
            return 70; // Minimum supported
        }

        /// <summary>
        /// Checks if ROCBlas library is available.
        /// </summary>
        private static bool CheckROCBlasAvailability()
        {
            try
            {
                // Try to detect ROCBlas by checking for library files or environment
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return System.IO.File.Exists("/opt/rocm/lib/librocblas.so") ||
                           System.IO.File.Exists("/usr/lib/librocblas.so") ||
                           Environment.GetEnvironmentVariable("ROCM_PATH") != null;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return System.IO.File.Exists("C:\\Program Files\\AMD\\ROCm\\bin\\rocblas.dll");
                }
                return false;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Checks if ROCFFT library is available.
        /// </summary>
        private static bool CheckROCFFTAvailability()
        {
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return System.IO.File.Exists("/opt/rocm/lib/librocfft.so") ||
                           System.IO.File.Exists("/usr/lib/librocfft.so");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return System.IO.File.Exists("C:\\Program Files\\AMD\\ROCm\\bin\\rocfft.dll");
                }
                return false;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Detects Intel OneAPI/SYCL capabilities.
        /// </summary>
        private static OneAPICapabilities DetectOneAPICapabilities()
        {
            try
            {
                var devices = IntelOneAPIDevice.GetDevices();
                return new OneAPICapabilities
                {
                    IsSupported = devices.Length > 0,
                    DeviceCount = devices.Length,
                    MaxArchitecture = devices.Length > 0 ? 
                        devices.Max(d => (int)d.Architecture) : 0,
                    TotalMemory = devices.Length > 0 ? devices.Max(d => d.MemorySize) : 0,
                    SupportsMKLSYCL = devices.Any(d => (int)d.Architecture >= 2), // XeLP equivalent
                    SupportsLevel0 = true, // Assume Level Zero support
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new OneAPICapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects Intel AMX capabilities.
        /// </summary>
        private static AMXCapabilities DetectAMXCapabilities()
        {
            try
            {
                var devices = IntelAMXDevice.GetDevices();
                return new AMXCapabilities
                {
                    IsSupported = devices.Length > 0,
                    SupportsBF16 = devices.Any(d => d.SupportsBF16),
                    SupportsINT8 = devices.Any(d => d.SupportsINT8),
                    SupportsMixedPrecision = devices.Any(d => d.SupportsMixedPrecision),
                    MaxTileSize = devices.Length > 0 ? devices.Max(d => d.MaxTileSize) : 0,
                    TileCount = devices.Length > 0 ? devices.Max(d => d.TileCount) : 0,
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new AMXCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects Apple hardware capabilities.
        /// </summary>
        private static AppleCapabilities DetectAppleCapabilities()
        {
            try
            {
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    return new AppleCapabilities { IsSupported = false };
                }

                var metalDevices = Array.Empty<Device>(); // TODO: AppleMetalDevice not implemented
                var aneDevices = Array.Empty<AppleNeuralEngineDevice>(); // TODO: AppleNeuralEngineDevice.GetDevices() not implemented

                return new AppleCapabilities
                {
                    IsSupported = metalDevices.Length > 0 || aneDevices.Length > 0,
                    SupportsMetalGPU = metalDevices.Length > 0,
                    SupportsNeuralEngine = aneDevices.Length > 0,
                    MetalDeviceCount = metalDevices.Length,
                    TotalGPUMemory = metalDevices.Length > 0 ? metalDevices.Max(d => d.MemorySize) : 0,
                    SupportsCoreML = aneDevices.Length > 0,
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new AppleCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects OpenCL capabilities.
        /// </summary>
        private static OpenCLCapabilities DetectOpenCLCapabilities()
        {
            try
            {
                var devices = Array.Empty<CLDevice>(); // TODO: CLDevice.GetDevices() not implemented
                return new OpenCLCapabilities
                {
                    IsSupported = devices.Length > 0,
                    DeviceCount = devices.Length,
                    MaxOpenCLVersion = devices.Length > 0 ? 
                        devices.Max(d => d.OpenCLMajorVersion * 10 + d.OpenCLMinorVersion) : 0,
                    TotalMemory = devices.Length > 0 ? devices.Max(d => d.MemorySize) : 0,
                    SupportsFP64 = devices.Any(d => d.SupportsDoublePrecision),
                    SupportsUnifiedMemory = devices.Any(d => d.SupportsUnifiedMemory),
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new OpenCLCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects Vulkan capabilities.
        /// </summary>
        private static VulkanCapabilities DetectVulkanCapabilities()
        {
            try
            {
                var devices = Array.Empty<VulkanDevice>(); // TODO: VulkanDevice.GetDevices() not implemented
                return new VulkanCapabilities
                {
                    IsSupported = devices.Length > 0,
                    DeviceCount = devices.Length,
                    MaxVulkanVersion = 0, // TODO: VulkanApiVersion property not available
                    TotalMemory = 0, // TODO: MemorySize property not available
                    SupportsCompute = false, // TODO: SupportsCompute property not available
                    SupportsRayTracing = false, // TODO: SupportsRayTracing property not available
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new VulkanCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Detects Velocity (SIMD) capabilities.
        /// </summary>
        private static VelocityCapabilities DetectVelocityCapabilities()
        {
            try
            {
                var devices = Array.Empty<VelocityDevice>(); // TODO: VelocityDevice.GetDevices() not implemented
                return new VelocityCapabilities
                {
                    IsSupported = true, // CPU SIMD always available
                    SupportsAVX2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported,
                    SupportsAVX512 = System.Runtime.Intrinsics.X86.Avx512F.IsSupported,
                    SupportsNEON = System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported,
                    VectorSize = 256, // Conservative default for AVX2
                    ErrorMessage = null
                };
            }
            catch (Exception ex)
            {
                return new VelocityCapabilities
                {
                    IsSupported = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Creates fallback capabilities when detection fails.
        /// </summary>
        private static HardwareCapabilities CreateFallbackCapabilities()
        {
            return new HardwareCapabilities
            {
                CUDA = new CUDACapabilities { IsSupported = false },
                ROCm = new ROCmCapabilities { IsSupported = false },
                OneAPI = new OneAPICapabilities { IsSupported = false },
                AMX = new AMXCapabilities { IsSupported = false },
                Apple = new AppleCapabilities { IsSupported = false },
                OpenCL = new OpenCLCapabilities { IsSupported = false },
                Vulkan = new VulkanCapabilities { IsSupported = false },
                Velocity = new VelocityCapabilities 
                { 
                    IsSupported = true, // CPU always available as fallback
                    SupportsAVX2 = false,
                    SupportsAVX512 = false,
                    SupportsNEON = false,
                    VectorSize = 128 // Conservative default
                }
            };
        }

        #endregion

        #region Accelerator Selection

        /// <summary>
        /// Gets the best available accelerator for a specific workload type.
        /// </summary>
        /// <param name="workloadType">The type of workload.</param>
        /// <param name="context">ILGPU context.</param>
        /// <returns>The best accelerator or null if none available.</returns>
        public static Accelerator? GetBestAccelerator(WorkloadType workloadType, Context context)
        {
            Initialize();

            return workloadType switch
            {
                WorkloadType.MatrixOperations => GetBestMatrixAccelerator(context),
                WorkloadType.FFTOperations => GetBestFFTAccelerator(context),
                WorkloadType.AIInference => GetBestAIAccelerator(context),
                WorkloadType.GeneralCompute => GetBestGeneralAccelerator(context),
                WorkloadType.ImageProcessing => GetBestImageAccelerator(context),
                _ => GetBestGeneralAccelerator(context)
            };
        }

        /// <summary>
        /// Gets the best accelerator for matrix operations.
        /// </summary>
        private static Accelerator? GetBestMatrixAccelerator(Context context)
        {
            // Priority: AMX > CUDA (cuBLAS) > ROCm (rocBLAS) > OneAPI (MKL) > OpenCL > CPU
            
            if (Capabilities.AMX.IsSupported && Capabilities.AMX.SupportsBF16)
            {
                // TODO: IntelAMXDevice.GetDefaultDevice() and CreateAMXAccelerator not implemented
                // var device = IntelAMXDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAMXAccelerator(device);
            }

            if (Capabilities.CUDA.IsSupported && Capabilities.CUDA.SupportsCUBLAS)
            {
                var devices = CudaDevice.GetDevices(_ => true);
                if (devices.Length > 0)
                    return context.CreateCudaAccelerator(devices[0]);
            }

            if (Capabilities.ROCm.IsSupported && Capabilities.ROCm.SupportsROCBLAS)
            {
                // TODO: ROCmDevice.GetBestDevice() and CreateROCmAccelerator not implemented
                // var device = ROCmDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateROCmAccelerator(device);
            }

            if (Capabilities.OneAPI.IsSupported && Capabilities.OneAPI.SupportsMKLSYCL)
            {
                // TODO: IntelOneAPIDevice.GetDefaultDevice() and CreateOneAPIAccelerator not implemented
                // var device = IntelOneAPIDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateOneAPIAccelerator(device);
            }

            return CreateFallbackAccelerator(context);
        }

        /// <summary>
        /// Gets the best accelerator for FFT operations.
        /// </summary>
        private static Accelerator? GetBestFFTAccelerator(Context context)
        {
            // Priority: CUDA (cuFFT) > ROCm (rocFFT) > OneAPI > OpenCL > CPU
            
            if (Capabilities.CUDA.IsSupported && Capabilities.CUDA.SupportsCUFFT)
            {
                var devices = CudaDevice.GetDevices(_ => true);
                if (devices.Length > 0)
                    return context.CreateCudaAccelerator(devices[0]);
            }

            if (Capabilities.ROCm.IsSupported && Capabilities.ROCm.SupportsROCFFT)
            {
                // TODO: ROCmDevice.GetBestDevice() and CreateROCmAccelerator not implemented
                // var device = ROCmDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateROCmAccelerator(device);
            }

            if (Capabilities.OneAPI.IsSupported)
            {
                // TODO: IntelOneAPIDevice.GetDefaultDevice() and CreateOneAPIAccelerator not implemented
                // var device = IntelOneAPIDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateOneAPIAccelerator(device);
            }

            return CreateFallbackAccelerator(context);
        }

        /// <summary>
        /// Gets the best accelerator for AI inference.
        /// </summary>
        private static Accelerator? GetBestAIAccelerator(Context context)
        {
            // Priority: Apple ANE > AMX > CUDA > Apple Metal > ROCm > OneAPI > CPU
            
            if (Capabilities.Apple.IsSupported && Capabilities.Apple.SupportsNeuralEngine)
            {
                // TODO: AppleNeuralEngineDevice.GetDefaultDevice() and CreateAppleNeuralEngineAccelerator not implemented
                // var device = AppleNeuralEngineDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAppleNeuralEngineAccelerator(device);
            }

            if (Capabilities.AMX.IsSupported)
            {
                // TODO: IntelAMXDevice.GetDefaultDevice() and CreateAMXAccelerator not implemented
                // var device = IntelAMXDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAMXAccelerator(device);
            }

            if (Capabilities.CUDA.IsSupported)
            {
                var devices = CudaDevice.GetDevices(_ => true);
                if (devices.Length > 0)
                    return context.CreateCudaAccelerator(devices[0]);
            }

            if (Capabilities.Apple.IsSupported && Capabilities.Apple.SupportsMetalGPU)
            {
                // TODO: AppleMetalDevice.GetDefaultDevice() and CreateAppleMetalAccelerator not implemented
                // var device = AppleMetalDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAppleMetalAccelerator(device);
            }

            return CreateFallbackAccelerator(context);
        }

        /// <summary>
        /// Gets the best general purpose accelerator.
        /// </summary>
        private static Accelerator? GetBestGeneralAccelerator(Context context)
        {
            // Priority: CUDA > ROCm > OneAPI > Apple Metal > Vulkan > OpenCL > CPU
            
            if (Capabilities.CUDA.IsSupported)
            {
                var devices = CudaDevice.GetDevices(_ => true);
                if (devices.Length > 0)
                    return context.CreateCudaAccelerator(devices[0]);
            }

            if (Capabilities.ROCm.IsSupported)
            {
                // TODO: ROCmDevice.GetBestDevice() and CreateROCmAccelerator not implemented
                // var device = ROCmDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateROCmAccelerator(device);
            }

            if (Capabilities.OneAPI.IsSupported)
            {
                // TODO: IntelOneAPIDevice.GetDefaultDevice() and CreateOneAPIAccelerator not implemented
                // var device = IntelOneAPIDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateOneAPIAccelerator(device);
            }

            if (Capabilities.Apple.IsSupported && Capabilities.Apple.SupportsMetalGPU)
            {
                // TODO: AppleMetalDevice.GetDefaultDevice() and CreateAppleMetalAccelerator not implemented
                // var device = AppleMetalDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAppleMetalAccelerator(device);
            }

            if (Capabilities.Vulkan.IsSupported && Capabilities.Vulkan.SupportsCompute)
            {
                // TODO: VulkanDevice.GetBestDevice() and CreateVulkanAccelerator not implemented
                // var device = VulkanDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateVulkanAccelerator(device);
            }

            if (Capabilities.OpenCL.IsSupported)
            {
                // TODO: CLDevice.GetBestDevice() and CreateCLAccelerator not implemented
                // var device = CLDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateCLAccelerator(device);
            }

            return CreateFallbackAccelerator(context);
        }

        /// <summary>
        /// Gets the best accelerator for image processing.
        /// </summary>
        private static Accelerator? GetBestImageAccelerator(Context context)
        {
            // Priority: CUDA > Apple Metal > ROCm > Vulkan > OpenCL > CPU
            
            if (Capabilities.CUDA.IsSupported)
            {
                var devices = CudaDevice.GetDevices(_ => true);
                if (devices.Length > 0)
                    return context.CreateCudaAccelerator(devices[0]);
            }

            if (Capabilities.Apple.IsSupported && Capabilities.Apple.SupportsMetalGPU)
            {
                // TODO: AppleMetalDevice.GetDefaultDevice() and CreateAppleMetalAccelerator not implemented
                // var device = AppleMetalDevice.GetDefaultDevice();
                // if (device != null)
                //     return context.CreateAppleMetalAccelerator(device);
            }

            if (Capabilities.ROCm.IsSupported)
            {
                // TODO: ROCmDevice.GetBestDevice() and CreateROCmAccelerator not implemented
                // var device = ROCmDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateROCmAccelerator(device);
            }

            if (Capabilities.Vulkan.IsSupported && Capabilities.Vulkan.SupportsCompute)
            {
                // TODO: VulkanDevice.GetBestDevice() and CreateVulkanAccelerator not implemented
                // var device = VulkanDevice.GetBestDevice();
                // if (device != null)
                //     return context.CreateVulkanAccelerator(device);
            }

            return CreateFallbackAccelerator(context);
        }

        /// <summary>
        /// Creates a fallback CPU accelerator.
        /// </summary>
        private static Accelerator CreateFallbackAccelerator(Context context)
        {
            if (Capabilities.Velocity.IsSupported)
            {
                return context.CreateVelocityAccelerator();
            }
            else
            {
                return context.CreateCPUAccelerator();
            }
        }

        #endregion

        #region Diagnostics

        /// <summary>
        /// Prints comprehensive hardware detection results.
        /// </summary>
        public static void PrintHardwareInfo()
        {
            Initialize();

            Console.WriteLine("=== Hardware Detection Results ===");
            Console.WriteLine();
            
            Console.WriteLine($"NVIDIA CUDA: {(Capabilities.CUDA.IsSupported ? "✓" : "✗")}");
            if (Capabilities.CUDA.IsSupported)
            {
                Console.WriteLine($"  Devices: {Capabilities.CUDA.DeviceCount}");
                Console.WriteLine($"  Max Compute: {Capabilities.CUDA.MaxComputeCapability / 10.0:F1}");
                Console.WriteLine($"  Max Memory: {Capabilities.CUDA.TotalMemory / (1024 * 1024)} MB");
                Console.WriteLine($"  cuBLAS: {(Capabilities.CUDA.SupportsCUBLAS ? "✓" : "✗")}");
                Console.WriteLine($"  cuFFT: {(Capabilities.CUDA.SupportsCUFFT ? "✓" : "✗")}");
            }
            else if (!string.IsNullOrEmpty(Capabilities.CUDA.ErrorMessage))
            {
                Console.WriteLine($"  Error: {Capabilities.CUDA.ErrorMessage}");
            }
            Console.WriteLine();

            Console.WriteLine($"AMD ROCm/HIP: {(Capabilities.ROCm.IsSupported ? "✓" : "✗")}");
            if (Capabilities.ROCm.IsSupported)
            {
                Console.WriteLine($"  Devices: {Capabilities.ROCm.DeviceCount}");
                Console.WriteLine($"  Max Compute: {Capabilities.ROCm.MaxComputeCapability / 10.0:F1}");
                Console.WriteLine($"  Max Memory: {Capabilities.ROCm.TotalMemory / (1024 * 1024)} MB");
                Console.WriteLine($"  rocBLAS: {(Capabilities.ROCm.SupportsROCBLAS ? "✓" : "✗")}");
                Console.WriteLine($"  rocFFT: {(Capabilities.ROCm.SupportsROCFFT ? "✓" : "✗")}");
            }
            Console.WriteLine();

            Console.WriteLine($"Intel OneAPI/SYCL: {(Capabilities.OneAPI.IsSupported ? "✓" : "✗")}");
            if (Capabilities.OneAPI.IsSupported)
            {
                Console.WriteLine($"  Devices: {Capabilities.OneAPI.DeviceCount}");
                Console.WriteLine($"  Max Architecture: {Capabilities.OneAPI.MaxArchitecture}");
                Console.WriteLine($"  Max Memory: {Capabilities.OneAPI.TotalMemory / (1024 * 1024)} MB");
                Console.WriteLine($"  MKL SYCL: {(Capabilities.OneAPI.SupportsMKLSYCL ? "✓" : "✗")}");
            }
            Console.WriteLine();

            Console.WriteLine($"Intel AMX: {(Capabilities.AMX.IsSupported ? "✓" : "✗")}");
            if (Capabilities.AMX.IsSupported)
            {
                Console.WriteLine($"  BF16: {(Capabilities.AMX.SupportsBF16 ? "✓" : "✗")}");
                Console.WriteLine($"  INT8: {(Capabilities.AMX.SupportsINT8 ? "✓" : "✗")}");
                Console.WriteLine($"  Mixed Precision: {(Capabilities.AMX.SupportsMixedPrecision ? "✓" : "✗")}");
                Console.WriteLine($"  Tile Size: {Capabilities.AMX.MaxTileSize}x{Capabilities.AMX.MaxTileSize}");
            }
            Console.WriteLine();

            Console.WriteLine($"Apple Hardware: {(Capabilities.Apple.IsSupported ? "✓" : "✗")}");
            if (Capabilities.Apple.IsSupported)
            {
                Console.WriteLine($"  Metal GPU: {(Capabilities.Apple.SupportsMetalGPU ? "✓" : "✗")}");
                Console.WriteLine($"  Neural Engine: {(Capabilities.Apple.SupportsNeuralEngine ? "✓" : "✗")}");
                Console.WriteLine($"  Metal Devices: {Capabilities.Apple.MetalDeviceCount}");
                Console.WriteLine($"  Max GPU Memory: {Capabilities.Apple.TotalGPUMemory / (1024 * 1024)} MB");
            }
            Console.WriteLine();

            Console.WriteLine($"OpenCL: {(Capabilities.OpenCL.IsSupported ? "✓" : "✗")}");
            if (Capabilities.OpenCL.IsSupported)
            {
                Console.WriteLine($"  Devices: {Capabilities.OpenCL.DeviceCount}");
                Console.WriteLine($"  Max Version: {Capabilities.OpenCL.MaxOpenCLVersion / 10.0:F1}");
                Console.WriteLine($"  FP64: {(Capabilities.OpenCL.SupportsFP64 ? "✓" : "✗")}");
                Console.WriteLine($"  Unified Memory: {(Capabilities.OpenCL.SupportsUnifiedMemory ? "✓" : "✗")}");
            }
            Console.WriteLine();

            Console.WriteLine($"Vulkan Compute: {(Capabilities.Vulkan.IsSupported ? "✓" : "✗")}");
            if (Capabilities.Vulkan.IsSupported)
            {
                Console.WriteLine($"  Devices: {Capabilities.Vulkan.DeviceCount}");
                Console.WriteLine($"  Compute Support: {(Capabilities.Vulkan.SupportsCompute ? "✓" : "✗")}");
                Console.WriteLine($"  Ray Tracing: {(Capabilities.Vulkan.SupportsRayTracing ? "✓" : "✗")}");
            }
            Console.WriteLine();

            Console.WriteLine($"Velocity (SIMD): {(Capabilities.Velocity.IsSupported ? "✓" : "✗")}");
            if (Capabilities.Velocity.IsSupported)
            {
                Console.WriteLine($"  AVX2: {(Capabilities.Velocity.SupportsAVX2 ? "✓" : "✗")}");
                Console.WriteLine($"  AVX-512: {(Capabilities.Velocity.SupportsAVX512 ? "✓" : "✗")}");
                Console.WriteLine($"  ARM NEON: {(Capabilities.Velocity.SupportsNEON ? "✓" : "✗")}");
                Console.WriteLine($"  Vector Size: {Capabilities.Velocity.VectorSize} bits");
            }

            Console.WriteLine("====================================");
        }

        #endregion
    }

    #region Capability Structures

    /// <summary>
    /// Overall hardware capabilities.
    /// </summary>
    public class HardwareCapabilities
    {
        public CUDACapabilities CUDA { get; set; } = new();
        public ROCmCapabilities ROCm { get; set; } = new();
        public OneAPICapabilities OneAPI { get; set; } = new();
        public AMXCapabilities AMX { get; set; } = new();
        public AppleCapabilities Apple { get; set; } = new();
        public OpenCLCapabilities OpenCL { get; set; } = new();
        public VulkanCapabilities Vulkan { get; set; } = new();
        public VelocityCapabilities Velocity { get; set; } = new();
    }

    public class CUDACapabilities
    {
        public bool IsSupported { get; set; }
        public int DeviceCount { get; set; }
        public int MaxComputeCapability { get; set; }
        public long TotalMemory { get; set; }
        public bool SupportsCooperativeKernels { get; set; }
        public bool SupportsCUBLAS { get; set; }
        public bool SupportsCUFFT { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class ROCmCapabilities
    {
        public bool IsSupported { get; set; }
        public int DeviceCount { get; set; }
        public int MaxComputeCapability { get; set; }
        public long TotalMemory { get; set; }
        public bool SupportsROCBLAS { get; set; }
        public bool SupportsROCFFT { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class OneAPICapabilities
    {
        public bool IsSupported { get; set; }
        public int DeviceCount { get; set; }
        public int MaxArchitecture { get; set; }
        public long TotalMemory { get; set; }
        public bool SupportsMKLSYCL { get; set; }
        public bool SupportsLevel0 { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class AMXCapabilities
    {
        public bool IsSupported { get; set; }
        public bool SupportsBF16 { get; set; }
        public bool SupportsINT8 { get; set; }
        public bool SupportsMixedPrecision { get; set; }
        public int MaxTileSize { get; set; }
        public int TileCount { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class AppleCapabilities
    {
        public bool IsSupported { get; set; }
        public bool SupportsMetalGPU { get; set; }
        public bool SupportsNeuralEngine { get; set; }
        public int MetalDeviceCount { get; set; }
        public long TotalGPUMemory { get; set; }
        public bool SupportsCoreML { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class OpenCLCapabilities
    {
        public bool IsSupported { get; set; }
        public int DeviceCount { get; set; }
        public int MaxOpenCLVersion { get; set; }
        public long TotalMemory { get; set; }
        public bool SupportsFP64 { get; set; }
        public bool SupportsUnifiedMemory { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class VulkanCapabilities
    {
        public bool IsSupported { get; set; }
        public int DeviceCount { get; set; }
        public uint MaxVulkanVersion { get; set; }
        public long TotalMemory { get; set; }
        public bool SupportsCompute { get; set; }
        public bool SupportsRayTracing { get; set; }
        public string? ErrorMessage { get; set; }
    }

    public class VelocityCapabilities
    {
        public bool IsSupported { get; set; }
        public bool SupportsAVX2 { get; set; }
        public bool SupportsAVX512 { get; set; }
        public bool SupportsNEON { get; set; }
        public int VectorSize { get; set; }
        public string? ErrorMessage { get; set; }
    }

    #endregion

    #region Enums

    /// <summary>
    /// Types of computational workloads.
    /// </summary>
    public enum WorkloadType
    {
        GeneralCompute,
        MatrixOperations,
        FFTOperations,
        AIInference,
        ImageProcessing
    }

    #endregion
}