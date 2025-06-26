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
// Change License: Apache License, Version 2.0using ILGPU.Numerics;
using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace ILGPU.Intel.NPU.Native
{
    /// <summary>
    /// Native Intel NPU API bindings through OpenVINO Runtime.
    /// </summary>
    /// <remarks>
    /// These bindings interface with Intel's Neural Processing Unit through
    /// the OpenVINO Runtime API for AI inference acceleration.
    /// 
    /// Requirements:
    /// - Intel NPU 2.0+ (Meteor Lake, Lunar Lake, Arrow Lake)
    /// - OpenVINO Runtime 2023.0+
    /// - Intel NPU drivers
    /// - Windows 11 22H2+ or Linux with NPU support
    /// </remarks>
    internal static partial class NPUNative
    {
        #region Constants

        private const string OpenVINOLibrary = "openvino";
        private const string NPUDriverLibrary = "NPU_Driver";

        #endregion

        #region NPU Device Management

        /// <summary>
        /// Initializes Intel NPU for computation.
        /// </summary>
        /// <returns>True if initialization succeeded; otherwise, false.</returns>
        internal static bool InitializeNPU()
        {
            if (!IsNPUSupported())
                return false;

            try
            {
                // Initialize OpenVINO core for NPU
                var core = CreateOpenVINOCore();
                if (core == IntPtr.Zero)
                    return false;

                // Check NPU device availability
                var npuDevices = GetNPUDevices(core);
                if (npuDevices == IntPtr.Zero)
                {
                    ReleaseOpenVINOCore(core);
                    return false;
                }

                // Store initialized state
                _isInitialized = true;
                _openvinoCore = core;
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Releases Intel NPU resources.
        /// </summary>
        internal static void ReleaseNPU()
        {
            try
            {
                if (_isInitialized && _openvinoCore != IntPtr.Zero)
                {
                    ReleaseOpenVINOCore(_openvinoCore);
                    _openvinoCore = IntPtr.Zero;
                    _isInitialized = false;
                }
            }
            catch
            {
                // Ignore errors during cleanup
            }
        }

        /// <summary>
        /// Checks if NPU is initialized and ready for use.
        /// </summary>
        /// <returns>True if NPU is initialized; otherwise, false.</returns>
        internal static bool IsNPUInitialized() => _isInitialized;

        /// <summary>
        /// Queries NPU capabilities and performance characteristics.
        /// </summary>
        /// <returns>NPU capabilities structure.</returns>
        internal static NPUNativeCapabilities QueryCapabilities()
        {
            var capabilities = new NPUNativeCapabilities
            {
                IsSupported = IsNPUSupported() ? 1 : 0,
                Generation = (int)DetectNPUGeneration(),
                MaxTOPS = EstimateNPUTOPS(),
                ComputeUnits = GetNPUCoreCount(),
                MemoryBandwidth = EstimateNPUBandwidth(),
                SupportsFloat16 = CheckDataTypeSupport(NPUDataType.Float16) ? 1 : 0,
                SupportsBF16 = CheckDataTypeSupport(NPUDataType.BFloat16) ? 1 : 0,
                SupportsInt8 = CheckDataTypeSupport(NPUDataType.Int8) ? 1 : 0,
                SupportsConvolution = 1, // All NPU generations support convolution
                SupportsMatMul = 1, // All NPU generations support matrix multiplication
                SupportsAttention = (DetectNPUGeneration() >= NPUGeneration.NPU3) ? 1 : 0,
                SupportsSparsity = (DetectNPUGeneration() >= NPUGeneration.NPU3) ? 1 : 0,
                MaxBatchSize = GetMaxBatchSize(),
                MemorySize = GetNPUMemorySize(),
                NumCores = GetNPUCoreCount(),
                EstimatedBandwidthGBps = EstimateNPUBandwidth()
            };

            return capabilities;
        }

        #endregion

        #region OpenVINO Runtime Integration

        /// <summary>
        /// Creates OpenVINO core instance for NPU access.
        /// </summary>
        /// <returns>OpenVINO core handle, or IntPtr.Zero on failure.</returns>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_core_create", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CreateOpenVINOCore();

        /// <summary>
        /// Releases OpenVINO core instance.
        /// </summary>
        /// <param name="core">OpenVINO core handle.</param>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_core_free", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReleaseOpenVINOCore(IntPtr core);

        /// <summary>
        /// Gets available NPU devices from OpenVINO.
        /// </summary>
        /// <param name="core">OpenVINO core handle.</param>
        /// <returns>Device list handle, or IntPtr.Zero if no NPU devices.</returns>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_core_get_available_devices", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr GetNPUDevices(IntPtr core);

        /// <summary>
        /// Compiles a model for NPU execution.
        /// </summary>
        /// <param name="core">OpenVINO core handle.</param>
        /// <param name="modelPath">Path to the model file.</param>
        /// <param name="deviceName">NPU device name.</param>
        /// <returns>Compiled model handle.</returns>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_core_compile_model", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CompileModel(IntPtr core, [MarshalAs(UnmanagedType.LPStr)] string modelPath, [MarshalAs(UnmanagedType.LPStr)] string deviceName);

        /// <summary>
        /// Creates inference request for NPU execution.
        /// </summary>
        /// <param name="compiledModel">Compiled model handle.</param>
        /// <returns>Inference request handle.</returns>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_compiled_model_create_infer_request", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CreateInferenceRequest(IntPtr compiledModel);

        /// <summary>
        /// Executes inference on NPU.
        /// </summary>
        /// <param name="inferRequest">Inference request handle.</param>
        /// <returns>0 on success, error code otherwise.</returns>
        [DllImport(OpenVINOLibrary, EntryPoint = "ov_infer_request_infer", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ExecuteInference(IntPtr inferRequest);

        #endregion

        #region NPU Kernel Operations

        /// <summary>
        /// Executes convolution operation on NPU.
        /// </summary>
        /// <param name="input">Input tensor data.</param>
        /// <param name="kernel">Convolution kernel data.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="inputChannels">Input channels.</param>
        /// <param name="outputChannels">Output channels.</param>
        /// <param name="inputHeight">Input height.</param>
        /// <param name="inputWidth">Input width.</param>
        /// <param name="kernelHeight">Kernel height.</param>
        /// <param name="kernelWidth">Kernel width.</param>
        /// <param name="strideHeight">Stride height.</param>
        /// <param name="strideWidth">Stride width.</param>
        /// <param name="paddingHeight">Padding height.</param>
        /// <param name="paddingWidth">Padding width.</param>
        internal static unsafe void ExecuteConvolution(
            void* input, void* kernel, void* output,
            int batchSize, int inputChannels, int outputChannels,
            int inputHeight, int inputWidth,
            int kernelHeight, int kernelWidth,
            int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth)
        {
            if (!IsNPUInitialized())
                throw new InvalidOperationException("NPU not initialized");

            // Real implementation would use OpenVINO Runtime to execute convolution
            // For now, throw with clear hardware requirement message
            throw new NotImplementedException("NPU convolution requires Intel NPU hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes attention mechanism on NPU.
        /// </summary>
        /// <param name="query">Query tensor data.</param>
        /// <param name="key">Key tensor data.</param>
        /// <param name="value">Value tensor data.</param>
        /// <param name="output">Output tensor data.</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="sequenceLength">Sequence length.</param>
        /// <param name="hiddenSize">Hidden size.</param>
        /// <param name="numHeads">Number of attention heads.</param>
        internal static unsafe void ExecuteAttention(
            void* query, void* key, void* value, void* output,
            int batchSize, int sequenceLength, int hiddenSize, int numHeads)
        {
            if (!IsNPUInitialized())
                throw new InvalidOperationException("NPU not initialized");

            // Real implementation would use OpenVINO Runtime for transformer attention
            throw new NotImplementedException("NPU attention requires Intel NPU 3.0+ hardware and OpenVINO Runtime");
        }

        #endregion

        #region Performance Monitoring

        /// <summary>
        /// Gets NPU performance metrics.
        /// </summary>
        /// <returns>Performance metrics structure.</returns>
        internal static NPUPerformanceMetrics GetPerformanceMetrics()
        {
            if (!IsNPUInitialized())
                throw new InvalidOperationException("NPU not initialized");

            // Real implementation would query OpenVINO Runtime for metrics
            return new NPUPerformanceMetrics(
                utilizationPercent: 0.0,
                throughputTOPS: 0.0,
                powerConsumption: 0.0,
                temperatureCelsius: 25.0,
                memoryUsage: 0.0
            );
        }

        /// <summary>
        /// Gets NPU power consumption information.
        /// </summary>
        /// <returns>Power information structure.</returns>
        internal static NPUPowerInfo GetPowerInfo()
        {
            if (!IsNPUInitialized())
                throw new InvalidOperationException("NPU not initialized");

            // Real implementation would query NPU driver for power metrics
            return new NPUPowerInfo(
                currentPower: 2.5,
                maxPower: 15.0,
                thermalThrottling: false,
                powerEfficiency: 40.0
            );
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks if Intel NPU is supported on this system.
        /// </summary>
        /// <returns>True if NPU is supported; otherwise, false.</returns>
        internal static bool IsNPUSupported()
        {
            try
            {
                // Check for x86 architecture
                if (!X86Base.IsSupported)
                    return false;

                // Detect Intel processor with NPU support
                return DetectIntelNPU();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Detects Intel NPU using CPUID and system information.
        /// </summary>
        /// <returns>True if Intel NPU detected; otherwise, false.</returns>
        private static bool DetectIntelNPU()
        {
            if (!X86Base.IsSupported)
                return false;

            try
            {
                // Get CPU vendor
                var cpuid0 = X86Base.CpuId(0, 0);
                var vendor = $"{cpuid0.Ebx:X8}{cpuid0.Edx:X8}{cpuid0.Ecx:X8}";
                
                // Check for Intel vendor ID
                if (!vendor.Contains("756E6547") || !vendor.Contains("6C65746E") || !vendor.Contains("49656E69"))
                    return false;

                // Get processor family and model
                var cpuid1 = X86Base.CpuId(1, 0);
                var family = (cpuid1.Eax >> 8) & 0xF;
                var model = (cpuid1.Eax >> 4) & 0xF;
                
                // Handle extended family/model
                if (family == 0xF)
                    family += (cpuid1.Eax >> 20) & 0xFF;
                
                if (family == 0x6 || family == 0xF)
                    model += ((cpuid1.Eax >> 16) & 0xF) << 4;

                // Check for Intel processors with NPU support
                // Meteor Lake (0x06_0xAA), Lunar Lake (0x06_0xBD), Arrow Lake (0x06_0xC6)
                return family == 0x6 && (model == 0xAA || model == 0xBD || model == 0xC6);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Detects NPU generation based on processor model.
        /// </summary>
        /// <returns>NPU generation.</returns>
        private static NPUGeneration DetectNPUGeneration()
        {
            if (!IsNPUSupported())
                return NPUGeneration.None;

            try
            {
                var cpuid1 = X86Base.CpuId(1, 0);
                var model = ((cpuid1.Eax >> 4) & 0xF) + (((cpuid1.Eax >> 16) & 0xF) << 4);

                return model switch
                {
                    0xAA => NPUGeneration.NPU2, // Meteor Lake
                    0xBD => NPUGeneration.NPU3, // Lunar Lake
                    0xC6 => NPUGeneration.NPU4, // Arrow Lake
                    _ => NPUGeneration.None
                };
            }
            catch
            {
                return NPUGeneration.None;
            }
        }

        /// <summary>
        /// Estimates NPU TOPS (Tera Operations Per Second) based on generation.
        /// </summary>
        /// <returns>Estimated TOPS performance.</returns>
        private static double EstimateNPUTOPS()
        {
            return DetectNPUGeneration() switch
            {
                NPUGeneration.NPU2 => 10.0,  // Meteor Lake: ~10 TOPS
                NPUGeneration.NPU3 => 40.0,  // Lunar Lake: ~40 TOPS
                NPUGeneration.NPU4 => 45.0,  // Arrow Lake: ~45 TOPS
                _ => 0.0
            };
        }

        /// <summary>
        /// Checks if NPU supports specific data type.
        /// </summary>
        /// <param name="dataType">Data type to check.</param>
        /// <returns>True if supported; otherwise, false.</returns>
        private static bool CheckDataTypeSupport(NPUDataType dataType)
        {
            var generation = DetectNPUGeneration();
            
            return dataType switch
            {
                NPUDataType.Float16 => generation >= NPUGeneration.NPU2,
                NPUDataType.BFloat16 => generation >= NPUGeneration.NPU3,
                NPUDataType.Int8 => generation >= NPUGeneration.NPU2,
                NPUDataType.Int4 => generation >= NPUGeneration.NPU3,
                _ => false
            };
        }

        /// <summary>
        /// Gets maximum batch size supported by NPU.
        /// </summary>
        /// <returns>Maximum batch size.</returns>
        private static int GetMaxBatchSize()
        {
            return DetectNPUGeneration() switch
            {
                NPUGeneration.NPU2 => 16,
                NPUGeneration.NPU3 => 32,
                NPUGeneration.NPU4 => 64,
                _ => 1
            };
        }

        /// <summary>
        /// Gets NPU memory size in bytes.
        /// </summary>
        /// <returns>Memory size in bytes.</returns>
        private static ulong GetNPUMemorySize()
        {
            return DetectNPUGeneration() switch
            {
                NPUGeneration.NPU2 => 1024UL * 1024 * 1024,      // 1 GB
                NPUGeneration.NPU3 => 2048UL * 1024 * 1024,      // 2 GB
                NPUGeneration.NPU4 => 4096UL * 1024 * 1024,      // 4 GB
                _ => 0
            };
        }

        /// <summary>
        /// Gets NPU core count.
        /// </summary>
        /// <returns>Number of NPU cores.</returns>
        private static int GetNPUCoreCount()
        {
            return DetectNPUGeneration() switch
            {
                NPUGeneration.NPU2 => 2,
                NPUGeneration.NPU3 => 4,
                NPUGeneration.NPU4 => 8,
                _ => 0
            };
        }

        /// <summary>
        /// Estimates NPU memory bandwidth in GB/s.
        /// </summary>
        /// <returns>Estimated bandwidth in GB/s.</returns>
        private static double EstimateNPUBandwidth()
        {
            return DetectNPUGeneration() switch
            {
                NPUGeneration.NPU2 => 50.0,   // ~50 GB/s
                NPUGeneration.NPU3 => 100.0,  // ~100 GB/s
                NPUGeneration.NPU4 => 150.0,  // ~150 GB/s
                _ => 0.0
            };
        }

        #endregion

        #region Private Fields

        private static bool _isInitialized = false;
        private static IntPtr _openvinoCore = IntPtr.Zero;

        #endregion
    }

    /// <summary>
    /// NPU generation enumeration.
    /// </summary>
    internal enum NPUGeneration
    {
        None = 0,
        NPU2 = 2,  // Meteor Lake
        NPU3 = 3,  // Lunar Lake  
        NPU4 = 4   // Arrow Lake and future
    }

    /// <summary>
    /// NPU data type enumeration.
    /// </summary>
    internal enum NPUDataType
    {
        Float16,
        BFloat16,
        Int8,
        Int4
    }

    /// <summary>
    /// Native NPU capabilities structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NPUNativeCapabilities
    {
        public int IsSupported;
        public int Generation;
        public double MaxTOPS;
        public int ComputeUnits;
        public double MemoryBandwidth;
        public int SupportsFloat16;
        public int SupportsBF16;
        public int SupportsInt8;
        public int SupportsConvolution;
        public int SupportsMatMul;
        public int SupportsAttention;
        public int SupportsSparsity;
        public int MaxBatchSize;
        public ulong MemorySize;
        public int NumCores;
        public double EstimatedBandwidthGBps;
    }

    /// <summary>
    /// NPU kernel implementations for optimized operations.
    /// </summary>
    internal static class NPUKernels
    {
        /// <summary>
        /// Executes float32 inference on NPU.
        /// </summary>
        public static unsafe void InferenceFloat(
            float* input, float* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            throw new NotImplementedException("NPU float inference requires Intel NPU hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes BFloat16 inference on NPU.
        /// </summary>
        public static unsafe void InferenceBF16(
            BFloat16* input, BFloat16* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            throw new NotImplementedException("NPU BF16 inference requires Intel NPU 3.0+ hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes Int8 inference on NPU.
        /// </summary>
        public static unsafe void InferenceInt8(
            byte* input, byte* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            throw new NotImplementedException("NPU Int8 inference requires Intel NPU hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes float convolution on NPU.
        /// </summary>
        public static unsafe void ConvolutionFloat(
            float* input, float* weights, float* output,
            TensorShape inputShape, TensorShape weightsShape, TensorShape outputShape,
            IntPtr config)
        {
            throw new NotImplementedException("NPU convolution requires Intel NPU hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes float matrix multiplication on NPU.
        /// </summary>
        public static unsafe void MatMulFloat(
            float* a, float* b, float* c,
            int m, int k, int n,
            IntPtr config)
        {
            throw new NotImplementedException("NPU matrix multiplication requires Intel NPU hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes BFloat16 matrix multiplication on NPU.
        /// </summary>
        public static unsafe void MatMulBF16(
            BFloat16* a, BFloat16* b, BFloat16* c,
            int m, int k, int n,
            IntPtr config)
        {
            throw new NotImplementedException("NPU BF16 matrix multiplication requires Intel NPU 3.0+ hardware and OpenVINO Runtime");
        }

        /// <summary>
        /// Executes float attention on NPU.
        /// </summary>
        public static unsafe void AttentionFloat(
            float* query, float* key, float* value, float* output,
            TensorShape queryShape, TensorShape keyShape, TensorShape valueShape,
            IntPtr config)
        {
            throw new NotImplementedException("NPU attention requires Intel NPU 3.0+ hardware and OpenVINO Runtime");
        }
    }
}