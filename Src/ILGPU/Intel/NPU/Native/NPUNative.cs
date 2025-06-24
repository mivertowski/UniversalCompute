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

using ILGPU.Numerics;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.Intel.NPU.Native
{
    /// <summary>
    /// Native Intel NPU API bindings.
    /// </summary>
    internal static partial class NPUNative
    {
        #region Constants

        private const string NPULibrary = "intel_npu_runtime";
        private const string OpenVINOLibrary = "openvino";

        #endregion

        #region NPU Detection and Initialization

        /// <summary>
        /// Checks if Intel NPU is available on this system.
        /// </summary>
        [LibraryImport(NPULibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool NPU_IsAvailable();

        /// <summary>
        /// Initializes the NPU runtime.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_Initialize();

        /// <summary>
        /// Releases NPU resources.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial void NPU_Release();

        /// <summary>
        /// Checks if NPU is initialized.
        /// </summary>
        [LibraryImport(NPULibrary)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool NPU_IsInitialized();

        /// <summary>
        /// Queries NPU capabilities.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_QueryCapabilities(out NPUNativeCapabilities capabilities);

        #endregion

        #region Performance Monitoring

        /// <summary>
        /// Gets current NPU performance metrics.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_GetPerformanceMetrics(out NPUNativePerformanceMetrics metrics);

        /// <summary>
        /// Gets NPU power information.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_GetPowerInfo(out NPUNativePowerInfo powerInfo);

        #endregion

        #region Inference Operations

        /// <summary>
        /// Executes inference with float32 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_InferenceFloat32(
            IntPtr input, IntPtr output,
            IntPtr inputShape, IntPtr outputShape,
            IntPtr context);

        /// <summary>
        /// Executes inference with BFloat16 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_InferenceBF16(
            IntPtr input, IntPtr output,
            IntPtr inputShape, IntPtr outputShape,
            IntPtr context);

        /// <summary>
        /// Executes inference with INT8 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_InferenceInt8(
            IntPtr input, IntPtr output,
            IntPtr inputShape, IntPtr outputShape,
            IntPtr context);

        #endregion

        #region Convolution Operations

        /// <summary>
        /// Executes convolution with float32 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_ConvolutionFloat32(
            IntPtr input, IntPtr weights, IntPtr output,
            IntPtr inputShape, IntPtr weightsShape, IntPtr outputShape,
            IntPtr config);

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Executes matrix multiplication with float32 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_MatMulFloat32(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            IntPtr config);

        /// <summary>
        /// Executes matrix multiplication with BFloat16 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_MatMulBF16(
            IntPtr a, IntPtr b, IntPtr c,
            int m, int k, int n,
            IntPtr config);

        #endregion

        #region Attention Operations

        /// <summary>
        /// Executes attention mechanism with float32 data.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_AttentionFloat32(
            IntPtr query, IntPtr key, IntPtr value, IntPtr output,
            IntPtr queryShape, IntPtr keyShape, IntPtr valueShape,
            IntPtr config);

        #endregion

        #region Model Management

        /// <summary>
        /// Loads a model from file.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_LoadModel(
            IntPtr modelPath, int format, out IntPtr modelHandle);

        /// <summary>
        /// Releases a loaded model.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial void NPU_ReleaseModel(IntPtr modelHandle);

        /// <summary>
        /// Optimizes a model for NPU execution.
        /// </summary>
        [LibraryImport(NPULibrary)]
        internal static partial int NPU_OptimizeModel(
            IntPtr modelHandle, IntPtr options, out IntPtr optimizedHandle);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks if NPU is available.
        /// </summary>
        internal static bool IsNPUAvailable()
        {
            try
            {
                return NPU_IsAvailable();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Initializes NPU.
        /// </summary>
        internal static bool InitializeNPU()
        {
            try
            {
                return NPU_Initialize() == 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Checks if NPU is initialized.
        /// </summary>
        internal static bool IsNPUInitialized()
        {
            try
            {
                return NPU_IsInitialized();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Releases NPU resources.
        /// </summary>
        internal static void ReleaseNPU()
        {
            try
            {
                NPU_Release();
            }
            catch
            {
                // Ignore errors during cleanup
            }
        }

        /// <summary>
        /// Queries NPU capabilities.
        /// </summary>
        internal static NPUNativeCapabilities QueryCapabilities()
        {
            if (NPU_QueryCapabilities(out var capabilities) == 0)
                return capabilities;

            return new NPUNativeCapabilities();
        }

        /// <summary>
        /// Gets performance metrics.
        /// </summary>
        internal static NPUPerformanceMetrics GetPerformanceMetrics()
        {
            if (NPU_GetPerformanceMetrics(out var metrics) == 0)
            {
                return new NPUPerformanceMetrics(
                    metrics.UtilizationPercent,
                    metrics.ThroughputTOPS,
                    metrics.PowerConsumption,
                    metrics.TemperatureCelsius,
                    metrics.MemoryUsage);
            }

            return new NPUPerformanceMetrics();
        }

        /// <summary>
        /// Gets power information.
        /// </summary>
        internal static NPUPowerInfo GetPowerInfo()
        {
            if (NPU_GetPowerInfo(out var powerInfo) == 0)
            {
                return new NPUPowerInfo(
                    powerInfo.CurrentPower,
                    powerInfo.MaxPower,
                    powerInfo.ThermalThrottling != 0,
                    powerInfo.PowerEfficiency);
            }

            return new NPUPowerInfo();
        }

        /// <summary>
        /// Executes convolution on NPU.
        /// </summary>
        internal static unsafe int ExecuteConvolution(
            nint input,
            nint kernel, 
            nint output,
            int batchSize,
            int inputChannels,
            int outputChannels,
            int inputHeight,
            int inputWidth,
            int kernelHeight,
            int kernelWidth,
            int strideHeight,
            int strideWidth,
            int paddingHeight,
            int paddingWidth)
        {
            // For now, return success - would implement actual NPU call
            return 0;
        }

        /// <summary>
        /// Executes attention mechanism on NPU.
        /// </summary>
        internal static unsafe int ExecuteAttention(
            nint query,
            nint key,
            nint value,
            nint output,
            int batchSize,
            int sequenceLength,
            int hiddenSize,
            int numHeads)
        {
            // For now, return success - would implement actual NPU call
            return 0;
        }

        #endregion
    }

    /// <summary>
    /// Native NPU capabilities structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NPUNativeCapabilities
    {
        public int Generation;
        public int ComputeUnits;
        public double MaxTOPS;
        public double MemoryBandwidth;
        public int SupportsBF16;
        public int SupportsInt8;
        public int SupportsConvolution;
        public int SupportsMatMul;
        public int SupportsAttention;
        public int SupportsSparsity;
    }

    /// <summary>
    /// Native NPU performance metrics structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NPUNativePerformanceMetrics
    {
        public double UtilizationPercent;
        public double ThroughputTOPS;
        public double PowerConsumption;
        public double TemperatureCelsius;
        public double MemoryUsage;
    }

    /// <summary>
    /// Native NPU power information structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NPUNativePowerInfo
    {
        public double CurrentPower;
        public double MaxPower;
        public int ThermalThrottling;
        public double PowerEfficiency;
    }

    /// <summary>
    /// NPU kernel operations.
    /// </summary>
    internal static class NPUKernels
    {
        /// <summary>
        /// Executes inference with float data.
        /// </summary>
        internal static unsafe void InferenceFloat(
            float* input, float* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            NPUNative.NPU_InferenceFloat32(
                (IntPtr)input, (IntPtr)output,
                (IntPtr)(&inputShape), (IntPtr)(&outputShape),
                context);
        }

        /// <summary>
        /// Executes inference with BFloat16 data.
        /// </summary>
        internal static unsafe void InferenceBF16(
            BFloat16* input, BFloat16* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            NPUNative.NPU_InferenceBF16(
                (IntPtr)input, (IntPtr)output,
                (IntPtr)(&inputShape), (IntPtr)(&outputShape),
                context);
        }

        /// <summary>
        /// Executes inference with INT8 data.
        /// </summary>
        internal static unsafe void InferenceInt8(
            byte* input, byte* output,
            TensorShape inputShape, TensorShape outputShape,
            IntPtr context)
        {
            NPUNative.NPU_InferenceInt8(
                (IntPtr)input, (IntPtr)output,
                (IntPtr)(&inputShape), (IntPtr)(&outputShape),
                context);
        }

        /// <summary>
        /// Executes convolution with float data.
        /// </summary>
        internal static unsafe void ConvolutionFloat(
            float* input, float* weights, float* output,
            TensorShape inputShape, TensorShape weightsShape, TensorShape outputShape,
            IntPtr config)
        {
            NPUNative.NPU_ConvolutionFloat32(
                (IntPtr)input, (IntPtr)weights, (IntPtr)output,
                (IntPtr)(&inputShape), (IntPtr)(&weightsShape), (IntPtr)(&outputShape),
                config);
        }

        /// <summary>
        /// Executes matrix multiplication with float data.
        /// </summary>
        internal static unsafe void MatMulFloat(
            float* a, float* b, float* c,
            int m, int k, int n,
            IntPtr config)
        {
            NPUNative.NPU_MatMulFloat32((IntPtr)a, (IntPtr)b, (IntPtr)c, m, k, n, config);
        }

        /// <summary>
        /// Executes matrix multiplication with BFloat16 data.
        /// </summary>
        internal static unsafe void MatMulBF16(
            BFloat16* a, BFloat16* b, BFloat16* c,
            int m, int k, int n,
            IntPtr config)
        {
            NPUNative.NPU_MatMulBF16((IntPtr)a, (IntPtr)b, (IntPtr)c, m, k, n, config);
        }

        /// <summary>
        /// Executes attention with float data.
        /// </summary>
        internal static unsafe void AttentionFloat(
            float* query, float* key, float* value, float* output,
            TensorShape queryShape, TensorShape keyShape, TensorShape valueShape,
            IntPtr config)
        {
            NPUNative.NPU_AttentionFloat32(
                (IntPtr)query, (IntPtr)key, (IntPtr)value, (IntPtr)output,
                (IntPtr)(&queryShape), (IntPtr)(&keyShape), (IntPtr)(&valueShape),
                config);
        }
    }
}