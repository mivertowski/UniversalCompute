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

using System;
using System.Runtime.InteropServices;
using ILGPU.Numerics.AI;

namespace ILGPU.Apple.NeuralEngine.Native
{
    /// <summary>
    /// Native Apple Neural Engine API bindings.
    /// </summary>
    internal static partial class ANENative
    {
        #region Constants

        private const string CoreMLFramework = "CoreML.framework/CoreML";
        private const string AppleNeuralEngineFramework = "AppleNeuralEngine.framework/AppleNeuralEngine";
        private const string CoreFoundationLibrary = "CoreFoundation.framework/CoreFoundation";

        #endregion

        #region Neural Engine Detection and Context

        /// <summary>
        /// Checks if the Apple Neural Engine is available on this system.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool IsNeuralEngineAvailable();

        /// <summary>
        /// Creates a Neural Engine context.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial IntPtr CreateContext();

        /// <summary>
        /// Releases a Neural Engine context.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial void ReleaseContext(IntPtr context);

        /// <summary>
        /// Queries Neural Engine capabilities.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial ANENativeCapabilities QueryCapabilities();

        #endregion

        #region Performance and Power Monitoring

        /// <summary>
        /// Gets Neural Engine performance metrics.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial ANEPerformanceMetrics GetPerformanceMetrics(IntPtr context);

        /// <summary>
        /// Gets Neural Engine power information.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial ANEPowerInfo GetPowerInfo(IntPtr context);

        /// <summary>
        /// Gets Neural Engine thermal state.
        /// </summary>
        [LibraryImport(AppleNeuralEngineFramework)]
        internal static partial ANEThermalState GetThermalState(IntPtr context);

        #endregion

        #region Core ML Integration

        /// <summary>
        /// Creates a Core ML model optimized for Neural Engine.
        /// </summary>
        [LibraryImport(CoreMLFramework)]
        internal static partial IntPtr MLModelCreateFromURL(IntPtr url, out IntPtr error);

        /// <summary>
        /// Configures a Core ML model for Neural Engine execution.
        /// </summary>
        [LibraryImport(CoreMLFramework)]
        internal static partial IntPtr MLModelConfigurationCreate();

        /// <summary>
        /// Sets Neural Engine compute units for Core ML.
        /// </summary>
        [LibraryImport(CoreMLFramework)]
        internal static partial void MLModelConfigurationSetComputeUnits(
            IntPtr configuration, int computeUnits);

        /// <summary>
        /// Creates a prediction from Core ML model.
        /// </summary>
        [LibraryImport(CoreMLFramework)]
        internal static partial IntPtr MLModelPredictionFromFeatures(
            IntPtr model, IntPtr features, IntPtr options, out IntPtr error);

        #endregion

        #region Memory Management

        /// <summary>
        /// Releases Core Foundation objects.
        /// </summary>
        [LibraryImport(CoreFoundationLibrary)]
        internal static partial void CFRelease(IntPtr obj);

        /// <summary>
        /// Creates a string from C string.
        /// </summary>
        [LibraryImport(CoreFoundationLibrary)]
        internal static partial IntPtr CFStringCreateWithCString(
            IntPtr allocator, IntPtr cStr, uint encoding);

        /// <summary>
        /// Creates a URL from file system path.
        /// </summary>
        [LibraryImport(CoreFoundationLibrary)]
        internal static partial IntPtr CFURLCreateFromFileSystemRepresentation(
            IntPtr allocator, IntPtr buffer, long bufferLength, [MarshalAs(UnmanagedType.Bool)] bool isDirectory);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets Neural Engine generation from system information.
        /// </summary>
        internal static ANEGeneration GetNeuralEngineGeneration()
        {
            try
            {
                var capabilities = QueryCapabilities();
                return (ANEGeneration)capabilities.Generation;
            }
            catch
            {
                return ANEGeneration.None;
            }
        }

        /// <summary>
        /// Creates Core ML configuration for Neural Engine.
        /// </summary>
        internal static IntPtr CreateNeuralEngineConfiguration()
        {
            var config = MLModelConfigurationCreate();
            if (config != IntPtr.Zero)
            {
                // Set compute units to Neural Engine (value 2)
                MLModelConfigurationSetComputeUnits(config, 2);
            }
            return config;
        }

        /// <summary>
        /// Creates a URL from file path.
        /// </summary>
        internal static IntPtr CreateURLFromPath(string path)
        {
            var pathPtr = Marshal.StringToHGlobalAnsi(path);
            try
            {
                return CFURLCreateFromFileSystemRepresentation(
                    IntPtr.Zero, pathPtr, path.Length, false);
            }
            finally
            {
                Marshal.FreeHGlobal(pathPtr);
            }
        }

        /// <summary>
        /// Loads a Core ML model from file path.
        /// </summary>
        internal static IntPtr LoadCoreMLModel(string modelPath)
        {
            var url = CreateURLFromPath(modelPath);
            if (url == IntPtr.Zero)
                return IntPtr.Zero;

            try
            {
                var model = MLModelCreateFromURL(url, out var error);
                if (error != IntPtr.Zero)
                {
                    CFRelease(error);
                    return IntPtr.Zero;
                }
                return model;
            }
            finally
            {
                CFRelease(url);
            }
        }

        #endregion
    }

    /// <summary>
    /// Native ANE capabilities structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct ANENativeCapabilities
    {
        public int IsAvailable;
        public int Generation;
        public double MaxTOPS;
        public int SupportsFloat16;
        public int SupportsInt8;
        public int SupportsConvolution;
        public int SupportsAttention;
        public int SupportsTransformer;
        public int SupportsCoreML;
        public int MaxBatchSize;
    }

    /// <summary>
    /// ANE kernel operations for direct Neural Engine execution.
    /// </summary>
    internal static class ANEKernels
    {
        private const string AppleNeuralEngineFramework = "AppleNeuralEngine.framework/AppleNeuralEngine";
        
        #region Convolution Operations

        /// <summary>
        /// Executes convolution operation on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteConvolution(
            float* input, float* output,
            int inputRank, int* inputDims, int outputRank, int* outputDims,
            ConvolutionParameters parameters, IntPtr context);

        /// <summary>
        /// Executes convolution with bias on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteConvolutionWithBias(
            float* input, float* weights, float* bias, float* output,
            int inputRank, int* inputDims, int weightsRank, int* weightsDims, 
            int outputRank, int* outputDims, ANEConvolutionParameters parameters, IntPtr context);

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Executes matrix multiplication on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteMatMul(
            float* input, float* output,
            int inputRank, int* inputDims, int outputRank, int* outputDims,
            IntPtr context);

        #endregion

        #region Attention Operations

        /// <summary>
        /// Executes attention mechanism on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteAttention(
            float* input, float* output,
            int inputRank, int* inputDims, int outputRank, int* outputDims,
            AttentionParameters parameters, IntPtr context);

        /// <summary>
        /// Executes multi-head attention on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteMultiHeadAttention(
            float* query, float* key, float* value, float* output,
            int queryRank, int* queryDims, int keyRank, int* keyDims, int valueRank, int* valueDims,
            ANEAttentionParameters parameters, IntPtr context);

        #endregion

        #region Core ML Execution

        /// <summary>
        /// Executes Core ML inference on Neural Engine.
        /// </summary>
        [DllImport(AppleNeuralEngineFramework)]
        internal static unsafe extern void ExecuteCoreMLInference(
            float* input, float* output,
            int inputRank, int* inputDims, int outputRank, int* outputDims,
            IntPtr modelHandle, IntPtr context);

        #endregion
    }

    /// <summary>
    /// ANE convolution parameters for native operations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ANEConvolutionParameters
    {
        public ANESize2D KernelSize;
        public ANESize2D Stride;
        public ANESize2D Padding;
        public ANESize2D Dilation;
        public int Groups;
        public ANEActivationType Activation;
    }

    /// <summary>
    /// ANE attention parameters for native operations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ANEAttentionParameters
    {
        public int NumHeads;
        public int HeadDimension;
        public float ScaleFactor;
        public int UseDropout;
        public float DropoutRate;
        public int UseCausalMask;
    }

    /// <summary>
    /// ANE 2D size structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ANESize2D(int width, int height)
    {
        public int Width = width;
        public int Height = height;
    }

    /// <summary>
    /// ANE activation function types.
    /// </summary>
    public enum ANEActivationType
    {
        None = 0,
        ReLU = 1,
        ReLU6 = 2,
        Sigmoid = 3,
        Tanh = 4,
        Swish = 5,
        GELU = 6
    }
}