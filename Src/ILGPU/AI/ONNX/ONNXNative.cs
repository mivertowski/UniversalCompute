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

using System;
using System.Runtime.InteropServices;

namespace ILGPU.AI.ONNX
{
    /// <summary>
    /// Native ONNX Runtime API bindings.
    /// </summary>
    /// <remarks>
    /// These bindings interface with the ONNX Runtime C API for cross-platform
    /// AI model inference support.
    /// 
    /// Requirements:
    /// - ONNX Runtime 1.16.0+ 
    /// - Platform-specific ONNX Runtime libraries
    /// - Optional: Execution providers (CUDA, DirectML, TensorRT, etc.)
    /// </remarks>
    internal static partial class ONNXNative
    {
        #region Constants

#if WINDOWS
        private const string ONNXRuntimeLibrary = "onnxruntime";
#elif MACOS
        private const string ONNXRuntimeLibrary = "libonnxruntime.dylib";
#else
        private const string ONNXRuntimeLibrary = "libonnxruntime.so";
#endif

        #endregion

        #region Environment and Session Management

        /// <summary>
        /// Creates an ONNX Runtime environment.
        /// </summary>
        /// <param name="logLevel">Logging level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal).</param>
        /// <param name="logId">Identifier for logging.</param>
        /// <returns>Environment handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr OrtCreateEnv(int logLevel, string logId);

        /// <summary>
        /// Releases an ONNX Runtime environment.
        /// </summary>
        /// <param name="env">Environment handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void OrtReleaseEnv(IntPtr env);

        /// <summary>
        /// Creates session options.
        /// </summary>
        /// <returns>Session options handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CreateSessionOptions();

        /// <summary>
        /// Releases session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReleaseSessionOptions(IntPtr options);

        /// <summary>
        /// Creates an inference session from a model file.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <param name="options">Session options.</param>
        /// <returns>Session handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern IntPtr CreateInferenceSession(string modelPath, IntPtr options);

        /// <summary>
        /// Releases an inference session.
        /// </summary>
        /// <param name="session">Session handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReleaseSession(IntPtr session);

        #endregion

        #region Session Configuration

        /// <summary>
        /// Sets the optimization level for the session.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="level">Optimization level (0=DisableAll, 1=Basic, 2=Extended, 3=All).</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SetOptimizationLevel(IntPtr options, int level);

        /// <summary>
        /// Sets the execution mode for the session.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="mode">Execution mode (0=Sequential, 1=Parallel).</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SetExecutionMode(IntPtr options, int mode);

        /// <summary>
        /// Enables memory optimization for the session.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="enable">Whether to enable memory optimization.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void EnableMemoryOptimization(IntPtr options, bool enable);

        /// <summary>
        /// Sets the number of intra-operation threads.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="numThreads">Number of threads.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void SetIntraOpNumThreads(IntPtr options, int numThreads);

        /// <summary>
        /// Enables profiling for the session.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="profileFile">Path to the profile output file.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern void EnableProfiling(IntPtr options, string profileFile);

        #endregion

        #region Execution Providers

        /// <summary>
        /// Appends CUDA execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="deviceId">CUDA device ID.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AppendExecutionProvider_CUDA(IntPtr options, int deviceId);

        /// <summary>
        /// Appends DirectML execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="deviceId">DirectML device ID.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AppendExecutionProvider_DML(IntPtr options, int deviceId);

        /// <summary>
        /// Appends OpenVINO execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="device">OpenVINO device (e.g., "CPU", "GPU", "NPU").</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern void AppendExecutionProvider_OpenVINO(IntPtr options, string device);

        /// <summary>
        /// Appends TensorRT execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="deviceId">TensorRT device ID.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AppendExecutionProvider_TensorRT(IntPtr options, int deviceId);

        /// <summary>
        /// Appends ROCm execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        /// <param name="deviceId">ROCm device ID.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AppendExecutionProvider_ROCm(IntPtr options, int deviceId);

        /// <summary>
        /// Appends CoreML execution provider to session options.
        /// </summary>
        /// <param name="options">Session options handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void AppendExecutionProvider_CoreML(IntPtr options);

        #endregion

        #region Model Metadata

        /// <summary>
        /// Gets the number of model inputs.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <returns>Number of inputs.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SessionGetInputCount(IntPtr session);

        /// <summary>
        /// Gets the number of model outputs.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <returns>Number of outputs.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SessionGetOutputCount(IntPtr session);

        /// <summary>
        /// Gets the name of an input tensor.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Input index.</param>
        /// <returns>Input name.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SessionGetInputName_Internal(IntPtr session, int index);

        /// <summary>
        /// Gets the name of an output tensor.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Output index.</param>
        /// <returns>Output name.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SessionGetOutputName_Internal(IntPtr session, int index);

        /// <summary>
        /// Gets type information for an input tensor.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Input index.</param>
        /// <returns>Type info handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SessionGetInputTypeInfo(IntPtr session, int index);

        /// <summary>
        /// Gets type information for an output tensor.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Output index.</param>
        /// <returns>Type info handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr SessionGetOutputTypeInfo(IntPtr session, int index);

        /// <summary>
        /// Releases type information.
        /// </summary>
        /// <param name="typeInfo">Type info handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReleaseTypeInfo(IntPtr typeInfo);

        #endregion

        #region Tensor Information

        /// <summary>
        /// Casts type info to tensor type info.
        /// </summary>
        /// <param name="typeInfo">Type info handle.</param>
        /// <returns>Tensor type info handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CastTypeInfoToTensorInfo(IntPtr typeInfo);

        /// <summary>
        /// Gets the element type of a tensor.
        /// </summary>
        /// <param name="tensorInfo">Tensor type info handle.</param>
        /// <returns>Element type.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GetTensorElementType(IntPtr tensorInfo);

        /// <summary>
        /// Gets the number of dimensions in tensor shape.
        /// </summary>
        /// <param name="tensorInfo">Tensor type info handle.</param>
        /// <returns>Number of dimensions.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GetTensorShapeElementCount(IntPtr tensorInfo);

        /// <summary>
        /// Gets a specific dimension value from tensor shape.
        /// </summary>
        /// <param name="tensorInfo">Tensor type info handle.</param>
        /// <param name="index">Dimension index.</param>
        /// <returns>Dimension value.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GetTensorShapeElementValue(IntPtr tensorInfo, int index);

        #endregion

        #region Tensor Operations

        /// <summary>
        /// Creates a tensor with CPU memory.
        /// </summary>
        /// <param name="data">Pointer to tensor data.</param>
        /// <param name="dataSize">Size of data in bytes.</param>
        /// <param name="shape">Tensor shape.</param>
        /// <param name="shapeLength">Number of dimensions.</param>
        /// <param name="elementType">Element type.</param>
        /// <returns>Tensor handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CreateTensorWithData(
            IntPtr data,
            ulong dataSize,
            long[] shape,
            ulong shapeLength,
            int elementType);

        /// <summary>
        /// Creates an empty tensor.
        /// </summary>
        /// <param name="shape">Tensor shape.</param>
        /// <param name="shapeLength">Number of dimensions.</param>
        /// <param name="elementType">Element type.</param>
        /// <returns>Tensor handle.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr CreateTensor(
            long[] shape,
            ulong shapeLength,
            int elementType);

        /// <summary>
        /// Gets tensor data pointer.
        /// </summary>
        /// <param name="tensor">Tensor handle.</param>
        /// <returns>Data pointer.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr GetTensorMutableData(IntPtr tensor);

        /// <summary>
        /// Releases a tensor.
        /// </summary>
        /// <param name="tensor">Tensor handle.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void ReleaseTensor(IntPtr tensor);

        #endregion

        #region Inference Execution

        /// <summary>
        /// Runs inference on the model.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="inputNames">Array of input names.</param>
        /// <param name="inputs">Array of input tensors.</param>
        /// <param name="inputCount">Number of inputs.</param>
        /// <param name="outputNames">Array of output names.</param>
        /// <param name="outputs">Array of output tensors.</param>
        /// <param name="outputCount">Number of outputs.</param>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void Run(
            IntPtr session,
            string[] inputNames,
            IntPtr[] inputs,
            int inputCount,
            string[] outputNames,
            IntPtr[] outputs,
            int outputCount);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Gets the name of an input tensor as a managed string.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Input index.</param>
        /// <returns>Input name.</returns>
        internal static string SessionGetInputName(IntPtr session, int index)
        {
            var ptr = SessionGetInputName_Internal(session, index);
            return Marshal.PtrToStringAnsi(ptr) ?? string.Empty;
        }

        /// <summary>
        /// Gets the name of an output tensor as a managed string.
        /// </summary>
        /// <param name="session">Session handle.</param>
        /// <param name="index">Output index.</param>
        /// <returns>Output name.</returns>
        internal static string SessionGetOutputName(IntPtr session, int index)
        {
            var ptr = SessionGetOutputName_Internal(session, index);
            return Marshal.PtrToStringAnsi(ptr) ?? string.Empty;
        }

        /// <summary>
        /// Checks if ONNX Runtime is available.
        /// </summary>
        /// <returns>True if ONNX Runtime is available; otherwise, false.</returns>
        internal static bool IsAvailable()
        {
            try
            {
                var env = OrtCreateEnv(2, "ILGPU-ONNX");
                if (env != IntPtr.Zero)
                {
                    OrtReleaseEnv(env);
                    return true;
                }
                return false;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the ONNX Runtime version.
        /// </summary>
        /// <returns>Version string.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr OrtGetVersionString();

        /// <summary>
        /// Gets the ONNX Runtime version as a managed string.
        /// </summary>
        /// <returns>Version string.</returns>
        internal static string GetVersionString()
        {
            try
            {
                var ptr = OrtGetVersionString();
                return Marshal.PtrToStringAnsi(ptr) ?? "Unknown";
            }
            catch
            {
                return "Unknown";
            }
        }

        #endregion

        #region Error Handling

        /// <summary>
        /// Gets the last error message.
        /// </summary>
        /// <returns>Error message.</returns>
        [DllImport(ONNXRuntimeLibrary, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr OrtGetLastErrorMessage();

        /// <summary>
        /// Gets the last error message as a managed string.
        /// </summary>
        /// <returns>Error message.</returns>
        internal static string GetLastErrorMessage()
        {
            try
            {
                var ptr = OrtGetLastErrorMessage();
                return Marshal.PtrToStringAnsi(ptr) ?? "Unknown error";
            }
            catch
            {
                return "Unknown error";
            }
        }

        #endregion
    }
}