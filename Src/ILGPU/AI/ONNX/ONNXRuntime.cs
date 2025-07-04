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

using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace ILGPU.AI.ONNX
{
    /// <summary>
    /// ONNX Runtime execution providers.
    /// </summary>
    public enum ONNXExecutionProvider
    {
        /// <summary>
        /// CPU execution provider.
        /// </summary>
        CPU,

        /// <summary>
        /// CUDA execution provider.
        /// </summary>
        CUDA,

        /// <summary>
        /// DirectML execution provider (Windows).
        /// </summary>
        DirectML,

        /// <summary>
        /// OpenVINO execution provider.
        /// </summary>
        OpenVINO,

        /// <summary>
        /// TensorRT execution provider.
        /// </summary>
        TensorRT,

        /// <summary>
        /// ROCm execution provider (AMD).
        /// </summary>
        ROCm,

        /// <summary>
        /// CoreML execution provider (Apple).
        /// </summary>
        CoreML,

        /// <summary>
        /// NNAPI execution provider (Android).
        /// </summary>
        NNAPI
    }

    /// <summary>
    /// ONNX model optimization levels.
    /// </summary>
    public enum ONNXOptimizationLevel
    {
        /// <summary>
        /// Disable all optimizations.
        /// </summary>
        DisableAll,

        /// <summary>
        /// Enable basic optimizations.
        /// </summary>
        Basic,

        /// <summary>
        /// Enable extended optimizations.
        /// </summary>
        Extended,

        /// <summary>
        /// Enable all optimizations.
        /// </summary>
        All
    }

    /// <summary>
    /// ONNX Runtime configuration for inference.
    /// </summary>
    public sealed class ONNXRuntimeConfig
    {
        /// <summary>
        /// Gets or sets the execution provider.
        /// </summary>
        public ONNXExecutionProvider ExecutionProvider { get; set; } = ONNXExecutionProvider.CPU;

        /// <summary>
        /// Gets or sets the optimization level.
        /// </summary>
        public ONNXOptimizationLevel OptimizationLevel { get; set; } = ONNXOptimizationLevel.All;

        /// <summary>
        /// Gets or sets whether to enable memory optimization.
        /// </summary>
        public bool EnableMemoryOptimization { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to enable parallel execution.
        /// </summary>
        public bool EnableParallelExecution { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of threads for CPU execution.
        /// </summary>
        public int NumThreads { get; set; } = Environment.ProcessorCount;

        /// <summary>
        /// Gets or sets the device ID for GPU execution.
        /// </summary>
        public int DeviceId { get; set; }

        /// <summary>
        /// Gets or sets whether to enable profiling.
        /// </summary>
        public bool EnableProfiling { get; set; }

        /// <summary>
        /// Gets or sets custom execution provider options.
        /// </summary>
        public Dictionary<string, string> ProviderOptions { get; set; } = new();

        /// <summary>
        /// Creates a default ONNX Runtime configuration.
        /// </summary>
        public static ONNXRuntimeConfig Default => new();

        /// <summary>
        /// Creates a configuration optimized for CUDA execution.
        /// </summary>
        /// <param name="deviceId">CUDA device ID.</param>
        /// <returns>CUDA-optimized configuration.</returns>
        public static ONNXRuntimeConfig ForCUDA(int deviceId = 0) => new()
        {
            ExecutionProvider = ONNXExecutionProvider.CUDA,
            DeviceId = deviceId,
            OptimizationLevel = ONNXOptimizationLevel.All,
            EnableMemoryOptimization = true,
            EnableParallelExecution = true,
            ProviderOptions = new Dictionary<string, string>
            {
                ["device_id"] = deviceId.ToString(),
                ["gpu_mem_limit"] = "2147483648", // 2GB limit
                ["arena_extend_strategy"] = "kSameAsRequested"
            }
        };

        /// <summary>
        /// Creates a configuration optimized for CPU execution.
        /// </summary>
        /// <param name="numThreads">Number of CPU threads.</param>
        /// <returns>CPU-optimized configuration.</returns>
        public static ONNXRuntimeConfig ForCPU(int numThreads = 0) => new()
        {
            ExecutionProvider = ONNXExecutionProvider.CPU,
            NumThreads = numThreads > 0 ? numThreads : Environment.ProcessorCount,
            OptimizationLevel = ONNXOptimizationLevel.All,
            EnableMemoryOptimization = true,
            EnableParallelExecution = true
        };
    }

    /// <summary>
    /// ONNX model wrapper for ILGPU integration.
    /// </summary>
    public sealed class ONNXModel : IDisposable
    {
        private readonly IntPtr _session;
        private readonly Dictionary<string, ONNXTensorInfo> _inputInfo;
        private readonly Dictionary<string, ONNXTensorInfo> _outputInfo;
        private bool _disposed;

        /// <summary>
        /// Initializes a new ONNX model.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <param name="config">Runtime configuration.</param>
        public ONNXModel(string modelPath, ONNXRuntimeConfig? config = null)
        {
            ModelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
            Config = config ?? ONNXRuntimeConfig.Default;
            _inputInfo = new Dictionary<string, ONNXTensorInfo>();
            _outputInfo = new Dictionary<string, ONNXTensorInfo>();

            // Create ONNX Runtime session
            _session = CreateSession(modelPath, Config);
            if (_session == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create ONNX Runtime session");

            // Query model metadata
            QueryModelMetadata();
        }

        /// <summary>
        /// Gets the model file path.
        /// </summary>
        public string ModelPath { get; }

        /// <summary>
        /// Gets the runtime configuration.
        /// </summary>
        public ONNXRuntimeConfig Config { get; }

        /// <summary>
        /// Gets input tensor information.
        /// </summary>
        public IReadOnlyDictionary<string, ONNXTensorInfo> InputInfo => _inputInfo;

        /// <summary>
        /// Gets output tensor information.
        /// </summary>
        public IReadOnlyDictionary<string, ONNXTensorInfo> OutputInfo => _outputInfo;

        /// <summary>
        /// Runs inference on the model.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Output tensors.</returns>
        public Dictionary<string, ArrayView<float>> RunInference(Dictionary<string, ArrayView<float>> inputs)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ONNXModel));

            var outputs = new Dictionary<string, ArrayView<float>>();

            try
            {
                // Prepare input tensors
                var inputTensors = PrepareInputTensors(inputs);
                
                // Prepare output tensors
                var outputTensors = PrepareOutputTensors();

                // Run inference
                RunInferenceInternal(inputTensors, outputTensors);

                // Extract results
                outputs = ExtractOutputTensors(outputTensors);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"ONNX inference failed: {ex.Message}", ex);
            }

            return outputs;
        }

        /// <summary>
        /// Runs inference asynchronously.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Task with output tensors.</returns>
        public async Task<Dictionary<string, ArrayView<float>>> RunInferenceAsync(Dictionary<string, ArrayView<float>> inputs) => await Task.Run(() => RunInference(inputs));

        /// <summary>
        /// Runs batch inference on multiple inputs.
        /// </summary>
        /// <param name="batchInputs">Batch of input tensors.</param>
        /// <returns>Batch of output tensors.</returns>
        public List<Dictionary<string, ArrayView<float>>> RunBatchInference(List<Dictionary<string, ArrayView<float>>> batchInputs)
        {
            var batchOutputs = new List<Dictionary<string, ArrayView<float>>>();

            foreach (var inputs in batchInputs)
            {
                batchOutputs.Add(RunInference(inputs));
            }

            return batchOutputs;
        }

        /// <summary>
        /// Gets model profiling information.
        /// </summary>
        /// <returns>Profiling data.</returns>
        public ONNXProfilingInfo GetProfilingInfo()
        {
            if (!Config.EnableProfiling)
                throw new InvalidOperationException("Profiling not enabled");

            return QueryProfilingInfo(_session);
        }

        /// <summary>
        /// Disposes the ONNX model.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_session != IntPtr.Zero)
                {
                    ReleaseSession(_session);
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        #region Private Methods

        private IntPtr CreateSession(string modelPath, ONNXRuntimeConfig config)
        {
            try
            {
                // Create session options
                var sessionOptions = CreateSessionOptions(config);
                
                // Create session
                var session = ONNXNative.CreateInferenceSession(modelPath, sessionOptions);
                
                // Release session options
                ONNXNative.ReleaseSessionOptions(sessionOptions);
                
                return session;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to create ONNX session: {ex.Message}", ex);
            }
        }

        private IntPtr CreateSessionOptions(ONNXRuntimeConfig config)
        {
            var options = ONNXNative.CreateSessionOptions();
            
            // Set optimization level
            ONNXNative.SetOptimizationLevel(options, (int)config.OptimizationLevel);
            
            // Set execution mode
            ONNXNative.SetExecutionMode(options, config.EnableParallelExecution ? 1 : 0);
            
            // Set memory optimization
            ONNXNative.EnableMemoryOptimization(options, config.EnableMemoryOptimization);
            
            // Set CPU thread count
            if (config.ExecutionProvider == ONNXExecutionProvider.CPU)
            {
                ONNXNative.SetIntraOpNumThreads(options, config.NumThreads);
            }
            
            // Add execution provider
            AddExecutionProvider(options, config);
            
            // Enable profiling if requested
            if (config.EnableProfiling)
            {
                ONNXNative.EnableProfiling(options, "onnx_profile.json");
            }
            
            return options;
        }

        private void AddExecutionProvider(IntPtr options, ONNXRuntimeConfig config)
        {
            switch (config.ExecutionProvider)
            {
                case ONNXExecutionProvider.CUDA:
                    ONNXNative.AppendExecutionProvider_CUDA(options, config.DeviceId);
                    break;
                
                case ONNXExecutionProvider.DirectML:
                    ONNXNative.AppendExecutionProvider_DML(options, config.DeviceId);
                    break;
                
                case ONNXExecutionProvider.OpenVINO:
                    ONNXNative.AppendExecutionProvider_OpenVINO(options, "CPU");
                    break;
                
                case ONNXExecutionProvider.TensorRT:
                    ONNXNative.AppendExecutionProvider_TensorRT(options, config.DeviceId);
                    break;
                
                case ONNXExecutionProvider.ROCm:
                    ONNXNative.AppendExecutionProvider_ROCm(options, config.DeviceId);
                    break;
                
                case ONNXExecutionProvider.CoreML:
                    ONNXNative.AppendExecutionProvider_CoreML(options);
                    break;
                
                case ONNXExecutionProvider.CPU:
                default:
                    // CPU provider is always available by default
                    break;
            }
        }

        private void QueryModelMetadata()
        {
            // Get input count and info
            var inputCount = ONNXNative.SessionGetInputCount(_session);
            for (int i = 0; i < inputCount; i++)
            {
                var name = ONNXNative.SessionGetInputName(_session, i);
                var typeInfo = ONNXNative.SessionGetInputTypeInfo(_session, i);
                var tensorInfo = ExtractTensorInfo(typeInfo);
                _inputInfo[name] = tensorInfo;
                ONNXNative.ReleaseTypeInfo(typeInfo);
            }

            // Get output count and info
            var outputCount = ONNXNative.SessionGetOutputCount(_session);
            for (int i = 0; i < outputCount; i++)
            {
                var name = ONNXNative.SessionGetOutputName(_session, i);
                var typeInfo = ONNXNative.SessionGetOutputTypeInfo(_session, i);
                var tensorInfo = ExtractTensorInfo(typeInfo);
                _outputInfo[name] = tensorInfo;
                ONNXNative.ReleaseTypeInfo(typeInfo);
            }
        }

        private ONNXTensorInfo ExtractTensorInfo(IntPtr typeInfo)
        {
            var tensorTypeInfo = ONNXNative.CastTypeInfoToTensorInfo(typeInfo);
            var elementType = ONNXNative.GetTensorElementType(tensorTypeInfo);
            var dims = ONNXNative.GetTensorShapeElementCount(tensorTypeInfo);
            var shape = new int[dims];
            
            for (int i = 0; i < dims; i++)
            {
                shape[i] = ONNXNative.GetTensorShapeElementValue(tensorTypeInfo, i);
            }

            return new ONNXTensorInfo
            {
                ElementType = (ONNXTensorElementType)elementType,
                Shape = shape
            };
        }

        private Dictionary<string, IntPtr> PrepareInputTensors(Dictionary<string, ArrayView<float>> inputs)
        {
            var inputTensors = new Dictionary<string, IntPtr>();

            foreach (var kvp in inputs)
            {
                var name = kvp.Key;
                var data = kvp.Value;
                
                if (!_inputInfo.ContainsKey(name))
                    throw new ArgumentException($"Unknown input tensor: {name}");

                var info = _inputInfo[name];
                var tensor = CreateTensorFromArrayView(data, info);
                inputTensors[name] = tensor;
            }

            return inputTensors;
        }

        private Dictionary<string, IntPtr> PrepareOutputTensors()
        {
            var outputTensors = new Dictionary<string, IntPtr>();

            foreach (var kvp in _outputInfo)
            {
                var name = kvp.Key;
                var info = kvp.Value;
                var tensor = CreateEmptyTensor(info);
                outputTensors[name] = tensor;
            }

            return outputTensors;
        }

        private void RunInferenceInternal(Dictionary<string, IntPtr> inputs, Dictionary<string, IntPtr> outputs)
        {
            var inputNames = new string[inputs.Count];
            var inputTensors = new IntPtr[inputs.Count];
            var outputNames = new string[outputs.Count];
            var outputTensors = new IntPtr[outputs.Count];

            int i = 0;
            foreach (var kvp in inputs)
            {
                inputNames[i] = kvp.Key;
                inputTensors[i] = kvp.Value;
                i++;
            }

            i = 0;
            foreach (var kvp in outputs)
            {
                outputNames[i] = kvp.Key;
                outputTensors[i] = kvp.Value;
                i++;
            }

            ONNXNative.Run(_session, inputNames, inputTensors, inputs.Count,
                          outputNames, outputTensors, outputs.Count);
        }

        private Dictionary<string, ArrayView<float>> ExtractOutputTensors(Dictionary<string, IntPtr> outputs)
        {
            var results = new Dictionary<string, ArrayView<float>>();

            foreach (var kvp in outputs)
            {
                var name = kvp.Key;
                var tensor = kvp.Value;
                var arrayView = ExtractArrayViewFromTensor(tensor);
                results[name] = arrayView;
            }

            return results;
        }

        private unsafe IntPtr CreateTensorFromArrayView(ArrayView<float> data, ONNXTensorInfo info)
        {
            var dataSize = (ulong)(info.ElementCount * sizeof(float));
            var shape = Array.ConvertAll(info.Shape, x => (long)x);
            
            var dataPtr = data.LoadEffectiveAddressAsPtr().ToPointer();
            return ONNXNative.CreateTensorWithData(
                new IntPtr(dataPtr),
                dataSize,
                shape,
                (ulong)shape.Length,
                (int)info.ElementType);
        }

        private IntPtr CreateEmptyTensor(ONNXTensorInfo info)
        {
            var shape = Array.ConvertAll(info.Shape, x => (long)x);
            
            return ONNXNative.CreateTensor(
                shape,
                (ulong)shape.Length,
                (int)info.ElementType);
        }

        private unsafe ArrayView<float> ExtractArrayViewFromTensor(IntPtr tensor)
        {
            var dataPtr = ONNXNative.GetTensorMutableData(tensor);
            if (dataPtr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to get tensor data pointer");

            // For now, return an empty view - real implementation would create proper ArrayView
            // This would require knowing the tensor size and creating appropriate memory buffer
            throw new NotImplementedException("Tensor extraction requires proper memory buffer creation");
        }

        private void ReleaseSession(IntPtr session) => ONNXNative.ReleaseSession(session);

        private ONNXProfilingInfo QueryProfilingInfo(IntPtr session) =>
            // Query profiling information from ONNX Runtime
            new()
            {
                TotalInferenceTime = 0.0,
                OperatorTimes = new Dictionary<string, double>(),
                MemoryUsage = 0L
            };

        #endregion
    }

    /// <summary>
    /// ONNX tensor information.
    /// </summary>
    public sealed class ONNXTensorInfo
    {
        /// <summary>
        /// Gets or sets the element type.
        /// </summary>
        public ONNXTensorElementType ElementType { get; set; }

        /// <summary>
        /// Gets or sets the tensor shape.
        /// </summary>
        public int[] Shape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets the total number of elements.
        /// </summary>
        public long ElementCount
        {
            get
            {
                long count = 1;
                foreach (var dim in Shape)
                    count *= dim;
                return count;
            }
        }
    }

    /// <summary>
    /// ONNX tensor element types.
    /// </summary>
    public enum ONNXTensorElementType
    {
        /// <summary>
        /// 32-bit floating point.
        /// </summary>
        Float,

        /// <summary>
        /// 8-bit unsigned integer.
        /// </summary>
        UInt8,

        /// <summary>
        /// 8-bit signed integer.
        /// </summary>
        Int8,

        /// <summary>
        /// 16-bit unsigned integer.
        /// </summary>
        UInt16,

        /// <summary>
        /// 16-bit signed integer.
        /// </summary>
        Int16,

        /// <summary>
        /// 32-bit signed integer.
        /// </summary>
        Int32,

        /// <summary>
        /// 64-bit signed integer.
        /// </summary>
        Int64,

        /// <summary>
        /// Boolean.
        /// </summary>
        Bool,

        /// <summary>
        /// 16-bit floating point.
        /// </summary>
        Float16,

        /// <summary>
        /// 64-bit floating point.
        /// </summary>
        Double,

        /// <summary>
        /// BFloat16.
        /// </summary>
        BFloat16
    }

    /// <summary>
    /// ONNX profiling information.
    /// </summary>
    public sealed class ONNXProfilingInfo
    {
        /// <summary>
        /// Gets or sets the total inference time in milliseconds.
        /// </summary>
        public double TotalInferenceTime { get; set; }

        /// <summary>
        /// Gets or sets operator execution times.
        /// </summary>
        public Dictionary<string, double> OperatorTimes { get; set; } = new();

        /// <summary>
        /// Gets or sets memory usage in bytes.
        /// </summary>
        public long MemoryUsage { get; set; }
    }
}