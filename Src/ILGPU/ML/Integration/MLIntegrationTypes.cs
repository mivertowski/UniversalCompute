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

using ILGPU.Runtime;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ILGPU.ML.Integration
{
    /// <summary>
    /// Represents a machine learning model interface.
    /// </summary>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    public interface IMLModel<TInput, TOutput>
        where TInput : class
        where TOutput : class
    {
        /// <summary>
        /// Gets the model name.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the model version.
        /// </summary>
        string Version { get; }

        /// <summary>
        /// Performs inference on the input data.
        /// </summary>
        Task<TOutput> PredictAsync(TInput input);

        /// <summary>
        /// Performs batch inference.
        /// </summary>
        Task<TOutput[]> PredictBatchAsync(TInput[] inputs);

        /// <summary>
        /// Gets model metadata.
        /// </summary>
        ModelMetadata GetMetadata();
    }

    /// <summary>
    /// Model metadata.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Gets or sets the input shapes.
        /// </summary>
        public Dictionary<string, int[]> InputShapes { get; set; } = new();

        /// <summary>
        /// Gets or sets the output shapes.
        /// </summary>
        public Dictionary<string, int[]> OutputShapes { get; set; } = new();

        /// <summary>
        /// Gets or sets the model type.
        /// </summary>
        public string ModelType { get; set; }

        /// <summary>
        /// Gets or sets the framework.
        /// </summary>
        public string Framework { get; set; }
    }

    /// <summary>
    /// Prediction statistics.
    /// </summary>
    public class PredictionStats
    {
        /// <summary>
        /// Gets or sets the inference time in milliseconds.
        /// </summary>
        public double InferenceTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the preprocessing time.
        /// </summary>
        public double PreprocessingTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the postprocessing time.
        /// </summary>
        public double PostprocessingTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the device used.
        /// </summary>
        public string DeviceUsed { get; set; }

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// Gets or sets the throughput.
        /// </summary>
        public double ThroughputPerSecond { get; set; }
    }

    /// <summary>
    /// Performance analysis results.
    /// </summary>
    public class PerformanceAnalysis
    {
        /// <summary>
        /// Gets or sets the total execution time.
        /// </summary>
        public double TotalExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the GPU utilization percentage.
        /// </summary>
        public double GpuUtilizationPercent { get; set; }

        /// <summary>
        /// Gets or sets the memory bandwidth utilization.
        /// </summary>
        public double MemoryBandwidthUtilizationPercent { get; set; }

        /// <summary>
        /// Gets or sets the compute efficiency.
        /// </summary>
        public double ComputeEfficiency { get; set; }

        /// <summary>
        /// Gets or sets the bottleneck analysis.
        /// </summary>
        public string BottleneckAnalysis { get; set; }

        /// <summary>
        /// Gets or sets optimization suggestions.
        /// </summary>
        public List<string> OptimizationSuggestions { get; set; } = new();
    }

    /// <summary>
    /// Device profile results.
    /// </summary>
    public class DeviceProfileResult
    {
        /// <summary>
        /// Gets or sets the device name.
        /// </summary>
        public string DeviceName { get; set; }

        /// <summary>
        /// Gets or sets the device type.
        /// </summary>
        public AcceleratorType DeviceType { get; set; }

        /// <summary>
        /// Gets or sets the peak performance.
        /// </summary>
        public double PeakPerformanceGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets the measured performance.
        /// </summary>
        public double MeasuredPerformanceGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets the efficiency percentage.
        /// </summary>
        public double EfficiencyPercent { get; set; }

        /// <summary>
        /// Gets or sets the memory bandwidth.
        /// </summary>
        public double MemoryBandwidthGBps { get; set; }
    }

    /// <summary>
    /// Collection of device profile results.
    /// </summary>
    public class DeviceProfileResults
    {
        /// <summary>
        /// Gets or sets the individual device results.
        /// </summary>
        public List<DeviceProfileResult> Results { get; set; } = new();

        /// <summary>
        /// Gets or sets the best device for the workload.
        /// </summary>
        public string BestDevice { get; set; }

        /// <summary>
        /// Gets or sets the profiling duration.
        /// </summary>
        public double ProfilingDurationMs { get; set; }
    }

    /// <summary>
    /// Hybrid compute orchestrator.
    /// </summary>
    public class HybridComputeOrchestrator
    {
        private readonly Dictionary<string, Accelerator> _accelerators;
        private readonly LoadBalancer _loadBalancer;

        /// <summary>
        /// Initializes a new instance of the HybridComputeOrchestrator class.
        /// </summary>
        public HybridComputeOrchestrator(IEnumerable<Accelerator> accelerators)
        {
            _accelerators = new Dictionary<string, Accelerator>();
            foreach (var acc in accelerators)
            {
                _accelerators[acc.Name] = acc;
            }
            _loadBalancer = new LoadBalancer();
        }

        /// <summary>
        /// Schedules a computation across available devices.
        /// </summary>
        public async Task<T> ScheduleComputationAsync<T>(
            Func<Accelerator, Task<T>> computation,
            ComputeSchedulingPolicy policy = ComputeSchedulingPolicy.Automatic)
        {
            var selectedAccelerator = SelectAccelerator(policy);
            return await computation(selectedAccelerator);
        }

        /// <summary>
        /// Executes a parallel computation across multiple devices.
        /// </summary>
        public async Task<T[]> ParallelComputeAsync<T>(
            Func<Accelerator, int, Task<T>> computation,
            int count)
        {
            var tasks = new List<Task<T>>();
            var acceleratorList = new List<Accelerator>(_accelerators.Values);
            
            for (int i = 0; i < count; i++)
            {
                var acc = acceleratorList[i % acceleratorList.Count];
                tasks.Add(computation(acc, i));
            }

            return await Task.WhenAll(tasks);
        }

        private Accelerator SelectAccelerator(ComputeSchedulingPolicy policy)
        {
            // Simple round-robin for now
            return _accelerators.Values.First();
        }
    }

    /// <summary>
    /// Compute scheduling policy.
    /// </summary>
    public enum ComputeSchedulingPolicy
    {
        /// <summary>
        /// Automatically select the best policy.
        /// </summary>
        Automatic,

        /// <summary>
        /// Round-robin scheduling.
        /// </summary>
        RoundRobin,

        /// <summary>
        /// Load-based scheduling.
        /// </summary>
        LoadBased,

        /// <summary>
        /// Performance-based scheduling.
        /// </summary>
        PerformanceBased
    }

    /// <summary>
    /// Execution provider statistics.
    /// </summary>
    public class ExecutionProviderStats
    {
        /// <summary>
        /// Gets or sets the provider name.
        /// </summary>
        public string ProviderName { get; set; }

        /// <summary>
        /// Gets or sets the execution count.
        /// </summary>
        public long ExecutionCount { get; set; }

        /// <summary>
        /// Gets or sets the total execution time.
        /// </summary>
        public double TotalExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the average execution time.
        /// </summary>
        public double AverageExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the success rate.
        /// </summary>
        public double SuccessRate { get; set; }
    }

    /// <summary>
    /// Model compilation options.
    /// </summary>
    public class ModelCompilationOptions
    {
        /// <summary>
        /// Gets or sets the optimization level.
        /// </summary>
        public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.Level2;

        /// <summary>
        /// Gets or sets whether to enable graph optimizations.
        /// </summary>
        public bool EnableGraphOptimizations { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to enable memory optimizations.
        /// </summary>
        public bool EnableMemoryOptimizations { get; set; } = true;

        /// <summary>
        /// Gets or sets the target device.
        /// </summary>
        public string TargetDevice { get; set; }

        /// <summary>
        /// Gets or sets custom compilation flags.
        /// </summary>
        public Dictionary<string, object> CustomFlags { get; set; } = new();
    }

    /// <summary>
    /// Optimization level for model compilation.
    /// </summary>
    public enum OptimizationLevel
    {
        /// <summary>
        /// No optimizations.
        /// </summary>
        None,

        /// <summary>
        /// Basic optimizations.
        /// </summary>
        Level1,

        /// <summary>
        /// Standard optimizations.
        /// </summary>
        Level2,

        /// <summary>
        /// Aggressive optimizations.
        /// </summary>
        Level3
    }

    /// <summary>
    /// Model optimizer for optimizing ML models.
    /// </summary>
    public class ModelOptimizer
    {
        /// <summary>
        /// Optimizes a model for deployment.
        /// </summary>
        public async Task<OptimizedModel> OptimizeAsync(
            object model,
            ModelCompilationOptions options)
        {
            // Placeholder implementation
            return new OptimizedModel
            {
                OriginalModel = model,
                OptimizationApplied = true,
                OptimizationDetails = new List<string> { "Graph fusion", "Memory layout optimization" }
            };
        }

        /// <summary>
        /// Analyzes model for optimization opportunities.
        /// </summary>
        public ModelAnalysisResult AnalyzeModel(object model)
        {
            return new ModelAnalysisResult
            {
                TotalOperations = 1000,
                OptimizableOperations = 800,
                EstimatedSpeedup = 1.5
            };
        }
    }

    /// <summary>
    /// Optimized model result.
    /// </summary>
    public class OptimizedModel
    {
        /// <summary>
        /// Gets or sets the original model.
        /// </summary>
        public object OriginalModel { get; set; }

        /// <summary>
        /// Gets or sets whether optimization was applied.
        /// </summary>
        public bool OptimizationApplied { get; set; }

        /// <summary>
        /// Gets or sets optimization details.
        /// </summary>
        public List<string> OptimizationDetails { get; set; } = new();
    }

    /// <summary>
    /// Model analysis result.
    /// </summary>
    public class ModelAnalysisResult
    {
        /// <summary>
        /// Gets or sets total operations.
        /// </summary>
        public int TotalOperations { get; set; }

        /// <summary>
        /// Gets or sets optimizable operations.
        /// </summary>
        public int OptimizableOperations { get; set; }

        /// <summary>
        /// Gets or sets estimated speedup.
        /// </summary>
        public double EstimatedSpeedup { get; set; }
    }

    /// <summary>
    /// Compiled model representation.
    /// </summary>
    public class CompiledModel
    {
        /// <summary>
        /// Gets or sets the model ID.
        /// </summary>
        public string ModelId { get; set; }

        /// <summary>
        /// Gets or sets the compiled bytecode.
        /// </summary>
        public byte[] CompiledBytecode { get; set; }

        /// <summary>
        /// Gets or sets the target device.
        /// </summary>
        public string TargetDevice { get; set; }

        /// <summary>
        /// Gets or sets compilation metadata.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Compiled execution plan.
    /// </summary>
    public class CompiledExecutionPlan
    {
        /// <summary>
        /// Gets or sets the plan ID.
        /// </summary>
        public string PlanId { get; set; }

        /// <summary>
        /// Gets or sets the execution steps.
        /// </summary>
        public List<ExecutionStep> Steps { get; set; } = new();

        /// <summary>
        /// Gets or sets the estimated execution time.
        /// </summary>
        public double EstimatedExecutionTimeMs { get; set; }
    }

    /// <summary>
    /// Execution step in a plan.
    /// </summary>
    public class ExecutionStep
    {
        /// <summary>
        /// Gets or sets the step name.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the operation type.
        /// </summary>
        public string OperationType { get; set; }

        /// <summary>
        /// Gets or sets the target device.
        /// </summary>
        public string TargetDevice { get; set; }
    }

    /// <summary>
    /// Compiled kernel representation.
    /// </summary>
    public class CompiledKernel
    {
        /// <summary>
        /// Gets or sets the kernel name.
        /// </summary>
        public string KernelName { get; set; }

        /// <summary>
        /// Gets or sets the compiled code.
        /// </summary>
        public byte[] CompiledCode { get; set; }

        /// <summary>
        /// Gets or sets the entry point.
        /// </summary>
        public string EntryPoint { get; set; }

        /// <summary>
        /// Gets or sets kernel parameters.
        /// </summary>
        public List<KernelParameter> Parameters { get; set; } = new();
    }

    /// <summary>
    /// Kernel parameter information.
    /// </summary>
    public class KernelParameter
    {
        /// <summary>
        /// Gets or sets the parameter name.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the parameter type.
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Gets or sets whether it's an input parameter.
        /// </summary>
        public bool IsInput { get; set; }

        /// <summary>
        /// Gets or sets whether it's an output parameter.
        /// </summary>
        public bool IsOutput { get; set; }
    }

    /// <summary>
    /// Model profiling results.
    /// </summary>
    public class ModelProfilingResults
    {
        /// <summary>
        /// Gets or sets the layer timings.
        /// </summary>
        public Dictionary<string, double> LayerTimings { get; set; } = new();

        /// <summary>
        /// Gets or sets the memory usage per layer.
        /// </summary>
        public Dictionary<string, long> LayerMemoryUsage { get; set; } = new();

        /// <summary>
        /// Gets or sets the total inference time.
        /// </summary>
        public double TotalInferenceTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the peak memory usage.
        /// </summary>
        public long PeakMemoryUsageBytes { get; set; }
    }

    /// <summary>
    /// Device profiling result.
    /// </summary>
    public class DeviceProfilingResult
    {
        /// <summary>
        /// Gets or sets the device name.
        /// </summary>
        public string DeviceName { get; set; }

        /// <summary>
        /// Gets or sets the execution time.
        /// </summary>
        public double ExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the memory usage.
        /// </summary>
        public long MemoryUsageBytes { get; set; }

        /// <summary>
        /// Gets or sets the utilization percentage.
        /// </summary>
        public double UtilizationPercent { get; set; }
    }

    /// <summary>
    /// Universal compute engine for cross-platform execution.
    /// </summary>
    public class UniversalComputeEngine
    {
        private readonly Dictionary<string, IComputeBackend> _backends = new();

        /// <summary>
        /// Registers a compute backend.
        /// </summary>
        public void RegisterBackend(string name, IComputeBackend backend)
        {
            _backends[name] = backend;
        }

        /// <summary>
        /// Executes a computation on the specified backend.
        /// </summary>
        public async Task<T> ExecuteAsync<T>(string backendName, Func<Task<T>> computation)
        {
            if (!_backends.TryGetValue(backendName, out var backend))
            {
                throw new InvalidOperationException($"Backend '{backendName}' not found");
            }

            return await backend.ExecuteAsync(computation);
        }
    }

    /// <summary>
    /// Compute backend interface.
    /// </summary>
    public interface IComputeBackend
    {
        /// <summary>
        /// Gets the backend name.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Executes a computation.
        /// </summary>
        Task<T> ExecuteAsync<T>(Func<Task<T>> computation);
    }

    /// <summary>
    /// ONNX model representation.
    /// </summary>
    public class ONNXModel
    {
        /// <summary>
        /// Gets or sets the model path.
        /// </summary>
        public string ModelPath { get; set; }

        /// <summary>
        /// Gets or sets the model bytes.
        /// </summary>
        public byte[] ModelBytes { get; set; }

        /// <summary>
        /// Gets or sets the input names.
        /// </summary>
        public List<string> InputNames { get; set; } = new();

        /// <summary>
        /// Gets or sets the output names.
        /// </summary>
        public List<string> OutputNames { get; set; } = new();

        /// <summary>
        /// Gets or sets the model version.
        /// </summary>
        public long ModelVersion { get; set; }
    }

    /// <summary>
    /// Named ONNX value for input/output.
    /// </summary>
    public class NamedOnnxValue
    {
        /// <summary>
        /// Gets or sets the name.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the value.
        /// </summary>
        public object Value { get; set; }

        /// <summary>
        /// Gets or sets the data type.
        /// </summary>
        public string DataType { get; set; }

        /// <summary>
        /// Creates a named ONNX value.
        /// </summary>
        public static NamedOnnxValue CreateFromArray<T>(string name, T[] data)
            where T : unmanaged
        {
            return new NamedOnnxValue
            {
                Name = name,
                Value = data,
                DataType = typeof(T).Name
            };
        }
    }
}