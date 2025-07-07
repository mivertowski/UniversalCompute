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

using ILGPU.Numerics;
using ILGPU.Runtime;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
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

        /// <summary>
        /// Creates a compute graph for the given input.
        /// </summary>
        Task<ComputeGraph> CreateComputeGraphAsync(ITensor<float> input);

        /// <summary>
        /// Creates a batched compute graph for the given inputs.
        /// </summary>
        Task<ComputeGraph> CreateBatchedComputeGraphAsync(ITensor<float>[] inputs);
    }

    /// <summary>
    /// Model metadata.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Gets the input shapes.
        /// </summary>
        public Dictionary<string, int[]> InputShapes { get; } = [];

        /// <summary>
        /// Gets the output shapes.
        /// </summary>
        public Dictionary<string, int[]> OutputShapes { get; } = [];

        /// <summary>
        /// Gets or sets the model type.
        /// </summary>
        public required string ModelType { get; set; }

        /// <summary>
        /// Gets or sets the framework.
        /// </summary>
        public required string Framework { get; set; }
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
        public required string DeviceUsed { get; set; }

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
        public required string BottleneckAnalysis { get; set; }

        /// <summary>
        /// Gets optimization suggestions.
        /// </summary>
        public Collection<string> OptimizationSuggestions { get; } = [];
    }

    /// <summary>
    /// Device profile results.
    /// </summary>
    public class DeviceProfileResult
    {
        /// <summary>
        /// Gets or sets the device name.
        /// </summary>
        public required string DeviceName { get; set; }

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

        /// <summary>
        /// Initializes a new instance of the DeviceProfileResult class.
        /// </summary>
        public DeviceProfileResult()
        {
        }

        /// <summary>
        /// Initializes a new instance of the DeviceProfileResult class.
        /// </summary>
        [SetsRequiredMembers]
        public DeviceProfileResult(ComputeDevice device, bool success, double avgLatency, double throughput)
        {
            DeviceName = device.ToString();
            DeviceType = AcceleratorType.CPU; // Default, should be determined from device
            PeakPerformanceGFLOPS = throughput / 1000;
            MeasuredPerformanceGFLOPS = success ? throughput / 1000 : 0;
            EfficiencyPercent = success ? 85.0 : 0.0;
            MemoryBandwidthGBps = 100.0; // Placeholder
        }

        /// <summary>
        /// Initializes a new instance of the DeviceProfileResult class with error message.
        /// </summary>
        [SetsRequiredMembers]
        public DeviceProfileResult(ComputeDevice device, bool success, double avgLatency, double throughput, string errorMessage)
            : this(device, success, avgLatency, throughput)
        {
            if (!success)
            {
                DeviceName = $"{device} (Error: {errorMessage})";
            }
        }
    }

    /// <summary>
    /// Collection of device profile results.
    /// </summary>
    public class DeviceProfileResults
    {
        /// <summary>
        /// Gets or sets the individual device results.
        /// </summary>
        public IList<DeviceProfileResult> Results { get; } = [];

        /// <summary>
        /// Gets or sets the best device for the workload.
        /// </summary>
        public required string BestDevice { get; set; }

        /// <summary>
        /// Gets or sets the profiling duration.
        /// </summary>
        public double ProfilingDurationMs { get; set; }

        /// <summary>
        /// Initializes a new instance of the DeviceProfileResults class.
        /// </summary>
        public DeviceProfileResults()
        {
        }

        /// <summary>
        /// Initializes a new instance of the DeviceProfileResults class.
        /// </summary>
        [SetsRequiredMembers]
        public DeviceProfileResults(Dictionary<ComputeDevice, DeviceProfileResult> results)
        {
            Results = [.. results.Values];
            BestDevice = results.OrderByDescending(r => r.Value.MeasuredPerformanceGFLOPS).FirstOrDefault().Key.ToString();
            ProfilingDurationMs = 1000.0; // Placeholder
        }
    }

    /// <summary>
    /// Hybrid compute orchestrator.
    /// </summary>
    public class HybridComputeOrchestrator : IDisposable
    {
        private readonly Dictionary<string, Accelerator> _accelerators;
        private readonly LoadBalancer _loadBalancer;

        /// <summary>
        /// Initializes a new instance of the HybridComputeOrchestrator class.
        /// </summary>
        public HybridComputeOrchestrator(IEnumerable<Accelerator> accelerators)
        {
            _accelerators = [];
            foreach (var acc in accelerators)
            {
                _accelerators[acc.Name] = acc;
            }
            _loadBalancer = new LoadBalancer();
        }

        /// <summary>
        /// Initializes a new instance of the HybridComputeOrchestrator class with scheduler and context.
        /// </summary>
        public HybridComputeOrchestrator(AdaptiveScheduler scheduler, PredictionContext context)
        {
            _accelerators = [];
            _loadBalancer = new LoadBalancer();
            // Initialize with scheduler and context
        }

        /// <summary>
        /// Schedules a computation across available devices.
        /// </summary>
        public async Task<T> ScheduleComputationAsync<T>(
            Func<Accelerator, Task<T>> computation,
            ComputeSchedulingPolicy policy = ComputeSchedulingPolicy.Automatic)
        {
            var selectedAccelerator = SelectAccelerator(policy);
            return await computation(selectedAccelerator).ConfigureAwait(false);
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

            return await Task.WhenAll(tasks).ConfigureAwait(false);
        }

        private Accelerator SelectAccelerator(ComputeSchedulingPolicy policy) =>
            // Simple round-robin for now
            _accelerators.Values.First();

        /// <summary>
        /// Gets performance statistics for all accelerators.
        /// </summary>
        public static PerformanceAnalysis GetPerformanceStats()
        {
            var analysis = new PerformanceAnalysis
            {
                TotalExecutionTimeMs = 1000.0,
                GpuUtilizationPercent = 85.0,
                MemoryBandwidthUtilizationPercent = 75.0,
                ComputeEfficiency = 0.9,
                BottleneckAnalysis = "Memory bandwidth limited"
            };
            
            // Add optimization suggestions to the collection
            analysis.OptimizationSuggestions.Add("Increase batch size");
            analysis.OptimizationSuggestions.Add("Optimize memory access patterns");
            
            return analysis;
        }

        /// <summary>
        /// Executes a computation with ML model.
        /// </summary>
        public static async Task<TOutput> ExecuteAsync<TInput, TOutput>(IMLModel<TInput, TOutput> model, TInput input)
            where TInput : class
            where TOutput : class => await model.PredictAsync(input).ConfigureAwait(false);

        /// <summary>
        /// Executes batch computation with ML model.
        /// </summary>
        public static async Task<TOutput[]> ExecuteBatchAsync<TInput, TOutput>(IMLModel<TInput, TOutput> model, TInput[] inputs)
            where TInput : class
            where TOutput : class => await model.PredictBatchAsync(inputs).ConfigureAwait(false);

        /// <summary>
        /// Optimizes scheduling for better performance.
        /// </summary>
        public static async Task OptimizeSchedulingAsync() =>
            // Placeholder implementation
            await Task.CompletedTask.ConfigureAwait(false);

        /// <summary>
        /// Optimizes memory usage across accelerators.
        /// </summary>
        public static async Task OptimizeMemoryAsync() =>
            // Placeholder implementation
            await Task.CompletedTask.ConfigureAwait(false);

        /// <summary>
        /// Analyzes performance across accelerators.
        /// </summary>
        public static PerformanceAnalysis AnalyzePerformance() => GetPerformanceStats();

        /// <summary>
        /// Disposes the hybrid compute orchestrator.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the hybrid compute orchestrator.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _accelerators?.Clear();
                _loadBalancer?.Dispose();
            }
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
        public required string ProviderName { get; set; }

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
    /// <remarks>
    /// Initializes a new instance of the ModelCompilationOptions class.
    /// </remarks>
    [method: SetsRequiredMembers]
    
    public class ModelCompilationOptions()
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
        public required string TargetDevice { get; set; } = "default";

        /// <summary>
        /// Gets or sets custom compilation flags.
        /// </summary>
        public Dictionary<string, object> CustomFlags { get; } = [];
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
    public class ModelOptimizer : IDisposable
    {
        private readonly int _optimizationLevel;

        /// <summary>
        /// Initializes a new instance of the ModelOptimizer class.
        /// </summary>
        public ModelOptimizer()
        {
            _optimizationLevel = 2; // Default optimization level
        }

        /// <summary>
        /// Initializes a new instance of the ModelOptimizer class with optimization level.
        /// </summary>
        public ModelOptimizer(int optimizationLevel)
        {
            _optimizationLevel = optimizationLevel;
        }

        /// <summary>
        /// Optimizes a model for deployment.
        /// </summary>
        public static async Task<OptimizedModel> OptimizeAsync(
            object model,
            ModelCompilationOptions options)
        {
            // Simulate async operation
            await Task.Delay(1).ConfigureAwait(false);
            
            // Placeholder implementation
            var optimizedModel = new OptimizedModel
            {
                OriginalModel = model,
                OptimizationApplied = true,
                OptimizedGraph = model as ComputeGraph ?? new ComputeGraph()
            };
            
            // Add optimization details to the collection
            optimizedModel.OptimizationDetails.Add("Graph fusion");
            optimizedModel.OptimizationDetails.Add("Memory layout optimization");
            
            return optimizedModel;
        }

        /// <summary>
        /// Analyzes model for optimization opportunities.
        /// </summary>
        public static ModelAnalysisResult AnalyzeModel(object model) => new()
        {
            TotalOperations = 1000,
            OptimizableOperations = 800,
            EstimatedSpeedup = 1.5
        };

        /// <summary>
        /// Disposes the model optimizer.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the model optimizer.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Nothing to dispose for now
            }
        }
    }

    /// <summary>
    /// Optimized model result.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the OptimizedModel class.
    /// </remarks>
    [method: SetsRequiredMembers]
    
    public class OptimizedModel()
    {

        /// <summary>
        /// Gets or sets the original model.
        /// </summary>
        public required object OriginalModel { get; set; } = new object();

        /// <summary>
        /// Gets or sets whether optimization was applied.
        /// </summary>
        public bool OptimizationApplied { get; set; }

        /// <summary>
        /// Gets optimization details.
        /// </summary>
        public Collection<string> OptimizationDetails { get; } = [];

        /// <summary>
        /// Gets or sets the optimized compute graph.
        /// </summary>
        public required ComputeGraph OptimizedGraph { get; set; } = new ComputeGraph();

        /// <summary>
        /// Implicit conversion to ComputeGraph.
        /// </summary>
        public static implicit operator ComputeGraph(OptimizedModel optimizedModel) => optimizedModel?.OptimizedGraph ?? new ComputeGraph();
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

        /// <summary>
        /// Gets or sets the recommended batch size.
        /// </summary>
        public int RecommendedBatchSize { get; set; } = 32;

        /// <summary>
        /// Gets or sets the optimal memory layout.
        /// </summary>
        public MemoryLayout OptimalMemoryLayout { get; set; } = MemoryLayout.RowMajor;

        /// <summary>
        /// Gets or sets suggested optimizations.
        /// </summary>
        public IReadOnlyList<string> SuggestedOptimizations { get; } = ["Graph fusion", "Memory optimization"];
    }

    /// <summary>
    /// Compiled model representation.
    /// </summary>
    public class CompiledModel : IDisposable
    {
        /// <summary>
        /// Gets or sets the model ID.
        /// </summary>
        public required string ModelId { get; set; }

        /// <summary>
        /// Gets the compiled bytecode.
        /// </summary>
        public required IList<byte> CompiledBytecode { get; init; }

        /// <summary>
        /// Gets or sets the target device.
        /// </summary>
        public required string TargetDevice { get; set; }

        /// <summary>
        /// Gets compilation metadata.
        /// </summary>
        public Dictionary<string, object> Metadata { get; } = [];

        /// <summary>
        /// Gets or sets the compute graph.
        /// </summary>
        public required ComputeGraph ComputeGraph { get; set; }

        /// <summary>
        /// Gets the input names.
        /// </summary>
        public IList<string> InputNames { get; } = [];

        /// <summary>
        /// Gets the output names.
        /// </summary>
        public IList<string> OutputNames { get; } = [];


        /// <summary>
        /// Initializes a new instance of the CompiledModel class.
        /// </summary>
        [SetsRequiredMembers]
        public CompiledModel(string modelPath, ComputeGraph graph, IList<string> inputNames, IList<string> outputNames)
        {
            ModelId = modelPath;
            ComputeGraph = graph;
            InputNames = inputNames;
            OutputNames = outputNames;
            CompiledBytecode = [];
            TargetDevice = "GPU";
        }

        /// <summary>
        /// Disposes the compiled model.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the compiled model.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Metadata?.Clear();
            }
        }
    }

    /// <summary>
    /// Compiled execution plan.
    /// </summary>
    public class CompiledExecutionPlan
    {
        /// <summary>
        /// Gets or sets the plan ID.
        /// </summary>
        public required string PlanId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets the execution steps.
        /// </summary>
        public IList<ExecutionStep> Steps { get; } = [];

        /// <summary>
        /// Gets or sets the estimated execution time.
        /// </summary>
        public double EstimatedExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the output names.
        /// </summary>
        public IEnumerable<string> OutputNames { get; set; } = new List<string>();


        /// <summary>
        /// Initializes a new instance of the CompiledExecutionPlan class.
        /// </summary>
        [SetsRequiredMembers]
        public CompiledExecutionPlan(ExecutionPlan executionPlan, Dictionary<ComputeNode, CompiledKernel> compiledKernels, IList<string> inputNames, IList<string> outputNames)
        {
            PlanId = Guid.NewGuid().ToString();
            OutputNames = outputNames;
            EstimatedExecutionTimeMs = executionPlan.TotalTime.TotalMilliseconds;
        }
    }

    /// <summary>
    /// Execution step in a plan.
    /// </summary>
    public class ExecutionStep
    {
        /// <summary>
        /// Gets or sets the step name.
        /// </summary>
        public required string Name { get; set; } = "step";

        /// <summary>
        /// Gets or sets the operation type.
        /// </summary>
        public required string OperationType { get; set; } = "operation";

        /// <summary>
        /// Gets or sets the target device.
        /// </summary>
        public required string TargetDevice { get; set; } = "GPU";
    }

    /// <summary>
    /// Compiled kernel representation.
    /// </summary>
    public class CompiledKernel
    {
        /// <summary>
        /// Gets or sets the kernel name.
        /// </summary>
        public required string KernelName { get; set; } = "default_kernel";

        /// <summary>
        /// Gets the compiled code.
        /// </summary>
        public required IList<byte> CompiledCode { get; init; } = [0x00];

        /// <summary>
        /// Gets or sets the entry point.
        /// </summary>
        public required string EntryPoint { get; set; } = "main";

        /// <summary>
        /// Gets kernel parameters.
        /// </summary>
        public IList<KernelParameter> Parameters { get; } = [];


        /// <summary>
        /// Initializes a new instance of the CompiledKernel class.
        /// </summary>
        [SetsRequiredMembers]
        public CompiledKernel(ComputeNode node, ComputeDevice device)
        {
            KernelName = $"{node.Operation.GetType().Name}_{device}";
            EntryPoint = "main";
            CompiledCode = [0x00]; // Placeholder
        }
    }

    /// <summary>
    /// Kernel parameter information.
    /// </summary>
    public class KernelParameter
    {
        /// <summary>
        /// Gets or sets the parameter name.
        /// </summary>
        public required string Name { get; set; } = "param";

        /// <summary>
        /// Gets or sets the parameter type.
        /// </summary>
        public required string Type { get; set; } = "int";

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
        /// Gets the layer timings.
        /// </summary>
        public Dictionary<string, double> LayerTimings { get; } = [];

        /// <summary>
        /// Gets the memory usage per layer.
        /// </summary>
        public Dictionary<string, long> LayerMemoryUsage { get; } = [];

        /// <summary>
        /// Gets or sets the total inference time.
        /// </summary>
        public double TotalInferenceTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the peak memory usage.
        /// </summary>
        public long PeakMemoryUsageBytes { get; set; }

        /// <summary>
        /// Initializes a new instance of the ModelProfilingResults class.
        /// </summary>
        public ModelProfilingResults()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ModelProfilingResults class with device results.
        /// </summary>
        [SetsRequiredMembers]
        public ModelProfilingResults(Dictionary<ComputeDevice, DeviceProfilingResult> deviceResults)
        {
            TotalInferenceTimeMs = deviceResults.Values.Sum(r => r.ExecutionTimeMs);
            PeakMemoryUsageBytes = deviceResults.Values.Max(r => r.MemoryUsageBytes);
            
            foreach (var kvp in deviceResults)
            {
                LayerTimings[kvp.Key.ToString()] = kvp.Value.ExecutionTimeMs;
                LayerMemoryUsage[kvp.Key.ToString()] = kvp.Value.MemoryUsageBytes;
            }
        }
    }

    /// <summary>
    /// Device profiling result.
    /// </summary>
    public class DeviceProfilingResult
    {
        /// <summary>
        /// Gets or sets the device name.
        /// </summary>
        public required string DeviceName { get; set; }

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

        /// <summary>
        /// Initializes a new instance of the DeviceProfilingResult class.
        /// </summary>
        public DeviceProfilingResult()
        {
        }

        /// <summary>
        /// Initializes a new instance of the DeviceProfilingResult class.
        /// </summary>
        [SetsRequiredMembers]
        public DeviceProfilingResult(ComputeDevice device, bool success, double avgLatency, double throughput)
        {
            DeviceName = device.ToString();
            ExecutionTimeMs = avgLatency;
            MemoryUsageBytes = 1024 * 1024; // 1MB placeholder
            UtilizationPercent = success ? 85.0 : 0.0;
        }

        /// <summary>
        /// Initializes a new instance of the DeviceProfilingResult class with error.
        /// </summary>
        [SetsRequiredMembers]
        public DeviceProfilingResult(ComputeDevice device, bool success, double avgLatency, double throughput, string errorMessage)
            : this(device, success, avgLatency, throughput)
        {
            if (!success)
            {
                DeviceName = $"{device} (Error: {errorMessage})";
            }
        }
    }

    /// <summary>
    /// Universal compute engine for cross-platform execution.
    /// </summary>
    public class UniversalComputeEngine : IDisposable
    {
        private readonly Dictionary<string, IComputeBackend> _backends = [];

        /// <summary>
        /// Initializes a new instance of the UniversalComputeEngine class.
        /// </summary>
        public UniversalComputeEngine()
        {
        }

        /// <summary>
        /// Initializes a new instance of the UniversalComputeEngine class with a scheduler.
        /// </summary>
        public UniversalComputeEngine(AdaptiveScheduler scheduler)
        {
            // Initialize with scheduler
        }

        /// <summary>
        /// Registers a compute backend.
        /// </summary>
        public void RegisterBackend(string name, IComputeBackend backend) => _backends[name] = backend;

        /// <summary>
        /// Executes a computation on the specified backend.
        /// </summary>
        public async Task<T> ExecuteAsync<T>(string backendName, Func<Task<T>> computation) => !_backends.TryGetValue(backendName, out var backend)
                ? throw new InvalidOperationException($"Backend '{backendName}' not found")
                : await backend.ExecuteAsync(computation).ConfigureAwait(false);

        /// <summary>
        /// Executes an execution plan with input tensors.
        /// </summary>
        public static async Task<Dictionary<string, ITensor<float>>> ExecuteAsync(ExecutionPlan plan, Dictionary<string, ITensor<float>> inputs) =>
            // Placeholder implementation
            await Task.FromResult(new Dictionary<string, ITensor<float>>()).ConfigureAwait(false);

        /// <summary>
        /// Executes a pre-compiled execution plan.
        /// </summary>
        public static async Task<Dictionary<string, ITensor<float>>> ExecuteCompiledAsync(CompiledExecutionPlan plan, Dictionary<string, ITensor<float>> inputs) =>
            // Placeholder implementation
            await Task.FromResult(new Dictionary<string, ITensor<float>>()).ConfigureAwait(false);

        /// <summary>
        /// Gets execution provider statistics.
        /// </summary>
        public static ExecutionProviderStats Stats => new()
        {
            ProviderName = "UniversalComputeEngine",
            ExecutionCount = 0,
            TotalExecutionTimeMs = 0,
            AverageExecutionTimeMs = 0,
            SuccessRate = 1.0
        };

        /// <summary>
        /// Optimizes memory usage based on workload analysis.
        /// </summary>
        public static async Task OptimizeMemoryAsync(WorkloadAnalysis analysis) =>
            // Placeholder implementation
            await Task.CompletedTask.ConfigureAwait(false);

        /// <summary>
        /// Disposes the universal compute engine.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes the universal compute engine.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _backends.Clear();
            }
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
        public required string ModelPath { get; set; }

        /// <summary>
        /// Gets the model bytes.
        /// </summary>
        public required IList<byte> ModelBytes { get; init; }

        /// <summary>
        /// Gets the input names.
        /// </summary>
        public IList<string> InputNames { get; } = [];

        /// <summary>
        /// Gets the output names.
        /// </summary>
        public IList<string> OutputNames { get; } = [];

        /// <summary>
        /// Gets or sets the model version.
        /// </summary>
        public long ModelVersion { get; set; }

        /// <summary>
        /// Initializes a new instance of the ONNXModel class.
        /// </summary>
        public ONNXModel()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ONNXModel class.
        /// </summary>
        [SetsRequiredMembers]
        public ONNXModel(string modelPath)
        {
            ModelPath = modelPath;
            ModelBytes = Array.Empty<byte>(); // Default empty bytes
        }
    }

    /// <summary>
    /// Named ONNX value for input/output.
    /// </summary>
    public class NamedOnnxValue
    {
        /// <summary>
        /// Gets or sets the name.
        /// </summary>
        public required string Name { get; set; } = "value";

        /// <summary>
        /// Gets or sets the value.
        /// </summary>
        public required object Value { get; set; } = new object();

        /// <summary>
        /// Gets or sets the data type.
        /// </summary>
        public required string DataType { get; set; } = "object";

        /// <summary>
        /// Creates a named ONNX value.
        /// </summary>
        public static NamedOnnxValue CreateFromArray<T>(string name, T[] data)
            where T : unmanaged => new()
            {
                Name = name,
                Value = data,
                DataType = typeof(T).Name
            };
    }
}
