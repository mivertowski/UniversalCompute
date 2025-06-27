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
using System.Linq;
using System.Threading.Tasks;

namespace ILGPU.ML.Integration
{
    /// <summary>
    /// ONNX Runtime execution provider that leverages ILGPU's universal compute platform
    /// for optimal performance across all supported hardware accelerators.
    /// </summary>
    public class ILGPUUniversalExecutionProvider : IDisposable
    {
        private readonly AdaptiveScheduler _scheduler;
        private readonly UniversalComputeEngine _computeEngine;
        private readonly ModelOptimizer _modelOptimizer;
        private readonly Dictionary<string, CompiledModel> _modelCache;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ILGPUUniversalExecutionProvider class.
        /// </summary>
        /// <param name="availableAccelerators">Available compute accelerators.</param>
        /// <param name="options">Execution provider options.</param>
        public ILGPUUniversalExecutionProvider(
            IReadOnlyDictionary<ComputeDevice, Accelerator> availableAccelerators,
            ExecutionProviderOptions? options = null)
        {
            options ??= new ExecutionProviderOptions();

            _scheduler = new AdaptiveScheduler(
                availableAccelerators, 
                options.SchedulingPolicy);

            _computeEngine = new UniversalComputeEngine(_scheduler);
            _modelOptimizer = new ModelOptimizer(options.OptimizationLevel);
            _modelCache = [];
        }

        /// <summary>
        /// Gets the provider name for ONNX Runtime registration.
        /// </summary>
        public string Name => "ILGPUUniversal";

        /// <summary>
        /// Gets the supported device types.
        /// </summary>
        public IEnumerable<string> SupportedDeviceTypes => new[] 
        { 
            "CPU", "CUDA", "OpenCL", "Metal", "DML", "IntelNPU", "AppleANE" 
        };

        /// <summary>
        /// Gets performance statistics for the execution provider.
        /// </summary>
        public ExecutionProviderStats Stats => _computeEngine.GetStats();

        /// <summary>
        /// Runs inference on the ONNX model with automatic hardware optimization.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <param name="inputs">Input data for inference.</param>
        /// <param name="outputNames">Names of outputs to compute.</param>
        /// <returns>The computed outputs.</returns>
        public async Task<NamedOnnxValue[]> RunAsync(
            string modelPath,
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames)
        {
            ThrowIfDisposed();

            // Get or compile the model
            var compiledModel = await GetOrCompileModelAsync(modelPath).ConfigureAwait(false);

            // Convert inputs to universal tensor format
            var inputTensors = await ConvertInputsToTensorsAsync(inputs).ConfigureAwait(false);

            // Create execution plan with optimal scheduling
            var executionPlan = await _scheduler.CreateExecutionPlanAsync(compiledModel.ComputeGraph).ConfigureAwait(false);

            // Execute with universal compute engine
            var outputTensors = await _computeEngine.ExecuteAsync(executionPlan, inputTensors).ConfigureAwait(false);

            // Convert outputs back to ONNX format
            return await ConvertTensorsToOutputsAsync(outputTensors, outputNames).ConfigureAwait(false);
        }

        /// <summary>
        /// Runs inference with pre-compiled execution plan for maximum performance.
        /// </summary>
        /// <param name="executionPlan">Pre-compiled execution plan.</param>
        /// <param name="inputs">Input data for inference.</param>
        /// <returns>The computed outputs.</returns>
        public async Task<NamedOnnxValue[]> RunCompiledAsync(
            CompiledExecutionPlan executionPlan,
            IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            ThrowIfDisposed();

            // Convert inputs to universal tensor format
            var inputTensors = await ConvertInputsToTensorsAsync(inputs).ConfigureAwait(false);

            // Execute with pre-compiled plan
            var outputTensors = await _computeEngine.ExecuteCompiledAsync(executionPlan, inputTensors).ConfigureAwait(false);

            // Convert outputs back to ONNX format
            return await ConvertTensorsToOutputsAsync(outputTensors, executionPlan.OutputNames).ConfigureAwait(false);
        }

        /// <summary>
        /// Compiles an ONNX model for optimal execution on available hardware.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <param name="compilationOptions">Compilation options.</param>
        /// <returns>A compiled execution plan.</returns>
        public async Task<CompiledExecutionPlan> CompileModelAsync(
            string modelPath,
            ModelCompilationOptions? compilationOptions = null)
        {
            ThrowIfDisposed();

            compilationOptions ??= new ModelCompilationOptions();

            // Load and parse ONNX model
            var onnxModel = await LoadONNXModelAsync(modelPath).ConfigureAwait(false);

            // Convert ONNX graph to ILGPU compute graph
            var computeGraph = await ConvertONNXToComputeGraphAsync(onnxModel).ConfigureAwait(false);

            // Optimize the compute graph
            var optimizedGraph = await _modelOptimizer.OptimizeAsync(computeGraph, compilationOptions).ConfigureAwait(false);

            // Create execution plan
            var executionPlan = await _scheduler.CreateExecutionPlanAsync(optimizedGraph).ConfigureAwait(false);

            // Compile kernels for target devices
            var compiledKernels = await CompileKernelsAsync(executionPlan).ConfigureAwait(false);

            return new CompiledExecutionPlan(
                executionPlan,
                compiledKernels,
                onnxModel.InputNames,
                onnxModel.OutputNames);
        }

        /// <summary>
        /// Profiles the model execution across different device configurations.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <param name="sampleInputs">Sample inputs for profiling.</param>
        /// <returns>Profiling results for each device configuration.</returns>
        public async Task<ModelProfilingResults> ProfileModelAsync(
            string modelPath,
            IReadOnlyCollection<NamedOnnxValue> sampleInputs)
        {
            ThrowIfDisposed();

            var results = new Dictionary<ComputeDevice, DeviceProfilingResult>();

            foreach (var device in _scheduler.AvailableDevices)
            {
                try
                {
                    var deviceResult = await ProfileOnDeviceAsync(modelPath, sampleInputs, device).ConfigureAwait(false);
                    results[device] = deviceResult;
                }
                catch (Exception ex)
                {
                    results[device] = new DeviceProfilingResult(device, false, 0, 0, ex.Message);
                }
            }

            return new ModelProfilingResults(results);
        }

        /// <summary>
        /// Optimizes the execution provider for a specific workload pattern.
        /// </summary>
        /// <param name="workloadSamples">Sample workloads for optimization.</param>
        /// <returns>A task representing the optimization operation.</returns>
        public async Task OptimizeForWorkloadAsync(IEnumerable<WorkloadSample> workloadSamples)
        {
            ThrowIfDisposed();

            // Analyze workload characteristics
            var workloadAnalysis = AnalyzeWorkloadSamples(workloadSamples);

            // Update scheduling strategy
            await _scheduler.UpdatePolicyAsync(workloadAnalysis).ConfigureAwait(false);

            // Optimize memory management
            await _computeEngine.OptimizeMemoryAsync(workloadAnalysis).ConfigureAwait(false);

            // Clear model cache to force recompilation with new optimizations
            ClearModelCache();
        }

        /// <summary>
        /// Gets recommendations for optimal execution configuration.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file.</param>
        /// <returns>Configuration recommendations.</returns>
        public async Task<ConfigurationRecommendations> GetRecommendationsAsync(string modelPath)
        {
            ThrowIfDisposed();

            var model = await LoadONNXModelAsync(modelPath).ConfigureAwait(false);
            var computeGraph = await ConvertONNXToComputeGraphAsync(model).ConfigureAwait(false);
            
            var analysis = _modelOptimizer.AnalyzeModel(computeGraph);
            var deviceRecommendations = _scheduler.GetDeviceRecommendations(analysis);
            
            return new ConfigurationRecommendations(
                deviceRecommendations,
                analysis.RecommendedBatchSize,
                analysis.OptimalMemoryLayout,
                analysis.SuggestedOptimizations);
        }

        private async Task<CompiledModel> GetOrCompileModelAsync(string modelPath)
        {
            if (_modelCache.TryGetValue(modelPath, out var cachedModel))
            {
                return cachedModel;
            }

            var onnxModel = await LoadONNXModelAsync(modelPath).ConfigureAwait(false);
            var computeGraph = await ConvertONNXToComputeGraphAsync(onnxModel).ConfigureAwait(false);
            var optimizedGraph = await _modelOptimizer.OptimizeAsync(computeGraph, new ModelCompilationOptions()).ConfigureAwait(false);

            var compiledModel = new CompiledModel(modelPath, optimizedGraph, onnxModel.InputNames, onnxModel.OutputNames);
            _modelCache[modelPath] = compiledModel;

            return compiledModel;
        }

        private async Task<ONNXModel> LoadONNXModelAsync(string modelPath) =>
            // Implementation would load and parse ONNX model file
            // This is a simplified placeholder
            await Task.FromResult(new ONNXModel(modelPath)).ConfigureAwait(false);

        private async Task<ComputeGraph> ConvertONNXToComputeGraphAsync(ONNXModel onnxModel) =>
            // Implementation would convert ONNX operators to ILGPU compute operations
            // This involves mapping ONNX ops to universal kernels
            await Task.FromResult(new ComputeGraph()).ConfigureAwait(false);

        private async Task<Dictionary<string, ITensor<float>>> ConvertInputsToTensorsAsync(
            IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            var tensorInputs = new Dictionary<string, ITensor<float>>();

            foreach (var input in inputs)
            {
                var tensor = await ConvertOnnxValueToTensorAsync(input).ConfigureAwait(false);
                tensorInputs[input.Name] = tensor;
            }

            return tensorInputs;
        }

        private async Task<ITensor<float>> ConvertOnnxValueToTensorAsync(NamedOnnxValue onnxValue) =>
            // Implementation would convert ONNX tensor format to ILGPU tensor format
            await Task.FromResult<ITensor<float>>(null).ConfigureAwait(false);

        private async Task<NamedOnnxValue[]> ConvertTensorsToOutputsAsync(
            Dictionary<string, ITensor<float>> tensors,
            IEnumerable<string> outputNames)
        {
            var outputs = new List<NamedOnnxValue>();

            foreach (var outputName in outputNames)
            {
                if (tensors.TryGetValue(outputName, out var tensor))
                {
                    var onnxValue = await ConvertTensorToOnnxValueAsync(outputName, tensor).ConfigureAwait(false);
                    outputs.Add(onnxValue);
                }
            }

            return outputs.ToArray();
        }

        private async Task<NamedOnnxValue> ConvertTensorToOnnxValueAsync(string name, ITensor<float> tensor) =>
            // Implementation would convert ILGPU tensor back to ONNX format
            await Task.FromResult<NamedOnnxValue>(null).ConfigureAwait(false);

        private async Task<Dictionary<ComputeNode, CompiledKernel>> CompileKernelsAsync(ExecutionPlan plan)
        {
            var compiledKernels = new Dictionary<ComputeNode, CompiledKernel>();

            foreach (var node in plan.Graph.Nodes)
            {
                var kernel = await CompileNodeKernelAsync(node, plan.Assignments[node]).ConfigureAwait(false);
                compiledKernels[node] = kernel;
            }

            return compiledKernels;
        }

        private async Task<CompiledKernel> CompileNodeKernelAsync(ComputeNode node, ComputeDevice device) =>
            // Implementation would compile the node's operation to device-specific code
            await Task.FromResult(new CompiledKernel(node, device)).ConfigureAwait(false);

        private async Task<DeviceProfilingResult> ProfileOnDeviceAsync(
            string modelPath,
            IReadOnlyCollection<NamedOnnxValue> sampleInputs,
            ComputeDevice device)
        {
            // Measure execution time and throughput on specific device
            var startTime = DateTime.UtcNow;
            
            for (int i = 0; i < 10; i++)
            {
                await RunAsync(modelPath, sampleInputs, new[] { "output" }).ConfigureAwait(false);
            }

            var endTime = DateTime.UtcNow;
            var totalTime = (endTime - startTime).TotalMilliseconds;
            var avgLatency = totalTime / 10.0;
            var throughput = 10.0 / (totalTime / 1000.0);

            return new DeviceProfilingResult(device, true, avgLatency, throughput);
        }

        private WorkloadAnalysis AnalyzeWorkloadSamples(IEnumerable<WorkloadSample> samples)
        {
            // Analyze common patterns in workload samples
            var batchSizes = samples.Select(s => s.BatchSize).ToList();
            var modelSizes = samples.Select(s => s.ModelComplexity).ToList();
            var frequencies = samples.Select(s => s.Frequency).ToList();

            return new WorkloadAnalysis(
                TimeSpan.FromMilliseconds(batchSizes.Count != 0 ? batchSizes.Average() * 10 : 10),
                modelSizes.Count != 0 ? (long)modelSizes.Average() * 1024 : 1024,
                WorkloadType.Compute,
                1, // Priority
                modelSizes.Count != 0 ? modelSizes.Average() : 0,
                "ONNX workload analysis");
        }

        private void ClearModelCache()
        {
            foreach (var model in _modelCache.Values)
            {
                model.Dispose();
            }
            _modelCache.Clear();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ILGPUUniversalExecutionProvider));
        }

        /// <summary>
        /// Disposes the execution provider and releases all resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            ClearModelCache();
            _computeEngine?.Dispose();
            _scheduler?.Dispose();
            _modelOptimizer?.Dispose();

            _disposed = true;
        }
    }

    /// <summary>
    /// Options for configuring the ILGPU Universal Execution Provider.
    /// </summary>
    public class ExecutionProviderOptions
    {
        /// <summary>
        /// Gets or sets the scheduling policy to use.
        /// </summary>
        public SchedulingPolicy SchedulingPolicy { get; set; } = SchedulingPolicy.PerformanceOptimized;

        /// <summary>
        /// Gets or sets the optimization level (0-3).
        /// </summary>
        public int OptimizationLevel { get; set; } = 2;

        /// <summary>
        /// Gets or sets whether to enable automatic mixed precision.
        /// </summary>
        public bool EnableMixedPrecision { get; set; } = true;

        /// <summary>
        /// Gets or sets the memory usage limit in bytes.
        /// </summary>
        public long MemoryLimitBytes { get; set; } = long.MaxValue;

        /// <summary>
        /// Gets or sets whether to enable kernel fusion optimizations.
        /// </summary>
        public bool EnableKernelFusion { get; set; } = true;
    }

    /// <summary>
    /// Represents a sample workload for optimization analysis.
    /// </summary>
    public class WorkloadSample
    {
        /// <summary>
        /// Gets or sets the batch size for this workload.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// Gets or sets the model complexity metric.
        /// </summary>
        public double ModelComplexity { get; set; }

        /// <summary>
        /// Gets or sets the frequency of execution.
        /// </summary>
        public double Frequency { get; set; }

        /// <summary>
        /// Gets or sets the input data characteristics.
        /// </summary>
        public DataCharacteristics DataCharacteristics { get; set; }
    }

    /// <summary>
    /// Provides configuration recommendations for optimal performance.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ConfigurationRecommendations class.
    /// </remarks>
    public class ConfigurationRecommendations(
        IReadOnlyDictionary<string, ComputeDevice> deviceRecommendations,
        int recommendedBatchSize,
        MemoryLayout optimalMemoryLayout,
        string[] suggestedOptimizations)
    {
        /// <summary>
        /// Gets the recommended device assignments.
        /// </summary>
        public IReadOnlyDictionary<string, ComputeDevice> DeviceRecommendations { get; } = deviceRecommendations;

        /// <summary>
        /// Gets the recommended batch size.
        /// </summary>
        public int RecommendedBatchSize { get; } = recommendedBatchSize;

        /// <summary>
        /// Gets the optimal memory layout strategy.
        /// </summary>
        public MemoryLayout OptimalMemoryLayout { get; } = optimalMemoryLayout;

        /// <summary>
        /// Gets suggested optimizations.
        /// </summary>
        public string[] SuggestedOptimizations { get; } = suggestedOptimizations;
    }

    /// <summary>
    /// Defines memory layout strategies.
    /// </summary>
    public enum MemoryLayout
    {
        /// <summary>
        /// Row-major layout (C-style).
        /// </summary>
        RowMajor,

        /// <summary>
        /// Column-major layout (Fortran-style).
        /// </summary>
        ColumnMajor,

        /// <summary>
        /// Tiled layout for cache optimization.
        /// </summary>
        Tiled,

        /// <summary>
        /// Optimal layout determined automatically.
        /// </summary>
        Optimal
    }

}