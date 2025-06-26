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
using ILGPU.Runtime.DependencyInjection;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ILGPU.ML.Integration
{
    /// <summary>
    /// Universal ML.NET predictor that leverages ILGPU's cross-platform acceleration capabilities.
    /// Automatically selects optimal hardware for different model operations.
    /// </summary>
    /// <typeparam name="TInput">The input data type.</typeparam>
    /// <typeparam name="TOutput">The output data type.</typeparam>
    public class ILGPUUniversalPredictor<TInput, TOutput> : IDisposable
        where TInput : class
        where TOutput : class
    {
        private readonly HybridComputeOrchestrator _orchestrator;
        private readonly IMLModel<TInput, TOutput> _model;
        private readonly AdaptiveScheduler _scheduler;
        private readonly PredictionContext _context;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ILGPUUniversalPredictor class.
        /// </summary>
        /// <param name="model">The ML model to use for predictions.</param>
        /// <param name="context">The prediction context with accelerator information.</param>
        public ILGPUUniversalPredictor(IMLModel<TInput, TOutput> model, PredictionContext context)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _context = context ?? throw new ArgumentNullException(nameof(context));
            
            _scheduler = new AdaptiveScheduler(
                context.AvailableAccelerators,
                SchedulingPolicy.PerformanceOptimized);
            
            _orchestrator = new HybridComputeOrchestrator(_scheduler, context);
        }

        /// <summary>
        /// Gets the model information.
        /// </summary>
        public IMLModel<TInput, TOutput> Model => _model;

        /// <summary>
        /// Gets performance statistics from recent predictions.
        /// </summary>
        public PredictionStats PerformanceStats => ConvertToPredictionStats(_orchestrator.GetPerformanceStats());

        private PredictionStats ConvertToPredictionStats(PerformanceAnalysis analysis) => new PredictionStats
        {
            InferenceTimeMs = analysis.TotalExecutionTimeMs,
            PreprocessingTimeMs = 0.0,
            PostprocessingTimeMs = 0.0,
            DeviceUsed = "Auto",
            BatchSize = 1,
            ThroughputPerSecond = 1000.0 / analysis.TotalExecutionTimeMs
        };

        /// <summary>
        /// Predicts output for a single input using optimal hardware acceleration.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <returns>The predicted output.</returns>
        public async ValueTask<TOutput> PredictAsync(TInput input)
        {
            ThrowIfDisposed();

            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Convert input to tensor format
            var inputTensor = await ConvertToTensorAsync(input);

            // Create compute graph for the model
            var computeGraph = await _model.CreateComputeGraphAsync(inputTensor);

            // Execute with optimal scheduling using the model
            var result = await _orchestrator.ExecuteAsync(_model, input);

            // Return the result directly
            return result;
        }

        /// <summary>
        /// Predicts outputs for a batch of inputs with optimal batching and parallelization.
        /// </summary>
        /// <param name="inputs">The input data batch.</param>
        /// <returns>The predicted outputs.</returns>
        public async ValueTask<TOutput[]> PredictBatchAsync(TInput[] inputs)
        {
            ThrowIfDisposed();

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (inputs.Length == 0)
                return Array.Empty<TOutput>();

            // Convert inputs to tensor batch
            var inputTensors = await ConvertToTensorBatchAsync(inputs);

            // Create batched compute graph
            var computeGraph = await _model.CreateBatchedComputeGraphAsync(inputTensors);

            // Execute with batch optimization
            var results = await _orchestrator.ExecuteBatchAsync(_model, inputs);

            // Return the results directly
            return results;
        }

        /// <summary>
        /// Streams predictions for large datasets with memory optimization.
        /// </summary>
        /// <param name="inputStream">The input data stream.</param>
        /// <param name="batchSize">The batch size for processing.</param>
        /// <returns>An async enumerable of predictions.</returns>
        public async IAsyncEnumerable<TOutput> PredictStreamAsync(
            IAsyncEnumerable<TInput> inputStream, 
            int batchSize = 32)
        {
            ThrowIfDisposed();

            var batch = new List<TInput>();

            await foreach (var input in inputStream)
            {
                batch.Add(input);

                if (batch.Count >= batchSize)
                {
                    var results = await PredictBatchAsync(batch.ToArray());
                    
                    foreach (var result in results)
                    {
                        yield return result;
                    }

                    batch.Clear();
                }
            }

            // Process remaining items
            if (batch.Count > 0)
            {
                var results = await PredictBatchAsync(batch.ToArray());
                
                foreach (var result in results)
                {
                    yield return result;
                }
            }
        }

        /// <summary>
        /// Optimizes the predictor for a specific dataset and usage pattern.
        /// </summary>
        /// <param name="sampleInputs">Sample inputs for optimization.</param>
        /// <param name="optimizationHints">Hints for optimization strategy.</param>
        /// <returns>A task representing the optimization operation.</returns>
        public async Task OptimizeAsync(TInput[] sampleInputs, OptimizationHints? optimizationHints = null)
        {
            ThrowIfDisposed();

            // Profile different device configurations with sample data
            var profileResults = await ProfileDevicesAsync(sampleInputs);

            // Update scheduling strategy based on profiling results
            await _orchestrator.OptimizeSchedulingAsync();

            // Optimize memory layout and transfer patterns
            await _orchestrator.OptimizeMemoryAsync();
        }

        /// <summary>
        /// Provides detailed performance analysis and recommendations.
        /// </summary>
        /// <returns>Performance analysis results.</returns>
        public PerformanceAnalysis AnalyzePerformance()
        {
            ThrowIfDisposed();
            return _orchestrator.AnalyzePerformance();
        }

        private async Task<ITensor<float>> ConvertToTensorAsync(TInput input) =>
            // Implementation would depend on the specific input type
            // This is a simplified version that assumes conversion is possible
            await _context.TensorFactory.CreateFromInputAsync(input);

        private async Task<ITensor<float>[]> ConvertToTensorBatchAsync(TInput[] inputs)
        {
            var tensors = new ITensor<float>[inputs.Length];
            
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = await ConvertToTensorAsync(inputs[i]);
            }

            return tensors;
        }

        private async Task<TOutput> ConvertFromTensorAsync(ITensor<float> tensor) =>
            // Implementation would depend on the specific output type
            await _context.TensorFactory.CreateOutputFromTensorAsync<TOutput>(tensor);

        private async Task<TOutput[]> ConvertFromTensorBatchAsync(ITensor<float>[] tensors)
        {
            var outputs = new TOutput[tensors.Length];
            
            for (int i = 0; i < tensors.Length; i++)
            {
                outputs[i] = await ConvertFromTensorAsync(tensors[i]);
            }

            return outputs;
        }

        private async Task<DeviceProfileResults> ProfileDevicesAsync(TInput[] sampleInputs)
        {
            var results = new Dictionary<ComputeDevice, DeviceProfileResult>();

            foreach (var device in _context.AvailableAccelerators.Keys)
            {
                try
                {
                    var profileResult = await ProfileDeviceAsync(device, sampleInputs);
                    results[device] = profileResult;
                }
                catch (Exception ex)
                {
                    // Log error and continue with other devices
                    results[device] = new DeviceProfileResult(device, false, 0, 0, ex.Message);
                }
            }

            return new DeviceProfileResults(results);
        }

        private async Task<DeviceProfileResult> ProfileDeviceAsync(ComputeDevice device, TInput[] sampleInputs)
        {
            const int warmupRuns = 3;
            const int measurementRuns = 10;

            // Warmup runs
            for (int i = 0; i < warmupRuns; i++)
            {
                foreach (var input in sampleInputs.Take(Math.Min(sampleInputs.Length, 4)))
                {
                    await PredictAsync(input);
                }
            }

            // Measurement runs
            var times = new List<double>();
            var startTime = DateTime.UtcNow;

            for (int i = 0; i < measurementRuns; i++)
            {
                var runStart = DateTime.UtcNow;
                
                foreach (var input in sampleInputs.Take(Math.Min(sampleInputs.Length, 4)))
                {
                    await PredictAsync(input);
                }
                
                var runEnd = DateTime.UtcNow;
                times.Add((runEnd - runStart).TotalMilliseconds);
            }

            var endTime = DateTime.UtcNow;
            var averageTime = times.Average();
            var throughput = sampleInputs.Length * measurementRuns / (endTime - startTime).TotalSeconds;

            return new DeviceProfileResult(device, true, averageTime, throughput);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ILGPUUniversalPredictor<TInput, TOutput>));
        }

        /// <summary>
        /// Disposes the predictor and releases all resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            _orchestrator?.Dispose();
            _scheduler?.Dispose();
            // _model doesn't implement IDisposable

            _disposed = true;
        }
    }

    /// <summary>
    /// Provides context information for ML predictions.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the PredictionContext class.
    /// </remarks>
    public class PredictionContext(
        IReadOnlyDictionary<ComputeDevice, Accelerator> accelerators,
        ITensorFactory tensorFactory,
        IMemoryManager memoryManager)
    {
        /// <summary>
        /// Gets the available accelerators for computation.
        /// </summary>
        public IReadOnlyDictionary<ComputeDevice, Accelerator> AvailableAccelerators { get; } = accelerators ?? throw new ArgumentNullException(nameof(accelerators));

        /// <summary>
        /// Gets the tensor factory for data conversion.
        /// </summary>
        public ITensorFactory TensorFactory { get; } = tensorFactory ?? throw new ArgumentNullException(nameof(tensorFactory));

        /// <summary>
        /// Gets the memory manager for efficient data handling.
        /// </summary>
        public IMemoryManager MemoryManager { get; } = memoryManager ?? throw new ArgumentNullException(nameof(memoryManager));
    }

    /// <summary>
    /// Provides optimization hints for the predictor.
    /// </summary>
    public class OptimizationHints
    {
        /// <summary>
        /// Gets or sets the expected batch size for optimization.
        /// </summary>
        public int ExpectedBatchSize { get; set; } = 1;

        /// <summary>
        /// Gets or sets whether to optimize for latency or throughput.
        /// </summary>
        public OptimizationTarget Target { get; set; } = OptimizationTarget.Throughput;

        /// <summary>
        /// Gets or sets the expected input data characteristics.
        /// </summary>
        public DataCharacteristics InputCharacteristics { get; set; } = DataCharacteristics.Unknown;

        /// <summary>
        /// Gets or sets memory usage constraints.
        /// </summary>
        public MemoryConstraints MemoryConstraints { get; set; } = MemoryConstraints.None;
    }

    /// <summary>
    /// Defines optimization targets.
    /// </summary>
    public enum OptimizationTarget
    {
        /// <summary>
        /// Optimize for minimum latency (best for real-time applications).
        /// </summary>
        Latency,

        /// <summary>
        /// Optimize for maximum throughput (best for batch processing).
        /// </summary>
        Throughput,

        /// <summary>
        /// Balance between latency and throughput.
        /// </summary>
        Balanced,

        /// <summary>
        /// Optimize for energy efficiency.
        /// </summary>
        EnergyEfficient
    }

    /// <summary>
    /// Describes input data characteristics for optimization.
    /// </summary>
    public enum DataCharacteristics
    {
        /// <summary>
        /// Unknown data characteristics.
        /// </summary>
        Unknown,

        /// <summary>
        /// Small, frequent inputs.
        /// </summary>
        SmallFrequent,

        /// <summary>
        /// Large, infrequent inputs.
        /// </summary>
        LargeInfrequent,

        /// <summary>
        /// Streaming data with variable sizes.
        /// </summary>
        Streaming,

        /// <summary>
        /// Batch processing with consistent sizes.
        /// </summary>
        BatchConsistent
    }

    /// <summary>
    /// Defines memory usage constraints.
    /// </summary>
    public enum MemoryConstraints
    {
        /// <summary>
        /// No memory constraints.
        /// </summary>
        None,

        /// <summary>
        /// Low memory usage required.
        /// </summary>
        Low,

        /// <summary>
        /// Moderate memory usage acceptable.
        /// </summary>
        Moderate,

        /// <summary>
        /// Strict memory limits.
        /// </summary>
        Strict
    }
}