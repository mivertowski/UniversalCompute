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
using ILGPU.Numerics.AI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Runtime.AI
{
    /// <summary>
    /// Intelligent workload orchestrator that distributes computations across available accelerators.
    /// </summary>
    public sealed class WorkloadOrchestrator : IDisposable
    {
        private readonly Context _context;
        private readonly List<AcceleratorProfile> _acceleratorProfiles;
        private readonly WorkloadScheduler _scheduler;
        private readonly PerformanceTracker _performanceTracker;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the WorkloadOrchestrator class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="accelerators">The accelerators to orchestrate across.</param>
        public WorkloadOrchestrator(Context context, IEnumerable<Accelerator> accelerators)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _acceleratorProfiles = new List<AcceleratorProfile>();
            _scheduler = new WorkloadScheduler();
            _performanceTracker = new PerformanceTracker();

            InitializeAcceleratorProfiles(accelerators);
        }

        /// <summary>
        /// Gets the available accelerator profiles.
        /// </summary>
        public IReadOnlyList<AcceleratorProfile> AcceleratorProfiles => _acceleratorProfiles;

        /// <summary>
        /// Gets the performance tracker.
        /// </summary>
        public PerformanceTracker PerformanceTracker => _performanceTracker;

        /// <summary>
        /// Executes a workload across optimal accelerators.
        /// </summary>
        /// <param name="workload">The workload to execute.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteWorkloadAsync(
            IWorkload workload,
            CancellationToken cancellationToken = default)
        {
            if (workload == null)
                throw new ArgumentNullException(nameof(workload));

            // Analyze workload and determine optimal execution strategy
            var strategy = await _scheduler.AnalyzeWorkloadAsync(workload, _acceleratorProfiles);
            
            // Execute workload using the determined strategy
            var executionContext = new WorkloadExecutionContext(strategy, _performanceTracker);
            
            await ExecuteStrategyAsync(workload, executionContext, cancellationToken);
        }

        /// <summary>
        /// Executes a matrix multiplication workload with automatic accelerator selection.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Result matrix C.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task<ITensor<T>> ExecuteMatMulAsync<T>(
            ITensor<T> a,
            ITensor<T> b,
            ITensor<T>? c = null,
            CancellationToken cancellationToken = default) where T : unmanaged
        {
            // Create workload
            var workload = new MatrixMultiplicationWorkload<T>(a, b, c);
            
            // Find optimal accelerator for this operation
            var profile = SelectOptimalAccelerator(workload);
            
            // Execute on selected accelerator
            var primitives = PerformancePrimitivesFactory.Create(profile.Accelerator);
            
            if (c == null)
            {
                var resultShape = new TensorShape(a.Shape[0], b.Shape[1]);
                c = TensorFactory.Create<T>(resultShape, ComputeLocation.Gpu);
            }
            
            await primitives.GemmAsync(a, b, c, GetOne<T>(), GetZero<T>(), cancellationToken);
            
            // Track performance
            _performanceTracker.RecordOperation(profile.Accelerator.AcceleratorType, 
                PrimitiveType.MatrixMultiplication, a.Shape.Length);
            
            return c;
        }

        /// <summary>
        /// Executes a convolution workload with automatic accelerator selection.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="input">Input tensor.</param>
        /// <param name="kernel">Convolution kernel.</param>
        /// <param name="parameters">Convolution parameters.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>The output tensor.</returns>
        public async Task<ITensor<T>> ExecuteConvolutionAsync<T>(
            ITensor<T> input,
            ITensor<T> kernel,
            ConvolutionParameters parameters,
            CancellationToken cancellationToken = default) where T : unmanaged
        {
            // Create workload
            var workload = new ConvolutionWorkload<T>(input, kernel, parameters);
            
            // Find optimal accelerator
            var profile = SelectOptimalAccelerator(workload);
            
            // Calculate output shape
            var outputShape = CalculateConvolutionOutputShape(input.Shape, kernel.Shape, parameters);
            var output = TensorFactory.Create<T>(outputShape, ComputeLocation.Gpu);
            
            // Execute convolution
            var primitives = PerformancePrimitivesFactory.Create(profile.Accelerator);
            await primitives.Conv2DAsync(input, kernel, output, parameters, cancellationToken);
            
            // Track performance
            _performanceTracker.RecordOperation(profile.Accelerator.AcceleratorType,
                PrimitiveType.Convolution, input.Shape.Length);
            
            return output;
        }

        /// <summary>
        /// Executes distributed workload across multiple accelerators.
        /// </summary>
        /// <param name="workload">The workload to distribute.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteDistributedWorkloadAsync(
            IDistributedWorkload workload,
            CancellationToken cancellationToken = default)
        {
            if (workload == null)
                throw new ArgumentNullException(nameof(workload));

            // Partition workload across available accelerators
            var partitions = await _scheduler.PartitionWorkloadAsync(workload, _acceleratorProfiles);
            
            // Execute partitions in parallel
            var tasks = partitions.Select(partition => 
                ExecutePartitionAsync(partition, cancellationToken));
            
            await Task.WhenAll(tasks);
            
            // Aggregate results if needed
            if (workload.RequiresAggregation)
            {
                await workload.AggregateResultsAsync(partitions);
            }
        }

        private void InitializeAcceleratorProfiles(IEnumerable<Accelerator> accelerators)
        {
            foreach (var accelerator in accelerators)
            {
                var primitives = PerformancePrimitivesFactory.Create(accelerator);
                var profile = new AcceleratorProfile(accelerator, primitives);
                
                // Benchmark accelerator if needed
                BenchmarkAccelerator(profile);
                
                _acceleratorProfiles.Add(profile);
            }
            
            // Sort by performance score
            _acceleratorProfiles.Sort((a, b) => b.PerformanceScore.CompareTo(a.PerformanceScore));
        }

        private void BenchmarkAccelerator(AcceleratorProfile profile)
        {
            // Quick benchmark to determine relative performance
            var score = 0.0;
            
            // Base score from capabilities
            var caps = profile.Primitives.Capabilities;
            score += caps.PeakTFLOPS * 10;
            
            if (caps.SupportsAcceleratedGemm) score += 100;
            if (caps.SupportsAcceleratedConvolution) score += 50;
            if (caps.SupportsAcceleratedAttention) score += 75;
            if (caps.HasTensorCores) score += 200;
            if (caps.SupportsFP16) score += 25;
            if (caps.SupportsBFloat16) score += 30;
            if (caps.SupportsInt8) score += 20;
            
            profile.PerformanceScore = score;
        }

        private AcceleratorProfile SelectOptimalAccelerator(IWorkload workload)
        {
            var suitableProfiles = _acceleratorProfiles
                .Where(p => p.CanExecute(workload))
                .ToList();
            
            if (suitableProfiles.Count == 0)
                return _acceleratorProfiles.First(); // Fallback to first available
            
            // Select based on workload type and current load
            return workload.WorkloadType switch
            {
                WorkloadType.MatrixMultiplication => suitableProfiles
                    .Where(p => p.Primitives.Capabilities.SupportsAcceleratedGemm)
                    .FirstOrDefault() ?? suitableProfiles.First(),
                WorkloadType.Convolution => suitableProfiles
                    .Where(p => p.Primitives.Capabilities.SupportsAcceleratedConvolution)
                    .FirstOrDefault() ?? suitableProfiles.First(),
                WorkloadType.Attention => suitableProfiles
                    .Where(p => p.Primitives.Capabilities.SupportsAcceleratedAttention)
                    .FirstOrDefault() ?? suitableProfiles.First(),
                _ => suitableProfiles.First()
            };
        }

        private async Task ExecuteStrategyAsync(
            IWorkload workload,
            WorkloadExecutionContext context,
            CancellationToken cancellationToken)
        {
            var startTime = DateTime.UtcNow;
            
            try
            {
                await workload.ExecuteAsync(context, cancellationToken);
                
                // Record successful execution
                var duration = DateTime.UtcNow - startTime;
                _performanceTracker.RecordExecution(workload.WorkloadType, duration, true);
            }
            catch (Exception ex)
            {
                // Record failed execution
                var duration = DateTime.UtcNow - startTime;
                _performanceTracker.RecordExecution(workload.WorkloadType, duration, false);
                throw;
            }
        }

        private async Task ExecutePartitionAsync(
            WorkloadPartition partition,
            CancellationToken cancellationToken)
        {
            var context = new WorkloadExecutionContext(partition.Strategy, _performanceTracker);
            await partition.Workload.ExecuteAsync(context, cancellationToken);
        }

        private static TensorShape CalculateConvolutionOutputShape(
            TensorShape inputShape,
            TensorShape kernelShape,
            ConvolutionParameters parameters)
        {
            var batch = inputShape[0];
            var inputHeight = inputShape[2];
            var inputWidth = inputShape[3];
            var outputChannels = kernelShape[0];
            
            var outputHeight = (inputHeight + 2 * parameters.Padding.Height - kernelShape[2]) / parameters.Stride.Height + 1;
            var outputWidth = (inputWidth + 2 * parameters.Padding.Width - kernelShape[3]) / parameters.Stride.Width + 1;
            
            return new TensorShape(batch, outputChannels, outputHeight, outputWidth);
        }

        private static T GetOne<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)1.0f;
            if (typeof(T) == typeof(double)) return (T)(object)1.0;
            if (typeof(T) == typeof(int)) return (T)(object)1;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        private static T GetZero<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float)) return (T)(object)0.0f;
            if (typeof(T) == typeof(double)) return (T)(object)0.0;
            if (typeof(T) == typeof(int)) return (T)(object)0;
            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        /// <summary>
        /// Disposes the workload orchestrator.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _performanceTracker?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Accelerator profile containing performance characteristics.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the AcceleratorProfile class.
    /// </remarks>
    /// <param name="accelerator">The accelerator.</param>
    /// <param name="primitives">The performance primitives.</param>
    public sealed class AcceleratorProfile(Accelerator accelerator, IPerformancePrimitives primitives)
    {

        /// <summary>
        /// Gets the accelerator.
        /// </summary>
        public Accelerator Accelerator { get; } = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

        /// <summary>
        /// Gets the performance primitives.
        /// </summary>
        public IPerformancePrimitives Primitives { get; } = primitives ?? throw new ArgumentNullException(nameof(primitives));

        /// <summary>
        /// Gets or sets the performance score.
        /// </summary>
        public double PerformanceScore { get; set; }

        /// <summary>
        /// Gets or sets the current load factor (0.0 to 1.0).
        /// </summary>
        public double LoadFactor { get; set; }

        /// <summary>
        /// Checks if this accelerator can execute the given workload.
        /// </summary>
        /// <param name="workload">The workload.</param>
        /// <returns>True if can execute; otherwise, false.</returns>
        public bool CanExecute(IWorkload workload)
        {
            if (workload == null) return false;
            
            return workload.WorkloadType switch
            {
                WorkloadType.MatrixMultiplication => Primitives.Capabilities.SupportsAcceleratedGemm,
                WorkloadType.Convolution => Primitives.Capabilities.SupportsAcceleratedConvolution,
                WorkloadType.Attention => Primitives.Capabilities.SupportsAcceleratedAttention,
                _ => true // Generic workloads can run on any accelerator
            };
        }
    }

    /// <summary>
    /// Workload execution context.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the WorkloadExecutionContext class.
    /// </remarks>
    /// <param name="strategy">The execution strategy.</param>
    /// <param name="performanceTracker">The performance tracker.</param>
    public sealed class WorkloadExecutionContext(ExecutionStrategy strategy, PerformanceTracker performanceTracker)
    {

        /// <summary>
        /// Gets the execution strategy.
        /// </summary>
        public ExecutionStrategy Strategy { get; } = strategy ?? throw new ArgumentNullException(nameof(strategy));

        /// <summary>
        /// Gets the performance tracker.
        /// </summary>
        public PerformanceTracker PerformanceTracker { get; } = performanceTracker ?? throw new ArgumentNullException(nameof(performanceTracker));
    }

    /// <summary>
    /// Execution strategy for workloads.
    /// </summary>
    public abstract class ExecutionStrategy
    {
        /// <summary>
        /// Gets the strategy type.
        /// </summary>
        public abstract StrategyType Type { get; }

        /// <summary>
        /// Gets the target accelerators.
        /// </summary>
        public abstract IReadOnlyList<Accelerator> TargetAccelerators { get; }
    }

    /// <summary>
    /// Single accelerator execution strategy.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the SingleAcceleratorStrategy class.
    /// </remarks>
    /// <param name="accelerator">The target accelerator.</param>
    public sealed class SingleAcceleratorStrategy(Accelerator accelerator) : ExecutionStrategy
    {

        /// <summary>
        /// Gets the target accelerator.
        /// </summary>
        public Accelerator TargetAccelerator { get; } = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

        /// <summary>
        /// Gets the strategy type.
        /// </summary>
        public override StrategyType Type => StrategyType.SingleAccelerator;

        /// <summary>
        /// Gets the target accelerators.
        /// </summary>
        public override IReadOnlyList<Accelerator> TargetAccelerators => new[] { TargetAccelerator };
    }

    /// <summary>
    /// Multi-accelerator execution strategy.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the MultiAcceleratorStrategy class.
    /// </remarks>
    /// <param name="accelerators">The target accelerators.</param>
    public sealed class MultiAcceleratorStrategy(IEnumerable<Accelerator> accelerators) : ExecutionStrategy
    {

        /// <summary>
        /// Gets the accelerators.
        /// </summary>
        public List<Accelerator> Accelerators { get; } = accelerators?.ToList() ?? throw new ArgumentNullException(nameof(accelerators));

        /// <summary>
        /// Gets the strategy type.
        /// </summary>
        public override StrategyType Type => StrategyType.MultiAccelerator;

        /// <summary>
        /// Gets the target accelerators.
        /// </summary>
        public override IReadOnlyList<Accelerator> TargetAccelerators => Accelerators;
    }

    /// <summary>
    /// Strategy types.
    /// </summary>
    public enum StrategyType
    {
        /// <summary>
        /// Execute on a single accelerator.
        /// </summary>
        SingleAccelerator,

        /// <summary>
        /// Execute across multiple accelerators.
        /// </summary>
        MultiAccelerator,

        /// <summary>
        /// Pipeline execution across accelerators.
        /// </summary>
        Pipeline
    }

    /// <summary>
    /// Workload types for orchestration.
    /// </summary>
    public enum WorkloadType
    {
        /// <summary>
        /// Matrix multiplication workload.
        /// </summary>
        MatrixMultiplication,

        /// <summary>
        /// Convolution workload.
        /// </summary>
        Convolution,

        /// <summary>
        /// Attention mechanism workload.
        /// </summary>
        Attention,

        /// <summary>
        /// Activation function workload.
        /// </summary>
        Activation,

        /// <summary>
        /// Normalization workload.
        /// </summary>
        Normalization,

        /// <summary>
        /// Generic compute workload.
        /// </summary>
        Generic
    }
}