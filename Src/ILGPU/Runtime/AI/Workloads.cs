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
using ILGPU.Numerics.AI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Runtime.AI
{
    /// <summary>
    /// Base interface for workloads that can be orchestrated.
    /// </summary>
    public interface IWorkload
    {
        /// <summary>
        /// Gets the workload type.
        /// </summary>
        WorkloadType WorkloadType { get; }

        /// <summary>
        /// Gets the estimated computational complexity.
        /// </summary>
        long EstimatedComplexity { get; }

        /// <summary>
        /// Gets the memory requirements in bytes.
        /// </summary>
        long MemoryRequirements { get; }

        /// <summary>
        /// Executes the workload.
        /// </summary>
        /// <param name="context">The execution context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        Task ExecuteAsync(WorkloadExecutionContext context, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Interface for workloads that can be distributed across multiple accelerators.
    /// </summary>
    public interface IDistributedWorkload : IWorkload
    {
        /// <summary>
        /// Gets whether this workload requires result aggregation.
        /// </summary>
        bool RequiresAggregation { get; }

        /// <summary>
        /// Partitions the workload for distributed execution.
        /// </summary>
        /// <param name="acceleratorCount">The number of accelerators.</param>
        /// <returns>A collection of workload partitions.</returns>
        IEnumerable<IWorkload> Partition(int acceleratorCount);

        /// <summary>
        /// Aggregates results from distributed execution.
        /// </summary>
        /// <param name="partitions">The executed partitions.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the aggregation.</returns>
        Task AggregateResultsAsync(IEnumerable<WorkloadPartition> partitions, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Matrix multiplication workload.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <remarks>
    /// Initializes a new instance of the MatrixMultiplicationWorkload class.
    /// </remarks>
    /// <param name="a">Matrix A.</param>
    /// <param name="b">Matrix B.</param>
    /// <param name="c">Result matrix C (optional).</param>
    public sealed class MatrixMultiplicationWorkload<T>(ITensor<T> a, ITensor<T> b, ITensor<T>? c = null) : IWorkload where T : unmanaged
    {

        /// <summary>
        /// Gets matrix A.
        /// </summary>
        public ITensor<T> A { get; } = a ?? throw new ArgumentNullException(nameof(a));

        /// <summary>
        /// Gets matrix B.
        /// </summary>
        public ITensor<T> B { get; } = b ?? throw new ArgumentNullException(nameof(b));

        /// <summary>
        /// Gets or sets the result matrix C.
        /// </summary>
        public ITensor<T>? C { get; set; } = c;

        /// <summary>
        /// Gets the workload type.
        /// </summary>
        public WorkloadType WorkloadType => WorkloadType.MatrixMultiplication;

        /// <summary>
        /// Gets the estimated computational complexity.
        /// </summary>
        public long EstimatedComplexity { get; } = 2L * a.Shape[0] * b.Shape[1] * a.Shape[1];

        /// <summary>
        /// Gets the memory requirements.
        /// </summary>
        public long MemoryRequirements { get; } = (a.Shape.Length + b.Shape.Length +
                                (c?.Shape.Length ?? a.Shape[0] * b.Shape[1])) * Interop.SizeOf<T>();

        /// <summary>
        /// Executes the matrix multiplication.
        /// </summary>
        /// <param name="context">The execution context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteAsync(WorkloadExecutionContext context, CancellationToken cancellationToken = default)
        {
            var strategy = context.Strategy as SingleAcceleratorStrategy ?? throw new InvalidOperationException("Matrix multiplication requires single accelerator strategy");
            var primitives = PerformancePrimitivesFactory.Create(strategy.TargetAccelerator);
            
            if (C == null)
            {
                var resultShape = new TensorShape(A.Shape[0], B.Shape[1]);
                C = TensorFactory.Create<T>(resultShape, ComputeLocation.Gpu);
            }

            var alpha = GetOne<T>();
            var beta = GetZero<T>();
            
            await primitives.GemmAsync(A, B, C, alpha, beta, cancellationToken).ConfigureAwait(false);
        }

        private static TElement GetOne<TElement>() where TElement : unmanaged
        {
            if (typeof(TElement) == typeof(float)) return (TElement)(object)1.0f;
            if (typeof(TElement) == typeof(double)) return (TElement)(object)1.0;
            return typeof(TElement) == typeof(int)
                ? (TElement)(object)1
                : throw new NotSupportedException($"Type {typeof(TElement)} not supported");
        }

        private static TElement GetZero<TElement>() where TElement : unmanaged
        {
            if (typeof(TElement) == typeof(float)) return (TElement)(object)0.0f;
            if (typeof(TElement) == typeof(double)) return (TElement)(object)0.0;
            return typeof(TElement) == typeof(int)
                ? (TElement)(object)0
                : throw new NotSupportedException($"Type {typeof(TElement)} not supported");
        }
    }

    /// <summary>
    /// Convolution workload.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class ConvolutionWorkload<T> : IWorkload where T : unmanaged
    {
        /// <summary>
        /// Initializes a new instance of the ConvolutionWorkload class.
        /// </summary>
        /// <param name="input">Input tensor.</param>
        /// <param name="kernel">Convolution kernel.</param>
        /// <param name="parameters">Convolution parameters.</param>
        public ConvolutionWorkload(ITensor<T> input, ITensor<T> kernel, ConvolutionParameters parameters)
        {
            Input = input ?? throw new ArgumentNullException(nameof(input));
            Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
            Parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            
            // Estimate complexity based on output size and kernel operations
            var outputHeight = (input.Shape[2] + 2 * parameters.Padding.Height - kernel.Shape[2]) / parameters.Stride.Height + 1;
            var outputWidth = (input.Shape[3] + 2 * parameters.Padding.Width - kernel.Shape[3]) / parameters.Stride.Width + 1;
            EstimatedComplexity = input.Shape[0] * kernel.Shape[0] * outputHeight * outputWidth * 
                                kernel.Shape[1] * kernel.Shape[2] * kernel.Shape[3] * 2L;
            
            MemoryRequirements = (input.Shape.Length + kernel.Shape.Length + 
                               input.Shape[0] * kernel.Shape[0] * outputHeight * outputWidth) * Interop.SizeOf<T>();
        }

        /// <summary>
        /// Gets the input tensor.
        /// </summary>
        public ITensor<T> Input { get; }

        /// <summary>
        /// Gets the convolution kernel.
        /// </summary>
        public ITensor<T> Kernel { get; }

        /// <summary>
        /// Gets the convolution parameters.
        /// </summary>
        public ConvolutionParameters Parameters { get; }

        /// <summary>
        /// Gets or sets the output tensor.
        /// </summary>
        public ITensor<T>? Output { get; set; }

        /// <summary>
        /// Gets the workload type.
        /// </summary>
        public WorkloadType WorkloadType => WorkloadType.Convolution;

        /// <summary>
        /// Gets the estimated computational complexity.
        /// </summary>
        public long EstimatedComplexity { get; }

        /// <summary>
        /// Gets the memory requirements.
        /// </summary>
        public long MemoryRequirements { get; }

        /// <summary>
        /// Executes the convolution.
        /// </summary>
        /// <param name="context">The execution context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public async Task ExecuteAsync(WorkloadExecutionContext context, CancellationToken cancellationToken = default)
        {
            var strategy = context.Strategy as SingleAcceleratorStrategy ?? throw new InvalidOperationException("Convolution requires single accelerator strategy");
            var primitives = PerformancePrimitivesFactory.Create(strategy.TargetAccelerator);
            
            if (Output == null)
            {
                var outputHeight = (Input.Shape[2] + 2 * Parameters.Padding.Height - Kernel.Shape[2]) / Parameters.Stride.Height + 1;
                var outputWidth = (Input.Shape[3] + 2 * Parameters.Padding.Width - Kernel.Shape[3]) / Parameters.Stride.Width + 1;
                var outputShape = new TensorShape(Input.Shape[0], Kernel.Shape[0], outputHeight, outputWidth);
                Output = TensorFactory.Create<T>(outputShape, ComputeLocation.Gpu);
            }

            await primitives.Conv2DAsync(Input, Kernel, Output, Parameters, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Distributed matrix multiplication workload.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <remarks>
    /// Initializes a new instance of the DistributedMatrixMultiplicationWorkload class.
    /// </remarks>
    /// <param name="a">Matrix A.</param>
    /// <param name="b">Matrix B.</param>
    /// <param name="c">Result matrix C (optional).</param>
    public sealed class DistributedMatrixMultiplicationWorkload<T>(ITensor<T> a, ITensor<T> b, ITensor<T>? c = null) : IDistributedWorkload where T : unmanaged
    {

        /// <summary>
        /// Gets matrix A.
        /// </summary>
        public ITensor<T> A { get; } = a ?? throw new ArgumentNullException(nameof(a));

        /// <summary>
        /// Gets matrix B.
        /// </summary>
        public ITensor<T> B { get; } = b ?? throw new ArgumentNullException(nameof(b));

        /// <summary>
        /// Gets or sets the result matrix C.
        /// </summary>
        public ITensor<T>? C { get; set; } = c;

        /// <summary>
        /// Gets the workload type.
        /// </summary>
        public WorkloadType WorkloadType => WorkloadType.MatrixMultiplication;

        /// <summary>
        /// Gets the estimated computational complexity.
        /// </summary>
        public long EstimatedComplexity { get; } = 2L * a.Shape[0] * b.Shape[1] * a.Shape[1];

        /// <summary>
        /// Gets the memory requirements.
        /// </summary>
        public long MemoryRequirements { get; } = (a.Shape.Length + b.Shape.Length +
                                (c?.Shape.Length ?? a.Shape[0] * b.Shape[1])) * Interop.SizeOf<T>();

        /// <summary>
        /// Gets whether this workload requires result aggregation.
        /// </summary>
        public bool RequiresAggregation => true;

        /// <summary>
        /// Partitions the matrix multiplication across accelerators.
        /// </summary>
        /// <param name="acceleratorCount">The number of accelerators.</param>
        /// <returns>A collection of workload partitions.</returns>
        public IEnumerable<IWorkload> Partition(int acceleratorCount)
        {
            // Partition along the M dimension (rows of A and C)
            var rowsPerPartition = A.Shape[0] / acceleratorCount;
            var remainingRows = A.Shape[0] % acceleratorCount;
            
            for (int i = 0; i < acceleratorCount; i++)
            {
                var startRow = i * rowsPerPartition;
                var endRow = startRow + rowsPerPartition + (i == acceleratorCount - 1 ? remainingRows : 0);
                
                // Create partition tensors (would use tensor slicing)
                var aPartition = DistributedMatrixMultiplicationWorkload<T>.CreateTensorSlice(A, startRow, endRow);
                var cPartition = C != null ? DistributedMatrixMultiplicationWorkload<T>.CreateTensorSlice(C, startRow, endRow) : null;
                
                yield return new MatrixMultiplicationWorkload<T>(aPartition, B, cPartition);
            }
        }

        /// <summary>
        /// Aggregates results from distributed execution.
        /// </summary>
        /// <param name="partitions">The executed partitions.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the aggregation.</returns>
        public async Task AggregateResultsAsync(IEnumerable<WorkloadPartition> partitions, CancellationToken cancellationToken = default) =>
            // Results are already in the correct positions in C
            // No additional aggregation needed for row-wise partitioning
            await Task.CompletedTask.ConfigureAwait(false);

        /// <summary>
        /// Executes the workload.
        /// </summary>
        /// <param name="context">The execution context.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        /// <returns>A task representing the operation.</returns>
        public Task ExecuteAsync(WorkloadExecutionContext context, CancellationToken cancellationToken = default) =>
            // For distributed workload, this should not be called directly
            Task.FromException(new InvalidOperationException("Use DistributedWorkloadOrchestrator for distributed execution"));

        private static ITensor<T> CreateTensorSlice(ITensor<T> tensor, int startRow, int endRow) =>
            // This would create a tensor slice/view
            // For now, return the original tensor as placeholder
            tensor;
    }

    /// <summary>
    /// Workload partition for distributed execution.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the WorkloadPartition class.
    /// </remarks>
    /// <param name="workload">The workload partition.</param>
    /// <param name="strategy">The execution strategy.</param>
    public sealed class WorkloadPartition(IWorkload workload, ExecutionStrategy strategy)
    {

        /// <summary>
        /// Gets the workload.
        /// </summary>
        public IWorkload Workload { get; } = workload ?? throw new ArgumentNullException(nameof(workload));

        /// <summary>
        /// Gets the execution strategy.
        /// </summary>
        public ExecutionStrategy Strategy { get; } = strategy ?? throw new ArgumentNullException(nameof(strategy));
    }

    /// <summary>
    /// Workload scheduler for determining optimal execution strategies.
    /// </summary>
    public sealed class WorkloadScheduler
    {
        /// <summary>
        /// Analyzes a workload and determines the optimal execution strategy.
        /// </summary>
        /// <param name="workload">The workload to analyze.</param>
        /// <param name="profiles">Available accelerator profiles.</param>
        /// <returns>The optimal execution strategy.</returns>
        public static async Task<ExecutionStrategy> AnalyzeWorkloadAsync(
            IWorkload workload,
            IReadOnlyList<AcceleratorProfile> profiles)
        {
            await Task.CompletedTask.ConfigureAwait(false); // Placeholder for async analysis
            
            // Simple strategy selection based on workload type and complexity
            if (workload.EstimatedComplexity > 1000000 && profiles.Count > 1)
            {
                // Use multiple accelerators for large workloads
                var suitableAccelerators = profiles
                    .Where(p => p.CanExecute(workload))
                    .Take(2) // Limit to 2 accelerators for now
                    .Select(p => p.Accelerator)
                    .ToList();
                
                if (suitableAccelerators.Count > 1)
                    return new MultiAcceleratorStrategy(suitableAccelerators);
            }
            
            // Default to single accelerator
            var bestProfile = profiles.FirstOrDefault(p => p.CanExecute(workload)) ?? profiles.First();
            return new SingleAcceleratorStrategy(bestProfile.Accelerator);
        }

        /// <summary>
        /// Partitions a distributed workload across accelerators.
        /// </summary>
        /// <param name="workload">The distributed workload.</param>
        /// <param name="profiles">Available accelerator profiles.</param>
        /// <returns>A collection of workload partitions.</returns>
        public static async Task<IEnumerable<WorkloadPartition>> PartitionWorkloadAsync(
            IDistributedWorkload workload,
            IReadOnlyList<AcceleratorProfile> profiles)
        {
            await Task.CompletedTask.ConfigureAwait(false); // Placeholder for async partitioning
            
            var suitableProfiles = profiles.Where(p => p.CanExecute(workload)).ToList();
            var partitions = workload.Partition(suitableProfiles.Count).ToList();
            
            var result = new List<WorkloadPartition>();
            for (int i = 0; i < partitions.Count && i < suitableProfiles.Count; i++)
            {
                var strategy = new SingleAcceleratorStrategy(suitableProfiles[i].Accelerator);
                result.Add(new WorkloadPartition(partitions[i], strategy));
            }
            
            return result;
        }
    }

    /// <summary>
    /// Performance tracker for monitoring execution performance.
    /// </summary>
    public sealed class PerformanceTracker : IDisposable
    {
        private readonly Dictionary<AcceleratorType, AcceleratorPerformanceMetrics> _metrics;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the PerformanceTracker class.
        /// </summary>
        public PerformanceTracker()
        {
            _metrics = [];
        }

        /// <summary>
        /// Records an operation execution.
        /// </summary>
        /// <param name="acceleratorType">The accelerator type.</param>
        /// <param name="primitiveType">The primitive type.</param>
        /// <param name="operationCount">The number of operations.</param>
        public void RecordOperation(AcceleratorType acceleratorType, PrimitiveType primitiveType, long operationCount)
        {
            if (!_metrics.TryGetValue(acceleratorType, out var metrics))
            {
                metrics = new AcceleratorPerformanceMetrics();
                _metrics[acceleratorType] = metrics;
            }
            
            metrics.RecordOperation(primitiveType, operationCount);
        }

        /// <summary>
        /// Records a workload execution.
        /// </summary>
        /// <param name="workloadType">The workload type.</param>
        /// <param name="duration">The execution duration.</param>
        /// <param name="success">Whether the execution was successful.</param>
        public static void RecordExecution(WorkloadType workloadType, TimeSpan duration, bool success)
        {
            // Record execution metrics
        }

        /// <summary>
        /// Gets performance metrics for an accelerator type.
        /// </summary>
        /// <param name="acceleratorType">The accelerator type.</param>
        /// <returns>The performance metrics.</returns>
        public AcceleratorPerformanceMetrics? GetMetrics(AcceleratorType acceleratorType) => _metrics.TryGetValue(acceleratorType, out var metrics) ? metrics : null;

        /// <summary>
        /// Disposes the performance tracker.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _metrics.Clear();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Performance metrics for an accelerator.
    /// </summary>
    public sealed class AcceleratorPerformanceMetrics
    {
        private readonly Dictionary<PrimitiveType, PrimitiveMetrics> _primitiveMetrics;

        /// <summary>
        /// Initializes a new instance of the AcceleratorPerformanceMetrics class.
        /// </summary>
        public AcceleratorPerformanceMetrics()
        {
            _primitiveMetrics = [];
        }

        /// <summary>
        /// Records an operation for a primitive type.
        /// </summary>
        /// <param name="primitiveType">The primitive type.</param>
        /// <param name="operationCount">The number of operations.</param>
        public void RecordOperation(PrimitiveType primitiveType, long operationCount)
        {
            if (!_primitiveMetrics.TryGetValue(primitiveType, out var metrics))
            {
                metrics = new PrimitiveMetrics();
                _primitiveMetrics[primitiveType] = metrics;
            }
            
            metrics.OperationCount += operationCount;
            metrics.ExecutionCount++;
        }

        /// <summary>
        /// Gets metrics for a primitive type.
        /// </summary>
        /// <param name="primitiveType">The primitive type.</param>
        /// <returns>The primitive metrics.</returns>
        public PrimitiveMetrics? GetPrimitiveMetrics(PrimitiveType primitiveType) => _primitiveMetrics.TryGetValue(primitiveType, out var metrics) ? metrics : null;
    }

    /// <summary>
    /// Metrics for a specific primitive type.
    /// </summary>
    public sealed class PrimitiveMetrics
    {
        /// <summary>
        /// Gets or sets the total operation count.
        /// </summary>
        public long OperationCount { get; set; }

        /// <summary>
        /// Gets or sets the execution count.
        /// </summary>
        public long ExecutionCount { get; set; }

        /// <summary>
        /// Gets the average operations per execution.
        /// </summary>
        public double AverageOperationsPerExecution => 
            ExecutionCount > 0 ? (double)OperationCount / ExecutionCount : 0;
    }
}