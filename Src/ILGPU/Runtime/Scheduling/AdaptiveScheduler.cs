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

using ILGPU.ML.Integration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ILGPU.Runtime.Scheduling
{
    /// <summary>
    /// Adaptive scheduler that intelligently distributes compute operations across
    /// heterogeneous accelerators based on performance characteristics and workload analysis.
    /// </summary>
    public class AdaptiveScheduler : IDisposable
    {
        private readonly IReadOnlyDictionary<ComputeDevice, Accelerator> _devices;
        private readonly PerformanceProfiler _profiler;
        private readonly LoadBalancer _loadBalancer;
        private readonly Dictionary<ComputeDevice, DevicePerformance> _devicePerformance;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the AdaptiveScheduler class.
        /// </summary>
        /// <param name="devices">Available compute devices and their accelerators.</param>
        /// <param name="policy">Scheduling policy to use.</param>
        public AdaptiveScheduler(
            IReadOnlyDictionary<ComputeDevice, Accelerator> devices,
            SchedulingPolicy policy = SchedulingPolicy.PerformanceOptimized)
        {
            _devices = devices ?? throw new ArgumentNullException(nameof(devices));
            Policy = policy;
            _profiler = new PerformanceProfiler();
            _loadBalancer = new LoadBalancer();
            _devicePerformance = [];

            // Initialize device performance characteristics
            InitializeDevicePerformance();
        }

        /// <summary>
        /// Gets the available compute devices.
        /// </summary>
        public IEnumerable<ComputeDevice> AvailableDevices => _devices.Keys;

        /// <summary>
        /// Gets the current scheduling policy.
        /// </summary>
        public SchedulingPolicy Policy { get; }

        /// <summary>
        /// Gets performance statistics for all devices.
        /// </summary>
        public IReadOnlyDictionary<ComputeDevice, DevicePerformance> DevicePerformance => _devicePerformance;

        /// <summary>
        /// Creates an optimized execution plan for the given compute graph.
        /// </summary>
        /// <param name="graph">The compute graph to schedule.</param>
        /// <returns>An execution plan that optimizes performance across all devices.</returns>
        public async Task<ExecutionPlan> CreateExecutionPlanAsync(ComputeGraph graph)
        {
            ThrowIfDisposed();

            if (graph == null)
                throw new ArgumentNullException(nameof(graph));

            // Optimize the graph structure
            graph.Optimize();

            // Analyze workload characteristics
            var workloadAnalysis = AnalyzeWorkload(graph);

            // Create device assignments
            var assignments = await CreateDeviceAssignmentsAsync(graph, workloadAnalysis).ConfigureAwait(false);

            // Generate execution schedule
            var schedule = GenerateExecutionSchedule(graph, assignments);

            // Optimize memory transfers
            var memoryPlan = OptimizeMemoryTransfers(graph, assignments);

            return new ExecutionPlan(graph, assignments, schedule, memoryPlan, workloadAnalysis);
        }

        /// <summary>
        /// Executes the given execution plan across all assigned devices.
        /// </summary>
        /// <param name="plan">The execution plan to execute.</param>
        /// <returns>A task representing the execution operation.</returns>
        public async Task ExecuteAsync(ExecutionPlan plan)
        {
            ThrowIfDisposed();

            if (plan == null)
                throw new ArgumentNullException(nameof(plan));

            // Start performance monitoring
            var executionId = $"execution_{DateTime.UtcNow.Ticks}";
            _profiler.StartExecution(executionId);

            try
            {
                // Execute memory transfers first
                await ExecuteMemoryTransfersAsync(plan.MemoryPlan).ConfigureAwait(false);

                // Execute compute operations according to schedule
                await ExecuteComputeOperationsAsync(plan.Schedule).ConfigureAwait(false);

                // Update performance statistics
                _profiler.EndExecution(executionId);
            }
            catch (Exception)
            {
                _profiler.EndExecution(executionId);
                throw;
            }
        }

        /// <summary>
        /// Selects the best device for a specific operation type and size.
        /// </summary>
        /// <param name="operation">The operation to be executed.</param>
        /// <returns>The recommended device for the operation.</returns>
        public ComputeDevice SelectBestDevice(IComputeOperation operation)
        {
            ThrowIfDisposed();

            return operation switch
            {
                // Large matrix multiplication - prefer devices with tensor cores
                MatMulOp op when op.Size > 1024 * 1024 => 
                    SelectBestTensorDevice(),
                
                // Convolution operations - prefer specialized AI accelerators
                ConvolutionOp conv => 
                    SelectBestConvolutionDevice(conv),
                
                // Small matrix operations - prefer fast matrix extensions
                MatMulOp op when op.Size < 1024 => 
                    SelectBestDevice(ComputeCapability.MatrixExtensions),
                
                // Vector operations - prefer SIMD-capable devices
                VectorOp => 
                    SelectBestDevice(ComputeCapability.SIMD),
                
                // Memory-bound operations - prefer high bandwidth devices
                MemoryOp => 
                    SelectBestDevice(ComputeCapability.HighBandwidth),
                
                _ => SelectBestDevice(ComputeCapability.General)
            };
        }

        private void InitializeDevicePerformance()
        {
            foreach (var (device, accelerator) in _devices)
            {
                var performance = _profiler.ProfileDevice(device, accelerator);
                _devicePerformance[device] = performance;
            }
        }

        private WorkloadAnalysis AnalyzeWorkload(ComputeGraph graph)
        {
            var totalFlops = 0.0;
            var totalMemoryOps = 0L;
            var operationTypes = new Dictionary<Type, int>();
            var dataTransferSize = 0L;

            foreach (var node in graph.Nodes)
            {
                // Count operation types
                var opType = node.Operation.GetType();
                operationTypes[opType] = operationTypes.GetValueOrDefault(opType, 0) + 1;

                // Accumulate computational complexity
                totalFlops += node.Operation.EstimatedFlops;
                totalMemoryOps += node.Operation.MemoryOperations;
            }

            foreach (var dependency in graph.Dependencies)
            {
                dataTransferSize += dependency.DataSize;
            }

            var computeIntensity = totalMemoryOps > 0 ? totalFlops / totalMemoryOps : 0.0;
            var parallelismLevel = CalculateParallelismLevel(graph);

            return new WorkloadAnalysis(
                TimeSpan.FromMilliseconds(totalFlops / 1000.0), // Convert FLOPS to estimated duration
                totalMemoryOps,
                WorkloadType.Compute,
                (int)parallelismLevel,
                computeIntensity,
                $"Mixed workload with {operationTypes.Count} operation types");
        }

        private async Task<Dictionary<ComputeNode, ComputeDevice>> CreateDeviceAssignmentsAsync(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            // Apply scheduling policy
            switch (Policy)
            {
                case SchedulingPolicy.PerformanceOptimized:
                    assignments = CreatePerformanceOptimizedAssignments(graph, analysis);
                    break;
                    
                case SchedulingPolicy.EnergyEfficient:
                    assignments = CreateEnergyEfficientAssignments(graph, analysis);
                    break;
                    
                case SchedulingPolicy.LoadBalanced:
                    assignments = await CreateLoadBalancedAssignmentsAsync(graph, analysis).ConfigureAwait(false);
                    break;
                    
                case SchedulingPolicy.LatencyOptimized:
                    assignments = CreateLatencyOptimizedAssignments(graph, analysis);
                    break;
                    
                default:
                    assignments = CreateDefaultAssignments(graph);
                    break;
            }

            return assignments;
        }

        private Dictionary<ComputeNode, ComputeDevice> CreatePerformanceOptimizedAssignments(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            foreach (var node in graph.Nodes)
            {
                var bestDevice = SelectBestDevice(node.Operation);
                assignments[node] = bestDevice;
            }

            return assignments;
        }

        private Dictionary<ComputeNode, ComputeDevice> CreateEnergyEfficientAssignments(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            foreach (var node in graph.Nodes)
            {
                // Select device with best performance per watt
                var bestDeviceKvp = _devicePerformance
                    .Where(kvp => CanExecuteOperation(kvp.Key, node.Operation))
                    .OrderByDescending(kvp => kvp.Value.PerformancePerWatt)
                    .FirstOrDefault();
                var bestDevice = bestDeviceKvp.Key != default ? bestDeviceKvp.Key : ComputeDevice.CPU;

                assignments[node] = bestDevice;
            }

            return assignments;
        }

        private async Task<Dictionary<ComputeNode, ComputeDevice>> CreateLoadBalancedAssignmentsAsync(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();
            var deviceLoads = _devices.Keys.ToDictionary(d => d, d => 0.0);

            var orderedNodes = graph.GetTopologicalOrder();

            foreach (var node in orderedNodes)
            {
                // Find device with minimum current load that can execute this operation
                var availableDevices = _devices.Keys
                    .Where(d => CanExecuteOperation(d, node.Operation))
                    .ToList();

                if (availableDevices.Count != 0)
                {
                    var bestDevice = availableDevices
                        .OrderBy(d => deviceLoads[d])
                        .First();

                    assignments[node] = bestDevice;
                    deviceLoads[bestDevice] += node.EstimatedTimeMs;
                }
                else
                {
                    // Fallback to CPU
                    assignments[node] = ComputeDevice.CPU;
                }
            }

            return assignments;
        }

        private Dictionary<ComputeNode, ComputeDevice> CreateLatencyOptimizedAssignments(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            foreach (var node in graph.Nodes)
            {
                // Select device with lowest latency for this operation type
                var bestDeviceKvp = _devicePerformance
                    .Where(kvp => CanExecuteOperation(kvp.Key, node.Operation))
                    .OrderBy(kvp => kvp.Value.AverageLatencyMs)
                    .FirstOrDefault();
                var bestDevice = bestDeviceKvp.Key != default ? bestDeviceKvp.Key : ComputeDevice.CPU;

                assignments[node] = bestDevice;
            }

            return assignments;
        }

        private Dictionary<ComputeNode, ComputeDevice> CreateDefaultAssignments(ComputeGraph graph)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            foreach (var node in graph.Nodes)
            {
                assignments[node] = node.PreferredDevice;
            }

            return assignments;
        }

        private ComputeDevice SelectBestTensorDevice()
        {
            var bestDeviceKvp = _devicePerformance
                .Where(kvp => kvp.Value.SupportsTensorCores)
                .OrderByDescending(kvp => kvp.Value.TensorPerformanceGFLOPS)
                .FirstOrDefault();
            return bestDeviceKvp.Key != default ? bestDeviceKvp.Key : ComputeDevice.GPU;
        }

        private ComputeDevice SelectBestConvolutionDevice(ConvolutionOp conv)
        {
            // Prefer AI/ML accelerators for convolution operations
            var aiDevices = _devicePerformance
                .Where(kvp => kvp.Value.HasAIAcceleration)
                .OrderByDescending(kvp => kvp.Value.AIPerformanceGOPS)
                .ToList();

            if (aiDevices.Count != 0)
                return aiDevices.First().Key;

            // Fallback to best tensor device
            return SelectBestTensorDevice();
        }

        private ComputeDevice SelectBestDevice(ComputeCapability capability) => capability switch
        {
            ComputeCapability.MatrixExtensions =>
                GetBestDevice(kvp => kvp.Value.SupportsMatrixExtensions,
                             kvp => kvp.Value.MatrixPerformanceGFLOPS,
                             ComputeDevice.CPU),

            ComputeCapability.SIMD =>
                GetBestDevice(kvp => kvp.Value.SIMDWidthBits > 0,
                             kvp => kvp.Value.SIMDPerformanceGFLOPS,
                             ComputeDevice.CPU),

            ComputeCapability.HighBandwidth =>
                GetBestDevice(_ => true,
                             kvp => kvp.Value.MemoryBandwidthGBps,
                             ComputeDevice.GPU),

            _ => ComputeDevice.Auto
        };

        private ComputeDevice GetBestDevice(
            Func<KeyValuePair<ComputeDevice, DevicePerformance>, bool> filter,
            Func<KeyValuePair<ComputeDevice, DevicePerformance>, double> selector,
            ComputeDevice fallback)
        {
            var bestDeviceKvp = _devicePerformance
                .Where(filter)
                .OrderByDescending(selector)
                .FirstOrDefault();
            return bestDeviceKvp.Key != default ? bestDeviceKvp.Key : fallback;
        }

        private bool CanExecuteOperation(ComputeDevice device, IComputeOperation operation)
        {
            if (!_devicePerformance.TryGetValue(device, out var performance))
                return false;

            return operation switch
            {
                MatMulOp => performance.SupportsMatrixOperations,
                ConvolutionOp => performance.SupportsConvolution,
                VectorOp => performance.SIMDWidthBits > 0,
                _ => true
            };
        }

        private double CalculateParallelismLevel(ComputeGraph graph)
        {
            var levels = graph.GetParallelExecutionLevels().ToList();
            var totalNodes = graph.Nodes.Count;
            var maxParallelNodes = levels.Max(level => level.Count());

            return totalNodes > 0 ? (double)maxParallelNodes / totalNodes : 0.0;
        }

        private ExecutionSchedule GenerateExecutionSchedule(
            ComputeGraph graph, 
            Dictionary<ComputeNode, ComputeDevice> assignments)
        {
            var levels = graph.GetParallelExecutionLevels().ToList();
            var scheduledLevels = new List<ScheduledExecutionLevel>();

            for (int i = 0; i < levels.Count; i++)
            {
                var level = levels[i];
                var scheduledNodes = level.Select(node => new ScheduledNode(
                    node.Id, 
                    _devices[assignments[node]], 
                    EstimateExecutionTime(node, assignments[node]))).ToList();

                scheduledLevels.Add(new ScheduledExecutionLevel(i, scheduledNodes));
            }

            return new ExecutionSchedule(scheduledLevels.SelectMany(l => l.Nodes).ToList());
        }

        private MemoryTransferPlan OptimizeMemoryTransfers(
            ComputeGraph graph, 
            Dictionary<ComputeNode, ComputeDevice> assignments)
        {
            var transfers = new List<MemoryTransfer>();

            foreach (var dependency in graph.Dependencies)
            {
                var sourceDevice = assignments[dependency.Producer];
                var targetDevice = assignments[dependency.Consumer];

                if (sourceDevice != targetDevice)
                {
                    var transfer = new MemoryTransfer(
                        _devices[sourceDevice],
                        _devices[targetDevice],
                        dependency.DataSize,
                        (int)dependency.AccessPattern);

                    transfers.Add(transfer);
                }
            }

            return new MemoryTransferPlan(transfers);
        }

        private async Task ExecuteMemoryTransfersAsync(MemoryTransferPlan plan)
        {
            var transferTasks = plan.Transfers
                .GroupBy(t => t.Priority)
                .OrderByDescending(g => g.Key)
                .SelectMany(g => g.Select(ExecuteMemoryTransferAsync));

            await Task.WhenAll(transferTasks).ConfigureAwait(false);
        }

        private async Task ExecuteComputeOperationsAsync(ExecutionSchedule schedule)
        {
            foreach (var level in schedule.Levels)
            {
                var levelTasks = level.Nodes.Select(ExecuteNodeAsync);
                await Task.WhenAll(levelTasks).ConfigureAwait(false);
            }
        }

        private async Task ExecuteMemoryTransferAsync(MemoryTransfer transfer) =>
            // Implementation would perform actual memory transfer
            // between the specified devices
            await Task.Delay(1).ConfigureAwait(false); // Placeholder

        private async Task ExecuteNodeAsync(ScheduledNode node) =>
            // Implementation would execute the compute node
            // on the assigned device
            await Task.Delay((int)node.EstimatedTimeMs).ConfigureAwait(false); // Placeholder

        private double EstimateExecutionTime(ComputeNode node, ComputeDevice device)
        {
            if (_devicePerformance.TryGetValue(device, out var performance))
            {
                return node.Operation.EstimatedFlops / performance.PeakGFLOPS / 1000.0;
            }

            return node.EstimatedTimeMs;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AdaptiveScheduler));
        }

        /// <summary>
        /// Updates the scheduling policy based on workload analysis.
        /// </summary>
        public async Task UpdatePolicyAsync(WorkloadAnalysis analysis)
        {
            ThrowIfDisposed();
            
            // Update scheduling policy based on workload analysis
            // This would adjust internal scheduling parameters
            await Task.CompletedTask.ConfigureAwait(false);
        }

        /// <summary>
        /// Gets device recommendations based on model analysis.
        /// </summary>
        public Dictionary<string, ComputeDevice> GetDeviceRecommendations(ModelAnalysisResult analysis)
        {
            ThrowIfDisposed();

            var recommendations = new Dictionary<string, ComputeDevice>();
            
            // For high operation count models, recommend GPU
            if (analysis.TotalOperations > 1000)
            {
                recommendations["primary"] = ComputeDevice.GPU;
            }
            else
            {
                recommendations["primary"] = ComputeDevice.CPU;
            }

            return recommendations;
        }

        /// <summary>
        /// Disposes the adaptive scheduler and releases resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            _profiler?.Dispose();
            _loadBalancer?.Dispose();
            _disposed = true;
        }
    }

    /// <summary>
    /// Defines scheduling policies for the adaptive scheduler.
    /// </summary>
    public enum SchedulingPolicy
    {
        /// <summary>
        /// Optimize for maximum performance.
        /// </summary>
        PerformanceOptimized,

        /// <summary>
        /// Optimize for energy efficiency.
        /// </summary>
        EnergyEfficient,

        /// <summary>
        /// Balance load across all available devices.
        /// </summary>
        LoadBalanced,

        /// <summary>
        /// Optimize for minimum latency.
        /// </summary>
        LatencyOptimized,

        /// <summary>
        /// Use device preferences specified in the compute graph.
        /// </summary>
        RespectHints
    }

    /// <summary>
    /// Defines compute capabilities for device selection.
    /// </summary>
    public enum ComputeCapability
    {
        /// <summary>
        /// General purpose computing.
        /// </summary>
        General,

        /// <summary>
        /// Matrix extensions (Intel AMX, Apple AMX).
        /// </summary>
        MatrixExtensions,

        /// <summary>
        /// SIMD vector operations.
        /// </summary>
        SIMD,

        /// <summary>
        /// High memory bandwidth.
        /// </summary>
        HighBandwidth,

        /// <summary>
        /// Tensor cores for ML operations.
        /// </summary>
        TensorCores,

        /// <summary>
        /// Specialized AI/ML acceleration.
        /// </summary>
        AIAcceleration
    }
}