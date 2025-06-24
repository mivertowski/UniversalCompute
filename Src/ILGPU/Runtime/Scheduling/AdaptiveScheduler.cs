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
using ILGPU.Runtime.Profiling;
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
        private readonly SchedulingPolicy _policy;
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
            _policy = policy;
            _profiler = new PerformanceProfiler();
            _loadBalancer = new LoadBalancer();
            _devicePerformance = new Dictionary<ComputeDevice, DevicePerformance>();

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
        public SchedulingPolicy Policy => _policy;

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
            var assignments = await CreateDeviceAssignmentsAsync(graph, workloadAnalysis);

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
            var executionId = _profiler.StartExecution(plan);

            try
            {
                // Execute memory transfers first
                await ExecuteMemoryTransfersAsync(plan.MemoryPlan);

                // Execute compute operations according to schedule
                await ExecuteComputeOperationsAsync(plan.Schedule);

                // Update performance statistics
                _profiler.EndExecution(executionId, success: true);
            }
            catch (Exception ex)
            {
                _profiler.EndExecution(executionId, success: false, exception: ex);
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
                totalFlops,
                totalMemoryOps,
                dataTransferSize,
                computeIntensity,
                parallelismLevel,
                operationTypes);
        }

        private async Task<Dictionary<ComputeNode, ComputeDevice>> CreateDeviceAssignmentsAsync(
            ComputeGraph graph, 
            WorkloadAnalysis analysis)
        {
            var assignments = new Dictionary<ComputeNode, ComputeDevice>();

            // Apply scheduling policy
            switch (_policy)
            {
                case SchedulingPolicy.PerformanceOptimized:
                    assignments = CreatePerformanceOptimizedAssignments(graph, analysis);
                    break;
                    
                case SchedulingPolicy.EnergyEfficient:
                    assignments = CreateEnergyEfficientAssignments(graph, analysis);
                    break;
                    
                case SchedulingPolicy.LoadBalanced:
                    assignments = await CreateLoadBalancedAssignmentsAsync(graph, analysis);
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
                var bestDevice = _devicePerformance
                    .Where(kvp => CanExecuteOperation(kvp.Key, node.Operation))
                    .OrderByDescending(kvp => kvp.Value.PerformancePerWatt)
                    .FirstOrDefault().Key ?? ComputeDevice.CPU;

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

                if (availableDevices.Any())
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
                var bestDevice = _devicePerformance
                    .Where(kvp => CanExecuteOperation(kvp.Key, node.Operation))
                    .OrderBy(kvp => kvp.Value.AverageLatencyMs)
                    .FirstOrDefault().Key ?? ComputeDevice.CPU;

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
            return _devicePerformance
                .Where(kvp => kvp.Value.SupportsTensorCores)
                .OrderByDescending(kvp => kvp.Value.TensorPerformanceGFLOPS)
                .FirstOrDefault().Key ?? ComputeDevice.GPU;
        }

        private ComputeDevice SelectBestConvolutionDevice(ConvolutionOp conv)
        {
            // Prefer AI/ML accelerators for convolution operations
            var aiDevices = _devicePerformance
                .Where(kvp => kvp.Value.HasAIAcceleration)
                .OrderByDescending(kvp => kvp.Value.AIPerformanceGOPS)
                .ToList();

            if (aiDevices.Any())
                return aiDevices.First().Key;

            // Fallback to best tensor device
            return SelectBestTensorDevice();
        }

        private ComputeDevice SelectBestDevice(ComputeCapability capability)
        {
            return capability switch
            {
                ComputeCapability.MatrixExtensions => 
                    _devicePerformance
                        .Where(kvp => kvp.Value.SupportsMatrixExtensions)
                        .OrderByDescending(kvp => kvp.Value.MatrixPerformanceGFLOPS)
                        .FirstOrDefault().Key ?? ComputeDevice.CPU,

                ComputeCapability.SIMD => 
                    _devicePerformance
                        .Where(kvp => kvp.Value.SIMDWidthBits > 0)
                        .OrderByDescending(kvp => kvp.Value.SIMDPerformanceGFLOPS)
                        .FirstOrDefault().Key ?? ComputeDevice.CPU,

                ComputeCapability.HighBandwidth => 
                    _devicePerformance
                        .OrderByDescending(kvp => kvp.Value.MemoryBandwidthGBps)
                        .FirstOrDefault().Key ?? ComputeDevice.GPU,

                _ => ComputeDevice.Auto
            };
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
            var levels = graph.GetParallelExecutionLevels();
            var scheduledLevels = new List<ScheduledExecutionLevel>();

            foreach (var level in levels)
            {
                var scheduledNodes = level.Select(node => new ScheduledNode(
                    node, 
                    assignments[node], 
                    EstimateExecutionTime(node, assignments[node]))).ToList();

                scheduledLevels.Add(new ScheduledExecutionLevel(scheduledNodes));
            }

            return new ExecutionSchedule(scheduledLevels);
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
                        sourceDevice,
                        targetDevice,
                        dependency.DataSize,
                        dependency.AccessPattern);

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

            await Task.WhenAll(transferTasks);
        }

        private async Task ExecuteComputeOperationsAsync(ExecutionSchedule schedule)
        {
            foreach (var level in schedule.Levels)
            {
                var levelTasks = level.Nodes.Select(ExecuteNodeAsync);
                await Task.WhenAll(levelTasks);
            }
        }

        private async Task ExecuteMemoryTransferAsync(MemoryTransfer transfer)
        {
            // Implementation would perform actual memory transfer
            // between the specified devices
            await Task.Delay(1); // Placeholder
        }

        private async Task ExecuteNodeAsync(ScheduledNode node)
        {
            // Implementation would execute the compute node
            // on the assigned device
            await Task.Delay((int)node.EstimatedTimeMs); // Placeholder
        }

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