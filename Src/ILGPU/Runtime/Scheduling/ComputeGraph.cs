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
using System.Collections.Generic;
using System.Linq;

namespace ILGPU.Runtime.Scheduling
{
    /// <summary>
    /// Represents a compute graph for scheduling operations across multiple accelerators.
    /// </summary>
    public class ComputeGraph
    {
        private readonly List<ComputeNode> _nodes;
        private readonly List<DataDependency> _dependencies;

        /// <summary>
        /// Initializes a new instance of the ComputeGraph class.
        /// </summary>
        public ComputeGraph()
        {
            _nodes = [];
            _dependencies = [];
        }

        /// <summary>
        /// Gets all nodes in the compute graph.
        /// </summary>
        public IReadOnlyList<ComputeNode> Nodes => _nodes.AsReadOnly();

        /// <summary>
        /// Gets all data dependencies in the compute graph.
        /// </summary>
        public IReadOnlyList<DataDependency> Dependencies => _dependencies.AsReadOnly();

        /// <summary>
        /// Adds a compute node to the graph.
        /// </summary>
        /// <param name="node">The compute node to add.</param>
        public void AddNode(ComputeNode node)
        {
            if (node == null)
                throw new ArgumentNullException(nameof(node));

            if (_nodes.Contains(node))
                throw new ArgumentException("Node already exists in the graph.");

            _nodes.Add(node);
        }

        /// <summary>
        /// Adds a data dependency between two nodes.
        /// </summary>
        /// <param name="producer">The node that produces the data.</param>
        /// <param name="consumer">The node that consumes the data.</param>
        /// <param name="dataSize">The size of the data dependency in bytes.</param>
        /// <param name="accessPattern">The expected access pattern.</param>
        public void AddDependency(
            ComputeNode producer, 
            ComputeNode consumer, 
            long dataSize,
            DataAccessPattern accessPattern = DataAccessPattern.Unknown)
        {
            if (producer == null)
                throw new ArgumentNullException(nameof(producer));
            if (consumer == null)
                throw new ArgumentNullException(nameof(consumer));

            if (!_nodes.Contains(producer))
                throw new ArgumentException("Producer node not found in graph.");
            if (!_nodes.Contains(consumer))
                throw new ArgumentException("Consumer node not found in graph.");

            var dependency = new DataDependency(producer, consumer, dataSize, accessPattern);
            _dependencies.Add(dependency);
        }

        /// <summary>
        /// Gets the topological order of nodes for execution.
        /// </summary>
        /// <returns>Nodes in topological order.</returns>
        public IEnumerable<ComputeNode> GetTopologicalOrder()
        {
            var visited = new HashSet<ComputeNode>();
            var visiting = new HashSet<ComputeNode>();
            var result = new List<ComputeNode>();

            foreach (var node in _nodes)
            {
                if (!visited.Contains(node))
                {
                    if (!TopologicalSortVisit(node, visited, visiting, result))
                        throw new InvalidOperationException("Circular dependency detected in compute graph.");
                }
            }

            result.Reverse();
            return result;
        }

        /// <summary>
        /// Gets nodes that can be executed in parallel (no dependencies between them).
        /// </summary>
        /// <returns>Groups of nodes that can execute in parallel.</returns>
        public IEnumerable<IEnumerable<ComputeNode>> GetParallelExecutionLevels()
        {
            var orderedNodes = GetTopologicalOrder().ToList();
            var levels = new List<List<ComputeNode>>();
            var processedNodes = new HashSet<ComputeNode>();

            while (processedNodes.Count < orderedNodes.Count)
            {
                var currentLevel = new List<ComputeNode>();

                foreach (var node in orderedNodes)
                {
                    if (processedNodes.Contains(node))
                        continue;

                    // Check if all dependencies are satisfied
                    var dependencies = GetNodeDependencies(node);
                    if (dependencies.All(dep => processedNodes.Contains(dep)))
                    {
                        currentLevel.Add(node);
                    }
                }

                if (currentLevel.Count == 0)
                    throw new InvalidOperationException("Unable to find executable nodes. Possible circular dependency.");

                levels.Add(currentLevel);
                foreach (var node in currentLevel)
                {
                    processedNodes.Add(node);
                }
            }

            return levels;
        }

        /// <summary>
        /// Gets the estimated execution time for the entire graph.
        /// </summary>
        /// <param name="devicePerformance">Performance characteristics of available devices.</param>
        /// <returns>Estimated execution time in milliseconds.</returns>
        public double GetEstimatedExecutionTime(IReadOnlyDictionary<ComputeDevice, DevicePerformance> devicePerformance)
        {
            var levels = GetParallelExecutionLevels();
            var totalTime = 0.0;

            foreach (var level in levels)
            {
                var maxLevelTime = 0.0;

                foreach (var node in level)
                {
                    var device = node.PreferredDevice;
                    if (devicePerformance.TryGetValue(device, out var performance))
                    {
                        var nodeTime = EstimateNodeExecutionTime(node, performance);
                        maxLevelTime = Math.Max(maxLevelTime, nodeTime);
                    }
                }

                totalTime += maxLevelTime;
            }

            return totalTime;
        }

        /// <summary>
        /// Optimizes the graph by merging compatible nodes and eliminating redundancies.
        /// </summary>
        public void Optimize()
        {
            // Merge compatible adjacent nodes
            MergeCompatibleNodes();

            // Eliminate redundant data transfers
            EliminateRedundantTransfers();

            // Optimize memory access patterns
            OptimizeMemoryAccess();
        }

        private bool TopologicalSortVisit(
            ComputeNode node, 
            HashSet<ComputeNode> visited, 
            HashSet<ComputeNode> visiting, 
            List<ComputeNode> result)
        {
            if (visiting.Contains(node))
                return false; // Circular dependency

            if (visited.Contains(node))
                return true;

            visiting.Add(node);

            var dependencies = GetNodeDependencies(node);
            foreach (var dependency in dependencies)
            {
                if (!TopologicalSortVisit(dependency, visited, visiting, result))
                    return false;
            }

            visiting.Remove(node);
            visited.Add(node);
            result.Add(node);

            return true;
        }

        private IEnumerable<ComputeNode> GetNodeDependencies(ComputeNode node) => _dependencies
                .Where(dep => dep.Consumer == node)
                .Select(dep => dep.Producer);

        private void MergeCompatibleNodes()
        {
            // Implementation for merging compatible nodes
            // This would identify nodes that can be fused for better performance
        }

        private void EliminateRedundantTransfers()
        {
            // Implementation for eliminating redundant data transfers
            // This would identify and remove unnecessary memory copies
        }

        private void OptimizeMemoryAccess()
        {
            // Implementation for optimizing memory access patterns
            // This would reorder operations to improve memory locality
        }

        private double EstimateNodeExecutionTime(ComputeNode node, DevicePerformance performance) => node.Operation switch
        {
            MatMulOp matmul => EstimateMatMulTime(matmul, performance),
            ConvolutionOp conv => EstimateConvolutionTime(conv, performance),
            VectorOp vector => EstimateVectorTime(vector, performance),
            _ => node.EstimatedTimeMs
        };

        private double EstimateMatMulTime(MatMulOp operation, DevicePerformance performance)
        {
            var flops = 2.0 * operation.M * operation.N * operation.K;
            return flops / performance.PeakGFLOPS / 1000.0; // Convert to milliseconds
        }

        private double EstimateConvolutionTime(ConvolutionOp operation, DevicePerformance performance)
        {
            // Simplified convolution time estimation
            var flops = operation.OutputSize * operation.KernelSize * operation.InputChannels * 2.0;
            return flops / performance.PeakGFLOPS / 1000.0;
        }

        private double EstimateVectorTime(VectorOp operation, DevicePerformance performance)
        {
            var elements = operation.Size;
            var throughput = performance.MemoryBandwidthGBps * 1e9; // Convert to bytes/sec
            var timeForMemory = (elements * operation.ElementSize * 2) / throughput; // Read + Write
            var timeForCompute = elements / performance.PeakGFLOPS / 1e6; // Simplified
            
            return Math.Max(timeForMemory, timeForCompute) * 1000.0; // Convert to milliseconds
        }
    }

    /// <summary>
    /// Represents a data dependency between two compute nodes.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the DataDependency class.
    /// </remarks>
    public class DataDependency(
        ComputeNode producer,
        ComputeNode consumer,
        long dataSize,
        DataAccessPattern accessPattern)
    {

        /// <summary>
        /// Gets the node that produces the data.
        /// </summary>
        public ComputeNode Producer { get; } = producer;

        /// <summary>
        /// Gets the node that consumes the data.
        /// </summary>
        public ComputeNode Consumer { get; } = consumer;

        /// <summary>
        /// Gets the size of the data dependency in bytes.
        /// </summary>
        public long DataSize { get; } = dataSize;

        /// <summary>
        /// Gets the expected data access pattern.
        /// </summary>
        public DataAccessPattern AccessPattern { get; } = accessPattern;

        /// <summary>
        /// Gets the estimated transfer time in milliseconds.
        /// </summary>
        /// <param name="bandwidth">Available bandwidth in GB/s.</param>
        /// <returns>Transfer time in milliseconds.</returns>
        public double GetEstimatedTransferTime(double bandwidth) => DataSize / (bandwidth * 1e9) * 1000.0;
    }

    /// <summary>
    /// Describes the expected data access pattern for optimization.
    /// </summary>
    public enum DataAccessPattern
    {
        /// <summary>
        /// Unknown access pattern.
        /// </summary>
        Unknown,

        /// <summary>
        /// Sequential access pattern.
        /// </summary>
        Sequential,

        /// <summary>
        /// Random access pattern.
        /// </summary>
        Random,

        /// <summary>
        /// Strided access pattern.
        /// </summary>
        Strided,

        /// <summary>
        /// Broadcast pattern (one-to-many).
        /// </summary>
        Broadcast,

        /// <summary>
        /// Reduction pattern (many-to-one).
        /// </summary>
        Reduction
    }
}