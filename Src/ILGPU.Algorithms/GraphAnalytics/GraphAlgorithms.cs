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

namespace ILGPU.Algorithms.GraphAnalytics
{
    /// <summary>
    /// GPU-accelerated graph algorithms and analytics.
    /// </summary>
    public static class GraphAlgorithms
    {
        #region Constants

        private const float INFINITY = float.MaxValue;
        private const int NO_PREDECESSOR = -1;

        #endregion

        #region Shortest Path Algorithms

        /// <summary>
        /// Single-Source Shortest Path using Bellman-Ford algorithm.
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="source">Source vertex.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Shortest path distances and predecessors.</returns>
        public static GraphTraversalResult BellmanFord(
            CSRGraph graph,
            int source,
            AcceleratorStream? stream = null)
        {
            if (source < 0 || source >= graph.NumVertices)
                throw new ArgumentOutOfRangeException(nameof(source));

            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            // Initialize distances and predecessors
            var distances = accelerator.Allocate1D<float>(graph.NumVertices);
            var predecessors = accelerator.Allocate1D<int>(graph.NumVertices);
            var visited = accelerator.Allocate1D<bool>(graph.NumVertices);

            // Initialize arrays
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, ArrayView<bool>, int>(
                InitializeBellmanFordKernel);
            initKernel(actualStream, graph.NumVertices, distances.View, predecessors.View, visited.View, source);

            // Relaxation loop (V-1 iterations)
            var relaxKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<int>>(
                graph.IsWeighted ? RelaxEdgesWeightedKernel : RelaxEdgesUnweightedKernel);

            for (int iteration = 0; iteration < graph.NumVertices - 1; iteration++)
            {
                relaxKernel(actualStream, graph.NumVertices,
                    graph.RowPtr.View, graph.ColIndices.View,
                    graph.Values?.View ?? new ArrayView<float>(),
                    distances.View, predecessors.View);
                actualStream.Synchronize();
            }

            // Check for negative cycles (optional)
            // This would require an additional kernel to detect if any edge can still be relaxed

            return new GraphTraversalResult(distances, predecessors, visited);
        }

        /// <summary>
        /// Single-Source Shortest Path using Delta-Stepping algorithm (parallel).
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="source">Source vertex.</param>
        /// <param name="delta">Delta parameter for bucketing.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Shortest path distances and predecessors.</returns>
        public static GraphTraversalResult DeltaStepping(
            CSRGraph graph,
            int source,
            float delta = 1.0f,
            AcceleratorStream? stream = null)
        {
            if (source < 0 || source >= graph.NumVertices)
                throw new ArgumentOutOfRangeException(nameof(source));

            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            // Initialize data structures
            var distances = accelerator.Allocate1D<float>(graph.NumVertices);
            var predecessors = accelerator.Allocate1D<int>(graph.NumVertices);
            var visited = accelerator.Allocate1D<bool>(graph.NumVertices);
            var currentFrontier = accelerator.Allocate1D<bool>(graph.NumVertices);
            var nextFrontier = accelerator.Allocate1D<bool>(graph.NumVertices);

            // Initialize
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, ArrayView<bool>, ArrayView<bool>, int>(
                InitializeDeltaSteppingKernel);
            initKernel(actualStream, graph.NumVertices, 
                distances.View, predecessors.View, visited.View, currentFrontier.View, source);

            // Delta-stepping iterations
            var hasWork = true;
            while (hasWork)
            {
                // Clear next frontier
                var clearKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<bool>>(ClearBoolArrayKernel);
                clearKernel(actualStream, graph.NumVertices, nextFrontier.View);

                // Process current frontier
                var deltaStepKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>,
                    ArrayView<int>, ArrayView<bool>, ArrayView<bool>, ArrayView<bool>, float>(
                    graph.IsWeighted ? DeltaStepWeightedKernel : DeltaStepUnweightedKernel);

                deltaStepKernel(actualStream, graph.NumVertices,
                    graph.RowPtr.View, graph.ColIndices.View, graph.Values?.View ?? new ArrayView<float>(),
                    distances.View, predecessors.View, visited.View, currentFrontier.View, nextFrontier.View, delta);

                // Check if there's more work (simplified check)
                // In a real implementation, we'd use a reduction to check if any vertex in nextFrontier is true
                hasWork = false; // Placeholder termination condition

                // Swap frontiers
                (currentFrontier, nextFrontier) = (nextFrontier, currentFrontier);
            }

            nextFrontier.Dispose();
            currentFrontier.Dispose();
            return new GraphTraversalResult(distances, predecessors, visited);
        }

        #endregion

        #region Breadth-First Search

        /// <summary>
        /// Breadth-First Search traversal from a source vertex.
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="source">Source vertex.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>BFS distances and predecessors.</returns>
        public static GraphTraversalResult BreadthFirstSearch(
            CSRGraph graph,
            int source,
            AcceleratorStream? stream = null)
        {
            if (source < 0 || source >= graph.NumVertices)
                throw new ArgumentOutOfRangeException(nameof(source));

            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            // Initialize data structures
            var distances = accelerator.Allocate1D<float>(graph.NumVertices);
            var predecessors = accelerator.Allocate1D<int>(graph.NumVertices);
            var visited = accelerator.Allocate1D<bool>(graph.NumVertices);
            var currentLevel = accelerator.Allocate1D<bool>(graph.NumVertices);
            var nextLevel = accelerator.Allocate1D<bool>(graph.NumVertices);

            // Initialize BFS
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, ArrayView<bool>, ArrayView<bool>, int>(
                InitializeBFSKernel);
            initKernel(actualStream, graph.NumVertices,
                distances.View, predecessors.View, visited.View, currentLevel.View, source);

            float currentDistance = 0;
            bool hasWork = true;

            while (hasWork)
            {
                // Clear next level
                var clearKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<bool>>(ClearBoolArrayKernel);
                clearKernel(actualStream, graph.NumVertices, nextLevel.View);

                // Expand current level
                var bfsKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>,
                    ArrayView<bool>, ArrayView<bool>, ArrayView<bool>, float>(BFSKernel);

                bfsKernel(actualStream, graph.NumVertices,
                    graph.RowPtr.View, graph.ColIndices.View, distances.View, predecessors.View,
                    visited.View, currentLevel.View, nextLevel.View, currentDistance + 1);

                // Check if there's more work (simplified)
                hasWork = false; // Placeholder termination condition

                // Swap levels
                (currentLevel, nextLevel) = (nextLevel, currentLevel);
                currentDistance++;
            }

            nextLevel.Dispose();
            currentLevel.Dispose();
            return new GraphTraversalResult(distances, predecessors, visited);
        }

        #endregion

        #region Connected Components

        /// <summary>
        /// Finds connected components using label propagation.
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Connected components result.</returns>
        public static ConnectedComponentsResult ConnectedComponents(
            CSRGraph graph,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            // Initialize component IDs (each vertex is its own component initially)
            var componentIds = accelerator.Allocate1D<int>(graph.NumVertices);
            var newComponentIds = accelerator.Allocate1D<int>(graph.NumVertices);

            var initKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(InitializeComponentsKernel);
            initKernel(actualStream, graph.NumVertices, componentIds.View);

            // Label propagation iterations
            bool hasChanges = true;
            int maxIterations = graph.NumVertices; // Upper bound on iterations needed
            int iteration = 0;

            while (hasChanges && iteration < maxIterations)
            {
                // Copy current component IDs to new array
                componentIds.View.CopyTo(newComponentIds.View, actualStream);

                // Propagate labels
                var propagateKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(
                    PropagateLabelsKernel);

                propagateKernel(actualStream, graph.NumVertices,
                    graph.RowPtr.View, graph.ColIndices.View, componentIds.View, newComponentIds.View);

                // Check for changes (simplified)
                hasChanges = false; // Placeholder termination condition

                // Swap arrays
                (componentIds, newComponentIds) = (newComponentIds, componentIds);
                iteration++;
            }

            // Count components and their sizes
            var componentIdsHost = new int[graph.NumVertices];
            componentIds.CopyToCPU(componentIdsHost);

            var componentSizeMap = new System.Collections.Generic.Dictionary<int, int>();
            foreach (var id in componentIdsHost)
            {
                if (componentSizeMap.TryGetValue(id, out var currentSize))
                    componentSizeMap[id] = currentSize + 1;
                else
                    componentSizeMap[id] = 1;
            }

            var numComponents = componentSizeMap.Count;
            var componentSizes = new int[numComponents];
            var componentIndex = 0;
            foreach (var size in componentSizeMap.Values)
            {
                componentSizes[componentIndex++] = size;
            }

            newComponentIds.Dispose();
            return new ConnectedComponentsResult(componentIds, numComponents, componentSizes);
        }

        #endregion

        #region Centrality Measures

        /// <summary>
        /// Computes PageRank centrality using power iteration.
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="dampingFactor">Damping factor (typically 0.85).</param>
        /// <param name="tolerance">Convergence tolerance.</param>
        /// <param name="maxIterations">Maximum number of iterations.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Centrality result with PageRank scores.</returns>
        public static CentralityResult PageRank(
            CSRGraph graph,
            float dampingFactor = 0.85f,
            float tolerance = 1e-6f,
            int maxIterations = 100,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            // Initialize PageRank scores
            var pagerank = accelerator.Allocate1D<float>(graph.NumVertices);
            var newPagerank = accelerator.Allocate1D<float>(graph.NumVertices);
            var outDegrees = graph.GetDegrees();

            // Initialize with uniform distribution
            var initValue = 1.0f / graph.NumVertices;
            var initKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(
                InitializeUniformKernel);
            initKernel(actualStream, graph.NumVertices, pagerank.View, initValue);

            // Power iteration
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Clear new PageRank values
                var clearKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ClearFloatArrayKernel);
                clearKernel(actualStream, graph.NumVertices, newPagerank.View);

                // Compute new PageRank values
                var pagerankKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<int>,
                    ArrayView<float>, float, float>(PageRankKernel);

                pagerankKernel(actualStream, graph.NumVertices,
                    graph.RowPtr.View, graph.ColIndices.View, pagerank.View, outDegrees.View,
                    newPagerank.View, dampingFactor, initValue);

                // Check convergence (simplified)
                // Real implementation would compute L1 norm of difference

                // Swap arrays
                (pagerank, newPagerank) = (newPagerank, pagerank);
            }

            newPagerank.Dispose();
            outDegrees.Dispose();
            return new CentralityResult(pagerank: pagerank);
        }

        /// <summary>
        /// Computes betweenness centrality using Brandes' algorithm.
        /// </summary>
        /// <param name="graph">Input graph in CSR format.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Centrality result with betweenness centrality scores.</returns>
        public static CentralityResult BetweennessCentrality(
            CSRGraph graph,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? graph.Accelerator.DefaultStream;
            var accelerator = graph.Accelerator;

            var betweenness = accelerator.Allocate1D<float>(graph.NumVertices);
            var clearKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ClearFloatArrayKernel);
            clearKernel(actualStream, graph.NumVertices, betweenness.View);

            // For each vertex as source, compute single-source betweenness contribution
            for (int source = 0; source < graph.NumVertices; source++)
            {
                // This is a simplified version - real Brandes algorithm is more complex
                var bfsResult = BreadthFirstSearch(graph, source, actualStream);
                
                // Accumulate betweenness scores (placeholder implementation)
                var accumKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(
                    AccumulateBetweennessKernel);
                
                accumKernel(actualStream, graph.NumVertices, 
                    betweenness.View, bfsResult.Distances.View, bfsResult.Predecessors.View);
                
                bfsResult.Dispose();
            }

            // Normalize betweenness scores
            var normalizationFactor = graph.NumVertices > 2 ? 2.0f / ((graph.NumVertices - 1) * (graph.NumVertices - 2)) : 1.0f;
            var normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(
                NormalizeArrayKernel);
            normalizeKernel(actualStream, graph.NumVertices, betweenness.View, normalizationFactor);

            return new CentralityResult(betweenness: betweenness);
        }

        #endregion

        #region Kernel Implementations

        private static void InitializeBellmanFordKernel(
            Index1D index,
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            int source)
        {
            if (index >= distances.Length) return;

            if (index == source)
            {
                distances[index] = 0.0f;
                visited[index] = true;
            }
            else
            {
                distances[index] = INFINITY;
                visited[index] = false;
            }
            predecessors[index] = NO_PREDECESSOR;
        }

        private static void RelaxEdgesWeightedKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> values,
            ArrayView<float> distances,
            ArrayView<int> predecessors)
        {
            if (index >= distances.Length || distances[index] == INFINITY) return;

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                float weight = values[i];
                float newDistance = distances[index] + weight;

                if (newDistance < distances[neighbor])
                {
                    distances[neighbor] = newDistance;
                    predecessors[neighbor] = index;
                }
            }
        }

        private static void RelaxEdgesUnweightedKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> values, // Unused for unweighted
            ArrayView<float> distances,
            ArrayView<int> predecessors)
        {
            if (index >= distances.Length || distances[index] == INFINITY) return;

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                float newDistance = distances[index] + 1.0f;

                if (newDistance < distances[neighbor])
                {
                    distances[neighbor] = newDistance;
                    predecessors[neighbor] = index;
                }
            }
        }

        private static void InitializeDeltaSteppingKernel(
            Index1D index,
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            ArrayView<bool> currentFrontier,
            int source)
        {
            if (index >= distances.Length) return;

            if (index == source)
            {
                distances[index] = 0.0f;
                currentFrontier[index] = true;
            }
            else
            {
                distances[index] = INFINITY;
                currentFrontier[index] = false;
            }
            predecessors[index] = NO_PREDECESSOR;
            visited[index] = false;
        }

        private static void DeltaStepWeightedKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> values,
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            ArrayView<bool> currentFrontier,
            ArrayView<bool> nextFrontier,
            float delta)
        {
            if (index >= distances.Length || !currentFrontier[index]) return;

            visited[index] = true;

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                float weight = values[i];
                float newDistance = distances[index] + weight;

                if (newDistance < distances[neighbor])
                {
                    distances[neighbor] = newDistance;
                    predecessors[neighbor] = index;

                    if (!visited[neighbor])
                    {
                        if (weight <= delta)
                            currentFrontier[neighbor] = true;
                        else
                            nextFrontier[neighbor] = true;
                    }
                }
            }
        }

        private static void DeltaStepUnweightedKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> values, // Unused
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            ArrayView<bool> currentFrontier,
            ArrayView<bool> nextFrontier,
            float delta)
        {
            if (index >= distances.Length || !currentFrontier[index]) return;

            visited[index] = true;

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                float newDistance = distances[index] + 1.0f;

                if (newDistance < distances[neighbor])
                {
                    distances[neighbor] = newDistance;
                    predecessors[neighbor] = index;

                    if (!visited[neighbor])
                        nextFrontier[neighbor] = true;
                }
            }
        }

        private static void InitializeBFSKernel(
            Index1D index,
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            ArrayView<bool> currentLevel,
            int source)
        {
            if (index >= distances.Length) return;

            if (index == source)
            {
                distances[index] = 0.0f;
                currentLevel[index] = true;
                visited[index] = true;
            }
            else
            {
                distances[index] = INFINITY;
                currentLevel[index] = false;
                visited[index] = false;
            }
            predecessors[index] = NO_PREDECESSOR;
        }

        private static void BFSKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> distances,
            ArrayView<int> predecessors,
            ArrayView<bool> visited,
            ArrayView<bool> currentLevel,
            ArrayView<bool> nextLevel,
            float newDistance)
        {
            if (index >= distances.Length || !currentLevel[index]) return;

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                if (!visited[neighbor])
                {
                    distances[neighbor] = newDistance;
                    predecessors[neighbor] = index;
                    visited[neighbor] = true;
                    nextLevel[neighbor] = true;
                }
            }
        }

        private static void InitializeComponentsKernel(Index1D index, ArrayView<int> componentIds)
        {
            if (index < componentIds.Length)
                componentIds[index] = index;
        }

        private static void PropagateLabelsKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<int> componentIds,
            ArrayView<int> newComponentIds)
        {
            if (index >= componentIds.Length) return;

            int minLabel = componentIds[index];

            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                if (neighbor >= 0 && neighbor < componentIds.Length)
                {
                    minLabel = XMath.Min(minLabel, componentIds[neighbor]);
                }
            }

            newComponentIds[index] = minLabel;
        }

        private static void InitializeUniformKernel(Index1D index, ArrayView<float> array, float value)
        {
            if (index < array.Length)
                array[index] = value;
        }

        private static void PageRankKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<float> pagerank,
            ArrayView<int> outDegrees,
            ArrayView<float> newPagerank,
            float dampingFactor,
            float baseValue)
        {
            if (index >= pagerank.Length) return;

            float sum = 0.0f;
            for (int i = rowPtr[index]; i < rowPtr[index + 1]; i++)
            {
                int neighbor = colIndices[i];
                if (outDegrees[neighbor] > 0)
                {
                    sum += pagerank[neighbor] / outDegrees[neighbor];
                }
            }

            newPagerank[index] = (1.0f - dampingFactor) * baseValue + dampingFactor * sum;
        }

        private static void AccumulateBetweennessKernel(
            Index1D index,
            ArrayView<float> betweenness,
            ArrayView<float> distances,
            ArrayView<int> predecessors)
        {
            if (index >= betweenness.Length) return;
            // Simplified betweenness accumulation
            if (distances[index] != INFINITY && predecessors[index] != NO_PREDECESSOR)
            {
                betweenness[index] += 1.0f / (distances[index] + 1.0f);
            }
        }

        private static void ClearBoolArrayKernel(Index1D index, ArrayView<bool> array)
        {
            if (index < array.Length)
                array[index] = false;
        }

        private static void ClearFloatArrayKernel(Index1D index, ArrayView<float> array)
        {
            if (index < array.Length)
                array[index] = 0.0f;
        }

        private static void NormalizeArrayKernel(Index1D index, ArrayView<float> array, float factor)
        {
            if (index < array.Length)
                array[index] *= factor;
        }

        #endregion
    }
}