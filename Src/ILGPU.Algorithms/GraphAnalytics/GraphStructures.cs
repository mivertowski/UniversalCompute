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
    /// Compressed Sparse Row (CSR) graph representation for GPU processing.
    /// </summary>
    public sealed class CSRGraph : IDisposable
    {
        private bool _disposed;

        /// <summary>
        /// Initializes a new CSR graph.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numVertices">Number of vertices.</param>
        /// <param name="numEdges">Number of edges.</param>
        /// <param name="rowPtr">Row pointer array (length: numVertices + 1).</param>
        /// <param name="colIndices">Column indices array (length: numEdges).</param>
        /// <param name="values">Edge weights array (optional, length: numEdges).</param>
        public CSRGraph(
            Accelerator accelerator, 
            int numVertices, 
            int numEdges,
            int[] rowPtr,
            int[] colIndices,
            float[]? values = null)
        {
            Accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            if (rowPtr.Length != numVertices + 1)
                throw new ArgumentException($"Row pointer array must have length {numVertices + 1}");
            if (colIndices.Length != numEdges)
                throw new ArgumentException($"Column indices array must have length {numEdges}");
            if (values != null && values.Length != numEdges)
                throw new ArgumentException($"Values array must have length {numEdges}");

            NumVertices = numVertices;
            NumEdges = numEdges;
            
            // Allocate GPU memory and copy data
            RowPtr = Accelerator.Allocate1D(rowPtr);
            ColIndices = Accelerator.Allocate1D(colIndices);
            
            if (values != null)
            {
                Values = Accelerator.Allocate1D(values);
                IsWeighted = true;
            }
            else
            {
                Values = null;
                IsWeighted = false;
            }
        }

        /// <summary>
        /// Gets the number of vertices in the graph.
        /// </summary>
        public int NumVertices { get; }

        /// <summary>
        /// Gets the number of edges in the graph.
        /// </summary>
        public int NumEdges { get; }

        /// <summary>
        /// Gets whether the graph is weighted.
        /// </summary>
        public bool IsWeighted { get; }

        /// <summary>
        /// Gets the row pointer array (CSR format).
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> RowPtr { get; }

        /// <summary>
        /// Gets the column indices array (CSR format).
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> ColIndices { get; }

        /// <summary>
        /// Gets the edge weights array (if weighted).
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? Values { get; }

        /// <summary>
        /// Gets the accelerator associated with this graph.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Creates a CSR graph from an edge list.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numVertices">Number of vertices.</param>
        /// <param name="edges">Array of edges (source, destination pairs).</param>
        /// <param name="weights">Optional edge weights.</param>
        /// <returns>CSR graph representation.</returns>
        public static CSRGraph FromEdgeList(
            Accelerator accelerator,
            int numVertices,
            (int source, int dest)[] edges,
            float[]? weights = null)
        {
            if (weights != null && weights.Length != edges.Length)
                throw new ArgumentException("Weights array must have same length as edges array");

            var numEdges = edges.Length;
            var rowPtr = new int[numVertices + 1];
            var colIndices = new int[numEdges];
            
            // Count out-degree of each vertex
            var outDegree = new int[numVertices];
            foreach (var (source, dest) in edges)
            {
                if (source >= 0 && source < numVertices)
                    outDegree[source]++;
            }
            
            // Build row pointer array (prefix sum)
            rowPtr[0] = 0;
            for (int i = 0; i < numVertices; i++)
            {
                rowPtr[i + 1] = rowPtr[i] + outDegree[i];
            }
            
            // Fill column indices (and weights if provided)
            var currentPos = new int[numVertices];
            Array.Copy(rowPtr, currentPos, numVertices);
            
            float[]? values = weights != null ? new float[numEdges] : null;
            
            for (int i = 0; i < edges.Length; i++)
            {
                var (source, dest) = edges[i];
                if (source >= 0 && source < numVertices)
                {
                    var pos = currentPos[source]++;
                    colIndices[pos] = dest;
                    if (values != null && weights != null)
                        values[pos] = weights[i];
                }
            }
            
            return new CSRGraph(accelerator, numVertices, numEdges, rowPtr, colIndices, values);
        }

        /// <summary>
        /// Creates an undirected graph from a directed graph by adding reverse edges.
        /// </summary>
        /// <returns>Undirected CSR graph.</returns>
        public CSRGraph ToUndirected()
        {
            // Copy data to CPU for processing
            var rowPtrHost = new int[NumVertices + 1];
            var colIndicesHost = new int[NumEdges];
            RowPtr.CopyToCPU(rowPtrHost);
            ColIndices.CopyToCPU(colIndicesHost);
            
            float[]? valuesHost = null;
            if (IsWeighted && Values != null)
            {
                valuesHost = new float[NumEdges];
                Values.CopyToCPU(valuesHost);
            }
            
            // Create edge list with reverse edges
            var edges = new System.Collections.Generic.List<(int, int)>();
            var weights = IsWeighted ? new System.Collections.Generic.List<float>() : null;
            
            for (int u = 0; u < NumVertices; u++)
            {
                for (int i = rowPtrHost[u]; i < rowPtrHost[u + 1]; i++)
                {
                    int v = colIndicesHost[i];
                    float weight = IsWeighted ? valuesHost![i] : 1.0f;
                    
                    // Add forward edge
                    edges.Add((u, v));
                    weights?.Add(weight);
                    
                    // Add reverse edge (if not already exists)
                    if (u != v) // Avoid self-loops duplication
                    {
                        edges.Add((v, u));
                        weights?.Add(weight);
                    }
                }
            }
            
            return FromEdgeList(Accelerator, NumVertices, edges.ToArray(), weights?.ToArray());
        }

        /// <summary>
        /// Gets the degree of each vertex.
        /// </summary>
        /// <returns>Array containing the degree of each vertex.</returns>
        public MemoryBuffer1D<int, Stride1D.Dense> GetDegrees()
        {
            var degrees = Accelerator.Allocate1D<int>(NumVertices);
            
            var kernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>>(GetDegreesKernel);
            
            kernel(Accelerator.DefaultStream, NumVertices, RowPtr.View, degrees.View);
            Accelerator.DefaultStream.Synchronize();
            
            return degrees;
        }

        /// <summary>
        /// Computes the transpose of the graph (reverse all edges).
        /// </summary>
        /// <returns>Transposed CSR graph.</returns>
        public CSRGraph Transpose()
        {
            var transposeRowPtr = new int[NumVertices + 1];
            var transposeColIndices = new int[NumEdges];
            float[]? transposeValues = IsWeighted ? new float[NumEdges] : null;
            
            // Copy data to CPU for transposition
            var rowPtrHost = new int[NumVertices + 1];
            var colIndicesHost = new int[NumEdges];
            RowPtr.CopyToCPU(rowPtrHost);
            ColIndices.CopyToCPU(colIndicesHost);
            
            float[]? valuesHost = null;
            if (IsWeighted && Values != null)
            {
                valuesHost = new float[NumEdges];
                Values.CopyToCPU(valuesHost);
            }
            
            // Count in-degree of each vertex (becomes out-degree in transpose)
            var inDegree = new int[NumVertices];
            for (int u = 0; u < NumVertices; u++)
            {
                for (int i = rowPtrHost[u]; i < rowPtrHost[u + 1]; i++)
                {
                    int v = colIndicesHost[i];
                    if (v >= 0 && v < NumVertices)
                        inDegree[v]++;
                }
            }
            
            // Build transpose row pointer array
            transposeRowPtr[0] = 0;
            for (int i = 0; i < NumVertices; i++)
            {
                transposeRowPtr[i + 1] = transposeRowPtr[i] + inDegree[i];
            }
            
            // Fill transpose adjacency lists
            var currentPos = new int[NumVertices];
            Array.Copy(transposeRowPtr, currentPos, NumVertices);
            
            for (int u = 0; u < NumVertices; u++)
            {
                for (int i = rowPtrHost[u]; i < rowPtrHost[u + 1]; i++)
                {
                    int v = colIndicesHost[i];
                    if (v >= 0 && v < NumVertices)
                    {
                        var pos = currentPos[v]++;
                        transposeColIndices[pos] = u;
                        if (transposeValues != null && valuesHost != null)
                            transposeValues[pos] = valuesHost[i];
                    }
                }
            }
            
            return new CSRGraph(Accelerator, NumVertices, NumEdges, transposeRowPtr, transposeColIndices, transposeValues);
        }

        /// <summary>
        /// Kernel to compute vertex degrees.
        /// </summary>
        private static void GetDegreesKernel(Index1D index, ArrayView<int> rowPtr, ArrayView<int> degrees)
        {
            if (index >= degrees.Length)
                return;
                
            degrees[index] = rowPtr[index + 1] - rowPtr[index];
        }

        /// <summary>
        /// Disposes the CSR graph and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                RowPtr?.Dispose();
                ColIndices?.Dispose();
                Values?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Graph traversal result containing distances and predecessors.
    /// </summary>
    public sealed class GraphTraversalResult : IDisposable
    {
        /// <summary>
        /// Distance from source to each vertex.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense> Distances { get; }

        /// <summary>
        /// Predecessor of each vertex in the shortest path tree.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> Predecessors { get; }

        /// <summary>
        /// Indicates whether each vertex was visited.
        /// </summary>
        public MemoryBuffer1D<bool, Stride1D.Dense> Visited { get; }

        /// <summary>
        /// Initializes a new graph traversal result.
        /// </summary>
        public GraphTraversalResult(
            MemoryBuffer1D<float, Stride1D.Dense> distances,
            MemoryBuffer1D<int, Stride1D.Dense> predecessors,
            MemoryBuffer1D<bool, Stride1D.Dense> visited)
        {
            Distances = distances ?? throw new ArgumentNullException(nameof(distances));
            Predecessors = predecessors ?? throw new ArgumentNullException(nameof(predecessors));
            Visited = visited ?? throw new ArgumentNullException(nameof(visited));
        }

        /// <summary>
        /// Disposes the traversal result and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            Distances?.Dispose();
            Predecessors?.Dispose();
            Visited?.Dispose();
        }
    }

    /// <summary>
    /// Result of connected components analysis.
    /// </summary>
    public sealed class ConnectedComponentsResult : IDisposable
    {
        /// <summary>
        /// Component ID for each vertex.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> ComponentIds { get; }

        /// <summary>
        /// Number of connected components found.
        /// </summary>
        public int NumComponents { get; }

        /// <summary>
        /// Size of each component.
        /// </summary>
#pragma warning disable CA1819 // Properties should not return arrays
        public int[] ComponentSizes { get; }
#pragma warning restore CA1819 // Properties should not return arrays

        /// <summary>
        /// Initializes a new connected components result.
        /// </summary>
        public ConnectedComponentsResult(
            MemoryBuffer1D<int, Stride1D.Dense> componentIds,
            int numComponents,
            int[] componentSizes)
        {
            ComponentIds = componentIds ?? throw new ArgumentNullException(nameof(componentIds));
            NumComponents = numComponents;
            ComponentSizes = componentSizes ?? throw new ArgumentNullException(nameof(componentSizes));
        }

        /// <summary>
        /// Disposes the connected components result.
        /// </summary>
        public void Dispose()
        {
            ComponentIds?.Dispose();
        }
    }

    /// <summary>
    /// Graph centrality measures.
    /// </summary>
    public sealed class CentralityResult : IDisposable
    {
        /// <summary>
        /// Betweenness centrality for each vertex.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? BetweennessCentrality { get; }

        /// <summary>
        /// Closeness centrality for each vertex.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? ClosenessCentrality { get; }

        /// <summary>
        /// Eigenvector centrality for each vertex.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? EigenvectorCentrality { get; }

        /// <summary>
        /// PageRank scores for each vertex.
        /// </summary>
        public MemoryBuffer1D<float, Stride1D.Dense>? PageRank { get; }

        /// <summary>
        /// Initializes a new centrality result.
        /// </summary>
        public CentralityResult(
            MemoryBuffer1D<float, Stride1D.Dense>? betweenness = null,
            MemoryBuffer1D<float, Stride1D.Dense>? closeness = null,
            MemoryBuffer1D<float, Stride1D.Dense>? eigenvector = null,
            MemoryBuffer1D<float, Stride1D.Dense>? pagerank = null)
        {
            BetweennessCentrality = betweenness;
            ClosenessCentrality = closeness;
            EigenvectorCentrality = eigenvector;
            PageRank = pagerank;
        }

        /// <summary>
        /// Disposes the centrality result.
        /// </summary>
        public void Dispose()
        {
            BetweennessCentrality?.Dispose();
            ClosenessCentrality?.Dispose();
            EigenvectorCentrality?.Dispose();
            PageRank?.Dispose();
        }
    }
}