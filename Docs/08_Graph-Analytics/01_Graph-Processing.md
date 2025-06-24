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

# Graph Processing with ILGPU

## Overview

ILGPU provides efficient graph processing capabilities through specialized algorithms and memory layouts optimized for parallel execution. This module supports common graph analytics workloads including traversal, shortest path, and community detection algorithms.

## Technical Background

### Graph Processing Challenges

Graph algorithms present unique challenges for parallel execution:

- **Irregular memory access patterns**: Graph traversal results in non-coalesced memory accesses
- **Load balancing**: Vertices with varying degrees create workload imbalances
- **Synchronization overhead**: Graph algorithms often require frequent synchronization
- **Memory bandwidth limitations**: Large graphs may exceed available device memory

### ILGPU Graph Processing Solution

ILGPU addresses these challenges through:

1. **Optimized memory layouts**: CSR (Compressed Sparse Row) and COO (Coordinate) formats
2. **Work-efficient algorithms**: Parallel algorithms designed for GPU execution patterns
3. **Memory management**: Streaming and partitioning for large graphs
4. **Cross-platform optimization**: Automatic optimization for different accelerator types

## Core Graph Algorithms

### Breadth-First Search (BFS)

```csharp
using ILGPU;
using ILGPU.Runtime;

public class GraphBFS
{
    static void BFSKernel(
        Index1D index,
        ArrayView<int> vertices,
        ArrayView<int> edges,
        ArrayView<int> distances,
        ArrayView<int> currentLevel,
        ArrayView<int> nextLevel,
        int level)
    {
        if (index >= currentLevel.Length)
            return;

        var vertex = currentLevel[index];
        if (vertex == -1)
            return;

        var start = vertices[vertex];
        var end = vertex + 1 < vertices.Length ? vertices[vertex + 1] : edges.Length;

        for (int i = start; i < end; i++)
        {
            var neighbor = edges[i];
            if (Atomic.CompareExchange(ref distances[neighbor], level + 1, -1) == -1)
            {
                // Successfully updated distance, add to next level
                var pos = Atomic.Add(ref nextLevel[0], 1);
                if (pos < nextLevel.Length - 1)
                {
                    nextLevel[pos + 1] = neighbor;
                }
            }
        }
    }

    public static int[] ComputeBFS(Context context, Accelerator accelerator, 
        int[] vertices, int[] edges, int sourceVertex)
    {
        var numVertices = vertices.Length - 1;
        
        using var verticesBuffer = accelerator.Allocate1D(vertices);
        using var edgesBuffer = accelerator.Allocate1D(edges);
        using var distancesBuffer = accelerator.Allocate1D<int>(numVertices);
        using var currentLevelBuffer = accelerator.Allocate1D<int>(numVertices + 1);
        using var nextLevelBuffer = accelerator.Allocate1D<int>(numVertices + 1);

        // Initialize distances to -1 (unvisited)
        var distances = new int[numVertices];
        for (int i = 0; i < numVertices; i++)
            distances[i] = -1;
        
        distances[sourceVertex] = 0;
        distancesBuffer.CopyFromCPU(distances);

        // Initialize first level
        var currentLevel = new int[numVertices + 1];
        currentLevel[0] = 1; // count
        currentLevel[1] = sourceVertex;
        currentLevelBuffer.CopyFromCPU(currentLevel);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, 
            ArrayView<int>, ArrayView<int>, int>(BFSKernel);

        int level = 0;
        while (true)
        {
            // Clear next level
            var nextLevel = new int[numVertices + 1];
            nextLevelBuffer.CopyFromCPU(nextLevel);

            // Process current level
            var currentCount = currentLevelBuffer.GetAsArray1D()[0];
            if (currentCount == 0)
                break;

            kernel(currentCount, verticesBuffer.View, edgesBuffer.View, 
                distancesBuffer.View, currentLevelBuffer.View, 
                nextLevelBuffer.View, level);
            
            accelerator.Synchronize();

            // Swap levels
            var temp = currentLevelBuffer;
            currentLevelBuffer = nextLevelBuffer;
            nextLevelBuffer = temp;

            level++;
        }

        return distancesBuffer.GetAsArray1D();
    }
}
```

### PageRank Algorithm

```csharp
public class PageRank
{
    static void PageRankKernel(
        Index1D index,
        ArrayView<int> vertices,
        ArrayView<int> edges,
        ArrayView<float> oldRanks,
        ArrayView<float> newRanks,
        ArrayView<int> outDegrees,
        float dampingFactor,
        int numVertices)
    {
        if (index >= numVertices)
            return;

        float rank = (1.0f - dampingFactor) / numVertices;
        
        var start = vertices[index];
        var end = index + 1 < vertices.Length ? vertices[index + 1] : edges.Length;

        for (int i = start; i < end; i++)
        {
            var neighbor = edges[i];
            rank += dampingFactor * oldRanks[neighbor] / outDegrees[neighbor];
        }

        newRanks[index] = rank;
    }

    public static float[] ComputePageRank(Context context, Accelerator accelerator,
        int[] vertices, int[] edges, int[] outDegrees, int iterations = 10)
    {
        var numVertices = vertices.Length - 1;
        const float dampingFactor = 0.85f;

        using var verticesBuffer = accelerator.Allocate1D(vertices);
        using var edgesBuffer = accelerator.Allocate1D(edges);
        using var outDegreesBuffer = accelerator.Allocate1D(outDegrees);
        using var oldRanksBuffer = accelerator.Allocate1D<float>(numVertices);
        using var newRanksBuffer = accelerator.Allocate1D<float>(numVertices);

        // Initialize ranks
        var initialRank = 1.0f / numVertices;
        var ranks = new float[numVertices];
        for (int i = 0; i < numVertices; i++)
            ranks[i] = initialRank;
        
        oldRanksBuffer.CopyFromCPU(ranks);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, 
            ArrayView<float>, ArrayView<int>, float, int>(PageRankKernel);

        for (int iter = 0; iter < iterations; iter++)
        {
            kernel(numVertices, verticesBuffer.View, edgesBuffer.View,
                oldRanksBuffer.View, newRanksBuffer.View, outDegreesBuffer.View,
                dampingFactor, numVertices);
            
            accelerator.Synchronize();

            // Swap buffers
            var temp = oldRanksBuffer;
            oldRanksBuffer = newRanksBuffer;
            newRanksBuffer = temp;
        }

        return oldRanksBuffer.GetAsArray1D();
    }
}
```

## Graph Data Structures

### Compressed Sparse Row (CSR) Format

```csharp
public class CSRGraph
{
    public int[] Vertices { get; }
    public int[] Edges { get; }
    public int NumVertices { get; }
    public int NumEdges { get; }

    public CSRGraph(int numVertices, List<(int, int)> edgeList)
    {
        NumVertices = numVertices;
        NumEdges = edgeList.Count;
        
        Vertices = new int[numVertices + 1];
        Edges = new int[NumEdges];

        // Sort edges by source vertex
        edgeList.Sort((a, b) => a.Item1.CompareTo(b.Item1));

        // Build CSR representation
        int edgeIndex = 0;
        for (int v = 0; v < numVertices; v++)
        {
            Vertices[v] = edgeIndex;
            while (edgeIndex < NumEdges && edgeList[edgeIndex].Item1 == v)
            {
                Edges[edgeIndex] = edgeList[edgeIndex].Item2;
                edgeIndex++;
            }
        }
        Vertices[numVertices] = NumEdges;
    }

    public void LoadToDevice(Accelerator accelerator, 
        out MemoryBuffer1D<int, Stride1D.Dense> verticesBuffer,
        out MemoryBuffer1D<int, Stride1D.Dense> edgesBuffer)
    {
        verticesBuffer = accelerator.Allocate1D(Vertices);
        edgesBuffer = accelerator.Allocate1D(Edges);
    }
}
```

## Performance Optimization

### Memory Coalescing

```csharp
public static class GraphOptimizations
{
    // Optimize memory access patterns for better coalescing
    static void CoalescedTraversalKernel(
        Index1D index,
        ArrayView<int> vertices,
        ArrayView<int> edges,
        ArrayView<float> values,
        ArrayView<float> results)
    {
        var vertex = index;
        if (vertex >= vertices.Length - 1)
            return;

        var start = vertices[vertex];
        var end = vertices[vertex + 1];
        
        float sum = 0.0f;
        
        // Process edges in chunks for better memory coalescing
        const int chunkSize = 32;
        for (int i = start; i < end; i += chunkSize)
        {
            var chunkEnd = Math.Min(i + chunkSize, end);
            
            // Coalesced memory access within chunk
            for (int j = i; j < chunkEnd; j++)
            {
                sum += values[edges[j]];
            }
        }
        
        results[vertex] = sum;
    }
}
```

### Load Balancing

```csharp
public static class LoadBalancing
{
    // Dynamic load balancing for irregular graphs
    static void BalancedProcessingKernel(
        Index1D index,
        ArrayView<int> vertices,
        ArrayView<int> edges,
        ArrayView<int> workQueue,
        ArrayView<int> workIndex,
        ArrayView<float> results)
    {
        while (true)
        {
            // Atomically get next work item
            var workPos = Atomic.Add(ref workIndex[0], 1);
            if (workPos >= workQueue.Length)
                break;

            var vertex = workQueue[workPos];
            if (vertex == -1)
                break;

            // Process vertex
            var start = vertices[vertex];
            var end = vertex + 1 < vertices.Length ? vertices[vertex + 1] : edges.Length;
            
            float sum = 0.0f;
            for (int i = start; i < end; i++)
            {
                sum += edges[i]; // Example computation
            }
            
            results[vertex] = sum;
        }
    }
}
```

## Large Graph Processing

### Graph Partitioning

```csharp
public class GraphPartitioner
{
    public static List<CSRGraph> PartitionGraph(CSRGraph graph, int numPartitions)
    {
        var partitions = new List<CSRGraph>();
        var verticesPerPartition = graph.NumVertices / numPartitions;
        
        for (int p = 0; p < numPartitions; p++)
        {
            var startVertex = p * verticesPerPartition;
            var endVertex = (p == numPartitions - 1) ? 
                graph.NumVertices : (p + 1) * verticesPerPartition;
            
            var partitionEdges = new List<(int, int)>();
            
            for (int v = startVertex; v < endVertex; v++)
            {
                var start = graph.Vertices[v];
                var end = v + 1 < graph.Vertices.Length ? 
                    graph.Vertices[v + 1] : graph.Edges.Length;
                
                for (int i = start; i < end; i++)
                {
                    var target = graph.Edges[i];
                    partitionEdges.Add((v - startVertex, target));
                }
            }
            
            partitions.Add(new CSRGraph(endVertex - startVertex, partitionEdges));
        }
        
        return partitions;
    }
}
```

## Usage Examples

### Complete Graph Processing Pipeline

```csharp
public class GraphProcessingExample
{
    public static void RunGraphAnalysis()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        // Create sample graph
        var edges = new List<(int, int)>
        {
            (0, 1), (0, 2), (1, 2), (1, 3), 
            (2, 3), (3, 4), (4, 0)
        };
        
        var graph = new CSRGraph(5, edges);
        
        // Compute BFS distances
        var distances = GraphBFS.ComputeBFS(context, accelerator, 
            graph.Vertices, graph.Edges, sourceVertex: 0);
        
        Console.WriteLine("BFS distances from vertex 0:");
        for (int i = 0; i < distances.Length; i++)
        {
            Console.WriteLine($"Vertex {i}: distance {distances[i]}");
        }
        
        // Compute PageRank
        var outDegrees = ComputeOutDegrees(graph);
        var pageRanks = PageRank.ComputePageRank(context, accelerator,
            graph.Vertices, graph.Edges, outDegrees);
        
        Console.WriteLine("\nPageRank values:");
        for (int i = 0; i < pageRanks.Length; i++)
        {
            Console.WriteLine($"Vertex {i}: rank {pageRanks[i]:F4}");
        }
    }
    
    private static int[] ComputeOutDegrees(CSRGraph graph)
    {
        var outDegrees = new int[graph.NumVertices];
        for (int v = 0; v < graph.NumVertices; v++)
        {
            var start = graph.Vertices[v];
            var end = v + 1 < graph.Vertices.Length ? 
                graph.Vertices[v + 1] : graph.Edges.Length;
            outDegrees[v] = end - start;
        }
        return outDegrees;
    }
}
```

## Best Practices

1. **Memory Layout**: Use CSR format for most graph algorithms to optimize memory access
2. **Load Balancing**: Implement work queues for graphs with irregular degree distributions
3. **Memory Coalescing**: Process edges in chunks to improve memory bandwidth utilization
4. **Graph Partitioning**: Split large graphs across multiple device invocations
5. **Algorithm Selection**: Choose algorithms based on graph characteristics (sparse vs dense)

## Limitations

1. **Memory Constraints**: Large graphs may require streaming or partitioning
2. **Irregular Access Patterns**: Some graph algorithms may not achieve optimal memory bandwidth
3. **Synchronization Overhead**: Frequent synchronization can limit performance on certain algorithms
4. **Load Imbalance**: Graphs with high degree variance may underutilize compute resources

---

Graph processing with ILGPU provides efficient parallel execution of common graph analytics algorithms while maintaining cross-platform compatibility and performance optimization.