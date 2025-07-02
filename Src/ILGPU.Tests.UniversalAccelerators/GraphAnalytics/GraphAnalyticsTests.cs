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

using ILGPU.Algorithms.GraphAnalytics;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.GraphAnalytics
{
    /// <summary>
    /// Tests for graph analytics algorithms.
    /// </summary>
    public class GraphAnalyticsTests : TestBase
    {
        #region Graph Traversal Tests

        [Fact]
        public void TestBreadthFirstSearch()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a simple graph: 0 -> 1 -> 2, 0 -> 3
            const int numVertices = 4;
            var edges = new int[] { 0, 1, 1, 2, 0, 3 }; // Edge pairs
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var distancesBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var visitedBuffer = accelerator!.Allocate1D<bool>(numVertices);
            
            // Initialize arrays
            var distances = new int[numVertices];
            var visited = new bool[numVertices];
            for (int i = 0; i < numVertices; i++)
            {
                distances[i] = int.MaxValue;
                visited[i] = false;
            }
            distances[0] = 0; // Start from vertex 0
            
            distancesBuffer.CopyFromCPU(distances);
            visitedBuffer.CopyFromCPU(visited);
            
            // Run BFS
            GraphTraversal.BreadthFirstSearch(
                graphBuffer, 
                0, // Start vertex
                distancesBuffer.View, 
                visitedBuffer.View,
                accelerator!.DefaultStream);
            
            var resultDistances = distancesBuffer.GetAsArray1D();
            var resultVisited = visitedBuffer.GetAsArray1D();
            
            // Verify BFS results
            Assert.Equal(0, resultDistances[0]); // Distance to self
            Assert.Equal(1, resultDistances[1]); // Distance to vertex 1
            Assert.Equal(2, resultDistances[2]); // Distance to vertex 2
            Assert.Equal(1, resultDistances[3]); // Distance to vertex 3
            
            // All vertices should be visited
            Assert.True(resultVisited.All(v => v), "Not all vertices were visited");
        }

        [Fact]
        public void TestDepthFirstSearch()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a tree: 0 -> 1, 1 -> 2, 1 -> 3
            const int numVertices = 4;
            var edges = new int[] { 0, 1, 1, 2, 1, 3 };
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var discoveryTimeBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var finishTimeBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var visitedBuffer = accelerator!.Allocate1D<bool>(numVertices);
            
            GraphTraversal.DepthFirstSearch(
                graphBuffer,
                0, // Start vertex
                discoveryTimeBuffer.View,
                finishTimeBuffer.View,
                visitedBuffer.View,
                accelerator!.DefaultStream);
            
            var discoveryTimes = discoveryTimeBuffer.GetAsArray1D();
            var finishTimes = finishTimeBuffer.GetAsArray1D();
            var visited = visitedBuffer.GetAsArray1D();
            
            // Verify DFS properties
            Assert.True(visited.All(v => v), "Not all vertices were visited");
            Assert.True(discoveryTimes[0] < finishTimes[0], "Invalid DFS timing for vertex 0");
            Assert.True(discoveryTimes.All(t => t >= 0), "Invalid discovery times");
            Assert.True(finishTimes.All(t => t >= 0), "Invalid finish times");
        }

        #endregion

        #region Shortest Path Tests

        [Fact]
        public void TestSingleSourceShortestPath()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create weighted graph: 0 -2-> 1 -3-> 2, 0 -4-> 2
            const int numVertices = 3;
            var edges = new int[] { 0, 1, 1, 2, 0, 2 };
            var weights = new float[] { 2.0f, 3.0f, 4.0f };
            var graph = new WeightedCSRGraph(numVertices, edges.Length / 2, weights);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var distancesBuffer = accelerator!.Allocate1D<float>(numVertices);
            using var predecessorsBuffer = accelerator!.Allocate1D<int>(numVertices);
            
            ShortestPath.SingleSourceShortestPath(
                graphBuffer,
                0, // Source vertex
                distancesBuffer.View,
                predecessorsBuffer.View,
                accelerator!.DefaultStream);
            
            var distances = distancesBuffer.GetAsArray1D();
            var predecessors = predecessorsBuffer.GetAsArray1D();
            
            // Verify shortest path distances
            Assert.Equal(0.0f, distances[0]); // Distance to self
            Assert.Equal(2.0f, distances[1]); // Shortest path 0 -> 1
            Assert.Equal(4.0f, distances[2]); // Direct path 0 -> 2 (shorter than 0->1->2)
            
            // Verify predecessor relationships
            Assert.Equal(-1, predecessors[0]); // Source has no predecessor
            Assert.Equal(0, predecessors[1]); // 1's predecessor is 0
            Assert.Equal(0, predecessors[2]); // 2's predecessor is 0 (direct path)
        }

        [Fact]
        public void TestAllPairsShortestPath()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numVertices = 4;
            // Create complete graph with unit weights
            var adjacencyMatrix = new float[numVertices * numVertices];
            for (int i = 0; i < numVertices; i++)
            {
                for (int j = 0; j < numVertices; j++)
                {
                    if (i == j)
                        adjacencyMatrix[i * numVertices + j] = 0.0f;
                    else
                        adjacencyMatrix[i * numVertices + j] = 1.0f;
                }
            }
            
            using var matrixBuffer = accelerator!.Allocate1D(adjacencyMatrix);
            using var resultBuffer = accelerator!.Allocate1D<float>(numVertices * numVertices);
            
            ShortestPath.AllPairsShortestPath(
                matrixBuffer.View,
                resultBuffer.View,
                numVertices,
                accelerator!.DefaultStream);
            
            var result = resultBuffer.GetAsArray1D();
            
            // Verify all-pairs shortest paths
            for (int i = 0; i < numVertices; i++)
            {
                for (int j = 0; j < numVertices; j++)
                {
                    var expected = (i == j) ? 0.0f : 1.0f; // Complete graph with unit weights
                    Assert.Equal(expected, result[i * numVertices + j]);
                }
            }
        }

        #endregion

        #region Centrality Tests

        [Fact]
        public void TestBetweennessCentrality()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a path graph: 0 - 1 - 2 - 3
            const int numVertices = 4;
            var edges = new int[] { 0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2 }; // Undirected edges
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var centralityBuffer = accelerator!.Allocate1D<float>(numVertices);
            
            CentralityMeasures.BetweennessCentrality(
                graphBuffer,
                centralityBuffer.View,
                accelerator!.DefaultStream);
            
            var centrality = centralityBuffer.GetAsArray1D();
            
            // In a path graph, middle vertices have higher betweenness centrality
            Assert.True(centrality[1] > centrality[0], "Vertex 1 should have higher centrality than vertex 0");
            Assert.True(centrality[2] > centrality[3], "Vertex 2 should have higher centrality than vertex 3");
            Assert.True(centrality[1] > 0, "Middle vertices should have positive centrality");
            Assert.True(centrality[2] > 0, "Middle vertices should have positive centrality");
        }

        [Fact]
        public void TestPageRank()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a simple directed graph: 0 -> 1, 1 -> 2, 2 -> 0
            const int numVertices = 3;
            var edges = new int[] { 0, 1, 1, 2, 2, 0 };
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var pageRankBuffer = accelerator!.Allocate1D<float>(numVertices);
            
            CentralityMeasures.PageRank(
                graphBuffer,
                pageRankBuffer.View,
                dampingFactor: 0.85f,
                tolerance: 1e-6f,
                maxIterations: 100,
                accelerator!.DefaultStream);
            
            var pageRank = pageRankBuffer.GetAsArray1D();
            
            // Verify PageRank properties
            var sum = pageRank.Sum();
            Assert.True(Math.Abs(sum - 1.0f) < 1e-5f, $"PageRank values should sum to 1, got {sum}");
            Assert.True(pageRank.All(pr => pr > 0), "All PageRank values should be positive");
            
            // In a symmetric cycle, all vertices should have equal PageRank
            var avgPageRank = sum / numVertices;
            for (int i = 0; i < numVertices; i++)
            {
                Assert.True(Math.Abs(pageRank[i] - avgPageRank) < 1e-4f,
                    $"PageRank values should be equal in symmetric graph, vertex {i}: {pageRank[i]} vs avg {avgPageRank}");
            }
        }

        [Fact]
        public void TestClosenessCentrality()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a star graph: center vertex connected to all others
            const int numVertices = 5;
            const int center = 0;
            var edges = new int[8 * (numVertices - 1)]; // Undirected star
            int edgeIndex = 0;
            
            for (int i = 1; i < numVertices; i++)
            {
                edges[edgeIndex++] = center;
                edges[edgeIndex++] = i;
                edges[edgeIndex++] = i;
                edges[edgeIndex++] = center;
            }
            
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var centralityBuffer = accelerator!.Allocate1D<float>(numVertices);
            
            CentralityMeasures.ClosenessCentrality(
                graphBuffer,
                centralityBuffer.View,
                accelerator!.DefaultStream);
            
            var centrality = centralityBuffer.GetAsArray1D();
            
            // Center vertex should have highest closeness centrality
            Assert.True(centrality[center] > centrality[1], "Center should have highest closeness centrality");
            
            // All non-center vertices should have equal centrality
            for (int i = 2; i < numVertices; i++)
            {
                Assert.True(Math.Abs(centrality[1] - centrality[i]) < 1e-5f,
                    $"Non-center vertices should have equal centrality: {centrality[1]} vs {centrality[i]}");
            }
        }

        #endregion

        #region Community Detection Tests

        [Fact]
        public void TestLouvainCommunityDetection()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create two disconnected cliques
            const int numVertices = 6;
            var edges = new int[] { 
                // First clique: 0, 1, 2
                0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 2, 1,
                // Second clique: 3, 4, 5
                3, 4, 4, 3, 3, 5, 5, 3, 4, 5, 5, 4
            };
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var communityBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var modularityBuffer = accelerator!.Allocate1D<float>(1);
            
            CommunityDetection.LouvainAlgorithm(
                graphBuffer,
                communityBuffer.View,
                modularityBuffer.View,
                accelerator!.DefaultStream);
            
            var communities = communityBuffer.GetAsArray1D();
            var modularity = modularityBuffer.GetAsArray1D();
            
            // Verify community structure
            // Vertices 0, 1, 2 should be in one community
            Assert.Equal(communities[0], communities[1]);
            Assert.Equal(communities[1], communities[2]);
            
            // Vertices 3, 4, 5 should be in another community
            Assert.Equal(communities[3], communities[4]);
            Assert.Equal(communities[4], communities[5]);
            
            // The two groups should be in different communities
            Assert.NotEqual(communities[0], communities[3]);
            
            // Modularity should be positive for good community structure
            Assert.True(modularity[0] > 0, $"Modularity should be positive, got {modularity[0]}");
        }

        [Fact]
        public void TestSpectralClustering()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numVertices = 4;
            // Create adjacency matrix for two pairs
            var adjacency = new float[]
            {
                0, 1, 0, 0,  // 0 connected to 1
                1, 0, 0, 0,  // 1 connected to 0
                0, 0, 0, 1,  // 2 connected to 3
                0, 0, 1, 0   // 3 connected to 2
            };
            
            using var adjacencyBuffer = accelerator!.Allocate1D(adjacency);
            using var clusterBuffer = accelerator!.Allocate1D<int>(numVertices);
            
            CommunityDetection.SpectralClustering(
                adjacencyBuffer.View,
                clusterBuffer.View,
                numVertices,
                numClusters: 2,
                accelerator!.DefaultStream);
            
            var clusters = clusterBuffer.GetAsArray1D();
            
            // Verify clustering: {0,1} and {2,3} should be in different clusters
            Assert.Equal(clusters[0], clusters[1]);
            Assert.Equal(clusters[2], clusters[3]);
            Assert.NotEqual(clusters[0], clusters[2]);
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestLargeGraphPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numVertices = 10000;
            const int avgDegree = 10;
            
            // Generate random graph
            var random = new Random(42);
            var edges = new System.Collections.Generic.List<int>();
            
            for (int i = 0; i < numVertices; i++)
            {
                int degree = random.Next(avgDegree / 2, avgDegree * 2);
                for (int j = 0; j < degree; j++)
                {
                    int target = random.Next(numVertices);
                    if (target != i)
                    {
                        edges.Add(i);
                        edges.Add(target);
                    }
                }
            }
            
            var graph = new CSRGraph(numVertices, edges.Count / 2);
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var distancesBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var visitedBuffer = accelerator!.Allocate1D<bool>(numVertices);
            
            // Measure BFS performance on large graph
            var bfsTime = MeasureTime(() =>
            {
                GraphTraversal.BreadthFirstSearch(
                    graphBuffer, 
                    0, 
                    distancesBuffer.View, 
                    visitedBuffer.View,
                    accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            Assert.True(bfsTime < 5000, $"BFS on large graph took {bfsTime}ms, expected < 5000ms");
            
            // Verify some vertices were reached
            var distances = distancesBuffer.GetAsArray1D();
            var reachableCount = distances.Count(d => d != int.MaxValue);
            Assert.True(reachableCount > numVertices / 10, 
                $"Too few vertices reachable: {reachableCount} out of {numVertices}");
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestGraphErrorHandling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test invalid vertex index
            const int numVertices = 3;
            var graph = new CSRGraph(numVertices, 0);
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var distancesBuffer = accelerator!.Allocate1D<int>(numVertices);
            using var visitedBuffer = accelerator!.Allocate1D<bool>(numVertices);
            
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                GraphTraversal.BreadthFirstSearch(
                    graphBuffer, 
                    numVertices, // Invalid start vertex
                    distancesBuffer.View, 
                    visitedBuffer.View,
                    accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestPageRankConvergence()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test PageRank with very strict tolerance
            const int numVertices = 3;
            var edges = new int[] { 0, 1, 1, 2, 2, 0 };
            var graph = new CSRGraph(numVertices, edges.Length / 2);
            
            using var graphBuffer = graph.ToGPUFormat(accelerator!);
            using var pageRankBuffer = accelerator!.Allocate1D<float>(numVertices);
            
            // Should converge even with strict tolerance
            CentralityMeasures.PageRank(
                graphBuffer,
                pageRankBuffer.View,
                dampingFactor: 0.85f,
                tolerance: 1e-10f, // Very strict
                maxIterations: 1000,
                accelerator!.DefaultStream);
            
            var pageRank = pageRankBuffer.GetAsArray1D();
            var sum = pageRank.Sum();
            
            Assert.True(Math.Abs(sum - 1.0f) < 1e-8f, 
                $"PageRank should converge to sum=1 even with strict tolerance, got {sum}");
        }

        #endregion
    }
}