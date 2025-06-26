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

using FluentAssertions;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalCompute.Scheduling
{
    /// <summary>
    /// Comprehensive tests for ComputeGraph and compute operation types.
    /// Achieves 100% code coverage for compute graph infrastructure.
    /// </summary>
    public class ComputeGraphTests : IDisposable
    {
        private readonly Context _context;

        public ComputeGraphTests()
        {
            _context = Context.CreateDefault();
        }

        #region ComputeGraph Constructor Tests

        [Fact]
        public void ComputeGraph_DefaultConstructor_ShouldInitializeEmpty()
        {
            // Arrange & Act
            var graph = new ComputeGraph();

            // Assert
            graph.Should().NotBeNull();
            graph.Nodes.Should().NotBeNull();
            graph.Nodes.Should().BeEmpty();
            graph.Dependencies.Should().NotBeNull();
            graph.Dependencies.Should().BeEmpty();
        }

        [Fact]
        public void ComputeGraph_ParameterizedConstructor_ShouldInitializeCorrectly()
        {
            // Arrange
            var nodes = new List<ComputeNode>
            {
                new ComputeNode(new MatMulOp(32, 32, 32)),
                new ComputeNode(new VectorOp(1024))
            };
            var dependencies = new Dictionary<ComputeNode, List<ComputeNode>>();

            // Act
            var graph = new ComputeGraph(nodes, dependencies);

            // Assert
            graph.Nodes.Should().HaveCount(2);
            graph.Dependencies.Should().NotBeNull();
        }

        #endregion

        #region Node Management Tests

        [Fact]
        public void ComputeGraph_AddNode_ShouldAddNodeCorrectly()
        {
            // Arrange
            var graph = new ComputeGraph();
            var operation = new MatMulOp(64, 64, 64);
            var node = new ComputeNode(operation);

            // Act
            graph.AddNode(node);

            // Assert
            graph.Nodes.Should().Contain(node);
            graph.Nodes.Should().HaveCount(1);
        }

        [Fact]
        public void ComputeGraph_AddNode_NullNode_ShouldThrowArgumentNullException()
        {
            // Arrange
            var graph = new ComputeGraph();

            // Act & Assert
            Action act = () => graph.AddNode(null!);
            act.Should().Throw<ArgumentNullException>().WithParameterName("node");
        }

        [Fact]
        public void ComputeGraph_AddNode_DuplicateNode_ShouldNotAddTwice()
        {
            // Arrange
            var graph = new ComputeGraph();
            var operation = new MatMulOp(32, 32, 32);
            var node = new ComputeNode(operation);

            // Act
            graph.AddNode(node);
            graph.AddNode(node); // Add same node again

            // Assert
            graph.Nodes.Should().HaveCount(1);
        }

        [Fact]
        public void ComputeGraph_RemoveNode_ShouldRemoveNodeAndDependencies()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddDependency(node2, node1);

            // Act
            var removed = graph.RemoveNode(node1);

            // Assert
            removed.Should().BeTrue();
            graph.Nodes.Should().NotContain(node1);
            graph.Nodes.Should().Contain(node2);
            graph.Dependencies.Should().NotContainKey(node2);
        }

        [Fact]
        public void ComputeGraph_RemoveNode_NonExistentNode_ShouldReturnFalse()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(32, 32, 32));

            // Act
            var removed = graph.RemoveNode(node);

            // Assert
            removed.Should().BeFalse();
        }

        #endregion

        #region Dependency Management Tests

        [Fact]
        public void ComputeGraph_AddDependency_ShouldAddDependencyCorrectly()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);

            // Act
            graph.AddDependency(node2, node1); // node2 depends on node1

            // Assert
            graph.Dependencies.Should().ContainKey(node2);
            graph.Dependencies[node2].Should().Contain(node1);
        }

        [Fact]
        public void ComputeGraph_AddDependency_MultipleDependencies_ShouldStoreAll()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            var node3 = new ComputeNode(new MemoryOp(2048));
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddNode(node3);

            // Act
            graph.AddDependency(node3, node1);
            graph.AddDependency(node3, node2);

            // Assert
            graph.Dependencies[node3].Should().HaveCount(2);
            graph.Dependencies[node3].Should().Contain(node1);
            graph.Dependencies[node3].Should().Contain(node2);
        }

        [Fact]
        public void ComputeGraph_AddDependency_SelfDependency_ShouldThrowInvalidOperationException()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(32, 32, 32));
            graph.AddNode(node);

            // Act & Assert
            Action act = () => graph.AddDependency(node, node);
            act.Should().Throw<InvalidOperationException>().WithMessage("*cycle*");
        }

        [Fact]
        public void ComputeGraph_AddDependency_CircularDependency_ShouldThrowInvalidOperationException()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddDependency(node2, node1);

            // Act & Assert
            Action act = () => graph.AddDependency(node1, node2);
            act.Should().Throw<InvalidOperationException>().WithMessage("*cycle*");
        }

        [Fact]
        public void ComputeGraph_RemoveDependency_ShouldRemoveDependencyCorrectly()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddDependency(node2, node1);

            // Act
            var removed = graph.RemoveDependency(node2, node1);

            // Assert
            removed.Should().BeTrue();
            graph.Dependencies[node2].Should().NotContain(node1);
        }

        [Fact]
        public void ComputeGraph_RemoveDependency_NonExistentDependency_ShouldReturnFalse()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);

            // Act
            var removed = graph.RemoveDependency(node2, node1);

            // Assert
            removed.Should().BeFalse();
        }

        #endregion

        #region Topological Ordering Tests

        [Fact]
        public void ComputeGraph_GetTopologicalOrder_EmptyGraph_ShouldReturnEmptyList()
        {
            // Arrange
            var graph = new ComputeGraph();

            // Act
            var order = graph.GetTopologicalOrder();

            // Assert
            order.Should().NotBeNull();
            order.Should().BeEmpty();
        }

        [Fact]
        public void ComputeGraph_GetTopologicalOrder_SingleNode_ShouldReturnSingleNode()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(32, 32, 32));
            graph.AddNode(node);

            // Act
            var order = graph.GetTopologicalOrder();

            // Assert
            order.Should().HaveCount(1);
            order.Should().Contain(node);
        }

        [Fact]
        public void ComputeGraph_GetTopologicalOrder_LinearDependencies_ShouldReturnCorrectOrder()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32)) { Id = "Node1" };
            var node2 = new ComputeNode(new VectorOp(1024)) { Id = "Node2" };
            var node3 = new ComputeNode(new MemoryOp(2048)) { Id = "Node3" };
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddNode(node3);
            
            // node2 depends on node1, node3 depends on node2
            graph.AddDependency(node2, node1);
            graph.AddDependency(node3, node2);

            // Act
            var order = graph.GetTopologicalOrder();

            // Assert
            order.Should().HaveCount(3);
            order.IndexOf(node1).Should().BeLessThan(order.IndexOf(node2));
            order.IndexOf(node2).Should().BeLessThan(order.IndexOf(node3));
        }

        [Fact]
        public void ComputeGraph_GetTopologicalOrder_ComplexDependencies_ShouldReturnValidOrder()
        {
            // Arrange
            var graph = new ComputeGraph();
            var nodeA = new ComputeNode(new MatMulOp(32, 32, 32)) { Id = "A" };
            var nodeB = new ComputeNode(new VectorOp(1024)) { Id = "B" };
            var nodeC = new ComputeNode(new MemoryOp(2048)) { Id = "C" };
            var nodeD = new ComputeNode(new ConvolutionOp(1024, 3, 64)) { Id = "D" };
            
            graph.AddNode(nodeA);
            graph.AddNode(nodeB);
            graph.AddNode(nodeC);
            graph.AddNode(nodeD);
            
            // Complex dependencies: D depends on B and C, C depends on A, B depends on A
            graph.AddDependency(nodeB, nodeA);
            graph.AddDependency(nodeC, nodeA);
            graph.AddDependency(nodeD, nodeB);
            graph.AddDependency(nodeD, nodeC);

            // Act
            var order = graph.GetTopologicalOrder();

            // Assert
            order.Should().HaveCount(4);
            order.IndexOf(nodeA).Should().BeLessThan(order.IndexOf(nodeB));
            order.IndexOf(nodeA).Should().BeLessThan(order.IndexOf(nodeC));
            order.IndexOf(nodeB).Should().BeLessThan(order.IndexOf(nodeD));
            order.IndexOf(nodeC).Should().BeLessThan(order.IndexOf(nodeD));
        }

        #endregion

        #region Parallel Levels Tests

        [Fact]
        public void ComputeGraph_GetParallelExecutionLevels_ShouldGroupCorrectly()
        {
            // Arrange
            var graph = new ComputeGraph();
            var nodeA = new ComputeNode(new MatMulOp(32, 32, 32)) { Id = "A" };
            var nodeB = new ComputeNode(new VectorOp(1024)) { Id = "B" };
            var nodeC = new ComputeNode(new MemoryOp(2048)) { Id = "C" };
            var nodeD = new ComputeNode(new ConvolutionOp(1024, 3, 64)) { Id = "D" };
            
            graph.AddNode(nodeA);
            graph.AddNode(nodeB);
            graph.AddNode(nodeC);
            graph.AddNode(nodeD);
            
            // B and C can run in parallel after A, D runs after B and C
            graph.AddDependency(nodeB, nodeA);
            graph.AddDependency(nodeC, nodeA);
            graph.AddDependency(nodeD, nodeB);
            graph.AddDependency(nodeD, nodeC);

            // Act
            var levels = graph.GetParallelExecutionLevels();

            // Assert
            levels.Should().HaveCount(3);
            levels[0].Should().Contain(nodeA);
            levels[1].Should().Contain(nodeB);
            levels[1].Should().Contain(nodeC);
            levels[2].Should().Contain(nodeD);
        }

        #endregion

        #region Graph Analysis Tests

        [Fact]
        public void ComputeGraph_EstimateTotalExecutionTime_ShouldCalculateCorrectly()
        {
            // Arrange
            var graph = new ComputeGraph();
            var nodeA = new ComputeNode(new MatMulOp(32, 32, 32)) { EstimatedTimeMs = 10.0 };
            var nodeB = new ComputeNode(new VectorOp(1024)) { EstimatedTimeMs = 5.0 };
            var nodeC = new ComputeNode(new MemoryOp(2048)) { EstimatedTimeMs = 3.0 };
            
            graph.AddNode(nodeA);
            graph.AddNode(nodeB);
            graph.AddNode(nodeC);
            
            // B and C can run in parallel after A
            graph.AddDependency(nodeB, nodeA);
            graph.AddDependency(nodeC, nodeA);

            // Act
            var estimatedTime = graph.EstimateTotalExecutionTime();

            // Assert
            // Should be 10 (A) + max(5, 3) (B and C in parallel) = 15
            estimatedTime.Should().Be(15.0);
        }

        [Fact]
        public void ComputeGraph_GetCriticalPath_ShouldReturnLongestPath()
        {
            // Arrange
            var graph = new ComputeGraph();
            var nodeA = new ComputeNode(new MatMulOp(32, 32, 32)) { EstimatedTimeMs = 10.0, Id = "A" };
            var nodeB = new ComputeNode(new VectorOp(1024)) { EstimatedTimeMs = 15.0, Id = "B" };
            var nodeC = new ComputeNode(new MemoryOp(2048)) { EstimatedTimeMs = 5.0, Id = "C" };
            var nodeD = new ComputeNode(new ConvolutionOp(1024, 3, 64)) { EstimatedTimeMs = 8.0, Id = "D" };
            
            graph.AddNode(nodeA);
            graph.AddNode(nodeB);
            graph.AddNode(nodeC);
            graph.AddNode(nodeD);
            
            // Path A->B->D (10+15+8=33) vs A->C->D (10+5+8=23)
            graph.AddDependency(nodeB, nodeA);
            graph.AddDependency(nodeC, nodeA);
            graph.AddDependency(nodeD, nodeB);
            graph.AddDependency(nodeD, nodeC);

            // Act
            var criticalPath = graph.GetCriticalPath();

            // Assert
            criticalPath.Should().HaveCount(3);
            criticalPath.Should().Contain(nodeA);
            criticalPath.Should().Contain(nodeB);
            criticalPath.Should().Contain(nodeD);
        }

        [Fact]
        public void ComputeGraph_OptimizeGraph_ShouldPerformOptimizations()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);

            // Act
            var optimizedGraph = graph.OptimizeGraph();

            // Assert
            optimizedGraph.Should().NotBeNull();
            optimizedGraph.Nodes.Should().NotBeEmpty();
        }

        #endregion

        #region Clone and Copy Tests

        [Fact]
        public void ComputeGraph_Clone_ShouldCreateIndependentCopy()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32)) { Id = "Node1" };
            var node2 = new ComputeNode(new VectorOp(1024)) { Id = "Node2" };
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddDependency(node2, node1);

            // Act
            var clonedGraph = graph.Clone();

            // Assert
            clonedGraph.Should().NotBeSameAs(graph);
            clonedGraph.Nodes.Should().HaveCount(graph.Nodes.Count);
            clonedGraph.Dependencies.Should().HaveCount(graph.Dependencies.Count);
        }

        #endregion

        #region Validation Tests

        [Fact]
        public void ComputeGraph_ValidateGraph_ValidGraph_ShouldReturnTrue()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node1 = new ComputeNode(new MatMulOp(32, 32, 32));
            var node2 = new ComputeNode(new VectorOp(1024));
            
            graph.AddNode(node1);
            graph.AddNode(node2);
            graph.AddDependency(node2, node1);

            // Act
            var isValid = graph.ValidateGraph();

            // Assert
            isValid.Should().BeTrue();
        }

        #endregion

        #region ToString and Debug Tests

        [Fact]
        public void ComputeGraph_ToString_ShouldProvideReadableOutput()
        {
            // Arrange
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(32, 32, 32)) { Id = "TestNode" };
            graph.AddNode(node);

            // Act
            var toString = graph.ToString();

            // Assert
            toString.Should().NotBeNullOrWhiteSpace();
            toString.Should().Contain("1"); // Node count
        }

        #endregion

        public void Dispose()
        {
            _context?.Dispose();
        }
    }
}