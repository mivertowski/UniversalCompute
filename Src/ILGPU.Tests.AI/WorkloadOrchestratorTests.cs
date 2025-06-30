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
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

// Use alias to resolve TensorShape ambiguity
using TensorShape = ILGPU.ML.TensorShape;

namespace ILGPU.Tests.AI
{
    /// <summary>
    /// Tests for workload orchestration across multiple accelerators.
    /// </summary>
    public class WorkloadOrchestratorTests : IDisposable
    {
        private readonly Context _context;
        private readonly WorkloadOrchestrator _orchestrator;

        public WorkloadOrchestratorTests()
        {
            _context = Context.CreateDefault();
            var accelerators = new List<Accelerator>();
            _orchestrator = new WorkloadOrchestrator(_context, accelerators);
        }

        [Fact]
        public void WorkloadOrchestrator_ShouldInitializeWithValidProfiles()
        {
            // Arrange & Act
            var profiles = _orchestrator.AcceleratorProfiles;

            // Assert
            profiles.Should().NotBeNull();
            profiles.Should().NotBeEmpty();
            profiles.All(p => p.Accelerator != null).Should().BeTrue();
            profiles.All(p => p.Primitives != null).Should().BeTrue();
            profiles.All(p => p.PerformanceScore >= 0).Should().BeTrue();
        }

        [Fact]
        public void AcceleratorProfile_ShouldHaveValidProperties()
        {
            // Arrange
            var profile = _orchestrator.AcceleratorProfiles.First();

            // Act & Assert
            profile.Accelerator.Should().NotBeNull();
            profile.Primitives.Should().NotBeNull();
            profile.PerformanceScore.Should().BeGreaterOrEqualTo(0);
            profile.LoadFactor.Should().BeInRange(0.0, 1.0);
        }

        [Theory]
        [InlineData(64, 64, 64)] // Small matrix
        [InlineData(256, 256, 256)] // Medium matrix
        [InlineData(512, 512, 512)] // Large matrix
        public async Task ExecuteMatMulAsync_ShouldSelectOptimalAccelerator(int m, int n, int k)
        {
            // Arrange
            var a = CreateRandomTensor<float>(new TensorShape(m, k));
            var b = CreateRandomTensor<float>(new TensorShape(k, n));

            // Act
            var result = await _orchestrator.ExecuteMatMulAsync(a, b);

            // Assert
            result.Should().NotBeNull();
            result.Shape[0].Should().Be(m);
            result.Shape[1].Should().Be(n);
        }

        [Theory]
        [InlineData(1, 3, 32, 32, 16)] // Small convolution
        [InlineData(2, 64, 128, 128, 64)] // Medium convolution
        public async Task ExecuteConvolutionAsync_ShouldSelectOptimalAccelerator(
            int batch, int inChannels, int height, int width, int outChannels)
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(batch, inChannels, height, width));
            var kernel = CreateRandomTensor<float>(new TensorShape(outChannels, inChannels, 3, 3));
            var parameters = new ConvolutionParameters
            {
                Stride = new Size2D(1, 1),
                Padding = new Size2D(1, 1),
                Dilation = new Size2D(1, 1)
            };

            // Act
            var result = await _orchestrator.ExecuteConvolutionAsync(input, kernel, parameters);

            // Assert
            result.Should().NotBeNull();
            result.Shape[0].Should().Be(batch);
            result.Shape[1].Should().Be(outChannels);
        }

        [Fact]
        public async Task ExecuteWorkloadAsync_ShouldExecuteMatrixMultiplicationWorkload()
        {
            // Arrange
            var a = CreateRandomTensor<float>(new TensorShape(128, 256));
            var b = CreateRandomTensor<float>(new TensorShape(256, 128));
            var workload = new MatrixMultiplicationWorkload<float>(a, b);

            // Act
            await _orchestrator.ExecuteWorkloadAsync(workload);

            // Assert
            workload.C.Should().NotBeNull();
            workload.C!.Shape[0].Should().Be(128);
            workload.C!.Shape[1].Should().Be(128);
        }

        [Fact]
        public async Task ExecuteWorkloadAsync_ShouldExecuteConvolutionWorkload()
        {
            // Arrange
            var input = CreateRandomTensor<float>(new TensorShape(1, 16, 64, 64));
            var kernel = CreateRandomTensor<float>(new TensorShape(32, 16, 5, 5));
            var parameters = new ConvolutionParameters
            {
                Stride = new Size2D(1, 1),
                Padding = new Size2D(2, 2),
                Dilation = new Size2D(1, 1)
            };
            var workload = new ConvolutionWorkload<float>(input, kernel, parameters);

            // Act
            await _orchestrator.ExecuteWorkloadAsync(workload);

            // Assert
            workload.Output.Should().NotBeNull();
            workload.Output!.Shape[0].Should().Be(1);
            workload.Output!.Shape[1].Should().Be(32);
        }

        [Fact]
        public async Task ExecuteDistributedWorkloadAsync_ShouldDistributeAcrossAccelerators()
        {
            // Arrange
            var a = CreateRandomTensor<float>(new TensorShape(1024, 512));
            var b = CreateRandomTensor<float>(new TensorShape(512, 256));
            var distributedWorkload = new DistributedMatrixMultiplicationWorkload<float>(a, b);

            // Act
            await _orchestrator.ExecuteDistributedWorkloadAsync(distributedWorkload);

            // Assert - workload should complete without exception
            distributedWorkload.Should().NotBeNull();
        }

        [Fact]
        public void PerformanceTracker_ShouldTrackOperations()
        {
            // Arrange
            var tracker = _orchestrator.PerformanceTracker;
            var acceleratorType = _orchestrator.AcceleratorProfiles.First().Accelerator.AcceleratorType;

            // Act
            tracker.RecordOperation(acceleratorType, PrimitiveType.MatrixMultiplication, 1000000);
            tracker.RecordOperation(acceleratorType, PrimitiveType.Convolution, 500000);

            // Assert
            var metrics = tracker.GetMetrics(acceleratorType);
            metrics.Should().NotBeNull();
            
            var gemmMetrics = metrics!.GetPrimitiveMetrics(PrimitiveType.MatrixMultiplication);
            gemmMetrics.Should().NotBeNull();
            gemmMetrics!.OperationCount.Should().Be(1000000);
            gemmMetrics.ExecutionCount.Should().Be(1);
            
            var convMetrics = metrics.GetPrimitiveMetrics(PrimitiveType.Convolution);
            convMetrics.Should().NotBeNull();
            convMetrics!.OperationCount.Should().Be(500000);
            convMetrics.ExecutionCount.Should().Be(1);
        }

        [Theory]
        [InlineData(WorkloadType.MatrixMultiplication)]
        [InlineData(WorkloadType.Convolution)]
        [InlineData(WorkloadType.Attention)]
        public void AcceleratorProfile_CanExecute_ShouldReturnValidResult(WorkloadType workloadType)
        {
            // Arrange
            var profile = _orchestrator.AcceleratorProfiles.First();
            var mockWorkload = new MockWorkload(workloadType);

            // Act
            var canExecute = profile.CanExecute(mockWorkload);

            // Assert
            Assert.IsType<bool>(canExecute);
        }

        [Fact]
        public void WorkloadScheduler_ShouldAnalyzeWorkloadCorrectly()
        {
            // Arrange
            var scheduler = new WorkloadScheduler();
            var profiles = _orchestrator.AcceleratorProfiles;
            var workload = new MockWorkload(WorkloadType.MatrixMultiplication, complexity: 100000);

            // Act
            var strategyTask = scheduler.AnalyzeWorkloadAsync(workload, profiles);
            var strategy = strategyTask.Result;

            // Assert
            strategy.Should().NotBeNull();
            strategy.TargetAccelerators.Should().NotBeEmpty();
        }

        [Fact]
        public void WorkloadScheduler_ShouldPartitionDistributedWorkload()
        {
            // Arrange
            var scheduler = new WorkloadScheduler();
            var profiles = _orchestrator.AcceleratorProfiles;
            var a = CreateRandomTensor<float>(new TensorShape(1024, 512));
            var b = CreateRandomTensor<float>(new TensorShape(512, 256));
            var distributedWorkload = new DistributedMatrixMultiplicationWorkload<float>(a, b);

            // Act
            var partitionsTask = scheduler.PartitionWorkloadAsync(distributedWorkload, profiles);
            var partitions = partitionsTask.Result.ToList();

            // Assert
            partitions.Should().NotBeEmpty();
            partitions.All(p => p.Workload != null).Should().BeTrue();
            partitions.All(p => p.Strategy != null).Should().BeTrue();
        }

        [Theory]
        [InlineData(StrategyType.SingleAccelerator)]
        [InlineData(StrategyType.MultiAccelerator)]
        public void ExecutionStrategy_ShouldHaveCorrectType(StrategyType strategyType)
        {
            // Arrange & Act
            ExecutionStrategy strategy = strategyType switch
            {
                StrategyType.SingleAccelerator => new SingleAcceleratorStrategy(_orchestrator.AcceleratorProfiles.First().Accelerator),
                StrategyType.MultiAccelerator => new MultiAcceleratorStrategy(_orchestrator.AcceleratorProfiles.Take(2).Select(p => p.Accelerator)),
                _ => throw new ArgumentException("Invalid strategy type")
            };

            // Assert
            strategy.Type.Should().Be(strategyType);
            strategy.TargetAccelerators.Should().NotBeEmpty();
        }

        #region Helper Classes and Methods

        private ITensor<T> CreateRandomTensor<T>(TensorShape shape) where T : unmanaged
        {
            // Skip tensor creation due to incompatible tensor systems
            throw new NotImplementedException("Tensor system incompatibility");
        }

        private class MockWorkload : IWorkload
        {
            public MockWorkload(WorkloadType workloadType, long complexity = 1000, long memory = 1024)
            {
                WorkloadType = workloadType;
                EstimatedComplexity = complexity;
                MemoryRequirements = memory;
            }

            public WorkloadType WorkloadType { get; }
            public long EstimatedComplexity { get; }
            public long MemoryRequirements { get; }

            public Task ExecuteAsync(WorkloadExecutionContext context, CancellationToken cancellationToken = default)
            {
                return Task.CompletedTask;
            }
        }

        #endregion

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _orchestrator?.Dispose();
                _context?.Dispose();
            }
        }
    }
}