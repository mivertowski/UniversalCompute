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
using ILGPU.CrossPlatform;
using ILGPU.Memory.Unified;
using ILGPU.ML.Integration;
using ILGPU.Runtime;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.Phase8.Integration
{
    /// <summary>
    /// Comprehensive integration tests for Phase 8 Universal Compute Platform.
    /// Tests end-to-end functionality across all Phase 8 components.
    /// </summary>
    public class Phase8IntegrationTests : IDisposable
    {
        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly Dictionary<ComputeDevice, IAccelerator> _availableAccelerators;

        public Phase8IntegrationTests()
        {
            _context = Context.CreateDefault();
            _accelerator = _context.CreateCPUAccelerator(0);
            _availableAccelerators = new Dictionary<ComputeDevice, IAccelerator>
            {
                { ComputeDevice.CPU, _accelerator }
            };
        }

        #region Universal Kernel End-to-End Tests

        [Fact]
        public void UniversalKernel_CompleteWorkflow_ShouldExecuteSuccessfully()
        {
            // Arrange - Create a universal kernel with platform optimizations
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>>(UniversalOptimizedKernel);
            using var buffer = _accelerator.Allocate1D<float>(1024);
            var data = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();
            buffer.CopyFromCPU(data);

            // Act - Execute the universal kernel
            kernel(buffer.View);
            _accelerator.Synchronize();

            // Assert - Verify results
            var result = buffer.GetAsArray1D();
            for (int i = 0; i < result.Length; i++)
            {
                result[i].Should().Be(data[i] * 2.0f);
            }
        }

        [UniversalKernel(EnableOptimizations = true, SupportsMixedPrecision = true)]
        [AppleOptimization(UseAMX = true)]
        [IntelOptimization(UseAVX512 = true)]
        [NvidiaOptimization(UseTensorCores = true)]
        private static void UniversalOptimizedKernel(ArrayView<float> data)
        {
            var index = UniversalGrid.GlobalIndex.X;
            if (index < data.Length)
            {
                data[index] = data[index] * 2.0f;
            }
        }

        #endregion

        #region Universal Memory Management Integration Tests

        [Fact]
        public void UniversalMemoryManager_CompleteWorkflow_ShouldManageMemoryEfficiently()
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);

            // Act - Allocate buffers with different strategies
            using var deviceBuffer = memoryManager.AllocateUniversal<float>(
                1024, MemoryPlacement.DeviceLocal, MemoryAccessPattern.Sequential);
            using var hostBuffer = memoryManager.AllocateUniversal<int>(
                512, MemoryPlacement.HostLocal, MemoryAccessPattern.Random);
            using var autoBuffer = memoryManager.AllocateUniversal<double>(
                256, MemoryPlacement.Auto, MemoryAccessPattern.Streaming);

            // Assert - Verify allocations
            deviceBuffer.Should().NotBeNull();
            deviceBuffer.Length.Should().Be(1024);
            deviceBuffer.Placement.Should().Be(MemoryPlacement.DeviceLocal);

            hostBuffer.Should().NotBeNull();
            hostBuffer.Length.Should().Be(512);
            hostBuffer.Placement.Should().Be(MemoryPlacement.HostLocal);

            autoBuffer.Should().NotBeNull();
            autoBuffer.Length.Should().Be(256);
            autoBuffer.Placement.Should().NotBe(MemoryPlacement.Auto); // Should be optimized

            // Verify memory statistics
            var stats = memoryManager.GetGlobalMemoryStatistics();
            stats.ActiveAllocations.Should().BeGreaterOrEqualTo(3);
            stats.TotalAllocatedBytes.Should().BeGreaterThan(0);
        }

        [Fact]
        public async Task UniversalMemoryManager_AsyncOperations_ShouldWorkCorrectly()
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);
            using var buffer = memoryManager.AllocateUniversal<float>(1024);
            var sourceData = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();

            // Act - Perform async memory operations
            await buffer.CopyFromAsync(sourceData);
            var resultData = new float[1024];
            await buffer.CopyToAsync(resultData);

            // Assert - Verify data integrity
            resultData.Should().Equal(sourceData);
        }

        #endregion

        #region Adaptive Scheduling Integration Tests

        [Fact]
        public async Task AdaptiveScheduler_CompleteWorkflow_ShouldOptimizeExecution()
        {
            // Arrange
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.PerformanceOptimized);

            // Create a compute graph with multiple operations
            var graph = new ComputeGraph();
            var matmulNode = new ComputeNode(new MatMulOp(256, 256, 256)) { Id = "MatMul" };
            var vectorNode = new ComputeNode(new VectorOp(1024)) { Id = "Vector" };
            var convNode = new ComputeNode(new ConvolutionOp(512, 3, 64)) { Id = "Conv" };

            graph.AddNode(matmulNode);
            graph.AddNode(vectorNode);
            graph.AddNode(convNode);

            // Add dependencies: conv depends on matmul and vector
            graph.AddDependency(convNode, matmulNode);
            graph.AddDependency(convNode, vectorNode);

            // Act - Create and execute execution plan
            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            await scheduler.ExecuteAsync(executionPlan);

            // Assert - Verify execution plan
            executionPlan.Should().NotBeNull();
            executionPlan.Graph.Should().Be(graph);
            executionPlan.Assignments.Should().HaveCount(3);

            // Verify scheduling statistics
            var stats = scheduler.GetPerformanceStatistics();
            stats.TotalExecutions.Should().BeGreaterThan(0);
        }

        [Theory]
        [InlineData(SchedulingPolicy.PerformanceOptimized)]
        [InlineData(SchedulingPolicy.EnergyEfficient)]
        [InlineData(SchedulingPolicy.LoadBalanced)]
        [InlineData(SchedulingPolicy.LatencyOptimized)]
        public async Task AdaptiveScheduler_DifferentPolicies_ShouldAdaptBehavior(SchedulingPolicy policy)
        {
            // Arrange
            var scheduler = new AdaptiveScheduler(_availableAccelerators, policy);
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(128, 128, 128));
            graph.AddNode(node);

            // Act
            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);

            // Assert
            executionPlan.Should().NotBeNull();
            scheduler.CurrentPolicy.Should().Be(policy);
        }

        #endregion

        #region ML Framework Integration Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_EndToEndWorkflow_ShouldPerformPredictions()
        {
            // Arrange
            var model = new TestMLModel();
            var predictionContext = new PredictionContext(_context);
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(model, predictionContext);

            var sampleInputs = new[]
            {
                new TestInput { Values = new float[] { 1.0f, 2.0f, 3.0f } },
                new TestInput { Values = new float[] { 4.0f, 5.0f, 6.0f } }
            };

            // Act - Optimize and perform predictions
            await predictor.OptimizeAsync(sampleInputs);

            var singleOutput = await predictor.PredictAsync(sampleInputs[0]);
            var batchOutputs = await predictor.PredictBatchAsync(sampleInputs);

            // Assert - Verify predictions
            singleOutput.Should().NotBeNull();
            singleOutput.Result.Should().NotBeEmpty();
            singleOutput.Confidence.Should().BeInRange(0.0f, 1.0f);

            batchOutputs.Should().HaveCount(2);
            batchOutputs.Should().AllSatisfy(output =>
            {
                output.Should().NotBeNull();
                output.Result.Should().NotBeEmpty();
            });

            predictor.IsOptimized.Should().BeTrue();

            // Verify performance statistics
            var stats = predictor.GetPerformanceStatistics();
            stats.TotalPredictions.Should().BeGreaterThan(0);
        }

        [Fact]
        public async Task ONNXRuntimeIntegration_CompleteWorkflow_ShouldExecuteModels()
        {
            // Arrange
            var options = new ExecutionProviderOptions
            {
                SchedulingPolicy = SchedulingPolicy.PerformanceOptimized,
                OptimizationLevel = 2,
                EnableMixedPrecision = true
            };
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, options);

            // Act - Compile and execute model
            var modelPath = "test_model.onnx";
            var compilationOptions = new ModelCompilationOptions
            {
                OptimizationLevel = 2,
                EnableKernelFusion = true,
                TargetDevices = new[] { ComputeDevice.CPU }
            };

            var compiledPlan = await provider.CompileModelAsync(modelPath, compilationOptions);
            var inputs = new List<NamedOnnxValue>
            {
                new NamedOnnxValue("input", new float[] { 1.0f, 2.0f, 3.0f })
            };

            var outputs = await provider.RunCompiledAsync(compiledPlan, inputs);

            // Assert - Verify execution
            compiledPlan.Should().NotBeNull();
            outputs.Should().NotBeNull();

            var stats = provider.Stats;
            stats.TotalInferences.Should().BeGreaterThan(0);

            provider.Dispose();
        }

        #endregion

        #region Cross-Component Integration Tests

        [Fact]
        public async Task Phase8_FullStackIntegration_ShouldWorkSeamlessly()
        {
            // Arrange - Set up the complete Phase 8 stack
            using var memoryManager = new UniversalMemoryManager(_context);
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.PerformanceOptimized);

            // Create universal memory buffers
            using var inputBuffer = memoryManager.AllocateUniversal<float>(
                1024, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);
            using var outputBuffer = memoryManager.AllocateUniversal<float>(
                1024, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);

            // Initialize input data
            var inputData = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();
            await inputBuffer.CopyFromAsync(inputData);

            // Create compute graph for processing
            var graph = new ComputeGraph();
            var processNode = new ComputeNode(new VectorOp(1024)) { Id = "Process" };
            graph.AddNode(processNode);

            // Act - Execute the complete workflow
            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            await scheduler.ExecuteAsync(executionPlan);

            // Execute universal kernel on the data
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<ArrayView<float>, ArrayView<float>>(
                ProcessingKernel);
            kernel(inputBuffer.GetView1D(), outputBuffer.GetView1D());
            _accelerator.Synchronize();

            // Retrieve results
            var outputData = new float[1024];
            await outputBuffer.CopyToAsync(outputData);

            // Assert - Verify complete workflow
            outputData.Should().NotBeEmpty();
            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i].Should().Be(inputData[i] * 3.0f);
            }

            // Verify memory statistics
            var memStats = memoryManager.GetGlobalMemoryStatistics();
            memStats.ActiveAllocations.Should().BeGreaterOrEqualTo(2);

            // Verify scheduling statistics
            var schedStats = scheduler.GetPerformanceStatistics();
            schedStats.TotalExecutions.Should().BeGreaterThan(0);

            scheduler.Dispose();
        }

        [UniversalKernel(EnableOptimizations = true)]
        private static void ProcessingKernel(ArrayView<float> input, ArrayView<float> output)
        {
            var index = UniversalGrid.GlobalIndex.X;
            if (index < input.Length && index < output.Length)
            {
                output[index] = input[index] * 3.0f;
            }
        }

        #endregion

        #region Performance and Scalability Tests

        [Theory]
        [InlineData(1024)]
        [InlineData(4096)]
        [InlineData(16384)]
        [InlineData(65536)]
        public async Task Phase8_ScalabilityTest_ShouldHandleLargeWorkloads(int dataSize)
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.PerformanceOptimized);

            // Act - Process different data sizes
            using var buffer = memoryManager.AllocateUniversal<float>(
                dataSize, MemoryPlacement.Auto, MemoryAccessPattern.Sequential);

            var data = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();
            await buffer.CopyFromAsync(data);

            var graph = new ComputeGraph();
            var node = new ComputeNode(new VectorOp(dataSize)) { Id = $"Vector_{dataSize}" };
            graph.AddNode(node);

            var executionPlan = await scheduler.CreateExecutionPlanAsync(graph);
            await scheduler.ExecuteAsync(executionPlan);

            // Assert - Verify scalability
            buffer.Length.Should().Be(dataSize);

            var stats = scheduler.GetPerformanceStatistics();
            stats.TotalExecutions.Should().BeGreaterThan(0);

            scheduler.Dispose();
        }

        [Fact]
        public async Task Phase8_ConcurrentOperations_ShouldHandleMultipleWorkflows()
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.LoadBalanced);

            // Act - Execute multiple concurrent workflows
            var tasks = new List<Task>();
            for (int i = 0; i < 5; i++)
            {
                var workflowId = i;
                tasks.Add(Task.Run(async () =>
                {
                    using var buffer = memoryManager.AllocateUniversal<float>(512);
                    var data = Enumerable.Range(0, 512).Select(j => (float)(j + workflowId * 1000)).ToArray();
                    await buffer.CopyFromAsync(data);

                    var graph = new ComputeGraph();
                    var node = new ComputeNode(new VectorOp(512)) { Id = $"Workflow_{workflowId}" };
                    graph.AddNode(node);

                    var plan = await scheduler.CreateExecutionPlanAsync(graph);
                    await scheduler.ExecuteAsync(plan);
                }));
            }

            await Task.WhenAll(tasks);

            // Assert - Verify concurrent execution
            var stats = scheduler.GetPerformanceStatistics();
            stats.TotalExecutions.Should().BeGreaterOrEqualTo(5);

            var memStats = memoryManager.GetGlobalMemoryStatistics();
            memStats.PeakAllocatedBytes.Should().BeGreaterThan(0);

            scheduler.Dispose();
        }

        #endregion

        #region Error Handling and Recovery Tests

        [Fact]
        public async Task Phase8_ErrorRecovery_ShouldHandleFailuresGracefully()
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.PerformanceOptimized);

            // Act - Try to allocate invalid memory and handle gracefully
            try
            {
                using var invalidBuffer = memoryManager.AllocateUniversal<float>(-100); // Invalid size
                Assert.Fail("Should have thrown an exception");
            }
            catch (ArgumentException)
            {
                // Expected exception
            }

            // Continue with valid operations after error
            using var validBuffer = memoryManager.AllocateUniversal<float>(1024);
            var graph = new ComputeGraph();
            var node = new ComputeNode(new VectorOp(1024));
            graph.AddNode(node);

            var plan = await scheduler.CreateExecutionPlanAsync(graph);

            // Assert - Verify recovery
            validBuffer.Should().NotBeNull();
            plan.Should().NotBeNull();

            scheduler.Dispose();
        }

        #endregion

        #region Configuration and Optimization Tests

        [Fact]
        public async Task Phase8_OptimizationPipeline_ShouldImprovePerformance()
        {
            // Arrange
            using var memoryManager = new UniversalMemoryManager(_context);
            var scheduler = new AdaptiveScheduler(_availableAccelerators, SchedulingPolicy.Balanced);

            // Act - Configure for different optimization strategies
            scheduler.OptimizeForThroughput();
            scheduler.CurrentPolicy.Should().Be(SchedulingPolicy.PerformanceOptimized);

            scheduler.OptimizeForLatency();
            scheduler.CurrentPolicy.Should().Be(SchedulingPolicy.LatencyOptimized);

            scheduler.OptimizeForPowerEfficiency();
            scheduler.CurrentPolicy.Should().Be(SchedulingPolicy.EnergyEfficient);

            // Execute workload with different optimizations
            var graph = new ComputeGraph();
            var node = new ComputeNode(new MatMulOp(128, 128, 128));
            graph.AddNode(node);

            var plan = await scheduler.CreateExecutionPlanAsync(graph);
            await scheduler.ExecuteAsync(plan);

            // Assert - Verify optimization effects
            var stats = scheduler.GetPerformanceStatistics();
            stats.Should().NotBeNull();

            memoryManager.OptimizeMemoryUsage();
            var memRecommendations = memoryManager.GetMemoryRecommendations();
            memRecommendations.Should().NotBeEmpty();

            scheduler.Dispose();
        }

        #endregion

        public void Dispose()
        {
            _accelerator?.Dispose();
            _context?.Dispose();
        }

        #region Test Helper Classes

        public class TestInput
        {
            public float[] Values { get; set; } = Array.Empty<float>();
        }

        public class TestOutput
        {
            public float[] Result { get; set; } = Array.Empty<float>();
            public float Confidence { get; set; }
        }

        public class TestMLModel : IMLModel<TestInput, TestOutput>
        {
            public string ModelName => "TestIntegrationModel";
            public string Version => "1.0.0";

            public async ValueTask<TestOutput> PredictAsync(TestInput input)
            {
                await Task.Delay(1);
                return new TestOutput
                {
                    Result = input.Values.Select(v => v * 2).ToArray(),
                    Confidence = 0.95f
                };
            }

            public async ValueTask<ComputeGraph> CreateComputeGraphAsync(ITensor inputTensor)
            {
                await Task.Delay(1);
                var graph = new ComputeGraph();
                var node = new ComputeNode(new VectorOp(inputTensor.ElementCount));
                graph.AddNode(node);
                return graph;
            }

            public void Dispose() { }
        }

        public class NamedOnnxValue
        {
            public string Name { get; }
            public object Value { get; }

            public NamedOnnxValue(string name, object value)
            {
                Name = name;
                Value = value;
            }
        }

        #endregion
    }
}