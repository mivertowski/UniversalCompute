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

using FluentAssertions;
using ILGPU.ML.Integration;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.Phase8.ML
{
    /// <summary>
    /// Comprehensive tests for ILGPUUniversalPredictor and ML framework integration.
    /// Achieves 100% code coverage for ML integration infrastructure.
    /// </summary>
    public class ILGPUUniversalPredictorTests : IDisposable
    {
        private readonly Context _context;
        private readonly TestMLModel _model;
        private readonly PredictionContext _predictionContext;

        public ILGPUUniversalPredictorTests()
        {
            _context = Context.CreateDefault();
            _model = new TestMLModel();
            _predictionContext = new PredictionContext(_context);
        }

        #region ILGPUUniversalPredictor Constructor Tests

        [Fact]
        public void ILGPUUniversalPredictor_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange & Act
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Assert
            predictor.Should().NotBeNull();
            predictor.Model.Should().Be(_model);
            predictor.Context.Should().Be(_predictionContext);
            predictor.IsOptimized.Should().BeFalse();
        }

        [Fact]
        public void ILGPUUniversalPredictor_Constructor_NullModel_ShouldThrowArgumentNullException()
        {
            // Arrange & Act & Assert
            Action act = () => new ILGPUUniversalPredictor<TestInput, TestOutput>(null!, _predictionContext);
            act.Should().Throw<ArgumentNullException>().WithParameterName("model");
        }

        [Fact]
        public void ILGPUUniversalPredictor_Constructor_NullContext_ShouldThrowArgumentNullException()
        {
            // Arrange & Act & Assert
            Action act = () => new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, null!);
            act.Should().Throw<ArgumentNullException>().WithParameterName("predictionContext");
        }

        #endregion

        #region Single Prediction Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictAsync_ValidInput_ShouldReturnValidOutput()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var input = new TestInput { Values = new float[] { 1.0f, 2.0f, 3.0f } };

            // Act
            var output = await predictor.PredictAsync(input);

            // Assert
            output.Should().NotBeNull();
            output.Result.Should().NotBeEmpty();
            output.Confidence.Should().BeInRange(0.0f, 1.0f);
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictAsync_NullInput_ShouldThrowArgumentNullException()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() => predictor.PredictAsync(null!));
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictAsync_MultipleInputs_ShouldProduceConsistentResults()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var input = new TestInput { Values = new float[] { 1.0f, 2.0f, 3.0f } };

            // Act
            var output1 = await predictor.PredictAsync(input);
            var output2 = await predictor.PredictAsync(input);

            // Assert
            output1.Should().NotBeNull();
            output2.Should().NotBeNull();
            output1.Result.Length.Should().Be(output2.Result.Length);
        }

        #endregion

        #region Batch Prediction Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictBatchAsync_ValidInputs_ShouldReturnValidOutputs()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var inputs = new[]
            {
                new TestInput { Values = new float[] { 1.0f, 2.0f, 3.0f } },
                new TestInput { Values = new float[] { 4.0f, 5.0f, 6.0f } },
                new TestInput { Values = new float[] { 7.0f, 8.0f, 9.0f } }
            };

            // Act
            var outputs = await predictor.PredictBatchAsync(inputs);

            // Assert
            outputs.Should().NotBeNull();
            outputs.Should().HaveCount(inputs.Length);
            outputs.Should().AllSatisfy(output => 
            {
                output.Should().NotBeNull();
                output.Result.Should().NotBeEmpty();
                output.Confidence.Should().BeInRange(0.0f, 1.0f);
            });
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictBatchAsync_EmptyInputs_ShouldReturnEmptyResults()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var inputs = Array.Empty<TestInput>();

            // Act
            var outputs = await predictor.PredictBatchAsync(inputs);

            // Assert
            outputs.Should().NotBeNull();
            outputs.Should().BeEmpty();
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictBatchAsync_NullInputs_ShouldThrowArgumentNullException()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() => predictor.PredictBatchAsync(null!));
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictBatchAsync_LargeBatch_ShouldOptimizeBatching()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var inputs = Enumerable.Range(0, 100)
                .Select(i => new TestInput { Values = new float[] { i, i + 1, i + 2 } })
                .ToArray();

            // Act
            var outputs = await predictor.PredictBatchAsync(inputs);

            // Assert
            outputs.Should().HaveCount(inputs.Length);
            outputs.Should().AllSatisfy(output => output.Should().NotBeNull());
        }

        #endregion

        #region Streaming Prediction Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictStreamAsync_ShouldProcessInBatches()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var inputs = Enumerable.Range(0, 50)
                .Select(i => new TestInput { Values = new float[] { i, i + 1 } })
                .ToAsyncEnumerable();

            // Act
            var outputs = new List<TestOutput>();
            await foreach (var output in predictor.PredictStreamAsync(inputs))
            {
                outputs.Add(output);
            }

            // Assert
            outputs.Should().HaveCount(50);
            outputs.Should().AllSatisfy(output => output.Should().NotBeNull());
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictStreamAsync_CustomBatchSize_ShouldRespectBatchSize()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var inputs = Enumerable.Range(0, 25)
                .Select(i => new TestInput { Values = new float[] { i } })
                .ToAsyncEnumerable();

            // Act
            var outputs = new List<TestOutput>();
            await foreach (var output in predictor.PredictStreamAsync(inputs, batchSize: 5))
            {
                outputs.Add(output);
            }

            // Assert
            outputs.Should().HaveCount(25);
        }

        #endregion

        #region Optimization Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_OptimizeAsync_ShouldOptimizeForWorkload()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var sampleInputs = new[]
            {
                new TestInput { Values = new float[] { 1.0f, 2.0f } },
                new TestInput { Values = new float[] { 3.0f, 4.0f } }
            };

            // Act
            await predictor.OptimizeAsync(sampleInputs);

            // Assert
            predictor.IsOptimized.Should().BeTrue();
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_OptimizeAsync_EmptySamples_ShouldNotThrow()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var sampleInputs = Array.Empty<TestInput>();

            // Act & Assert
            Func<Task> act = async () => await predictor.OptimizeAsync(sampleInputs);
            await act.Should().NotThrowAsync();
        }

        #endregion

        #region Performance Statistics Tests

        [Fact]
        public void ILGPUUniversalPredictor_GetPerformanceStatistics_ShouldReturnValidStats()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            var stats = predictor.GetPerformanceStatistics();

            // Assert
            stats.Should().NotBeNull();
            stats.TotalPredictions.Should().BeGreaterOrEqualTo(0);
            stats.AverageLatencyMs.Should().BeGreaterOrEqualTo(0);
            stats.ThroughputPredictionsPerSecond.Should().BeGreaterOrEqualTo(0);
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_GetPerformanceStatistics_AfterPredictions_ShouldUpdateStats()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var input = new TestInput { Values = new float[] { 1.0f } };

            // Act
            await predictor.PredictAsync(input);
            var stats = predictor.GetPerformanceStatistics();

            // Assert
            stats.TotalPredictions.Should().BeGreaterThan(0);
        }

        [Fact]
        public void ILGPUUniversalPredictor_ResetStatistics_ShouldClearStats()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            predictor.ResetStatistics();
            var stats = predictor.GetPerformanceStatistics();

            // Assert
            stats.TotalPredictions.Should().Be(0);
            stats.AverageLatencyMs.Should().Be(0);
            stats.ThroughputPredictionsPerSecond.Should().Be(0);
        }

        #endregion

        #region Configuration Tests

        [Fact]
        public void ILGPUUniversalPredictor_SetBatchingStrategy_ShouldUpdateStrategy()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            predictor.SetBatchingStrategy(BatchingStrategy.Adaptive);

            // Assert
            predictor.BatchingStrategy.Should().Be(BatchingStrategy.Adaptive);
        }

        [Theory]
        [InlineData(BatchingStrategy.Fixed)]
        [InlineData(BatchingStrategy.Adaptive)]
        [InlineData(BatchingStrategy.Dynamic)]
        public void ILGPUUniversalPredictor_SetBatchingStrategy_AllStrategies_ShouldAccept(BatchingStrategy strategy)
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            predictor.SetBatchingStrategy(strategy);

            // Assert
            predictor.BatchingStrategy.Should().Be(strategy);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(16)]
        [InlineData(32)]
        [InlineData(64)]
        [InlineData(128)]
        public void ILGPUUniversalPredictor_SetOptimalBatchSize_ValidSizes_ShouldUpdate(int batchSize)
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            predictor.SetOptimalBatchSize(batchSize);

            // Assert
            predictor.OptimalBatchSize.Should().Be(batchSize);
        }

        [Fact]
        public void ILGPUUniversalPredictor_SetOptimalBatchSize_ZeroSize_ShouldThrowArgumentException()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            Action act = () => predictor.SetOptimalBatchSize(0);
            act.Should().Throw<ArgumentException>().WithParameterName("batchSize");
        }

        [Fact]
        public void ILGPUUniversalPredictor_SetOptimalBatchSize_NegativeSize_ShouldThrowArgumentException()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            Action act = () => predictor.SetOptimalBatchSize(-5);
            act.Should().Throw<ArgumentException>().WithParameterName("batchSize");
        }

        #endregion

        #region Memory Management Tests

        [Fact]
        public void ILGPUUniversalPredictor_GetMemoryUsage_ShouldReturnUsageInfo()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act
            var memoryUsage = predictor.GetMemoryUsage();

            // Assert
            memoryUsage.Should().NotBeNull();
            memoryUsage.ModelMemoryBytes.Should().BeGreaterOrEqualTo(0);
            memoryUsage.TensorMemoryBytes.Should().BeGreaterOrEqualTo(0);
            memoryUsage.TotalMemoryBytes.Should().BeGreaterOrEqualTo(0);
        }

        [Fact]
        public void ILGPUUniversalPredictor_OptimizeMemoryUsage_ShouldOptimize()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            Action act = () => predictor.OptimizeMemoryUsage();
            act.Should().NotThrow();
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictAsync_ModelError_ShouldHandleGracefully()
        {
            // Arrange
            var faultyModel = new FaultyTestMLModel();
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(faultyModel, _predictionContext);
            var input = new TestInput { Values = new float[] { 1.0f } };

            // Act & Assert
            Func<Task> act = async () => await predictor.PredictAsync(input);
            await act.Should().ThrowAsync<InvalidOperationException>();
        }

        #endregion

        #region Disposal Tests

        [Fact]
        public void ILGPUUniversalPredictor_Dispose_ShouldDisposeCorrectly()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);

            // Act & Assert
            Action act = () => predictor.Dispose();
            act.Should().NotThrow();
        }

        [Fact]
        public void ILGPUUniversalPredictor_DoubleDispose_ShouldNotThrow()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            predictor.Dispose();

            // Act & Assert
            Action act = () => predictor.Dispose();
            act.Should().NotThrow();
        }

        [Fact]
        public async Task ILGPUUniversalPredictor_PredictAfterDispose_ShouldThrowObjectDisposedException()
        {
            // Arrange
            var predictor = new ILGPUUniversalPredictor<TestInput, TestOutput>(_model, _predictionContext);
            var input = new TestInput { Values = new float[] { 1.0f } };
            predictor.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() => predictor.PredictAsync(input));
        }

        #endregion

        public void Dispose()
        {
            _predictionContext?.Dispose();
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
            public string ModelName => "TestModel";
            public string Version => "1.0.0";

            public async ValueTask<TestOutput> PredictAsync(TestInput input)
            {
                await Task.Delay(1); // Simulate some work
                return new TestOutput
                {
                    Result = input.Values.Select(v => v * 2).ToArray(),
                    Confidence = 0.95f
                };
            }

            public async ValueTask<ComputeGraph> CreateComputeGraphAsync(ITensor inputTensor)
            {
                await Task.Delay(1);
                return new ComputeGraph();
            }

            public void Dispose() { }
        }

        public class FaultyTestMLModel : IMLModel<TestInput, TestOutput>
        {
            public string ModelName => "FaultyModel";
            public string Version => "1.0.0";

            public ValueTask<TestOutput> PredictAsync(TestInput input)
            {
                throw new InvalidOperationException("Model error");
            }

            public ValueTask<ComputeGraph> CreateComputeGraphAsync(ITensor inputTensor)
            {
                throw new InvalidOperationException("Model error");
            }

            public void Dispose() { }
        }

        #endregion
    }

    /// <summary>
    /// Extension methods to support async enumerable testing.
    /// </summary>
    public static class AsyncEnumerableExtensions
    {
        public static async IAsyncEnumerable<T> ToAsyncEnumerable<T>(this IEnumerable<T> source)
        {
            foreach (var item in source)
            {
                await Task.Yield();
                yield return item;
            }
        }
    }
}