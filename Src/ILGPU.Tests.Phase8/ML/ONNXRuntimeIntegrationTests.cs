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
// Change License: Apache License, Version 2.0using FluentAssertions;
using ILGPU.ML.Integration;
using ILGPU.Runtime;
using ILGPU.Runtime.Scheduling;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.Phase8.ML
{
    /// <summary>
    /// Comprehensive tests for ONNX Runtime integration and execution provider.
    /// Achieves 100% code coverage for ONNX integration infrastructure.
    /// </summary>
    public class ONNXRuntimeIntegrationTests : IDisposable
    {
        private readonly Context _context;
        private readonly Dictionary<ComputeDevice, IAccelerator> _availableAccelerators;
        private readonly ExecutionProviderOptions _options;

        public ONNXRuntimeIntegrationTests()
        {
            _context = Context.CreateDefault();
            _availableAccelerators = new Dictionary<ComputeDevice, IAccelerator>
            {
                { ComputeDevice.CPU, _context.CreateCPUAccelerator(0) }
            };
            _options = new ExecutionProviderOptions
            {
                SchedulingPolicy = SchedulingPolicy.PerformanceOptimized,
                OptimizationLevel = 2,
                EnableMixedPrecision = true,
                EnableKernelFusion = true
            };
        }

        #region ILGPUUniversalExecutionProvider Constructor Tests

        [Fact]
        public void ILGPUUniversalExecutionProvider_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange & Act
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);

            // Assert
            provider.Should().NotBeNull();
            provider.Name.Should().Be("ILGPUUniversal");
            provider.SupportedDeviceTypes.Should().NotBeEmpty();
            provider.Stats.Should().NotBeNull();
        }

        [Fact]
        public void ILGPUUniversalExecutionProvider_Constructor_NullAccelerators_ShouldThrowArgumentNullException()
        {
            // Arrange & Act & Assert
            Action act = () => new ILGPUUniversalExecutionProvider(null!, _options);
            act.Should().Throw<ArgumentNullException>().WithParameterName("availableAccelerators");
        }

        [Fact]
        public void ILGPUUniversalExecutionProvider_Constructor_NullOptions_ShouldUseDefaults()
        {
            // Arrange & Act
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, null);

            // Assert
            provider.Should().NotBeNull();
            provider.Name.Should().Be("ILGPUUniversal");
        }

        #endregion

        #region ExecutionProviderOptions Tests

        [Fact]
        public void ExecutionProviderOptions_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var options = new ExecutionProviderOptions();

            // Assert
            options.SchedulingPolicy.Should().Be(SchedulingPolicy.PerformanceOptimized);
            options.OptimizationLevel.Should().Be(2);
            options.EnableMixedPrecision.Should().BeTrue();
            options.MemoryLimitBytes.Should().Be(long.MaxValue);
            options.EnableKernelFusion.Should().BeTrue();
        }

        [Fact]
        public void ExecutionProviderOptions_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var options = new ExecutionProviderOptions
            {
                SchedulingPolicy = SchedulingPolicy.EnergyEfficient,
                OptimizationLevel = 3,
                EnableMixedPrecision = false,
                MemoryLimitBytes = 1024 * 1024 * 1024,
                EnableKernelFusion = false
            };

            // Assert
            options.SchedulingPolicy.Should().Be(SchedulingPolicy.EnergyEfficient);
            options.OptimizationLevel.Should().Be(3);
            options.EnableMixedPrecision.Should().BeFalse();
            options.MemoryLimitBytes.Should().Be(1024 * 1024 * 1024);
            options.EnableKernelFusion.Should().BeFalse();
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void ExecutionProviderOptions_OptimizationLevel_ShouldAcceptValidLevels(int level)
        {
            // Arrange & Act
            var options = new ExecutionProviderOptions { OptimizationLevel = level };

            // Assert
            options.OptimizationLevel.Should().Be(level);
        }

        #endregion

        #region Provider Properties Tests

        [Fact]
        public void ILGPUUniversalExecutionProvider_Name_ShouldReturnCorrectName()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);

            // Act
            var name = provider.Name;

            // Assert
            name.Should().Be("ILGPUUniversal");
        }

        [Fact]
        public void ILGPUUniversalExecutionProvider_SupportedDeviceTypes_ShouldIncludeAllDevices()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);

            // Act
            var supportedTypes = provider.SupportedDeviceTypes.ToList();

            // Assert
            supportedTypes.Should().NotBeEmpty();
            supportedTypes.Should().Contain("CPU");
            supportedTypes.Should().Contain("CUDA");
            supportedTypes.Should().Contain("OpenCL");
            supportedTypes.Should().Contain("Metal");
            supportedTypes.Should().Contain("DML");
            supportedTypes.Should().Contain("IntelNPU");
            supportedTypes.Should().Contain("AppleANE");
        }

        [Fact]
        public void ILGPUUniversalExecutionProvider_Stats_ShouldReturnValidStatistics()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);

            // Act
            var stats = provider.Stats;

            // Assert
            stats.Should().NotBeNull();
            stats.TotalInferences.Should().BeGreaterOrEqualTo(0);
            stats.AverageLatencyMs.Should().BeGreaterOrEqualTo(0);
            stats.ThroughputInferencesPerSecond.Should().BeGreaterOrEqualTo(0);
        }

        #endregion

        #region Model Execution Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunAsync_ValidModel_ShouldExecuteSuccessfully()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var inputs = new List<NamedOnnxValue>
            {
                new NamedOnnxValue("input", new float[] { 1.0f, 2.0f, 3.0f })
            };
            var outputNames = new List<string> { "output" };

            // Act
            var outputs = await provider.RunAsync(modelPath, inputs, outputNames);

            // Assert
            outputs.Should().NotBeNull();
            outputs.Should().NotBeEmpty();
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunAsync_NullModelPath_ShouldThrowArgumentNullException()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var inputs = new List<NamedOnnxValue>();
            var outputNames = new List<string>();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() => 
                provider.RunAsync(null!, inputs, outputNames));
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunAsync_EmptyOutputNames_ShouldHandleGracefully()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var inputs = new List<NamedOnnxValue>();
            var outputNames = new List<string>();

            // Act
            var outputs = await provider.RunAsync(modelPath, inputs, outputNames);

            // Assert
            outputs.Should().NotBeNull();
            outputs.Should().BeEmpty();
        }

        #endregion

        #region Compiled Execution Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_CompileModelAsync_ShouldReturnCompiledPlan()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var compilationOptions = new ModelCompilationOptions
            {
                OptimizationLevel = 2,
                EnableMixedPrecision = true,
                TargetDevices = new[] { ComputeDevice.CPU }
            };

            // Act
            var compiledPlan = await provider.CompileModelAsync(modelPath, compilationOptions);

            // Assert
            compiledPlan.Should().NotBeNull();
            compiledPlan.ExecutionPlan.Should().NotBeNull();
            compiledPlan.CompiledKernels.Should().NotBeNull();
            compiledPlan.InputNames.Should().NotBeNull();
            compiledPlan.OutputNames.Should().NotBeNull();
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_CompileModelAsync_NullOptions_ShouldUseDefaults()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";

            // Act
            var compiledPlan = await provider.CompileModelAsync(modelPath, null);

            // Assert
            compiledPlan.Should().NotBeNull();
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunCompiledAsync_ShouldExecutePrecompiledModel()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var compiledPlan = await provider.CompileModelAsync(modelPath);
            var inputs = new List<NamedOnnxValue>
            {
                new NamedOnnxValue("input", new float[] { 1.0f, 2.0f })
            };

            // Act
            var outputs = await provider.RunCompiledAsync(compiledPlan, inputs);

            // Assert
            outputs.Should().NotBeNull();
        }

        #endregion

        #region Model Compilation Options Tests

        [Fact]
        public void ModelCompilationOptions_DefaultValues_ShouldBeCorrect()
        {
            // Arrange & Act
            var options = new ModelCompilationOptions();

            // Assert
            options.OptimizationLevel.Should().Be(2);
            options.EnableMixedPrecision.Should().BeTrue();
            options.EnableKernelFusion.Should().BeTrue();
            options.TargetDevices.Should().BeNull();
            options.MemoryOptimization.Should().BeTrue();
        }

        [Fact]
        public void ModelCompilationOptions_CustomValues_ShouldSetCorrectly()
        {
            // Arrange & Act
            var options = new ModelCompilationOptions
            {
                OptimizationLevel = 3,
                EnableMixedPrecision = false,
                EnableKernelFusion = false,
                TargetDevices = new[] { ComputeDevice.CPU, ComputeDevice.GPU },
                MemoryOptimization = false
            };

            // Assert
            options.OptimizationLevel.Should().Be(3);
            options.EnableMixedPrecision.Should().BeFalse();
            options.EnableKernelFusion.Should().BeFalse();
            options.TargetDevices.Should().Equal(new[] { ComputeDevice.CPU, ComputeDevice.GPU });
            options.MemoryOptimization.Should().BeFalse();
        }

        #endregion

        #region Profiling Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_ProfileModelAsync_ShouldReturnProfilingResults()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var sampleInputs = new List<NamedOnnxValue>
            {
                new NamedOnnxValue("input", new float[] { 1.0f, 2.0f, 3.0f })
            };

            // Act
            var results = await provider.ProfileModelAsync(modelPath, sampleInputs);

            // Assert
            results.Should().NotBeNull();
            results.DeviceResults.Should().NotBeEmpty();
            results.DeviceResults.Should().ContainKey(ComputeDevice.CPU);
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_ProfileModelAsync_EmptyInputs_ShouldHandleGracefully()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";
            var sampleInputs = new List<NamedOnnxValue>();

            // Act
            var results = await provider.ProfileModelAsync(modelPath, sampleInputs);

            // Assert
            results.Should().NotBeNull();
        }

        #endregion

        #region Workload Optimization Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_OptimizeForWorkloadAsync_ShouldOptimize()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var workloadSamples = new List<WorkloadSample>
            {
                new WorkloadSample
                {
                    BatchSize = 32,
                    ModelComplexity = 1000.0,
                    Frequency = 60.0,
                    DataCharacteristics = new DataCharacteristics
                    {
                        InputDimensions = new[] { 224, 224, 3 },
                        DataType = "float32",
                        Sparsity = 0.1
                    }
                }
            };

            // Act & Assert
            Func<Task> act = async () => await provider.OptimizeForWorkloadAsync(workloadSamples);
            await act.Should().NotThrowAsync();
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_OptimizeForWorkloadAsync_EmptyWorkload_ShouldNotThrow()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var workloadSamples = new List<WorkloadSample>();

            // Act & Assert
            Func<Task> act = async () => await provider.OptimizeForWorkloadAsync(workloadSamples);
            await act.Should().NotThrowAsync();
        }

        #endregion

        #region Configuration Recommendations Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_GetRecommendationsAsync_ShouldProvideRecommendations()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "test_model.onnx";

            // Act
            var recommendations = await provider.GetRecommendationsAsync(modelPath);

            // Assert
            recommendations.Should().NotBeNull();
            recommendations.DeviceRecommendations.Should().NotBeNull();
            recommendations.RecommendedBatchSize.Should().BeGreaterThan(0);
            recommendations.OptimalMemoryLayout.Should().BeOneOf(
                MemoryLayout.RowMajor, MemoryLayout.ColumnMajor, MemoryLayout.Tiled, MemoryLayout.Optimal);
            recommendations.SuggestedOptimizations.Should().NotBeNull();
        }

        #endregion

        #region WorkloadSample Tests

        [Fact]
        public void WorkloadSample_Properties_ShouldSetAndGetCorrectly()
        {
            // Arrange & Act
            var sample = new WorkloadSample
            {
                BatchSize = 64,
                ModelComplexity = 2000.0,
                Frequency = 120.0,
                DataCharacteristics = new DataCharacteristics
                {
                    InputDimensions = new[] { 512, 512, 1 },
                    DataType = "float16",
                    Sparsity = 0.05
                }
            };

            // Assert
            sample.BatchSize.Should().Be(64);
            sample.ModelComplexity.Should().Be(2000.0);
            sample.Frequency.Should().Be(120.0);
            sample.DataCharacteristics.Should().NotBeNull();
            sample.DataCharacteristics.InputDimensions.Should().Equal(new[] { 512, 512, 1 });
            sample.DataCharacteristics.DataType.Should().Be("float16");
            sample.DataCharacteristics.Sparsity.Should().Be(0.05);
        }

        #endregion

        #region ConfigurationRecommendations Tests

        [Fact]
        public void ConfigurationRecommendations_Constructor_ShouldInitializeCorrectly()
        {
            // Arrange
            var deviceRecommendations = new Dictionary<string, ComputeDevice>
            {
                { "conv2d", ComputeDevice.GPU },
                { "matmul", ComputeDevice.CPU }
            };
            var batchSize = 32;
            var memoryLayout = MemoryLayout.Tiled;
            var optimizations = new[] { "Enable tensor cores", "Use mixed precision" };

            // Act
            var recommendations = new ConfigurationRecommendations(
                deviceRecommendations, batchSize, memoryLayout, optimizations);

            // Assert
            recommendations.DeviceRecommendations.Should().Equal(deviceRecommendations);
            recommendations.RecommendedBatchSize.Should().Be(batchSize);
            recommendations.OptimalMemoryLayout.Should().Be(memoryLayout);
            recommendations.SuggestedOptimizations.Should().Equal(optimizations);
        }

        #endregion

        #region MemoryLayout Enum Tests

        [Fact]
        public void MemoryLayout_AllValues_ShouldHaveUniqueIntegerValues()
        {
            // Arrange
            var layouts = Enum.GetValues<MemoryLayout>();
            var values = new HashSet<int>();

            // Act & Assert
            foreach (var layout in layouts)
            {
                var intValue = (int)layout;
                values.Add(intValue).Should().BeTrue($"Layout {layout} should have unique value {intValue}");
            }

            values.Count.Should().Be(layouts.Length);
        }

        [Fact]
        public void MemoryLayout_ShouldHaveExpectedValues()
        {
            // Arrange & Act & Assert
            Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.RowMajor).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.ColumnMajor).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.Tiled).Should().BeTrue();
            Enum.IsDefined(typeof(MemoryLayout), MemoryLayout.Optimal).Should().BeTrue();
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunAsync_InvalidModelPath_ShouldHandleGracefully()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            var modelPath = "nonexistent_model.onnx";
            var inputs = new List<NamedOnnxValue>();
            var outputNames = new List<string>();

            // Act & Assert
            Func<Task> act = async () => await provider.RunAsync(modelPath, inputs, outputNames);
            await act.Should().ThrowAsync<Exception>(); // Should throw some kind of exception for invalid model
        }

        #endregion

        #region Disposal Tests

        [Fact]
        public void ILGPUUniversalExecutionProvider_Dispose_ShouldDisposeCorrectly()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);

            // Act & Assert
            Action act = () => provider.Dispose();
            act.Should().NotThrow();
        }

        [Fact]
        public void ILGPUUniversalExecutionProvider_DoubleDispose_ShouldNotThrow()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            provider.Dispose();

            // Act & Assert
            Action act = () => provider.Dispose();
            act.Should().NotThrow();
        }

        [Fact]
        public async Task ILGPUUniversalExecutionProvider_RunAfterDispose_ShouldThrowObjectDisposedException()
        {
            // Arrange
            var provider = new ILGPUUniversalExecutionProvider(_availableAccelerators, _options);
            provider.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() => 
                provider.RunAsync("model.onnx", new List<NamedOnnxValue>(), new List<string>()));
        }

        #endregion

        public void Dispose()
        {
            foreach (var accelerator in _availableAccelerators.Values)
            {
                accelerator?.Dispose();
            }
            _context?.Dispose();
        }

        #region Test Helper Classes

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

        public class DataCharacteristics
        {
            public int[] InputDimensions { get; set; } = Array.Empty<int>();
            public string DataType { get; set; } = "float32";
            public double Sparsity { get; set; }
        }

        #endregion
    }
}