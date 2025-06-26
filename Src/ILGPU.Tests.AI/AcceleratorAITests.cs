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
using ILGPU.Intel.NPU;
using ILGPU.Numerics;
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;
using ILGPU.Runtime.CPU;
using System;
using System.Reflection;
using System.Threading.Tasks;
using Xunit;

// Use alias to resolve TensorShape ambiguity
using TensorShape = ILGPU.Numerics.TensorShape;

namespace ILGPU.Tests.AI
{
    /// <summary>
    /// Tests for AI-specific accelerator implementations.
    /// </summary>
    public class AcceleratorAITests : IDisposable
    {
        private readonly Context _context;

        public AcceleratorAITests()
        {
            _context = Context.CreateDefault();
        }

        [Fact]
        public void IntelNPU_ShouldDetectAvailability()
        {
            // Arrange - Create a CPU device for testing
            var cpuDevice = _context.GetCPUDevice(0);
            
            // Act
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);

            // Assert
            npuAccelerator.Should().NotBeNull();
            npuAccelerator.Device.Should().Be(cpuDevice);
            // Note: IsAvailable may be false on systems without NPU
        }

        [Fact]
        public void NPUCapabilities_ShouldHaveValidProperties()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);

            // Act
            var capabilities = npuAccelerator.Capabilities;

            // Assert
            capabilities.Should().NotBeNull();
            capabilities.Generation.Should().BeDefined();
        }

        [Fact]
        public async Task IntelNPU_InferenceAsync_ShouldExecuteWhenAvailable()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.SupportsNPU) return; // Skip test if NPU not available

            var network = new NeuralNetwork("TestNetwork");
            var input = TensorFactory.Create<float>(new TensorShape(1, 224, 224, 3), ComputeLocation.Cpu);

            // Act
            var result = await npuAccelerator.InferenceAsync(network, input);

            // Assert
            result.Should().NotBeNull();
        }

        [Fact]
        public async Task IntelNPU_ConvolutionAsync_ShouldExecuteWhenSupported()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.Capabilities.SupportsConvolution) return; // Skip test if convolution not supported

            var input = TensorFactory.Create<float>(new TensorShape(1, 64, 32, 32), ComputeLocation.Cpu);
            var weights = TensorFactory.Create<float>(new TensorShape(128, 64, 3, 3), ComputeLocation.Cpu);
            var parameters = new ConvolutionParameters
            {
                Stride = new Size2D(1, 1),
                Padding = new Size2D(1, 1),
                Dilation = new Size2D(1, 1)
            };

            // Act
            var result = await npuAccelerator.ConvolutionAsync(input, weights, parameters);

            // Assert
            result.Should().NotBeNull();
        }

        [Fact]
        public async Task IntelNPU_MatMulAsync_ShouldExecuteWhenSupported()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.Capabilities.SupportsMatMul) return; // Skip test if matmul not supported

            var a = TensorFactory.Create<float>(new TensorShape(128, 256), ComputeLocation.Cpu);
            var b = TensorFactory.Create<float>(new TensorShape(256, 512), ComputeLocation.Cpu);

            // Act
            var result = await npuAccelerator.MatMulAsync(a, b);

            // Assert
            result.Should().NotBeNull();
            result.Shape.Should().Be(new TensorShape(128, 512));
        }

        [Fact]
        public async Task IntelNPU_AttentionAsync_ShouldExecuteWhenSupported()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.Capabilities.SupportsAttention) return; // Skip test if attention not supported

            var seqLen = 128;
            var hiddenSize = 512;
            var query = TensorFactory.Create<float>(new TensorShape(1, seqLen, hiddenSize), ComputeLocation.Cpu);
            var key = TensorFactory.Create<float>(new TensorShape(1, seqLen, hiddenSize), ComputeLocation.Cpu);
            var value = TensorFactory.Create<float>(new TensorShape(1, seqLen, hiddenSize), ComputeLocation.Cpu);
            var parameters = new AttentionParameters { NumHeads = 8 };

            // Act
            var result = await npuAccelerator.AttentionAsync(query, key, value, parameters);

            // Assert
            result.Should().NotBeNull();
            result.Shape.Should().Be(new TensorShape(1, seqLen, hiddenSize));
        }

        [Fact]
        public async Task IntelNPU_LoadModelAsync_ShouldReturnModel()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            var modelPath = "test_model.onnx";

            // Act
            var model = await npuAccelerator.LoadModelAsync(modelPath, ModelFormat.ONNX);

            // Assert
            model.Should().NotBeNull();
        }

        [Fact]
        public async Task IntelNPU_OptimizeModelAsync_ShouldReturnOptimizedModel()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            var originalModel = new NeuralNetwork("TestModel");
            var options = new OptimizationOptions();

            // Act
            var optimizedModel = await npuAccelerator.OptimizeModelAsync(originalModel, options);

            // Assert
            optimizedModel.Should().NotBeNull();
        }

        [Fact]
        public void IntelNPU_GetPerformanceMetrics_ShouldReturnMetrics()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.SupportsNPU) return; // Skip test if NPU not available

            // Act
            var metrics = npuAccelerator.GetPerformanceMetrics();

            // Assert
            metrics.Should().NotBeNull();
        }

        [Fact]
        public void IntelNPU_GetPowerInfo_ShouldReturnPowerInfo()
        {
            // Arrange
            var cpuDevice = _context.GetCPUDevice(0);
            var npuAccelerator = CreateTestNPUAccelerator(cpuDevice);
            
            if (!npuAccelerator.SupportsNPU) return; // Skip test if NPU not available

            // Act
            var powerInfo = npuAccelerator.GetPowerInfo();

            // Assert
            powerInfo.Should().NotBeNull();
        }

        [Theory]
        [InlineData(ModelFormat.ONNX)]
        [InlineData(ModelFormat.OpenVINO)]
        [InlineData(ModelFormat.TensorFlow)]
        [InlineData(ModelFormat.PyTorch)]
        public void ModelFormat_ShouldHaveValidValues(ModelFormat format)
        {
            // Arrange & Act & Assert
            format.Should().BeDefined();
        }

        [Theory]
        [InlineData(NPUGeneration.None)]
        [InlineData(NPUGeneration.NPU2)]
        [InlineData(NPUGeneration.NPU3)]
        [InlineData(NPUGeneration.NPU4)]
        public void NPUGeneration_ShouldHaveValidValues(NPUGeneration generation)
        {
            // Arrange & Act & Assert
            generation.Should().BeDefined();
        }

        private static IntelNPUAccelerator CreateTestNPUAccelerator(Device device)
        {
            // Use reflection to create IntelNPUAccelerator since constructor is internal
            var constructor = typeof(IntelNPUAccelerator).GetConstructor(
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance,
                null, new[] { typeof(Device) }, null);
            return (IntelNPUAccelerator)constructor!.Invoke(new object[] { device });
        }

        public void Dispose()
        {
            _context?.Dispose();
        }
    }

    /// <summary>
    /// Tests for AI parameter structures and data types.
    /// </summary>
    public class AIParameterTests
    {
        [Fact]
        public void ConvolutionParameters_ShouldHaveValidDefaults()
        {
            // Arrange & Act
            var parameters = new ConvolutionParameters();

            // Assert
            parameters.Should().NotBeNull();
            parameters.Stride.Should().NotBeNull();
            parameters.Padding.Should().NotBeNull();
            parameters.Dilation.Should().NotBeNull();
        }

        [Fact]
        public void AttentionParameters_ShouldHaveValidDefaults()
        {
            // Arrange & Act
            var parameters = new AttentionParameters();

            // Assert
            parameters.Should().NotBeNull();
            parameters.NumHeads.Should().BeGreaterThan(0);
        }

        [Fact]
        public void Size2D_ShouldSupportEquality()
        {
            // Arrange
            var size1 = new Size2D(3, 4);
            var size2 = new Size2D(3, 4);
            var size3 = new Size2D(4, 3);

            // Act & Assert
            size1.Should().Be(size2);
            size1.Should().NotBe(size3);
            (size1 == size2).Should().BeTrue();
            (size1 != size3).Should().BeTrue();
        }

        [Fact]
        public void Size2D_ShouldHaveValidProperties()
        {
            // Arrange & Act
            var size = new Size2D(5, 7);

            // Assert
            size.Width.Should().Be(5);
            size.Height.Should().Be(7);
        }

        [Fact]
        public void NeuralNetwork_ShouldCreateWithName()
        {
            // Arrange & Act
            var network = new NeuralNetwork("TestNetwork");

            // Assert
            network.Should().NotBeNull();
        }

        [Fact]
        public void OptimizationOptions_ShouldHaveValidDefaults()
        {
            // Arrange & Act
            var options = new OptimizationOptions();

            // Assert
            options.Should().NotBeNull();
        }
    }

    /// <summary>
    /// Tests for workload orchestration and scheduling.
    /// </summary>
    public class WorkloadTests : IDisposable
    {
        private readonly Context _context;

        public WorkloadTests()
        {
            _context = Context.CreateDefault();
        }

        [Fact]
        public void MatrixMultiplicationWorkload_ShouldCalculateComplexity()
        {
            // Arrange
            var a = TensorFactory.Create<float>(new TensorShape(128, 256), ComputeLocation.Cpu);
            var b = TensorFactory.Create<float>(new TensorShape(256, 512), ComputeLocation.Cpu);

            // Act
            var workload = new MatrixMultiplicationWorkload<float>(a, b);

            // Assert
            workload.WorkloadType.Should().Be(WorkloadType.MatrixMultiplication);
            workload.EstimatedComplexity.Should().BeGreaterThan(0);
            workload.MemoryRequirements.Should().BeGreaterThan(0);
        }

        [Fact]
        public void ConvolutionWorkload_ShouldCalculateComplexity()
        {
            // Arrange
            var input = TensorFactory.Create<float>(new TensorShape(1, 64, 224, 224), ComputeLocation.Cpu);
            var kernel = TensorFactory.Create<float>(new TensorShape(128, 64, 3, 3), ComputeLocation.Cpu);
            var parameters = new ConvolutionParameters
            {
                Stride = new Size2D(1, 1),
                Padding = new Size2D(1, 1)
            };

            // Act
            var workload = new ConvolutionWorkload<float>(input, kernel, parameters);

            // Assert
            workload.WorkloadType.Should().Be(WorkloadType.Convolution);
            workload.EstimatedComplexity.Should().BeGreaterThan(0);
            workload.MemoryRequirements.Should().BeGreaterThan(0);
        }

        [Fact]
        public void DistributedMatrixMultiplicationWorkload_ShouldSupportPartitioning()
        {
            // Arrange
            var a = TensorFactory.Create<float>(new TensorShape(128, 256), ComputeLocation.Cpu);
            var b = TensorFactory.Create<float>(new TensorShape(256, 512), ComputeLocation.Cpu);

            // Act
            var workload = new DistributedMatrixMultiplicationWorkload<float>(a, b);
            var partitions = workload.Partition(4).ToList();

            // Assert
            workload.RequiresAggregation.Should().BeTrue();
            partitions.Should().HaveCount(4);
        }

        [Fact]
        public async Task WorkloadScheduler_AnalyzeWorkloadAsync_ShouldReturnStrategy()
        {
            // Arrange
            var scheduler = new WorkloadScheduler();
            var a = TensorFactory.Create<float>(new TensorShape(64, 64), ComputeLocation.Cpu);
            var b = TensorFactory.Create<float>(new TensorShape(64, 64), ComputeLocation.Cpu);
            var workload = new MatrixMultiplicationWorkload<float>(a, b);
            
            var profiles = new List<AcceleratorProfile>();

            // Act
            var strategy = await scheduler.AnalyzeWorkloadAsync(workload, profiles);

            // Assert
            strategy.Should().NotBeNull();
        }

        [Theory]
        [InlineData(WorkloadType.MatrixMultiplication)]
        [InlineData(WorkloadType.Convolution)]
        [InlineData(WorkloadType.MatrixMultiplication)]
        public void WorkloadType_ShouldHaveValidValues(WorkloadType workloadType)
        {
            // Arrange & Act & Assert
            workloadType.Should().BeDefined();
        }

        public void Dispose()
        {
            _context?.Dispose();
        }
    }
}