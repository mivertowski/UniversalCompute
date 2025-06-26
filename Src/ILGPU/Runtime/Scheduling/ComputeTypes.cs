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
// Change License: Apache License, Version 2.0using System;

namespace ILGPU.Runtime.Scheduling
{
    /// <summary>
    /// Represents different types of compute devices.
    /// </summary>
    public enum ComputeDevice
    {
        /// <summary>
        /// Automatically select the best device.
        /// </summary>
        Auto,

        /// <summary>
        /// CPU with SIMD capabilities.
        /// </summary>
        CPU,

        /// <summary>
        /// General purpose GPU.
        /// </summary>
        GPU,

        /// <summary>
        /// NVIDIA GPU with CUDA support.
        /// </summary>
        CUDA,

        /// <summary>
        /// OpenCL-compatible device.
        /// </summary>
        OpenCL,

        /// <summary>
        /// Apple Metal device.
        /// </summary>
        Metal,

        /// <summary>
        /// Intel integrated GPU.
        /// </summary>
        IntelGPU,

        /// <summary>
        /// Intel Neural Processing Unit.
        /// </summary>
        IntelNPU,

        /// <summary>
        /// Intel Advanced Matrix Extensions.
        /// </summary>
        IntelAMX,

        /// <summary>
        /// Apple Neural Engine.
        /// </summary>
        AppleNeuralEngine,

        /// <summary>
        /// Apple Matrix coprocessor.
        /// </summary>
        AppleAMX
    }

    /// <summary>
    /// Represents a compute node in the execution graph.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ComputeNode class.
    /// </remarks>
    public class ComputeNode(IComputeOperation operation)
    {
        /// <summary>
        /// Gets or sets the operation to be executed.
        /// </summary>
        public IComputeOperation Operation { get; set; } = operation ?? throw new ArgumentNullException(nameof(operation));

        /// <summary>
        /// Gets or sets the preferred device for execution.
        /// </summary>
        public ComputeDevice PreferredDevice { get; set; } = ComputeDevice.Auto;

        /// <summary>
        /// Gets or sets the estimated execution time in milliseconds.
        /// </summary>
        public double EstimatedTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the node identifier.
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();
    }

    /// <summary>
    /// Base interface for compute operations.
    /// </summary>
    public interface IComputeOperation
    {
        /// <summary>
        /// Gets the estimated floating-point operations for this operation.
        /// </summary>
        double EstimatedFlops { get; }

        /// <summary>
        /// Gets the estimated memory operations for this operation.
        /// </summary>
        long MemoryOperations { get; }

        /// <summary>
        /// Gets the operation type name.
        /// </summary>
        string OperationType { get; }
    }

    /// <summary>
    /// Represents a matrix multiplication operation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the MatMulOp class.
    /// </remarks>
    public class MatMulOp(int m, int n, int k) : IComputeOperation
    {
        /// <summary>
        /// Gets the M dimension (rows of A and C).
        /// </summary>
        public int M { get; } = m;

        /// <summary>
        /// Gets the N dimension (columns of B and C).
        /// </summary>
        public int N { get; } = n;

        /// <summary>
        /// Gets the K dimension (columns of A and rows of B).
        /// </summary>
        public int K { get; } = k;

        /// <summary>
        /// Gets the total size of the operation.
        /// </summary>
        public long Size => (long)M * N * K;

        /// <inheritdoc/>
        public double EstimatedFlops => 2.0 * M * N * K;

        /// <inheritdoc/>
        public long MemoryOperations => (long)M * K + (long)K * N + (long)M * N;

        /// <inheritdoc/>
        public string OperationType => "MatMul";
    }

    /// <summary>
    /// Represents a convolution operation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the ConvolutionOp class.
    /// </remarks>
    public class ConvolutionOp(long outputSize, int kernelSize, int inputChannels) : IComputeOperation
    {
        /// <summary>
        /// Gets the output size.
        /// </summary>
        public long OutputSize { get; } = outputSize;

        /// <summary>
        /// Gets the kernel size.
        /// </summary>
        public int KernelSize { get; } = kernelSize;

        /// <summary>
        /// Gets the number of input channels.
        /// </summary>
        public int InputChannels { get; } = inputChannels;

        /// <inheritdoc/>
        public double EstimatedFlops => OutputSize * KernelSize * KernelSize * InputChannels * 2.0;

        /// <inheritdoc/>
        public long MemoryOperations => OutputSize + (KernelSize * KernelSize * InputChannels);

        /// <inheritdoc/>
        public string OperationType => "Convolution";
    }

    /// <summary>
    /// Represents a vector operation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the VectorOp class.
    /// </remarks>
    public class VectorOp(long size, int elementSize = 4) : IComputeOperation
    {
        /// <summary>
        /// Gets the size of the vector operation.
        /// </summary>
        public long Size { get; } = size;

        /// <summary>
        /// Gets the element size in bytes.
        /// </summary>
        public int ElementSize { get; } = elementSize;

        /// <inheritdoc/>
        public double EstimatedFlops => Size;

        /// <inheritdoc/>
        public long MemoryOperations => Size * 2; // Read + Write

        /// <inheritdoc/>
        public string OperationType => "Vector";
    }

    /// <summary>
    /// Represents a memory operation.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the MemoryOp class.
    /// </remarks>
    public class MemoryOp(long sizeBytes) : IComputeOperation
    {
        /// <summary>
        /// Gets the size of the memory operation in bytes.
        /// </summary>
        public long SizeBytes { get; } = sizeBytes;

        /// <inheritdoc/>
        public double EstimatedFlops => 0; // Pure memory operation

        /// <inheritdoc/>
        public long MemoryOperations => SizeBytes;

        /// <inheritdoc/>
        public string OperationType => "Memory";
    }

    /// <summary>
    /// Represents device performance characteristics.
    /// </summary>
    public class DevicePerformance
    {
        /// <summary>
        /// Gets or sets the peak GFLOPS performance.
        /// </summary>
        public double PeakGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets the memory bandwidth in GB/s.
        /// </summary>
        public double MemoryBandwidthGBps { get; set; }

        /// <summary>
        /// Gets or sets whether tensor cores are supported.
        /// </summary>
        public bool SupportsTensorCores { get; set; }

        /// <summary>
        /// Gets or sets the tensor performance in GFLOPS.
        /// </summary>
        public double TensorPerformanceGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets whether AI acceleration is available.
        /// </summary>
        public bool HasAIAcceleration { get; set; }

        /// <summary>
        /// Gets or sets the AI performance in GOPS.
        /// </summary>
        public double AIPerformanceGOPS { get; set; }

        /// <summary>
        /// Gets or sets whether matrix extensions are supported.
        /// </summary>
        public bool SupportsMatrixExtensions { get; set; }

        /// <summary>
        /// Gets or sets the matrix performance in GFLOPS.
        /// </summary>
        public double MatrixPerformanceGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets the SIMD width in bits.
        /// </summary>
        public int SIMDWidthBits { get; set; }

        /// <summary>
        /// Gets or sets the SIMD performance in GFLOPS.
        /// </summary>
        public double SIMDPerformanceGFLOPS { get; set; }

        /// <summary>
        /// Gets or sets the average latency in milliseconds.
        /// </summary>
        public double AverageLatencyMs { get; set; }

        /// <summary>
        /// Gets or sets the performance per watt.
        /// </summary>
        public double PerformancePerWatt { get; set; }

        /// <summary>
        /// Gets or sets whether matrix operations are supported.
        /// </summary>
        public bool SupportsMatrixOperations { get; set; }

        /// <summary>
        /// Gets or sets whether convolution operations are supported.
        /// </summary>
        public bool SupportsConvolution { get; set; }
    }
}