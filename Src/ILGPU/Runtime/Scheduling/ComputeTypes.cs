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

using System;
using System.Collections.Generic;

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
    public class ComputeNode
    {
        /// <summary>
        /// Gets or sets the operation to be executed.
        /// </summary>
        public IComputeOperation Operation { get; set; }

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

        /// <summary>
        /// Initializes a new instance of the ComputeNode class.
        /// </summary>
        public ComputeNode(IComputeOperation operation)
        {
            Operation = operation ?? throw new ArgumentNullException(nameof(operation));
        }
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
    public class MatMulOp : IComputeOperation
    {
        /// <summary>
        /// Gets the M dimension (rows of A and C).
        /// </summary>
        public int M { get; }

        /// <summary>
        /// Gets the N dimension (columns of B and C).
        /// </summary>
        public int N { get; }

        /// <summary>
        /// Gets the K dimension (columns of A and rows of B).
        /// </summary>
        public int K { get; }

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

        /// <summary>
        /// Initializes a new instance of the MatMulOp class.
        /// </summary>
        public MatMulOp(int m, int n, int k)
        {
            M = m;
            N = n;
            K = k;
        }
    }

    /// <summary>
    /// Represents a convolution operation.
    /// </summary>
    public class ConvolutionOp : IComputeOperation
    {
        /// <summary>
        /// Gets the output size.
        /// </summary>
        public long OutputSize { get; }

        /// <summary>
        /// Gets the kernel size.
        /// </summary>
        public int KernelSize { get; }

        /// <summary>
        /// Gets the number of input channels.
        /// </summary>
        public int InputChannels { get; }

        /// <inheritdoc/>
        public double EstimatedFlops => OutputSize * KernelSize * KernelSize * InputChannels * 2.0;

        /// <inheritdoc/>
        public long MemoryOperations => OutputSize + (KernelSize * KernelSize * InputChannels);

        /// <inheritdoc/>
        public string OperationType => "Convolution";

        /// <summary>
        /// Initializes a new instance of the ConvolutionOp class.
        /// </summary>
        public ConvolutionOp(long outputSize, int kernelSize, int inputChannels)
        {
            OutputSize = outputSize;
            KernelSize = kernelSize;
            InputChannels = inputChannels;
        }
    }

    /// <summary>
    /// Represents a vector operation.
    /// </summary>
    public class VectorOp : IComputeOperation
    {
        /// <summary>
        /// Gets the size of the vector operation.
        /// </summary>
        public long Size { get; }

        /// <summary>
        /// Gets the element size in bytes.
        /// </summary>
        public int ElementSize { get; }

        /// <inheritdoc/>
        public double EstimatedFlops => Size;

        /// <inheritdoc/>
        public long MemoryOperations => Size * 2; // Read + Write

        /// <inheritdoc/>
        public string OperationType => "Vector";

        /// <summary>
        /// Initializes a new instance of the VectorOp class.
        /// </summary>
        public VectorOp(long size, int elementSize = 4)
        {
            Size = size;
            ElementSize = elementSize;
        }
    }

    /// <summary>
    /// Represents a memory operation.
    /// </summary>
    public class MemoryOp : IComputeOperation
    {
        /// <summary>
        /// Gets the size of the memory operation in bytes.
        /// </summary>
        public long SizeBytes { get; }

        /// <inheritdoc/>
        public double EstimatedFlops => 0; // Pure memory operation

        /// <inheritdoc/>
        public long MemoryOperations => SizeBytes;

        /// <inheritdoc/>
        public string OperationType => "Memory";

        /// <summary>
        /// Initializes a new instance of the MemoryOp class.
        /// </summary>
        public MemoryOp(long sizeBytes)
        {
            SizeBytes = sizeBytes;
        }
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