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

using BenchmarkDotNet.Attributes;
using ILGPU.AI.Quantization;
using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.NPU;
using ILGPU.Runtime;

namespace ILGPU.Benchmarks.UniversalAccelerators;

/// <summary>
/// Performance benchmarks for ML accelerators (ANE, NPU).
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class MLAcceleratorBenchmarks
{
    private Context? _context;
    private ANEAccelerator? _aneAccelerator;
    private IntelNPUAccelerator? _npuAccelerator;
    private MemoryBuffer1D<float, Stride1D.Dense>? _inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _outputBuffer;

    [Params(1024, 4096, 16384)]
    public int TensorSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = Context.CreateDefault();
        
        // Try to create ANE accelerator (Apple Neural Engine)
        try
        {
            _aneAccelerator = _context.CreateANEAccelerator(0);
        }
        catch
        {
            // ANE not available on this platform
        }

        // Try to create NPU accelerator (Intel Neural Processing Unit)
        try
        {
            _npuAccelerator = _context.CreateIntelNPUAccelerator(0);
        }
        catch
        {
            // NPU not available on this platform
        }

        // Use CPU as fallback if no ML accelerators available
        var accelerator = _aneAccelerator ?? _npuAccelerator ?? _context.CreateCPUAccelerator(0);
        
        _inputBuffer = accelerator.Allocate1D<float>(TensorSize);
        _outputBuffer = accelerator.Allocate1D<float>(TensorSize);

        // Initialize with random data
        var random = new Random(42);
        var inputData = new float[TensorSize];
        for (int i = 0; i < TensorSize; i++)
        {
            inputData[i] = (float)random.NextDouble();
        }
        _inputBuffer.CopyFromCPU(inputData);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _inputBuffer?.Dispose();
        _outputBuffer?.Dispose();
        _aneAccelerator?.Dispose();
        _npuAccelerator?.Dispose();
        _context?.Dispose();
    }

    [Benchmark]
    public void ANE_TensorOperation()
    {
        if (_aneAccelerator == null || _inputBuffer == null || _outputBuffer == null)
            return;

        // Perform a simple tensor operation on ANE
        var stream = _aneAccelerator.CreateStream();
        _outputBuffer.CopyFrom(_inputBuffer, stream);
        stream.Synchronize();
        stream.Dispose();
    }

    [Benchmark]
    public void NPU_TensorOperation()
    {
        if (_npuAccelerator == null || _inputBuffer == null || _outputBuffer == null)
            return;

        // Perform a simple tensor operation on NPU
        var stream = _npuAccelerator.CreateStream();
        _outputBuffer.CopyFrom(_inputBuffer, stream);
        stream.Synchronize();
        stream.Dispose();
    }

    [Benchmark]
    public void Quantization_INT8()
    {
        if (_inputBuffer == null || _outputBuffer == null)
            return;

        var accelerator = _aneAccelerator ?? _npuAccelerator ?? _context?.CreateCPUAccelerator(0);
        if (accelerator == null) return;

        // Benchmark INT8 quantization
        var quantizedBuffer = accelerator.Allocate1D<byte>(TensorSize);
        
        // Simulate quantization (actual implementation would use QuantizationKernels)
        var stream = accelerator.CreateStream();
        quantizedBuffer.MemSetToZero(stream);
        stream.Synchronize();
        
        quantizedBuffer.Dispose();
        stream.Dispose();
    }

    [Benchmark]
    public void Quantization_FP16()
    {
        if (_inputBuffer == null || _outputBuffer == null)
            return;

        var accelerator = _aneAccelerator ?? _npuAccelerator ?? _context?.CreateCPUAccelerator(0);
        if (accelerator == null) return;

        // Benchmark FP16 quantization
        var fp16Buffer = accelerator.Allocate1D<Half>(TensorSize);
        
        var stream = accelerator.CreateStream();
        fp16Buffer.MemSetToZero(stream);
        stream.Synchronize();
        
        fp16Buffer.Dispose();
        stream.Dispose();
    }
}