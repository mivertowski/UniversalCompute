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
using ILGPU.Algorithms.FFT;
using ILGPU.Runtime;

namespace ILGPU.Benchmarks.FFT;

/// <summary>
/// Performance benchmarks for FFT algorithms.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class FFTBenchmarks
{
    private Context? _context;
    private Accelerator? _accelerator;
    private MemoryBuffer1D<float, Stride1D.Dense>? _inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _outputBuffer;

    [Params(256, 1024, 4096, 16384)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.CreateCPUAccelerator(0);
        
        _inputBuffer = _accelerator.Allocate1D<float>(Size);
        _outputBuffer = _accelerator.Allocate1D<float>(Size);

        // Initialize with test data
        var inputData = new float[Size];
        for (int i = 0; i < Size; i++)
        {
            inputData[i] = (float)Math.Sin(2 * Math.PI * i / Size);
        }
        _inputBuffer.CopyFromCPU(inputData);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _inputBuffer?.Dispose();
        _outputBuffer?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }

    [Benchmark]
    public void FFT1D_Forward()
    {
        if (_accelerator == null || _inputBuffer == null || _outputBuffer == null)
            return;

        var fft = new FFT<float>(_accelerator);
        fft.Forward(_inputBuffer.View, _outputBuffer.View);
        _accelerator.Synchronize();
    }

    [Benchmark]
    public void FFT1D_Inverse()
    {
        if (_accelerator == null || _inputBuffer == null || _outputBuffer == null)
            return;

        var fft = new FFT<float>(_accelerator);
        fft.Inverse(_inputBuffer.View, _outputBuffer.View);
        _accelerator.Synchronize();
    }

    [Benchmark]
    public void FFT1D_PowerOfTwo_Optimized()
    {
        if (_accelerator == null || _inputBuffer == null || _outputBuffer == null)
            return;

        var fft = new FFT<float>(_accelerator);
        fft.ForwardPowerOfTwo(_inputBuffer.View, _outputBuffer.View);
        _accelerator.Synchronize();
    }
}