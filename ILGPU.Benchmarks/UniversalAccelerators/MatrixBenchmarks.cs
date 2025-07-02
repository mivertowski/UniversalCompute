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
using ILGPU.Algorithms.SparseMatrix;
using ILGPU.Runtime;

namespace ILGPU.Benchmarks.UniversalAccelerators;

/// <summary>
/// Performance benchmarks for matrix operations and sparse matrix algorithms.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class MatrixBenchmarks
{
    private Context? _context;
    private Accelerator? _accelerator;
    private MemoryBuffer2D<float, Stride2D.DenseX>? _matrixA;
    private MemoryBuffer2D<float, Stride2D.DenseX>? _matrixB;
    private MemoryBuffer2D<float, Stride2D.DenseX>? _matrixC;

    [Params(64, 128, 256, 512)]
    public int MatrixSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _context = Context.CreateDefault();
        _accelerator = _context.CreateCPUAccelerator(0);
        
        _matrixA = _accelerator.Allocate2DDenseX<float>(new Index2D(MatrixSize, MatrixSize));
        _matrixB = _accelerator.Allocate2DDenseX<float>(new Index2D(MatrixSize, MatrixSize));
        _matrixC = _accelerator.Allocate2DDenseX<float>(new Index2D(MatrixSize, MatrixSize));

        // Initialize matrices with test data
        var random = new Random(42);
        var dataA = new float[MatrixSize, MatrixSize];
        var dataB = new float[MatrixSize, MatrixSize];

        for (int i = 0; i < MatrixSize; i++)
        {
            for (int j = 0; j < MatrixSize; j++)
            {
                dataA[i, j] = (float)random.NextDouble();
                dataB[i, j] = (float)random.NextDouble();
            }
        }

        _matrixA.CopyFromCPU(dataA);
        _matrixB.CopyFromCPU(dataB);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _matrixA?.Dispose();
        _matrixB?.Dispose();
        _matrixC?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }

    [Benchmark]
    public void MatrixMultiplication()
    {
        if (_accelerator == null || _matrixA == null || _matrixB == null || _matrixC == null)
            return;

        // Simple matrix multiplication benchmark
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(MatrixMultiplyKernel);

        kernel(new Index2D(MatrixSize, MatrixSize), _matrixA.View, _matrixB.View, _matrixC.View);
        _accelerator.Synchronize();
    }

    [Benchmark]
    public void MatrixTranspose()
    {
        if (_accelerator == null || _matrixA == null || _matrixC == null)
            return;

        // Matrix transpose benchmark
        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(MatrixTransposeKernel);

        kernel(new Index2D(MatrixSize, MatrixSize), _matrixA.View, _matrixC.View);
        _accelerator.Synchronize();
    }

    [Benchmark]
    public void SparseMatrixOperations()
    {
        if (_accelerator == null || _matrixA == null)
            return;

        // Simulate sparse matrix operations
        var sparseMatrix = new SparseMatrix<float>(_accelerator, MatrixSize, MatrixSize);
        
        // Add some sparse elements
        for (int i = 0; i < MatrixSize; i += 10)
        {
            for (int j = 0; j < MatrixSize; j += 10)
            {
                sparseMatrix.SetValue(i, j, 1.0f);
            }
        }

        // Perform sparse matrix-vector multiplication
        var vector = _accelerator.Allocate1D<float>(MatrixSize);
        var result = _accelerator.Allocate1D<float>(MatrixSize);

        sparseMatrix.MultiplyVector(vector.View, result.View);
        _accelerator.Synchronize();

        vector.Dispose();
        result.Dispose();
        sparseMatrix.Dispose();
    }

    /// <summary>
    /// GPU kernel for matrix multiplication.
    /// </summary>
    private static void MatrixMultiplyKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        var x = index.X;
        var y = index.Y;

        if (x >= c.IntExtent.X || y >= c.IntExtent.Y)
            return;

        float sum = 0.0f;
        for (int k = 0; k < a.IntExtent.Y; k++)
        {
            sum += a[x, k] * b[k, y];
        }
        c[x, y] = sum;
    }

    /// <summary>
    /// GPU kernel for matrix transpose.
    /// </summary>
    private static void MatrixTransposeKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var x = index.X;
        var y = index.Y;

        if (x >= input.IntExtent.X || y >= input.IntExtent.Y)
            return;

        output[y, x] = input[x, y];
    }
}