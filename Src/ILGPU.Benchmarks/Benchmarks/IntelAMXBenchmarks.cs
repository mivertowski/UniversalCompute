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
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Benchmarks.Infrastructure;
using ILGPU.Intel.AMX;

namespace ILGPU.Benchmarks.Benchmarks;

/// <summary>
/// Benchmarks for Intel Advanced Matrix Extensions (AMX) operations.
/// AMX operations are simulated when actual AMX hardware is not available.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class IntelAMXBenchmarks : IDisposable
{
    private Context? context;
    private Accelerator? accelerator;
    private AMXAccelerator? amxAccelerator;
    private bool hasRealAMX;
    private MemoryBuffer2D<float, Stride2D.DenseX>? matrixA;
    private MemoryBuffer2D<float, Stride2D.DenseX>? matrixB;
    private MemoryBuffer2D<float, Stride2D.DenseX>? matrixC;
    private MemoryBuffer1D<sbyte, Stride1D.Dense>? int8DataA;
    private MemoryBuffer1D<sbyte, Stride1D.Dense>? int8DataB;

    [Params(16, 32, 64, 128)]
    public int TileSize { get; set; }

    [Params(1, 4, 8, 16)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            context = Context.CreateDefault();
            
            // Check for real Intel AMX hardware first
            hasRealAMX = AMXCapabilities.IsAMXSupported();
            
            if (hasRealAMX)
            {
                Console.WriteLine("🚀 Detected Intel AMX - using real hardware acceleration!");
                try
                {
                    accelerator = context.CreateAMXAccelerator(0);
                    amxAccelerator = accelerator as AMXAccelerator;
                    if (amxAccelerator == null)
                    {
                        Console.WriteLine("⚠️ AMX hardware detected but accelerator creation failed, falling back to simulation");
                        hasRealAMX = false;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️ AMX hardware detected but not accessible: {ex.Message}, falling back to simulation");
                    hasRealAMX = false;
                }
            }
            else
            {
                Console.WriteLine("ℹ️ Intel AMX not detected - using ILGPU simulation");
            }
            
            // Fallback to regular ILGPU accelerator if AMX not available
            if (!hasRealAMX)
            {
                var device = context.GetPreferredDevice(preferCPU: true);
                accelerator = device.CreateAccelerator(context);
            }

            // Allocate memory for matrix operations
            matrixA = accelerator.Allocate2DDenseX<float>(new Index2D(TileSize, TileSize));
            matrixB = accelerator.Allocate2DDenseX<float>(new Index2D(TileSize, TileSize));
            matrixC = accelerator.Allocate2DDenseX<float>(new Index2D(TileSize, TileSize));
            
            int8DataA = accelerator.Allocate1D<sbyte>(TileSize * TileSize);
            int8DataB = accelerator.Allocate1D<sbyte>(TileSize * TileSize);

            InitializeTestData();
            
            // Print AMX capabilities if available
            if (hasRealAMX)
            {
                var caps = AMXCapabilities.Query();
                Console.WriteLine($"💻 AMX Tile Config: {caps.MaxTileRows}x{caps.MaxTileColumns} (Max Size: {caps.MaxTileBytes})");
                Console.WriteLine($"🔧 Tile Registers: {caps.MaxTiles}, BF16 Support: {caps.SupportsBF16}");
            }
        }
        catch (Exception ex)
        {
            throw new NotSupportedException($"Failed to initialize AMX benchmark environment: {ex.Message}", ex);
        }
    }

    private void InitializeTestData()
    {
        var random = new Random(42);
        var totalElements = TileSize * TileSize;
        
        // Initialize FP32 matrices
        var matrixDataA = new float[totalElements];
        var matrixDataB = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            matrixDataA[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            matrixDataB[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }
        
        // Initialize INT8 data
        var int8A = new sbyte[totalElements];
        var int8B = new sbyte[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            int8A[i] = (sbyte)random.Next(-128, 127);
            int8B[i] = (sbyte)random.Next(-128, 127);
        }
        
        if (matrixA != null)
        {
            var index = 0;
            for (int y = 0; y < matrixA.IntExtent.Y && index < matrixDataA.Length; y++)
                for (int x = 0; x < matrixA.IntExtent.X && index < matrixDataA.Length; x++)
                    matrixA.View[y, x] = matrixDataA[index++];
        }
        if (matrixB != null)
        {
            var index = 0;
            for (int y = 0; y < matrixB.IntExtent.Y && index < matrixDataB.Length; y++)
                for (int x = 0; x < matrixB.IntExtent.X && index < matrixDataB.Length; x++)
                    matrixB.View[y, x] = matrixDataB[index++];
        }
        int8DataA?.View.CopyFromCPU(int8A);
        int8DataB?.View.CopyFromCPU(int8B);
    }

    [Benchmark(Baseline = true)]
    public float StandardMatrixMultiplication()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, int>(StandardMatMulKernel);

        if (matrixA == null || matrixB == null || matrixC == null)
            return 0.0f;
            
        kernel(new Index2D(TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    [Benchmark]
    public float AMXRealHardware()
    {
        if (!hasRealAMX || amxAccelerator == null)
        {
            // Fall back to simulation when real AMX not available
            return AMXSimulatedFP32();
        }

        try
        {
            // Use real Intel AMX hardware
            var kernel = amxAccelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, int>(AMXHardwareKernel);

            if (matrixA == null || matrixB == null || matrixC == null)
                return 0.0f;
                
            kernel(new Index2D(TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize);
            amxAccelerator.Synchronize();
            
            var result = matrixC.GetAsArray2D();
            Console.WriteLine("🚀 Executed on real Intel AMX hardware");
            return result[0, 0];
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Real AMX execution failed: {ex.Message}");
            // Fall back to simulation
            return AMXSimulatedFP32();
        }
    }

    [Benchmark]
    public float AMXSimulatedFP32()
    {
        // Simulate AMX FP32 matrix multiplication with tiled operations
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, int>(AMXTiledMatMulKernel);

        if (matrixA == null || matrixB == null || matrixC == null)
            return 0.0f;
            
        kernel(new Index2D(TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    [Benchmark]
    public float AMXSimulatedBF16()
    {
        // Simulate AMX BF16 operations with reduced precision
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, int>(AMXBFloat16Kernel);

        if (matrixA == null || matrixB == null || matrixC == null)
            return 0.0f;
            
        kernel(new Index2D(TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    [Benchmark]
    public float AMXSimulatedINT8()
    {
        // Simulate AMX INT8 quantized matrix multiplication
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<sbyte>, ArrayView<sbyte>, ArrayView<float>, int>(
            AMXQuantizedKernel);

        if (int8DataA == null || int8DataB == null || matrixC == null)
            return 0.0f;
            
        kernel(TileSize * TileSize, int8DataA.View, int8DataB.View, matrixC.View.AsLinearView(), TileSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    [Benchmark]
    public float AMXBatchedOperations()
    {
        // Simulate batched AMX operations
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index3D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, int, int>(BatchedAMXKernel);

        if (matrixA == null || matrixB == null || matrixC == null)
            return 0.0f;
            
        kernel(new Index3D(BatchSize, TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize, BatchSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    [Benchmark]
    public float AMXTileConfigOptimization()
    {
        // Simulate optimized tile configuration for different data patterns
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, int>(OptimizedTileKernel);

        if (matrixA == null || matrixB == null || matrixC == null)
            return 0.0f;
            
        kernel(new Index2D(TileSize, TileSize), matrixA.View, matrixB.View, matrixC.View, TileSize);
        accelerator!.Synchronize();
        
        var result = matrixC.GetAsArray2D();
        return result[0, 0];
    }

    #region AMX Kernels

    private static void AMXHardwareKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        // This kernel will be executed on real AMX hardware via ILGPU
        // The AMX accelerator will automatically optimize tile operations
        float sum = 0.0f;
        for (int k = 0; k < size; k++)
        {
            sum += a[index.X, k] * b[k, index.Y];
        }
        c[index.X, index.Y] = sum;
    }

    private static void StandardMatMulKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        float sum = 0.0f;
        for (int k = 0; k < size; k++)
        {
            sum += a[index.X, k] * b[k, index.Y];
        }
        c[index.X, index.Y] = sum;
    }

    private static void AMXTiledMatMulKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        // Simulate AMX 16x16 tile operations
        const int tileSize = 16;
        var tileX = index.X / tileSize;
        var tileY = index.Y / tileSize;
        var localX = index.X % tileSize;
        var localY = index.Y % tileSize;
        
        float sum = 0.0f;
        
        // Process in tile-sized chunks (AMX optimization)
        for (int tileK = 0; tileK < (size + tileSize - 1) / tileSize; tileK++)
        {
            for (int k = 0; k < tileSize; k++)
            {
                var globalK = tileK * tileSize + k;
                if (globalK < size)
                {
                    sum += a[index.X, globalK] * b[globalK, index.Y];
                }
            }
        }
        
        c[index.X, index.Y] = sum;
    }

    private static void AMXBFloat16Kernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        float sum = 0.0f;
        for (int k = 0; k < size; k++)
        {
            // Simulate BF16 precision by truncating mantissa
            var aVal = TruncateToBF16(a[index.X, k]);
            var bVal = TruncateToBF16(b[k, index.Y]);
            sum += aVal * bVal;
        }
        c[index.X, index.Y] = sum;
    }

    private static void AMXQuantizedKernel(
        Index1D index,
        ArrayView<sbyte> a,
        ArrayView<sbyte> b,
        ArrayView<float> c,
        int matrixSize)
    {
        if (index >= matrixSize * matrixSize)
            return;
            
        var row = index / matrixSize;
        var col = index % matrixSize;
        
        int sum = 0;
        for (int k = 0; k < matrixSize; k++)
        {
            var aIdx = row * matrixSize + k;
            var bIdx = k * matrixSize + col;
            
            if (aIdx < a.Length && bIdx < b.Length)
            {
                sum += a[aIdx] * b[bIdx];
            }
        }
        
        // Simulate dequantization (typical scale factor for INT8)
        c[index] = sum * 0.00390625f; // 1/256 scale factor
    }

    private static void BatchedAMXKernel(
        Index3D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int tileSize,
        int batchSize)
    {
        var batch = index.X;
        var row = index.Y;
        var col = index.Z;
        
        if (batch >= batchSize || row >= tileSize || col >= tileSize)
            return;
            
        // Real AMX-style batched matrix multiplication with tile register usage
        const int AMX_TILE_SIZE = 16; // AMX supports 16x16 tiles
        
        // Process in AMX-sized subtiles for optimal register utilization
        var tileRow = row / AMX_TILE_SIZE * AMX_TILE_SIZE;
        var tileCol = col / AMX_TILE_SIZE * AMX_TILE_SIZE;
        var localRow = row % AMX_TILE_SIZE;
        var localCol = col % AMX_TILE_SIZE;
        
        // Accumulator for this batch element
        float accumulator = 0.0f;
        
        // Process in tile-sized chunks as AMX would
        for (int tileK = 0; tileK < tileSize; tileK += AMX_TILE_SIZE)
        {
            float tileSum = 0.0f;
            var effectiveKSize = IntrinsicMath.Min(AMX_TILE_SIZE, tileSize - tileK);
            
            // Inner tile computation (simulates AMX TMUL operation)
            for (int k = 0; k < effectiveKSize; k++)
            {
                var globalK = tileK + k;
                if (globalK < tileSize)
                {
                    // Batch-specific data layout addressing
                    var aIdx = (tileRow + localRow, globalK);
                    var bIdx = (globalK, tileCol + localCol);
                    
                    // AMX operates on BF16 with accumulation in FP32
                    var aVal = TruncateToBF16(a[aIdx.Item1, aIdx.Item2]);
                    var bVal = TruncateToBF16(b[bIdx.Item1, bIdx.Item2]);
                    
                    tileSum += aVal * bVal;
                }
            }
            
            accumulator += tileSum;
        }
        
        // Apply batch-specific scaling (simulates different batch processing)
        var batchScale = 1.0f + (batch * 0.01f); // Small batch-dependent scaling
        accumulator *= batchScale;
        
        // Store result with AMX saturation behavior
        c[row, col] = IntrinsicMath.Max(-3.4e38f, IntrinsicMath.Min(3.4e38f, accumulator));
    }

    private static void OptimizedTileKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        // Simulate optimized tile configuration based on data locality
        const int optimalTileSize = 32; // Larger tiles for better cache utilization
        
        float sum = 0.0f;
        
        // Process in optimized tile blocks
        for (int tileK = 0; tileK < size; tileK += optimalTileSize)
        {
            var endK = IntrinsicMath.Min(tileK + optimalTileSize, size);
            
            for (int k = tileK; k < endK; k++)
            {
                sum += a[index.X, k] * b[k, index.Y];
            }
        }
        
        c[index.X, index.Y] = sum;
    }

    private static float TruncateToBF16(float value)
    {
        // Simulate BF16 precision by truncating mantissa bits
        var bits = BitConverter.SingleToUInt32Bits(value);
        var bf16Bits = bits & 0xFFFF0000; // Keep sign + exponent + 7 mantissa bits
        return BitConverter.UInt32BitsToSingle(bf16Bits);
    }

    #endregion

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        matrixA?.Dispose();
        matrixB?.Dispose();
        matrixC?.Dispose();
        int8DataA?.Dispose();
        int8DataB?.Dispose();
        amxAccelerator?.Dispose();
        accelerator?.Dispose();
        context?.Dispose();
    }
}