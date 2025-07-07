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

namespace ILGPU.Benchmarks.Benchmarks
{
    /// <summary>
    /// Benchmarks for Phase 7 hybrid processing capabilities across multiple accelerators.
    /// </summary>
    [MemoryDiagnoser]
    [SimpleJob]
    public class HybridProcessingBenchmarks : IDisposable
    {
        private Context? context;
        private Accelerator? cpuAccelerator;
        private Accelerator? gpuAccelerator;
        private MemoryBuffer1D<float, Stride1D.Dense>? cpuBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense>? gpuBuffer;
        private float[]? hostData;

        [Params(1024, 4096, 16384, 65536)]
        public int ProblemSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            // Setup CPU accelerator  
            context = SharedBenchmarkContext.GetOrCreateContext();
            cpuAccelerator = context.GetPreferredDevice(preferCPU: true).CreateAccelerator(context);
            
            // Try to setup GPU accelerator
            foreach (var device in context.Devices)
            {
                if (device.AcceleratorType != AcceleratorType.CPU)
                {
                    gpuAccelerator = device.CreateAccelerator(context);
                    break;
                }
            }

            hostData = new float[ProblemSize];
            var random = new Random(42);
            for (int i = 0; i < ProblemSize; i++)
            {
                hostData[i] = (float)random.NextDouble();
            }

            cpuBuffer = cpuAccelerator.Allocate1D<float>(ProblemSize);
            cpuBuffer.CopyFromCPU(hostData);

            if (gpuAccelerator != null)
            {
                gpuBuffer = gpuAccelerator.Allocate1D<float>(ProblemSize);
                gpuBuffer.CopyFromCPU(hostData);
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            cpuBuffer?.Dispose();
            gpuBuffer?.Dispose();
            cpuAccelerator?.Dispose();
            gpuAccelerator?.Dispose();
            context?.Dispose();
        }

        [Benchmark(Baseline = true)]
        public float CPU_Only_VectorAdd()
        {
            var kernel = cpuAccelerator!.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (index, a, b, c) => c[index] = a[index] + b[index]);

            using var result = cpuAccelerator.Allocate1D<float>(ProblemSize);
            kernel((Index1D)ProblemSize, cpuBuffer!.View, cpuBuffer.View, result.View);
            cpuAccelerator.Synchronize();

            var output = result.GetAsArray1D();
            return output[0];
        }

        [Benchmark]
        public float GPU_Only_VectorAdd()
        {
            if (gpuAccelerator == null)
            {
                return 0f;
            }

            var kernel = gpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (index, a, b, c) => c[index] = a[index] + b[index]);

            using var result = gpuAccelerator.Allocate1D<float>(ProblemSize);
            kernel((Index1D)ProblemSize, gpuBuffer!.View, gpuBuffer.View, result.View);
            gpuAccelerator.Synchronize();

            var output = result.GetAsArray1D();
            return output[0];
        }

        [Benchmark]
        public float Hybrid_CPUPreprocess_GPUCompute()
        {
            if (gpuAccelerator == null)
            {
                return CPU_Only_VectorAdd();
            }

            // CPU preprocessing: normalize data
            var preprocessKernel = cpuAccelerator!.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = input[index] / 2.0f);

            using var preprocessed = cpuAccelerator.Allocate1D<float>(ProblemSize);
            preprocessKernel((Index1D)ProblemSize, cpuBuffer!.View, preprocessed.View);
            cpuAccelerator.Synchronize();

            // Transfer to GPU
            using var gpuPreprocessed = gpuAccelerator.Allocate1D<float>(ProblemSize);
            var hostTemp = preprocessed.GetAsArray1D();
            gpuPreprocessed.CopyFromCPU(hostTemp);

            // GPU computation: complex operation
            var computeKernel = gpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = input[index] * input[index] + 1.0f); // Simplified operation

            using var result = gpuAccelerator.Allocate1D<float>(ProblemSize);
            computeKernel((Index1D)ProblemSize, gpuPreprocessed.View, result.View);
            gpuAccelerator.Synchronize();

            var output = result.GetAsArray1D();
            return output[0];
        }

        [Benchmark]
        public float Hybrid_GPUCompute_CPUPostprocess()
        {
            if (gpuAccelerator == null)
            {
                return CPU_Only_VectorAdd();
            }

            // GPU computation: heavy mathematical operation
            var computeKernel = gpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = input[index] * input[index] * input[index] + input[index] + 1.0f);

            using var gpuResult = gpuAccelerator.Allocate1D<float>(ProblemSize);
            computeKernel((Index1D)ProblemSize, gpuBuffer!.View, gpuResult.View);
            gpuAccelerator.Synchronize();

            // Transfer to CPU
            var hostTemp = gpuResult.GetAsArray1D();
            using var cpuResult = cpuAccelerator!.Allocate1D<float>(ProblemSize);
            cpuResult.CopyFromCPU(hostTemp);

            // CPU postprocessing: validation and filtering
            var postprocessKernel = cpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = (input[index] != input[index]) ? 0.0f : (input[index] < 1000.0f ? input[index] : 1000.0f));

            using var finalResult = cpuAccelerator.Allocate1D<float>(ProblemSize);
            postprocessKernel((Index1D)ProblemSize, cpuResult.View, finalResult.View);
            cpuAccelerator.Synchronize();

            var output = finalResult.GetAsArray1D();
            return output[0];
        }

        [Benchmark]
        public float Hybrid_Parallel_Execution()
        {
            if (gpuAccelerator == null)
            {
                return CPU_Only_VectorAdd();
            }

            var halfSize = ProblemSize / 2;

            // Split data processing between CPU and GPU
            using var cpuPart = cpuAccelerator!.Allocate1D<float>(halfSize);
            using var gpuPart = gpuAccelerator.Allocate1D<float>(halfSize);
            
            // Copy data splits
            var cpuPartData = new float[halfSize];
            var gpuPartData = new float[halfSize];
            Array.Copy(hostData!, 0, cpuPartData, 0, halfSize);
            Array.Copy(hostData!, halfSize, gpuPartData, 0, halfSize);
            cpuPart.CopyFromCPU(cpuPartData);
            gpuPart.CopyFromCPU(gpuPartData);

            // Define kernels
            var cpuKernel = cpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = input[index] * 2.0f + 1.0f);

            var gpuKernel = gpuAccelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                (index, input, output) => output[index] = input[index] * input[index] + 1.0f);

            using var cpuResult = cpuAccelerator.Allocate1D<float>(halfSize);
            using var gpuResult = gpuAccelerator.Allocate1D<float>(halfSize);

            // Execute in parallel (asynchronously)
            cpuKernel((Index1D)halfSize, cpuPart.View, cpuResult.View);
            gpuKernel((Index1D)halfSize, gpuPart.View, gpuResult.View);

            // Synchronize both
            cpuAccelerator.Synchronize();
            gpuAccelerator.Synchronize();

            // Combine results
            var cpuOutput = cpuResult.GetAsArray1D();
            var gpuOutput = gpuResult.GetAsArray1D();

            return cpuOutput[0] + gpuOutput[0];
        }

        [Benchmark]
        public float HybridTensor_CPUSetup_GPUCompute()
        {
            if (gpuAccelerator == null)
            {
                return CPU_Only_VectorAdd();
            }

            try
            {
                // Simulate hybrid tensor processing workflow
                var tileSize = 32;
                
                // Create CPU tensors
                using var cpuMatrixA = cpuAccelerator!.Allocate2DDenseX<float>(new Index2D(tileSize, tileSize));
                using var cpuMatrixB = cpuAccelerator.Allocate2DDenseX<float>(new Index2D(tileSize, tileSize));
                
                // Initialize data on CPU
                var temp2D = new float[tileSize, tileSize];
                for (int i = 0; i < tileSize; i++)
                {
                    for (int j = 0; j < tileSize; j++)
                    {
                        temp2D[i, j] = hostData![i * tileSize + j];
                    }
                }
                cpuMatrixA.CopyFromCPU(temp2D);
                cpuMatrixB.CopyFromCPU(temp2D);

                // Transfer to GPU for computation
                using var gpuMatrixA = gpuAccelerator.Allocate2DDenseX<float>(new Index2D(tileSize, tileSize));
                using var gpuMatrixB = gpuAccelerator.Allocate2DDenseX<float>(new Index2D(tileSize, tileSize));
                using var gpuResult = gpuAccelerator.Allocate2DDenseX<float>(new Index2D(tileSize, tileSize));

                var cpuDataA = cpuMatrixA.GetAsArray2D();
                var cpuDataB = cpuMatrixB.GetAsArray2D();
                gpuMatrixA.CopyFromCPU(cpuDataA);
                gpuMatrixB.CopyFromCPU(cpuDataB);

                // Perform matrix multiplication on GPU
                var kernel = gpuAccelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(
                    (index, a, b, c, size) =>
                    {
                        var sum = 0.0f;
                        for (int k = 0; k < size; k++)
                        {
                            sum += a[index.X, k] * b[k, index.Y];
                        }
                        c[index.X, index.Y] = sum;
                    });

                kernel(new Index2D(tileSize, tileSize), gpuMatrixA.View, gpuMatrixB.View, gpuResult.View, tileSize);
                gpuAccelerator.Synchronize();

                var result = gpuResult.GetAsArray2D();
                return result[0, 0];
            }
            catch
            {
                // Fallback if hybrid processing not available
                return CPU_Only_VectorAdd();
            }
        }

        public void Dispose()
        {
            Cleanup();
        }
    }
}