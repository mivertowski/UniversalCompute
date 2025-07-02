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

using BenchmarkDotNet.Running;
using ILGPU.Benchmarks.FFT;
using ILGPU.Benchmarks.UniversalAccelerators;

namespace ILGPU.Benchmarks;

/// <summary>
/// Main entry point for ILGPU performance benchmarks.
/// </summary>
public class Program
{
    /// <summary>
    /// Runs all performance benchmarks.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    public static void Main(string[] args)
    {
        Console.WriteLine("ILGPU Universal Accelerator Performance Benchmarks");
        Console.WriteLine("===================================================");

        if (args.Length > 0)
        {
            switch (args[0].ToLowerInvariant())
            {
                case "fft":
                    BenchmarkRunner.Run<FFTBenchmarks>();
                    break;
                case "ml":
                    BenchmarkRunner.Run<MLAcceleratorBenchmarks>();
                    break;
                case "matrix":
                    BenchmarkRunner.Run<MatrixBenchmarks>();
                    break;
                case "all":
                default:
                    RunAllBenchmarks();
                    break;
            }
        }
        else
        {
            RunAllBenchmarks();
        }
    }

    private static void RunAllBenchmarks()
    {
        Console.WriteLine("Running all benchmark suites...\n");
        
        BenchmarkRunner.Run<FFTBenchmarks>();
        BenchmarkRunner.Run<MLAcceleratorBenchmarks>();
        BenchmarkRunner.Run<MatrixBenchmarks>();
    }
}