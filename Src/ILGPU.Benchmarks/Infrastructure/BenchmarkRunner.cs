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

using ILGPU.Benchmarks.Benchmarks;
using Microsoft.Extensions.Logging;
using Spectre.Console;

namespace ILGPU.Benchmarks.Infrastructure;

/// <summary>
/// Orchestrates the execution of different benchmark suites.
/// </summary>
public class BenchmarkRunner(ILogger<BenchmarkRunner> logger, BenchmarkConfig config)
{
    private readonly ILogger<BenchmarkRunner> logger = logger;
    private readonly BenchmarkConfig config = config;

    /// <summary>
    /// Runs a quick performance suite for basic validation.
    /// </summary>
    public async Task RunQuickSuiteAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Quick Performance Suite[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Green));

        await AnsiConsole.Progress()
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new ElapsedTimeColumn())
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[green]Running quick benchmarks...[/]", maxValue: 100);

                // Basic SIMD operations
                task.Description = "[green]SIMD Vector Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<SimdVectorBenchmarks>(config.QuickConfig);
                task.Increment(25);

                // Basic tensor operations
                task.Description = "[green]Basic Tensor Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<TensorCoreBenchmarks>(config.QuickConfig);
                task.Increment(25);

                // Memory operations
                task.Description = "[green]Memory Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<MemoryBenchmarks>(config.QuickConfig);
                task.Increment(25);

                // AI performance primitives
                task.Description = "[green]GPU-Only Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<GpuOnlyBenchmarks>(config.QuickConfig);
                task.Increment(25);

                task.Description = "[green]Quick suite completed![/]";
                await Task.Delay(500); // Brief pause to show completion
            });
    }

    /// <summary>
    /// Runs comprehensive tensor core benchmarks.
    /// </summary>
    public async Task RunTensorCoreBenchmarksAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Tensor Core Performance Benchmarks[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Blue));

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[blue]Running tensor core benchmarks...[/]", maxValue: 100);

                task.Description = "[blue]Matrix Multiply-Accumulate (MMA)[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<TensorCoreBenchmarks>(config.StandardConfig);
                task.Increment(33);

                task.Description = "[blue]Mixed Precision Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<MixedPrecisionBenchmarks>(config.StandardConfig);
                task.Increment(33);

                task.Description = "[blue]BFloat16 Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<BFloat16Benchmarks>(config.StandardConfig);
                task.Increment(34);

                task.Description = "[blue]Tensor core benchmarks completed![/]";
                await Task.Delay(500);
            });
    }

    /// <summary>
    /// Runs SIMD performance benchmarks.
    /// </summary>
    public async Task RunSimdBenchmarksAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]SIMD Performance Benchmarks[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Yellow));

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[yellow]Running SIMD benchmarks...[/]", maxValue: 100);

                task.Description = "[yellow]Vector Operations (Add, Multiply, Dot Product)[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<SimdVectorBenchmarks>(config.StandardConfig);
                task.Increment(25);

                task.Description = "[yellow]Platform-Specific Intrinsics (AVX, SSE, NEON)[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<PlatformIntrinsicsBenchmarks>(config.StandardConfig);
                task.Increment(25);

                task.Description = "[yellow]Matrix-Vector Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<MatrixVectorBenchmarks>(config.StandardConfig);
                task.Increment(25);

                task.Description = "[yellow]CPU vs GPU Vectorization[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<CpuGpuComparisonBenchmarks>(config.StandardConfig);
                task.Increment(25);

                task.Description = "[yellow]SIMD benchmarks completed![/]";
                await Task.Delay(500);
            });
    }

    /// <summary>
    /// Runs hybrid processing benchmarks.
    /// </summary>
    public async Task RunHybridBenchmarksAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Hybrid Processing Benchmarks[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Purple));

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[magenta]Running hybrid benchmarks...[/]", maxValue: 100);

                task.Description = "[magenta]AI Performance Primitives[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<AIPerformancePrimitivesBenchmarks>(config.StandardConfig);
                task.Increment(50);

                task.Description = "[magenta]CPU/GPU Pipeline Performance[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<PipelineBenchmarks>(config.StandardConfig);
                task.Increment(50);

                task.Description = "[magenta]Hybrid benchmarks completed![/]";
                await Task.Delay(500);
            });
    }

    /// <summary>
    /// Runs memory operation benchmarks.
    /// </summary>
    public async Task RunMemoryBenchmarksAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Memory Operations Benchmarks[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Orange1));

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[orange1]Running memory benchmarks...[/]", maxValue: 100);

                task.Description = "[orange1]Zero-Copy Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<MemoryBenchmarks>(config.StandardConfig);
                task.Increment(33);

                task.Description = "[orange1]Memory Layout Optimization[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<MemoryLayoutBenchmarks>(config.StandardConfig);
                task.Increment(33);

                task.Description = "[orange1]Unified Memory Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<UnifiedMemoryBenchmarks>(config.StandardConfig);
                task.Increment(34);

                task.Description = "[orange1]Memory benchmarks completed![/]";
                await Task.Delay(500);
            });
    }

    /// <summary>
    /// Runs specialized hardware benchmarks (NPU, AMX, Apple Neural Engine).
    /// </summary>
    public async Task RunSpecializedHardwareBenchmarksAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Specialized Hardware Benchmarks[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Purple));

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("[magenta]Running specialized hardware benchmarks...[/]", maxValue: 100);

                task.Description = "[magenta]Hardware Accelerator Comparison (ILGPU)[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<HardwareAcceleratorComparison>(config.StandardConfig);
                task.Increment(20);

                task.Description = "[magenta]Intel NPU Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<IntelNPUBenchmarks>(config.StandardConfig);
                task.Increment(20);

                task.Description = "[magenta]Intel AMX Matrix Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<IntelAMXBenchmarks>(config.StandardConfig);
                task.Increment(20);

                task.Description = "[magenta]Apple Neural Engine Operations[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<AppleNeuralEngineBenchmarks>(config.StandardConfig);
                task.Increment(20);

                task.Description = "[magenta]GPU-Only Performance[/]";
                BenchmarkDotNet.Running.BenchmarkRunner.Run<GpuOnlyBenchmarks>(config.StandardConfig);
                task.Increment(20);

                task.Description = "[magenta]Specialized hardware benchmarks completed![/]";
                await Task.Delay(500);
            });
    }

    /// <summary>
    /// Runs the comprehensive benchmark suite.
    /// </summary>
    public async Task RunComprehensiveSuiteAsync()
    {
        AnsiConsole.Write(
            new Panel("[cyan1]Comprehensive Benchmark Suite[/]")
                .Border(BoxBorder.Rounded)
                .BorderColor(Color.Red));

        AnsiConsole.MarkupLine("[yellow]Warning: This will run all benchmarks and may take several hours.[/]");
        
        if (!AnsiConsole.Confirm("Continue with comprehensive benchmarks?"))
        {
            return;
        }

        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var mainTask = ctx.AddTask("[red]Comprehensive Benchmarks[/]", maxValue: 500);

                // Run all benchmark suites
                mainTask.Description = "[red]SIMD Benchmarks[/]";
                await RunSimdBenchmarksAsync();
                mainTask.Increment(100);

                mainTask.Description = "[red]Tensor Core Benchmarks[/]";
                await RunTensorCoreBenchmarksAsync();
                mainTask.Increment(100);

                mainTask.Description = "[red]Hybrid Processing Benchmarks[/]";
                await RunHybridBenchmarksAsync();
                mainTask.Increment(100);

                mainTask.Description = "[red]Memory Benchmarks[/]";
                await RunMemoryBenchmarksAsync();
                mainTask.Increment(80);

                mainTask.Description = "[red]Specialized Hardware Benchmarks[/]";
                await RunSpecializedHardwareBenchmarksAsync();
                mainTask.Increment(120);

                mainTask.Description = "[red]All benchmarks completed![/]";
                await Task.Delay(1000);
            });

        AnsiConsole.MarkupLine("[green]Comprehensive benchmark suite completed! Check BenchmarkDotNet results for detailed analysis.[/]");
    }
}
