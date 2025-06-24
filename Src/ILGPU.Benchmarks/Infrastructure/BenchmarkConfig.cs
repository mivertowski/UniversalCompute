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

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Csv;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Order;
using BenchmarkDotNet.Toolchains.InProcess.Emit;

namespace ILGPU.Benchmarks.Infrastructure;

/// <summary>
/// Configuration for different types of benchmarks.
/// </summary>
public class BenchmarkConfig
{
    /// <summary>
    /// Quick configuration for fast validation benchmarks.
    /// </summary>
    public IConfig QuickConfig { get; }

    /// <summary>
    /// Standard configuration for regular benchmarks.
    /// </summary>
    public IConfig StandardConfig { get; }

    /// <summary>
    /// Comprehensive configuration for detailed analysis.
    /// </summary>
    public IConfig ComprehensiveConfig { get; }

    /// <summary>
    /// Burn-in test configuration for maximum load testing.
    /// </summary>
    public IConfig BurnInConfig { get; }

    public BenchmarkConfig()
    {
        QuickConfig = CreateQuickConfig();
        StandardConfig = CreateStandardConfig();
        ComprehensiveConfig = CreateComprehensiveConfig();
        BurnInConfig = CreateBurnInConfig();
    }

    private static IConfig CreateQuickConfig()
    {
        return ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(Job.Dry
                .WithWarmupCount(1)
                .WithIterationCount(1)
                .WithInvocationCount(1)
                .WithUnrollFactor(1))
            .AddExporter(HtmlExporter.Default)
            .AddExporter(CsvExporter.Default)
            .AddLogger(ConsoleLogger.Default)
            .AddDiagnoser(MemoryDiagnoser.Default)
            .WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest))
            .WithSummaryStyle(BenchmarkDotNet.Reports.SummaryStyle.Default.WithRatioStyle(BenchmarkDotNet.Columns.RatioStyle.Trend));
    }

    private static IConfig CreateStandardConfig()
    {
        return ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(Job.Default
                .WithWarmupCount(3)
                .WithIterationCount(5)
                .WithInvocationCount(16)
                .WithUnrollFactor(16))
            .AddExporter(HtmlExporter.Default)
            .AddExporter(CsvExporter.Default)
            .AddExporter(MarkdownExporter.GitHub)
            .AddLogger(ConsoleLogger.Default)
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddDiagnoser(ThreadingDiagnoser.Default)
            .WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest))
            .WithSummaryStyle(BenchmarkDotNet.Reports.SummaryStyle.Default.WithRatioStyle(BenchmarkDotNet.Columns.RatioStyle.Trend));
    }

    private static IConfig CreateComprehensiveConfig()
    {
        return ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(Job.LongRun
                .WithWarmupCount(5)
                .WithIterationCount(10)
                .WithInvocationCount(16)
                .WithUnrollFactor(16))
            .AddExporter(HtmlExporter.Default)
            .AddExporter(CsvExporter.Default)
            .AddExporter(MarkdownExporter.GitHub)
            .AddExporter(JsonExporter.Default)
            .AddLogger(ConsoleLogger.Default)
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddDiagnoser(ThreadingDiagnoser.Default)
            // Hardware counters may not be available on all systems
            // .AddDiagnoser(HardwareCounters.BranchMispredictions)
            // .AddDiagnoser(HardwareCounters.CacheMisses)
            .WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest))
            .WithSummaryStyle(BenchmarkDotNet.Reports.SummaryStyle.Default.WithRatioStyle(BenchmarkDotNet.Columns.RatioStyle.Trend));
    }

    private static IConfig CreateBurnInConfig()
    {
        return ManualConfig.Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddJob(Job.Default
                .WithToolchain(InProcessEmitToolchain.Instance)
                .WithWarmupCount(1)
                .WithIterationCount(1)
                .WithInvocationCount(1000000) // High iteration count for burn-in
                .WithUnrollFactor(1))
            .AddExporter(HtmlExporter.Default)
            .AddLogger(ConsoleLogger.Default)
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddDiagnoser(ThreadingDiagnoser.Default)
            .WithOrderer(new DefaultOrderer(SummaryOrderPolicy.Method))
            .WithSummaryStyle(BenchmarkDotNet.Reports.SummaryStyle.Default.WithRatioStyle(BenchmarkDotNet.Columns.RatioStyle.Percentage));
    }
}
