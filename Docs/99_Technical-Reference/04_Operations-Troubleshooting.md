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

# Operations and Troubleshooting Guide

## Operational Best Practices

### 1. Production Monitoring

```csharp
// Comprehensive monitoring system for ILGPU applications
public class ILGPUMonitoringService : IHostedService
{
    private readonly ILogger<ILGPUMonitoringService> logger;
    private readonly ILGPUMetrics metrics;
    private readonly ISystemMonitor systemMonitor;
    private readonly Timer monitoringTimer;
    private readonly ConcurrentDictionary<string, AcceleratorHealthInfo> acceleratorHealth;
    
    public ILGPUMonitoringService(
        ILogger<ILGPUMonitoringService> logger,
        ILGPUMetrics metrics,
        ISystemMonitor systemMonitor)
    {
        this.logger = logger;
        this.metrics = metrics;
        this.systemMonitor = systemMonitor;
        this.acceleratorHealth = new ConcurrentDictionary<string, AcceleratorHealthInfo>();
        this.monitoringTimer = new Timer(MonitoringCallback, null, Timeout.Infinite, Timeout.Infinite);
    }
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        logger.LogInformation("Starting ILGPU monitoring service");
        monitoringTimer.Change(TimeSpan.Zero, TimeSpan.FromSeconds(30));
        return Task.CompletedTask;
    }
    
    public Task StopAsync(CancellationToken cancellationToken)
    {
        logger.LogInformation("Stopping ILGPU monitoring service");
        monitoringTimer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }
    
    private async void MonitoringCallback(object state)
    {
        try
        {
            await MonitorSystemHealth();
            await MonitorAcceleratorHealth();
            await MonitorMemoryUsage();
            await MonitorPerformanceMetrics();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error during monitoring cycle");
        }
    }
    
    private async Task MonitorSystemHealth()
    {
        var systemHealth = await systemMonitor.GetSystemHealthAsync();
        
        metrics.RecordSystemMetric("cpu_usage_percent", systemHealth.CpuUsage);
        metrics.RecordSystemMetric("memory_usage_percent", systemHealth.MemoryUsage);
        metrics.RecordSystemMetric("disk_usage_percent", systemHealth.DiskUsage);
        
        if (systemHealth.CpuUsage > 90)
        {
            logger.LogWarning("High CPU usage detected: {CpuUsage}%", systemHealth.CpuUsage);
        }
        
        if (systemHealth.MemoryUsage > 85)
        {
            logger.LogWarning("High memory usage detected: {MemoryUsage}%", systemHealth.MemoryUsage);
        }
    }
    
    private async Task MonitorAcceleratorHealth()
    {
        using var context = Context.CreateDefault();
        
        foreach (var device in context)
        {
            try
            {
                var health = await CheckAcceleratorHealth(device);
                var key = $"{device.AcceleratorType}_{device.DeviceId}";
                
                acceleratorHealth.AddOrUpdate(key, health, (k, v) => health);
                
                metrics.RecordAcceleratorHealth(
                    device.AcceleratorType.ToString(),
                    health.IsHealthy ? 1 : 0);
                
                if (!health.IsHealthy)
                {
                    logger.LogWarning("Accelerator {AcceleratorType} {DeviceId} is unhealthy: {Reason}",
                        device.AcceleratorType, device.DeviceId, health.Reason);
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to check health for {AcceleratorType} {DeviceId}",
                    device.AcceleratorType, device.DeviceId);
            }
        }
    }
    
    private async Task<AcceleratorHealthInfo> CheckAcceleratorHealth(Device device)
    {
        try
        {
            using var accelerator = device.CreateAccelerator(Context.CreateDefault());
            
            // Test basic functionality
            const int testSize = 100;
            using var testBuffer = accelerator.Allocate1D<float>(testSize);
            
            var testData = Enumerable.Range(0, testSize).Select(i => (float)i).ToArray();
            testBuffer.CopyFromCPU(testData);
            
            var result = testBuffer.GetAsArray1D();
            var isValid = result.SequenceEqual(testData);
            
            return new AcceleratorHealthInfo
            {
                IsHealthy = isValid,
                LastCheck = DateTime.UtcNow,
                Reason = isValid ? null : "Data integrity test failed",
                MemorySize = accelerator.MemorySize,
                MaxThreadsPerGroup = accelerator.MaxNumThreadsPerGroup
            };
        }
        catch (Exception ex)
        {
            return new AcceleratorHealthInfo
            {
                IsHealthy = false,
                LastCheck = DateTime.UtcNow,
                Reason = ex.Message
            };
        }
    }
    
    private async Task MonitorMemoryUsage()
    {
        var gcMemory = GC.GetTotalMemory(false);
        var workingSet = Environment.WorkingSet;
        
        metrics.RecordMemoryMetric("gc_memory_bytes", gcMemory);
        metrics.RecordMemoryMetric("working_set_bytes", workingSet);
        
        // Check for memory leaks
        if (gcMemory > 2L * 1024 * 1024 * 1024) // 2GB threshold
        {
            logger.LogWarning("High GC memory usage: {MemoryMB} MB", gcMemory / (1024 * 1024));
            
            // Force garbage collection and re-check
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            
            var afterGC = GC.GetTotalMemory(false);
            if (afterGC > gcMemory * 0.8) // Less than 20% reduction
            {
                logger.LogError("Potential memory leak detected. Memory after GC: {MemoryMB} MB",
                    afterGC / (1024 * 1024));
            }
        }
        
        await Task.CompletedTask;
    }
    
    private async Task MonitorPerformanceMetrics()
    {
        // Collect and analyze performance trends
        var recentMetrics = await GetRecentPerformanceMetrics();
        
        foreach (var metric in recentMetrics)
        {
            if (metric.HasPerformanceRegression())
            {
                logger.LogWarning("Performance regression detected for {Operation}: {CurrentValue} vs baseline {BaselineValue}",
                    metric.Operation, metric.CurrentValue, metric.BaselineValue);
            }
        }
    }
    
    public void Dispose()
    {
        monitoringTimer?.Dispose();
    }
}

public class AcceleratorHealthInfo
{
    public bool IsHealthy { get; set; }
    public DateTime LastCheck { get; set; }
    public string Reason { get; set; }
    public long MemorySize { get; set; }
    public int MaxThreadsPerGroup { get; set; }
}
```

### 2. Automated Recovery and Self-Healing

```csharp
// Self-healing mechanisms for ILGPU applications
public class ILGPUSelfHealingService : IHostedService
{
    private readonly ILogger<ILGPUSelfHealingService> logger;
    private readonly ILGPUContextManager contextManager;
    private readonly IMemoryManager memoryManager;
    private readonly Timer healingTimer;
    private readonly ConcurrentDictionary<string, FailureInfo> failures;
    
    public ILGPUSelfHealingService(
        ILogger<ILGPUSelfHealingService> logger,
        ILGPUContextManager contextManager,
        IMemoryManager memoryManager)
    {
        this.logger = logger;
        this.contextManager = contextManager;
        this.memoryManager = memoryManager;
        this.failures = new ConcurrentDictionary<string, FailureInfo>();
        this.healingTimer = new Timer(HealingCallback, null, Timeout.Infinite, Timeout.Infinite);
    }
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        logger.LogInformation("Starting ILGPU self-healing service");
        healingTimer.Change(TimeSpan.Zero, TimeSpan.FromMinutes(5));
        return Task.CompletedTask;
    }
    
    public Task StopAsync(CancellationToken cancellationToken)
    {
        logger.LogInformation("Stopping ILGPU self-healing service");
        healingTimer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }
    
    private async void HealingCallback(object state)
    {
        try
        {
            await AttemptMemoryRecovery();
            await AttemptAcceleratorRecovery();
            await CleanupStaleResources();
            await OptimizePerformance();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error during self-healing cycle");
        }
    }
    
    private async Task AttemptMemoryRecovery()
    {
        var memoryPressure = GC.GetTotalMemory(false) / (double)Environment.WorkingSet;
        
        if (memoryPressure > 0.8) // High memory pressure
        {
            logger.LogInformation("High memory pressure detected, attempting recovery");
            
            // Force garbage collection
            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true);
            
            // Clear memory caches
            await memoryManager.ClearCachesAsync();
            
            // Defragment memory pools
            await memoryManager.DefragmentAsync();
            
            var afterRecovery = GC.GetTotalMemory(false);
            logger.LogInformation("Memory recovery completed. Memory usage: {MemoryMB} MB",
                afterRecovery / (1024 * 1024));
        }
    }
    
    private async Task AttemptAcceleratorRecovery()
    {
        var failedAccelerators = failures.Values
            .Where(f => f.FailureType == FailureType.AcceleratorFailure)
            .Where(f => DateTime.UtcNow - f.LastFailure > TimeSpan.FromMinutes(10))
            .ToList();
        
        foreach (var failure in failedAccelerators)
        {
            try
            {
                logger.LogInformation("Attempting to recover accelerator: {AcceleratorId}", failure.AcceleratorId);
                
                var success = await TestAcceleratorRecovery(failure.AcceleratorId);
                
                if (success)
                {
                    failures.TryRemove(failure.AcceleratorId, out _);
                    logger.LogInformation("Successfully recovered accelerator: {AcceleratorId}", failure.AcceleratorId);
                }
                else
                {
                    failure.RecoveryAttempts++;
                    logger.LogWarning("Failed to recover accelerator: {AcceleratorId}, attempt {Attempt}",
                        failure.AcceleratorId, failure.RecoveryAttempts);
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error during accelerator recovery: {AcceleratorId}", failure.AcceleratorId);
            }
        }
    }
    
    private async Task<bool> TestAcceleratorRecovery(string acceleratorId)
    {
        try
        {
            var accelerator = contextManager.GetAcceleratorById(acceleratorId);
            
            // Test basic operation
            const int testSize = 10;
            using var testBuffer = accelerator.Allocate1D<float>(testSize);
            
            var testData = new float[testSize];
            for (int i = 0; i < testSize; i++)
                testData[i] = i;
            
            testBuffer.CopyFromCPU(testData);
            accelerator.Synchronize();
            
            var result = testBuffer.GetAsArray1D();
            return result.SequenceEqual(testData);
        }
        catch
        {
            return false;
        }
    }
    
    private async Task CleanupStaleResources()
    {
        // Cleanup expired kernel caches
        await contextManager.CleanupExpiredCaches();
        
        // Cleanup orphaned memory allocations
        await memoryManager.CleanupOrphanedAllocations();
        
        // Clear old performance metrics
        ClearOldMetrics();
    }
    
    private async Task OptimizePerformance()
    {
        // Adaptive performance optimization
        var performanceData = await GetPerformanceData();
        
        if (performanceData.NeedsOptimization)
        {
            logger.LogInformation("Applying performance optimizations");
            
            // Adjust memory pool sizes
            await memoryManager.OptimizePoolSizes(performanceData.MemoryUsagePattern);
            
            // Update kernel configurations
            await contextManager.OptimizeKernelConfigurations(performanceData.KernelPerformance);
            
            // Adjust garbage collection settings
            OptimizeGarbageCollection(performanceData.GCPressure);
        }
    }
    
    public void RecordFailure(string acceleratorId, FailureType failureType, Exception exception)
    {
        failures.AddOrUpdate(acceleratorId, 
            new FailureInfo
            {
                AcceleratorId = acceleratorId,
                FailureType = failureType,
                LastFailure = DateTime.UtcNow,
                Exception = exception,
                FailureCount = 1
            },
            (key, existing) =>
            {
                existing.LastFailure = DateTime.UtcNow;
                existing.FailureCount++;
                existing.Exception = exception;
                return existing;
            });
    }
    
    public void Dispose()
    {
        healingTimer?.Dispose();
    }
}

public class FailureInfo
{
    public string AcceleratorId { get; set; }
    public FailureType FailureType { get; set; }
    public DateTime LastFailure { get; set; }
    public Exception Exception { get; set; }
    public int FailureCount { get; set; }
    public int RecoveryAttempts { get; set; }
}

public enum FailureType
{
    AcceleratorFailure,
    MemoryFailure,
    KernelCompilationFailure,
    ExecutionFailure
}
```

## Troubleshooting Guide

### 1. Common Issues and Solutions

```csharp
// Comprehensive troubleshooting system
public class ILGPUTroubleshooter
{
    private readonly ILogger<ILGPUTroubleshooter> logger;
    private readonly Dictionary<string, TroubleshootingRule> rules;
    
    public ILGPUTroubleshooter(ILogger<ILGPUTroubleshooter> logger)
    {
        this.logger = logger;
        this.rules = LoadTroubleshootingRules();
    }
    
    public async Task<DiagnosticReport> DiagnoseAsync(Exception exception, Context context = null)
    {
        var report = new DiagnosticReport
        {
            Exception = exception,
            Timestamp = DateTime.UtcNow,
            SystemInfo = await GatherSystemInfo(),
            ILGPUInfo = await GatherILGPUInfo(context)
        };
        
        // Apply troubleshooting rules
        foreach (var rule in rules.Values)
        {
            if (rule.Matches(exception))
            {
                var recommendation = await rule.GenerateRecommendationAsync(report);
                report.Recommendations.Add(recommendation);
                
                logger.LogInformation("Applied troubleshooting rule: {RuleName}", rule.Name);
            }
        }
        
        // Generate automatic fixes if possible
        var autoFixes = await GenerateAutoFixes(report);
        report.AutoFixes.AddRange(autoFixes);
        
        return report;
    }
    
    private Dictionary<string, TroubleshootingRule> LoadTroubleshootingRules()
    {
        return new Dictionary<string, TroubleshootingRule>
        {
            ["OutOfMemoryRule"] = new OutOfMemoryRule(),
            ["CompilationFailureRule"] = new CompilationFailureRule(),
            ["AcceleratorNotFoundRule"] = new AcceleratorNotFoundRule(),
            ["KernelExecutionFailureRule"] = new KernelExecutionFailureRule(),
            ["InvalidOperationRule"] = new InvalidOperationRule(),
            ["PlatformNotSupportedRule"] = new PlatformNotSupportedRule(),
            ["PerformanceRegressionRule"] = new PerformanceRegressionRule()
        };
    }
    
    private async Task<SystemInfo> GatherSystemInfo()
    {
        return new SystemInfo
        {
            OSVersion = Environment.OSVersion.ToString(),
            ProcessorCount = Environment.ProcessorCount,
            WorkingSet = Environment.WorkingSet,
            DotNetVersion = Environment.Version.ToString(),
            Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            AvailableMemory = GC.GetTotalMemory(false),
            CPUUsage = await GetCPUUsage(),
            GPUInfo = await GetGPUInfo()
        };
    }
    
    private async Task<ILGPUInfo> GatherILGPUInfo(Context context)
    {
        var info = new ILGPUInfo();
        
        if (context != null)
        {
            info.AvailableDevices = context.Select(d => new DeviceInfo
            {
                AcceleratorType = d.AcceleratorType.ToString(),
                Name = d.Name,
                MemorySize = d.MemorySize,
                MaxThreadsPerGroup = d.MaxNumThreadsPerGroup,
                DeviceId = d.DeviceId
            }).ToList();
        }
        
        return info;
    }
}

// Specific troubleshooting rules
public class OutOfMemoryRule : TroubleshootingRule
{
    public override string Name => "OutOfMemoryRule";
    
    public override bool Matches(Exception exception)
    {
        return exception is OutOfMemoryException ||
               exception.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase);
    }
    
    public override async Task<Recommendation> GenerateRecommendationAsync(DiagnosticReport report)
    {
        var recommendation = new Recommendation
        {
            Title = "Out of Memory Error",
            Severity = Severity.High,
            Category = "Memory Management"
        };
        
        var availableMemory = report.SystemInfo.AvailableMemory;
        var workingSet = report.SystemInfo.WorkingSet;
        var memoryPressure = (double)availableMemory / workingSet;
        
        if (memoryPressure > 0.8)
        {
            recommendation.Description = "System is under high memory pressure. Consider reducing memory usage or increasing available memory.";
            recommendation.Actions.Add("Reduce batch sizes in ILGPU operations");
            recommendation.Actions.Add("Enable memory pooling to reduce allocation overhead");
            recommendation.Actions.Add("Use streaming for large datasets");
            recommendation.Actions.Add("Consider using smaller data types where possible");
        }
        else
        {
            recommendation.Description = "Memory fragmentation or memory leak detected.";
            recommendation.Actions.Add("Force garbage collection: GC.Collect()");
            recommendation.Actions.Add("Check for memory leaks in buffer management");
            recommendation.Actions.Add("Ensure proper disposal of ILGPU resources");
            recommendation.Actions.Add("Use memory profiling tools to identify leaks");
        }
        
        return recommendation;
    }
}

public class CompilationFailureRule : TroubleshootingRule
{
    public override string Name => "CompilationFailureRule";
    
    public override bool Matches(Exception exception)
    {
        return exception is CompilationException ||
               exception.Message.Contains("compilation", StringComparison.OrdinalIgnoreCase);
    }
    
    public override async Task<Recommendation> GenerateRecommendationAsync(DiagnosticReport report)
    {
        return new Recommendation
        {
            Title = "Kernel Compilation Failure",
            Severity = Severity.High,
            Category = "Compilation",
            Description = "ILGPU kernel compilation failed. This usually indicates unsupported operations or syntax in the kernel code.",
            Actions = new List<string>
            {
                "Verify kernel method signature (must have Index parameter as first argument)",
                "Remove unsupported operations (file I/O, network operations, etc.)",
                "Check for unsupported .NET types (reference types, nullable types)",
                "Test kernel on CPU accelerator for debugging",
                "Enable debug mode for detailed compilation errors",
                "Simplify complex control flow or method calls",
                "Ensure all used types are unmanaged value types"
            }
        };
    }
}

public class AcceleratorNotFoundRule : TroubleshootingRule
{
    public override string Name => "AcceleratorNotFoundRule";
    
    public override bool Matches(Exception exception)
    {
        return exception.Message.Contains("accelerator", StringComparison.OrdinalIgnoreCase) &&
               exception.Message.Contains("not found", StringComparison.OrdinalIgnoreCase);
    }
    
    public override async Task<Recommendation> GenerateRecommendationAsync(DiagnosticReport report)
    {
        var recommendation = new Recommendation
        {
            Title = "Accelerator Not Found",
            Severity = Severity.Medium,
            Category = "Hardware",
            Description = "No suitable accelerator found for the requested operation."
        };
        
        if (!report.ILGPUInfo.AvailableDevices.Any(d => d.AcceleratorType == "Cuda"))
        {
            recommendation.Actions.Add("Install NVIDIA GPU drivers and CUDA toolkit");
            recommendation.Actions.Add("Verify GPU is CUDA-compatible");
        }
        
        if (!report.ILGPUInfo.AvailableDevices.Any(d => d.AcceleratorType == "OpenCL"))
        {
            recommendation.Actions.Add("Install OpenCL drivers for your GPU");
            recommendation.Actions.Add("Verify OpenCL runtime is available");
        }
        
        recommendation.Actions.Add("Use CPU accelerator as fallback");
        recommendation.Actions.Add("Check device enumeration with Context.Create().AllAccelerators()");
        
        return recommendation;
    }
}
```

### 2. Performance Troubleshooting

```csharp
// Performance analysis and optimization suggestions
public class PerformanceTroubleshooter
{
    private readonly ILogger<PerformanceTroubleshooter> logger;
    private readonly IMetricsCollector metricsCollector;
    
    public PerformanceTroubleshooter(ILogger<PerformanceTroubleshooter> logger, IMetricsCollector metricsCollector)
    {
        this.logger = logger;
        this.metricsCollector = metricsCollector;
    }
    
    public async Task<PerformanceAnalysis> AnalyzePerformanceAsync(
        string operationName,
        TimeSpan actualDuration,
        long dataSize,
        AcceleratorType acceleratorType)
    {
        var analysis = new PerformanceAnalysis
        {
            OperationName = operationName,
            ActualDuration = actualDuration,
            DataSize = dataSize,
            AcceleratorType = acceleratorType,
            Timestamp = DateTime.UtcNow
        };
        
        // Calculate expected performance
        var expectedDuration = CalculateExpectedDuration(operationName, dataSize, acceleratorType);
        analysis.ExpectedDuration = expectedDuration;
        analysis.PerformanceRatio = actualDuration.TotalMilliseconds / expectedDuration.TotalMilliseconds;
        
        // Analyze performance characteristics
        analysis.Issues = await IdentifyPerformanceIssues(analysis);
        analysis.Recommendations = GeneratePerformanceRecommendations(analysis);
        
        // Record metrics for trend analysis
        await metricsCollector.RecordPerformanceMetric(analysis);
        
        return analysis;
    }
    
    private async Task<List<PerformanceIssue>> IdentifyPerformanceIssues(PerformanceAnalysis analysis)
    {
        var issues = new List<PerformanceIssue>();
        
        // Check for significant performance regression
        if (analysis.PerformanceRatio > 2.0)
        {
            issues.Add(new PerformanceIssue
            {
                Type = PerformanceIssueType.Regression,
                Severity = Severity.High,
                Description = $"Performance is {analysis.PerformanceRatio:F1}x slower than expected",
                PotentialCauses = new List<string>
                {
                    "Memory bandwidth bottleneck",
                    "Poor memory access patterns",
                    "Inadequate parallelization",
                    "Thermal throttling",
                    "Resource contention"
                }
            });
        }
        
        // Check memory bandwidth utilization
        var bandwidthUtilization = await CalculateBandwidthUtilization(analysis);
        if (bandwidthUtilization < 0.3)
        {
            issues.Add(new PerformanceIssue
            {
                Type = PerformanceIssueType.MemoryBandwidth,
                Severity = Severity.Medium,
                Description = $"Low memory bandwidth utilization: {bandwidthUtilization:P1}",
                PotentialCauses = new List<string>
                {
                    "Non-coalesced memory access",
                    "Cache misses",
                    "Memory bank conflicts",
                    "Insufficient data reuse"
                }
            });
        }
        
        // Check occupancy
        var occupancy = await EstimateOccupancy(analysis);
        if (occupancy < 0.5)
        {
            issues.Add(new PerformanceIssue
            {
                Type = PerformanceIssueType.Occupancy,
                Severity = Severity.Medium,
                Description = $"Low occupancy: {occupancy:P1}",
                PotentialCauses = new List<string>
                {
                    "Too few threads per block",
                    "High register usage",
                    "Excessive shared memory usage",
                    "Poor work distribution"
                }
            });
        }
        
        return issues;
    }
    
    private List<PerformanceRecommendation> GeneratePerformanceRecommendations(PerformanceAnalysis analysis)
    {
        var recommendations = new List<PerformanceRecommendation>();
        
        foreach (var issue in analysis.Issues)
        {
            switch (issue.Type)
            {
                case PerformanceIssueType.Regression:
                    recommendations.Add(new PerformanceRecommendation
                    {
                        Title = "Address Performance Regression",
                        Priority = Priority.High,
                        Actions = new List<string>
                        {
                            "Profile the application to identify bottlenecks",
                            "Compare with previous implementation",
                            "Check for resource contention",
                            "Verify hardware health and thermal conditions",
                            "Review recent code changes"
                        }
                    });
                    break;
                
                case PerformanceIssueType.MemoryBandwidth:
                    recommendations.Add(new PerformanceRecommendation
                    {
                        Title = "Optimize Memory Access",
                        Priority = Priority.Medium,
                        Actions = new List<string>
                        {
                            "Ensure coalesced memory access patterns",
                            "Use shared memory for data reuse",
                            "Minimize memory transfers between host and device",
                            "Consider memory layout optimization",
                            "Use appropriate data types (avoid unnecessary precision)"
                        }
                    });
                    break;
                
                case PerformanceIssueType.Occupancy:
                    recommendations.Add(new PerformanceRecommendation
                    {
                        Title = "Improve Occupancy",
                        Priority = Priority.Medium,
                        Actions = new List<string>
                        {
                            "Optimize block size for target hardware",
                            "Reduce register usage in kernels",
                            "Minimize shared memory usage",
                            "Consider kernel fusion to increase work per thread",
                            "Balance work distribution across threads"
                        }
                    });
                    break;
            }
        }
        
        return recommendations;
    }
    
    private async Task<double> CalculateBandwidthUtilization(PerformanceAnalysis analysis)
    {
        // Simplified bandwidth calculation
        var theoreticalBandwidth = GetTheoreticalBandwidth(analysis.AcceleratorType);
        var actualBandwidth = (analysis.DataSize * 2) / analysis.ActualDuration.TotalSeconds / (1024 * 1024 * 1024);
        
        return actualBandwidth / theoreticalBandwidth;
    }
    
    private double GetTheoreticalBandwidth(AcceleratorType acceleratorType)
    {
        return acceleratorType switch
        {
            AcceleratorType.Cuda => 900.0, // GB/s for high-end GPU
            AcceleratorType.OpenCL => 500.0,
            AcceleratorType.CPU => 50.0,
            _ => 100.0
        };
    }
}

public class PerformanceAnalysis
{
    public string OperationName { get; set; }
    public TimeSpan ActualDuration { get; set; }
    public TimeSpan ExpectedDuration { get; set; }
    public long DataSize { get; set; }
    public AcceleratorType AcceleratorType { get; set; }
    public double PerformanceRatio { get; set; }
    public DateTime Timestamp { get; set; }
    public List<PerformanceIssue> Issues { get; set; } = new();
    public List<PerformanceRecommendation> Recommendations { get; set; } = new();
}
```

### 3. Diagnostic Tools

```csharp
// Comprehensive diagnostic utilities
public class ILGPUDiagnostics
{
    private readonly ILogger<ILGPUDiagnostics> logger;
    
    public ILGPUDiagnostics(ILogger<ILGPUDiagnostics> logger)
    {
        this.logger = logger;
    }
    
    public async Task<DiagnosticResults> RunFullDiagnosticsAsync()
    {
        var results = new DiagnosticResults
        {
            Timestamp = DateTime.UtcNow
        };
        
        logger.LogInformation("Starting ILGPU diagnostics");
        
        // Hardware detection
        results.HardwareDiagnostics = await DiagnoseHardware();
        
        // Driver verification
        results.DriverDiagnostics = await DiagnoseDrivers();
        
        // ILGPU functionality test
        results.FunctionalityDiagnostics = await DiagnoseFunctionality();
        
        // Performance benchmarks
        results.PerformanceDiagnostics = await DiagnosePerformance();
        
        // Memory diagnostics
        results.MemoryDiagnostics = await DiagnoseMemory();
        
        // Generate overall health score
        results.OverallHealthScore = CalculateHealthScore(results);
        
        logger.LogInformation("ILGPU diagnostics completed. Health score: {HealthScore}/100", 
            results.OverallHealthScore);
        
        return results;
    }
    
    private async Task<HardwareDiagnostics> DiagnoseHardware()
    {
        var diagnostics = new HardwareDiagnostics();
        
        try
        {
            using var context = Context.CreateDefault();
            
            diagnostics.DetectedDevices = context.Select(device => new DetectedDevice
            {
                Type = device.AcceleratorType.ToString(),
                Name = device.Name,
                DeviceId = device.DeviceId,
                MemorySize = device.MemorySize,
                MaxThreadsPerGroup = device.MaxNumThreadsPerGroup,
                IsAvailable = true
            }).ToList();
            
            diagnostics.HasCudaDevice = context.GetCudaDevices().Any();
            diagnostics.HasOpenCLDevice = context.GetCLDevices().Any();
            diagnostics.HasCPUDevice = context.GetCPUDevices().Any();
            
            diagnostics.Status = DiagnosticStatus.Success;
        }
        catch (Exception ex)
        {
            diagnostics.Status = DiagnosticStatus.Failed;
            diagnostics.ErrorMessage = ex.Message;
            logger.LogError(ex, "Hardware diagnostics failed");
        }
        
        return diagnostics;
    }
    
    private async Task<DriverDiagnostics> DiagnoseDrivers()
    {
        var diagnostics = new DriverDiagnostics();
        
        try
        {
            // Check CUDA driver
            diagnostics.CudaDriverVersion = await GetCudaDriverVersion();
            diagnostics.IsCudaDriverInstalled = !string.IsNullOrEmpty(diagnostics.CudaDriverVersion);
            
            // Check OpenCL driver
            diagnostics.OpenCLDriverVersion = await GetOpenCLDriverVersion();
            diagnostics.IsOpenCLDriverInstalled = !string.IsNullOrEmpty(diagnostics.OpenCLDriverVersion);
            
            // Check .NET compatibility
            diagnostics.DotNetVersion = Environment.Version.ToString();
            diagnostics.IsDotNetCompatible = Environment.Version.Major >= 6;
            
            diagnostics.Status = DiagnosticStatus.Success;
        }
        catch (Exception ex)
        {
            diagnostics.Status = DiagnosticStatus.Failed;
            diagnostics.ErrorMessage = ex.Message;
            logger.LogError(ex, "Driver diagnostics failed");
        }
        
        return diagnostics;
    }
    
    private async Task<FunctionalityDiagnostics> DiagnoseFunctionality()
    {
        var diagnostics = new FunctionalityDiagnostics();
        
        try
        {
            using var context = Context.CreateDefault();
            
            foreach (var device in context)
            {
                var test = await TestDeviceFunctionality(device, context);
                diagnostics.DeviceTests.Add(test);
            }
            
            diagnostics.Status = diagnostics.DeviceTests.All(t => t.Success) 
                ? DiagnosticStatus.Success 
                : DiagnosticStatus.Warning;
        }
        catch (Exception ex)
        {
            diagnostics.Status = DiagnosticStatus.Failed;
            diagnostics.ErrorMessage = ex.Message;
            logger.LogError(ex, "Functionality diagnostics failed");
        }
        
        return diagnostics;
    }
    
    private async Task<DeviceTest> TestDeviceFunctionality(Device device, Context context)
    {
        var test = new DeviceTest
        {
            DeviceType = device.AcceleratorType.ToString(),
            DeviceName = device.Name
        };
        
        try
        {
            using var accelerator = device.CreateAccelerator(context);
            
            // Test basic memory allocation
            using var buffer = accelerator.Allocate1D<float>(100);
            test.MemoryAllocationTest = true;
            
            // Test data transfer
            var testData = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            buffer.CopyFromCPU(testData);
            var result = buffer.GetAsArray1D();
            test.DataTransferTest = result.SequenceEqual(testData);
            
            // Test kernel compilation and execution
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(TestKernel);
            
            using var outputBuffer = accelerator.Allocate1D<float>(100);
            kernel(100, buffer.View, outputBuffer.View);
            accelerator.Synchronize();
            
            var kernelResult = outputBuffer.GetAsArray1D();
            test.KernelExecutionTest = kernelResult.SequenceEqual(testData.Select(x => x * 2));
            
            test.Success = test.MemoryAllocationTest && test.DataTransferTest && test.KernelExecutionTest;
        }
        catch (Exception ex)
        {
            test.Success = false;
            test.ErrorMessage = ex.Message;
            logger.LogWarning("Device test failed for {DeviceType}: {Error}", device.AcceleratorType, ex.Message);
        }
        
        return test;
    }
    
    static void TestKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index < input.Length)
        {
            output[index] = input[index] * 2.0f;
        }
    }
    
    private int CalculateHealthScore(DiagnosticResults results)
    {
        var score = 100;
        
        // Deduct points for failures
        if (results.HardwareDiagnostics.Status == DiagnosticStatus.Failed) score -= 30;
        if (results.DriverDiagnostics.Status == DiagnosticStatus.Failed) score -= 20;
        if (results.FunctionalityDiagnostics.Status == DiagnosticStatus.Failed) score -= 25;
        if (results.PerformanceDiagnostics.Status == DiagnosticStatus.Failed) score -= 15;
        if (results.MemoryDiagnostics.Status == DiagnosticStatus.Failed) score -= 10;
        
        // Deduct points for warnings
        if (results.HardwareDiagnostics.Status == DiagnosticStatus.Warning) score -= 10;
        if (results.FunctionalityDiagnostics.Status == DiagnosticStatus.Warning) score -= 5;
        
        return Math.Max(0, score);
    }
}

// Diagnostic result classes
public class DiagnosticResults
{
    public DateTime Timestamp { get; set; }
    public HardwareDiagnostics HardwareDiagnostics { get; set; }
    public DriverDiagnostics DriverDiagnostics { get; set; }
    public FunctionalityDiagnostics FunctionalityDiagnostics { get; set; }
    public PerformanceDiagnostics PerformanceDiagnostics { get; set; }
    public MemoryDiagnostics MemoryDiagnostics { get; set; }
    public int OverallHealthScore { get; set; }
}

public enum DiagnosticStatus
{
    Success,
    Warning,
    Failed
}
```

### 4. Debug Configuration

```csharp
// Debug configuration for development and troubleshooting
public static class ILGPUDebugConfiguration
{
    public static Context CreateDebugContext()
    {
        return Context.Create(builder =>
        {
            builder.CPU() // Always include CPU for debugging
                   .EnableAssertions() // Enable runtime assertions
                   .OptimizationLevel(OptimizationLevel.Debug) // Debug optimization
                   .EnableKernelCaching(false); // Disable caching for debugging
            
            // Add GPU accelerators in debug mode if available
            try
            {
                builder.Cuda();
            }
            catch
            {
                // CUDA not available, continue
            }
            
            try
            {
                builder.OpenCL();
            }
            catch
            {
                // OpenCL not available, continue
            }
        });
    }
    
    public static void ConfigureDebugLogging(IServiceCollection services)
    {
        services.AddLogging(builder =>
        {
            builder.AddConsole()
                   .AddDebug()
                   .SetMinimumLevel(LogLevel.Debug);
            
            // Add custom ILGPU logging filters
            builder.AddFilter("ILGPU", LogLevel.Trace);
            builder.AddFilter("ILGPU.Compilation", LogLevel.Debug);
            builder.AddFilter("ILGPU.Memory", LogLevel.Debug);
        });
    }
    
    public static void EnableDetailedExceptions()
    {
        // Enable detailed exception information
        AppDomain.CurrentDomain.FirstChanceException += (sender, e) =>
        {
            if (e.Exception.Source?.Contains("ILGPU") == true)
            {
                Console.WriteLine($"ILGPU First Chance Exception: {e.Exception}");
            }
        };
    }
}
```

This operations and troubleshooting guide provides comprehensive monitoring, self-healing, diagnostic tools, and troubleshooting procedures for enterprise ILGPU deployments. The complete technical reference now covers architecture, implementation patterns, integration/deployment, and operations - providing architects and senior developers with all necessary information for successful ILGPU enterprise integration.