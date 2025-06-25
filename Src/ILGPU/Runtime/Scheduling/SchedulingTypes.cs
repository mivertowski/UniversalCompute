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
    public class WorkloadAnalysis
    {
        public TimeSpan EstimatedDuration { get; set; }
        public long MemoryRequirement { get; set; }
        public WorkloadType Type { get; set; }
        public int Priority { get; set; }
        public Dictionary<string, object> Properties { get; set; } = [];

        public WorkloadAnalysis()
        {
        }

        public WorkloadAnalysis(TimeSpan estimatedDuration, long memoryRequirement, WorkloadType type, int priority, double complexity, string description)
        {
            EstimatedDuration = estimatedDuration;
            MemoryRequirement = memoryRequirement;
            Type = type;
            Priority = priority;
            Properties["Complexity"] = complexity;
            Properties["Description"] = description;
        }
    }

    public enum WorkloadType
    {
        Compute,
        Memory,
        IO,
        Mixed
    }

    public class ExecutionSchedule
    {
        public List<ScheduledNode> Nodes { get; set; } = [];
        public TimeSpan TotalDuration { get; set; }
        public double Efficiency { get; set; }
        public List<ParallelExecutionGroup> ParallelGroups { get; set; } = [];
        public List<ScheduledExecutionLevel> Levels { get; set; } = [];

        public ExecutionSchedule()
        {
        }

        public ExecutionSchedule(List<ScheduledNode> nodes)
        {
            Nodes = nodes ?? throw new ArgumentNullException(nameof(nodes));
        }
    }

    public class ScheduledNode
    {
        public string Id { get; set; } = string.Empty;
        public TimeSpan StartTime { get; set; }
        public TimeSpan Duration { get; set; }
        public Accelerator TargetAccelerator { get; set; } = null!;
        public List<string> Dependencies { get; set; } = [];
        public double EstimatedTimeMs { get; set; }

        public ScheduledNode()
        {
        }

        public ScheduledNode(string id, Accelerator accelerator, double estimatedTimeMs)
        {
            Id = id ?? throw new ArgumentNullException(nameof(id));
            TargetAccelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            EstimatedTimeMs = estimatedTimeMs;
        }
    }

    public class ParallelExecutionGroup
    {
        public List<ScheduledNode> Nodes { get; set; } = [];
        public TimeSpan StartTime { get; set; }
        public TimeSpan EndTime { get; set; }
    }

    public class ScheduledExecutionLevel
    {
        public int Level { get; set; }
        public List<ScheduledNode> Nodes { get; set; } = [];
        public TimeSpan StartTime { get; set; }
        public TimeSpan EndTime { get; set; }
        public double ParallelismFactor { get; set; }

        public ScheduledExecutionLevel()
        {
        }

        public ScheduledExecutionLevel(int level, List<ScheduledNode> nodes)
        {
            Level = level;
            Nodes = nodes ?? throw new ArgumentNullException(nameof(nodes));
        }
    }

    public class MemoryTransferPlan
    {
        public List<MemoryTransfer> Transfers { get; set; } = [];
        public long TotalBytes { get; set; }
        public TimeSpan EstimatedTime { get; set; }
        public double Bandwidth { get; set; }

        public MemoryTransferPlan()
        {
        }

        public MemoryTransferPlan(List<MemoryTransfer> transfers)
        {
            Transfers = transfers ?? throw new ArgumentNullException(nameof(transfers));
        }
    }

    public class MemoryTransfer
    {
        public Accelerator Source { get; set; } = null!;
        public Accelerator Destination { get; set; } = null!;
        public long Size { get; set; }
        public TimeSpan EstimatedTime { get; set; }
        public int Priority { get; set; }

        public MemoryTransfer()
        {
        }

        public MemoryTransfer(Accelerator source, Accelerator destination, long size, int priority)
        {
            Source = source ?? throw new ArgumentNullException(nameof(source));
            Destination = destination ?? throw new ArgumentNullException(nameof(destination));
            Size = size;
            Priority = priority;
        }
    }

    public class ExecutionPlan
    {
        public ExecutionSchedule Schedule { get; set; } = new();
        public MemoryTransferPlan MemoryPlan { get; set; } = new();
        public double OverallEfficiency { get; set; }
        public TimeSpan TotalTime { get; set; }
        public ComputeGraph Graph { get; set; } = new();
        public Dictionary<ComputeNode, ComputeDevice> Assignments { get; set; } = [];

        public ExecutionPlan()
        {
        }

        public ExecutionPlan(ComputeGraph graph, Dictionary<ComputeNode, ComputeDevice> assignments, ExecutionSchedule schedule, MemoryTransferPlan memoryPlan, WorkloadAnalysis analysis)
        {
            Graph = graph;
            Assignments = assignments;
            Schedule = schedule;
            MemoryPlan = memoryPlan;
            TotalTime = analysis.EstimatedDuration;
            OverallEfficiency = 0.85; // Placeholder
        }
    }

    public class LoadBalancer : IDisposable
    {
        private readonly List<Accelerator> accelerators;
        private readonly Dictionary<Accelerator, double> currentLoads;
        private bool disposed;

        public LoadBalancer()
        {
            this.accelerators = [];
            this.currentLoads = [];
        }

        public LoadBalancer(IEnumerable<Accelerator> accelerators)
        {
            this.accelerators = [.. accelerators];
            this.currentLoads = [];
            foreach (var acc in this.accelerators)
            {
                this.currentLoads[acc] = 0.0;
            }
        }

        public Accelerator SelectOptimalAccelerator(WorkloadAnalysis workload)
        {
            var bestAccelerator = accelerators[0];
            var lowestLoad = double.MaxValue;

            foreach (var accelerator in accelerators)
            {
                var currentLoad = currentLoads[accelerator];
                if (currentLoad < lowestLoad)
                {
                    lowestLoad = currentLoad;
                    bestAccelerator = accelerator;
                }
            }

            return bestAccelerator;
        }

        public void UpdateLoad(Accelerator accelerator, double loadDelta)
        {
            if (currentLoads.ContainsKey(accelerator))
            {
                currentLoads[accelerator] = Math.Max(0, currentLoads[accelerator] + loadDelta);
            }
        }

        public void RebalanceLoads()
        {
            var totalLoad = 0.0;
            foreach (var load in currentLoads.Values)
            {
                totalLoad += load;
            }

            var averageLoad = totalLoad / accelerators.Count;
            
            foreach (var accelerator in accelerators)
            {
                currentLoads[accelerator] = averageLoad;
            }
        }

        public void Dispose()
        {
            if (!disposed)
            {
                accelerators.Clear();
                currentLoads.Clear();
                disposed = true;
            }
        }
    }

    public class PerformanceProfiler : IDisposable
    {
        private readonly Dictionary<string, List<TimeSpan>> executionTimes;
        private readonly Dictionary<string, DateTime> activeExecutions;
        private bool disposed;

        public PerformanceProfiler()
        {
            executionTimes = [];
            activeExecutions = [];
        }

        public void RecordExecution(string operationName, TimeSpan duration)
        {
            if (!executionTimes.ContainsKey(operationName))
            {
                executionTimes[operationName] = [];
            }
            executionTimes[operationName].Add(duration);
        }

        public void StartExecution(string operationName) => activeExecutions[operationName] = DateTime.UtcNow;

        public void EndExecution(string operationName)
        {
            if (activeExecutions.TryGetValue(operationName, out var startTime))
            {
                var duration = DateTime.UtcNow - startTime;
                RecordExecution(operationName, duration);
                activeExecutions.Remove(operationName);
            }
        }

        public DevicePerformanceMetrics ProfileDevice(Accelerator accelerator) => new DevicePerformanceMetrics
        {
            AverageLatencyMs = GetAverageTime("execution").TotalMilliseconds,
            ThroughputOpsPerSecond = 1000.0 / Math.Max(1, GetAverageTime("execution").TotalMilliseconds),
            MemoryBandwidthGBps = 100.0, // Placeholder
            CacheHitRate = 0.85, // Placeholder
            MemoryCapacityBytes = accelerator.MemorySize
        };

        public DevicePerformance ProfileDevice(ComputeDevice device, Accelerator accelerator)
        {
            var metrics = ProfileDevice(accelerator);
            return new DevicePerformance
            {
                PeakGFLOPS = metrics.TensorPerformanceGFLOPS + metrics.SIMDPerformanceGFLOPS,
                MemoryBandwidthGBps = metrics.MemoryBandwidthGBps,
                SupportsTensorCores = metrics.SupportsTensorCores,
                TensorPerformanceGFLOPS = metrics.TensorPerformanceGFLOPS,
                HasAIAcceleration = metrics.SupportsTensorCores,
                AIPerformanceGOPS = metrics.TensorPerformanceGFLOPS * 1000,
                SupportsMatrixExtensions = metrics.SupportsMatrixExtensions,
                MatrixPerformanceGFLOPS = metrics.MatrixPerformanceGFLOPS,
                SIMDWidthBits = metrics.SIMDWidthBits,
                SIMDPerformanceGFLOPS = metrics.SIMDPerformanceGFLOPS,
                AverageLatencyMs = metrics.AverageLatencyMs,
                PerformancePerWatt = metrics.PerformancePerWatt
            };
        }

        public TimeSpan GetAverageTime(string operationName)
        {
            if (!executionTimes.ContainsKey(operationName) || executionTimes[operationName].Count == 0)
            {
                return TimeSpan.Zero;
            }

            var total = TimeSpan.Zero;
            foreach (var time in executionTimes[operationName])
            {
                total = total.Add(time);
            }

            return new TimeSpan(total.Ticks / executionTimes[operationName].Count);
        }

        public double GetEfficiencyScore(string operationName)
        {
            var avgTime = GetAverageTime(operationName);
            return avgTime.TotalMilliseconds > 0 ? 1000.0 / avgTime.TotalMilliseconds : 0.0;
        }

        public void Dispose()
        {
            if (!disposed)
            {
                executionTimes.Clear();
                activeExecutions.Clear();
                disposed = true;
            }
        }
    }

    public class ExecutionSlot
    {
        public TimeSpan StartTime { get; set; }
        public TimeSpan EndTime { get; set; }
        public Accelerator Accelerator { get; set; } = null!;
        public string OperationId { get; set; } = string.Empty;
        public bool IsAvailable { get; set; } = true;
    }

    public class DevicePerformanceMetrics
    {
        public double PerformancePerWatt { get; set; }
        public double AverageLatencyMs { get; set; }
        public bool SupportsTensorCores { get; set; }
        public double TensorPerformanceGFLOPS { get; set; }
        public bool SupportsMatrixExtensions { get; set; }
        public double MatrixPerformanceGFLOPS { get; set; }
        public int SIMDWidthBits { get; set; }
        public double SIMDPerformanceGFLOPS { get; set; }
        public double MemoryBandwidthGBps { get; set; }
        public double ThroughputOpsPerSecond { get; set; }
        public double PowerConsumptionWatts { get; set; }
        public double CacheHitRate { get; set; }
        public long MemoryCapacityBytes { get; set; }
    }

}