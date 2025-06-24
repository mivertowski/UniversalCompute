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
using System.Threading.Tasks;

namespace ILGPU.Runtime.Scheduling
{
    public class WorkloadAnalysis
    {
        public TimeSpan EstimatedDuration { get; set; }
        public long MemoryRequirement { get; set; }
        public WorkloadType Type { get; set; }
        public int Priority { get; set; }
        public Dictionary<string, object> Properties { get; set; } = new();
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
        public List<ScheduledNode> Nodes { get; set; } = new();
        public TimeSpan TotalDuration { get; set; }
        public double Efficiency { get; set; }
        public List<ParallelExecutionGroup> ParallelGroups { get; set; } = new();
    }

    public class ScheduledNode
    {
        public string Id { get; set; } = string.Empty;
        public TimeSpan StartTime { get; set; }
        public TimeSpan Duration { get; set; }
        public Accelerator TargetAccelerator { get; set; } = null!;
        public List<string> Dependencies { get; set; } = new();
    }

    public class ParallelExecutionGroup
    {
        public List<ScheduledNode> Nodes { get; set; } = new();
        public TimeSpan StartTime { get; set; }
        public TimeSpan EndTime { get; set; }
    }

    public class MemoryTransferPlan
    {
        public List<MemoryTransfer> Transfers { get; set; } = new();
        public long TotalBytes { get; set; }
        public TimeSpan EstimatedTime { get; set; }
        public double Bandwidth { get; set; }
    }

    public class MemoryTransfer
    {
        public Accelerator Source { get; set; } = null!;
        public Accelerator Destination { get; set; } = null!;
        public long Size { get; set; }
        public TimeSpan EstimatedTime { get; set; }
        public int Priority { get; set; }
    }

    public class ExecutionPlan
    {
        public ExecutionSchedule Schedule { get; set; } = new();
        public MemoryTransferPlan MemoryPlan { get; set; } = new();
        public double OverallEfficiency { get; set; }
        public TimeSpan TotalTime { get; set; }
    }

    public class LoadBalancer
    {
        private readonly List<Accelerator> accelerators;
        private readonly Dictionary<Accelerator, double> currentLoads;

        public LoadBalancer(IEnumerable<Accelerator> accelerators)
        {
            this.accelerators = new List<Accelerator>(accelerators);
            this.currentLoads = new Dictionary<Accelerator, double>();
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
    }

    public class PerformanceProfiler
    {
        private readonly Dictionary<string, List<TimeSpan>> executionTimes;

        public PerformanceProfiler()
        {
            executionTimes = new Dictionary<string, List<TimeSpan>>();
        }

        public void RecordExecution(string operationName, TimeSpan duration)
        {
            if (!executionTimes.ContainsKey(operationName))
            {
                executionTimes[operationName] = new List<TimeSpan>();
            }
            executionTimes[operationName].Add(duration);
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
    }

    public class ExecutionSlot
    {
        public TimeSpan StartTime { get; set; }
        public TimeSpan EndTime { get; set; }
        public Accelerator Accelerator { get; set; } = null!;
        public string OperationId { get; set; } = string.Empty;
        public bool IsAvailable { get; set; } = true;
    }
}