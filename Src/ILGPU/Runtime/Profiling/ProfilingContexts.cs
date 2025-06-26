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
// Change License: Apache License, Version 2.0using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace ILGPU.Runtime.Profiling
{
    /// <summary>
    /// Implementation of kernel profiling context.
    /// </summary>
    internal sealed class KernelProfilingContext(PerformanceProfiler profiler, string executionId, string kernelName, Index3D gridSize, Index3D groupSize) : IKernelProfilingContext
    {
        private readonly PerformanceProfiler profiler = profiler;
        private readonly string executionId = executionId;
        private readonly Stopwatch stopwatch = Stopwatch.StartNew();
        private readonly Dictionary<string, object> metadata = [];
        private TimeSpan compilationTime;
        private int sharedMemorySize;
        private int registerCount;
        private bool disposed;

        public string KernelName { get; } = kernelName;
        public Index3D GridSize { get; } = gridSize;
        public Index3D GroupSize { get; } = groupSize;
        public DateTime StartTime { get; } = DateTime.UtcNow;

        public void RecordCompilation(TimeSpan compilationTime) => this.compilationTime = compilationTime;

        public void RecordLaunchParameters(int sharedMemorySize, int registerCount)
        {
            this.sharedMemorySize = sharedMemorySize;
            this.registerCount = registerCount;
        }

        public void RecordExecution(TimeSpan executionTime, double? throughput = null)
        {
            if (disposed)
                return;

            var endTime = StartTime.Add(executionTime);
            
            var record = new KernelExecutionRecord
            {
                ExecutionId = executionId,
                KernelName = KernelName,
                StartTime = StartTime,
                EndTime = endTime,
                CompilationTime = compilationTime,
                ExecutionTime = executionTime,
                GridSize = GridSize,
                GroupSize = GroupSize,
                SharedMemorySize = sharedMemorySize,
                RegisterCount = registerCount,
                Throughput = throughput,
                WasFromCache = compilationTime == TimeSpan.Zero,
                Metadata = new Dictionary<string, object>(metadata)
            };

            profiler.CompleteKernelExecution(executionId, record);
        }

        public void AddMetadata(string key, object value) => metadata[key] = value;

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            stopwatch.Stop();

            // If execution wasn't explicitly recorded, record it now
            if (!disposed)
            {
                RecordExecution(stopwatch.Elapsed);
            }
        }
    }

    /// <summary>
    /// Implementation of memory operation profiling context.
    /// </summary>
    internal sealed class MemoryProfilingContext(
        PerformanceProfiler profiler,
        string operationId,
        MemoryOperationType operationType,
        long sizeInBytes,
        string source,
        string destination) : IMemoryProfilingContext
    {
        private readonly PerformanceProfiler profiler = profiler;
        private readonly string operationId = operationId;
        private readonly string source = source;
        private readonly string destination = destination;
        private readonly Stopwatch stopwatch = Stopwatch.StartNew();
        private bool disposed;

        public MemoryOperationType OperationType { get; } = operationType;
        public long SizeInBytes { get; } = sizeInBytes;
        public DateTime StartTime { get; } = DateTime.UtcNow;

        public void RecordCompletion(TimeSpan actualDuration, double? bandwidth = null)
        {
            if (disposed)
                return;

            var endTime = StartTime.Add(actualDuration);
            
            var record = new MemoryOperationRecord
            {
                OperationId = operationId,
                OperationType = OperationType,
                StartTime = StartTime,
                EndTime = endTime,
                SizeInBytes = SizeInBytes,
                Bandwidth = bandwidth,
                Source = source,
                Destination = destination,
                WasFromPool = source.Contains("pool", StringComparison.OrdinalIgnoreCase) || 
                             destination.Contains("pool", StringComparison.OrdinalIgnoreCase)
            };

            profiler.CompleteMemoryOperation(operationId, record);
        }

        public void RecordError(Exception error)
        {
            if (disposed)
                return;

            var endTime = DateTime.UtcNow;
            
            var record = new MemoryOperationRecord
            {
                OperationId = operationId,
                OperationType = OperationType,
                StartTime = StartTime,
                EndTime = endTime,
                SizeInBytes = SizeInBytes,
                Source = source,
                Destination = destination,
                Error = error.Message
            };

            profiler.CompleteMemoryOperation(operationId, record);
        }

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            stopwatch.Stop();

            // If completion wasn't explicitly recorded, record it now
            RecordCompletion(stopwatch.Elapsed);
        }
    }

    /// <summary>
    /// No-operation kernel profiling context for when profiling is disabled.
    /// </summary>
    public sealed class NoOpKernelProfilingContext : IKernelProfilingContext
    {
        public string KernelName => "";
        public DateTime StartTime => DateTime.MinValue;

        public void RecordCompilation(TimeSpan compilationTime) { }
        public void RecordLaunchParameters(int sharedMemorySize, int registerCount) { }
        public void RecordExecution(TimeSpan executionTime, double? throughput = null) { }
        public void AddMetadata(string key, object value) { }
        public void Dispose() { }
    }

    /// <summary>
    /// No-operation memory profiling context for when profiling is disabled.
    /// </summary>
    public sealed class NoOpMemoryProfilingContext : IMemoryProfilingContext
    {
        public MemoryOperationType OperationType => MemoryOperationType.Allocation;
        public long SizeInBytes => 0;
        public DateTime StartTime => DateTime.MinValue;

        public void RecordCompletion(TimeSpan actualDuration, double? bandwidth = null) { }
        public void RecordError(Exception error) { }
        public void Dispose() { }
    }
}
