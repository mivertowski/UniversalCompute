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

using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Manages memory allocation and deallocation for accelerators.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the AcceleratorMemoryManager class.
    /// </remarks>
    public class AcceleratorMemoryManager(Accelerator accelerator) : IDisposable
    {
        private readonly Dictionary<IntPtr, MemoryAllocation> _allocations = [];
        private long _totalAllocatedBytes;
        private readonly object _syncLock = new();

        /// <summary>
        /// Gets the associated accelerator.
        /// </summary>
        public Accelerator Accelerator { get; } = accelerator ?? throw new ArgumentNullException(nameof(accelerator));

        /// <summary>
        /// Gets the total allocated memory in bytes.
        /// </summary>
        public long TotalAllocatedBytes => Interlocked.Read(ref _totalAllocatedBytes);

        /// <summary>
        /// Allocates memory on the accelerator.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Allocate<T>(long length)
            where T : unmanaged
        {
            var buffer = Accelerator.Allocate1D<T>(length);
            
            lock (_syncLock)
            {
                var allocation = new MemoryAllocation
                {
                    Address = buffer.NativePtr,
                    SizeBytes = length * Interop.SizeOf<T>(),
                    AllocationTime = DateTime.UtcNow
                };
                
                _allocations[buffer.NativePtr] = allocation;
                Interlocked.Add(ref _totalAllocatedBytes, allocation.SizeBytes);
            }
            
            return buffer;
        }

        /// <summary>
        /// Tracks a memory deallocation.
        /// </summary>
        public void NotifyDeallocation(IntPtr address)
        {
            lock (_syncLock)
            {
                if (_allocations.TryGetValue(address, out var allocation))
                {
                    _allocations.Remove(address);
                    Interlocked.Add(ref _totalAllocatedBytes, -allocation.SizeBytes);
                }
            }
        }

        /// <summary>
        /// Gets memory usage statistics.
        /// </summary>
        public MemoryUsageStats GetStats()
        {
            lock (_syncLock)
            {
                return new MemoryUsageStats
                {
                    TotalAllocatedBytes = TotalAllocatedBytes,
                    AllocationCount = _allocations.Count,
                    PeakAllocationBytes = _allocations.Count > 0 
                        ? _allocations.Values.Max(a => a.SizeBytes) 
                        : 0
                };
            }
        }

        /// <summary>
        /// Performs cleanup operations on the memory manager.
        /// </summary>
        public void Cleanup()
        {
            lock (_syncLock)
            {
                // Remove any orphaned allocations
                var orphanedAllocations = _allocations
                    .Where(kvp => DateTime.UtcNow - kvp.Value.AllocationTime > TimeSpan.FromMinutes(10))
                    .ToList();

                foreach (var orphaned in orphanedAllocations)
                {
                    _allocations.Remove(orphaned.Key);
                    Interlocked.Add(ref _totalAllocatedBytes, -orphaned.Value.SizeBytes);
                }
            }
        }

        /// <summary>
        /// Optimizes memory usage asynchronously.
        /// </summary>
        public async Task OptimizeAsync()
        {
            // Perform optimization operations
            Cleanup();
            await Task.CompletedTask.ConfigureAwait(false);
        }

        /// <summary>
        /// Disposes the memory manager.
        /// </summary>
        public void Dispose()
        {
            // Clean up any remaining allocations
            lock (_syncLock)
            {
                _allocations.Clear();
                _totalAllocatedBytes = 0;
            }
        }

        private class MemoryAllocation
        {
            public IntPtr Address { get; set; }
            public long SizeBytes { get; set; }
            public DateTime AllocationTime { get; set; }
        }
    }

    /// <summary>
    /// Tracks memory usage across accelerators.
    /// </summary>
    public class MemoryUsageTracker : IDisposable
    {
        private readonly Dictionary<Accelerator, MemoryUsageStats> _usageStats = [];
        private readonly object _syncLock = new();

        /// <summary>
        /// Updates usage statistics for an accelerator.
        /// </summary>
        public void UpdateUsage(Accelerator accelerator, MemoryUsageStats stats)
        {
            lock (_syncLock)
            {
                _usageStats[accelerator] = stats;
            }
        }

        /// <summary>
        /// Gets usage statistics for an accelerator.
        /// </summary>
        public MemoryUsageStats GetUsage(Accelerator accelerator)
        {
            lock (_syncLock)
            {
                return _usageStats.GetValueOrDefault(accelerator, new MemoryUsageStats());
            }
        }

        /// <summary>
        /// Gets total memory usage across all accelerators.
        /// </summary>
        public long GetTotalUsageBytes()
        {
            lock (_syncLock)
            {
                return _usageStats.Values.Sum(s => s.TotalAllocatedBytes);
            }
        }

        /// <summary>
        /// Finds the accelerator with most available memory.
        /// </summary>
        public Accelerator? FindBestAcceleratorForAllocation(long requiredBytes)
        {
            lock (_syncLock)
            {
                return _usageStats
                    .OrderBy(kvp => kvp.Value.TotalAllocatedBytes)
                    .Select(kvp => kvp.Key)
                    .FirstOrDefault();
            }
        }

        /// <summary>
        /// Gets global memory statistics across all accelerators.
        /// </summary>
        public GlobalMemoryStats GetGlobalStats()
        {
            lock (_syncLock)
            {
                var totalAllocated = _usageStats.Values.Sum(s => s.TotalAllocatedBytes);
                var totalAllocations = _usageStats.Values.Sum(s => s.AllocationCount);
                var peakUsage = _usageStats.Values.Max(s => s.PeakAllocationBytes);

                return new GlobalMemoryStats
                {
                    TotalAllocatedBytes = totalAllocated,
                    TotalAllocations = totalAllocations,
                    PeakUsageBytes = peakUsage,
                    AcceleratorCount = _usageStats.Count
                };
            }
        }

        /// <summary>
        /// Gets current memory usage information.
        /// </summary>
        public MemoryUsageInfo GetCurrentUsage()
        {
            lock (_syncLock)
            {
                var totalMemory = _usageStats.Values.Sum(s => s.TotalAllocatedBytes);
                var availableMemory = Math.Max(0, totalMemory * 0.8); // Assume 80% utilization threshold

                return new MemoryUsageInfo(totalMemory, (long)availableMemory);
            }
        }

        /// <summary>
        /// Records an allocation for tracking.
        /// </summary>
        public void RecordAllocation(Accelerator accelerator, long bytes, MemoryPlacement placement = MemoryPlacement.Auto)
        {
            lock (_syncLock)
            {
                if (_usageStats.TryGetValue(accelerator, out var stats))
                {
                    stats.TotalAllocatedBytes += bytes;
                    stats.AllocationCount++;
                    stats.PeakAllocationBytes = Math.Max(stats.PeakAllocationBytes, stats.TotalAllocatedBytes);
                    _usageStats[accelerator] = stats;
                }
                else
                {
                    _usageStats[accelerator] = new MemoryUsageStats
                    {
                        TotalAllocatedBytes = bytes,
                        AllocationCount = 1,
                        PeakAllocationBytes = bytes
                    };
                }
            }
        }

        /// <summary>
        /// Records a deallocation for tracking.
        /// </summary>
        public void RecordDeallocation(Accelerator accelerator, long bytes, MemoryPlacement placement = MemoryPlacement.Auto)
        {
            lock (_syncLock)
            {
                if (_usageStats.TryGetValue(accelerator, out var stats))
                {
                    stats.TotalAllocatedBytes = Math.Max(0, stats.TotalAllocatedBytes - bytes);
                    stats.AllocationCount = Math.Max(0, stats.AllocationCount - 1);
                    _usageStats[accelerator] = stats;
                }
            }
        }

        /// <summary>
        /// Disposes the memory usage tracker.
        /// </summary>
        public void Dispose()
        {
            lock (_syncLock)
            {
                _usageStats.Clear();
            }
        }
    }

    /// <summary>
    /// Memory usage statistics.
    /// </summary>
    public struct MemoryUsageStats
    {
        /// <summary>
        /// Total allocated memory in bytes.
        /// </summary>
        public long TotalAllocatedBytes { get; set; }

        /// <summary>
        /// Number of active allocations.
        /// </summary>
        public int AllocationCount { get; set; }

        /// <summary>
        /// Size of the largest allocation.
        /// </summary>
        public long PeakAllocationBytes { get; set; }
    }

    /// <summary>
    /// Optimizes memory placement across accelerators.
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the MemoryPlacementOptimizer class.
    /// </remarks>
    public class MemoryPlacementOptimizer(
        IEnumerable<Accelerator> accelerators,
        MemoryUsageTracker usageTracker) : IDisposable
    {
        private readonly List<Accelerator> _accelerators = accelerators?.ToList() ?? throw new ArgumentNullException(nameof(accelerators));
        private readonly MemoryUsageTracker _usageTracker = usageTracker ?? throw new ArgumentNullException(nameof(usageTracker));

        /// <summary>
        /// Determines optimal placement for a memory allocation.
        /// </summary>
        public MemoryPlacementDecision DeterminePlacement<T>(
            long length,
            MemoryAccessPattern accessPattern)
            where T : unmanaged
        {
            var requiredBytes = length * Interop.SizeOf<T>();
            
            // Simple strategy: find accelerator with most free memory
            var bestAccelerator = _usageTracker.FindBestAcceleratorForAllocation(requiredBytes);
            
            if (bestAccelerator == null && _accelerators.Count > 0)
            {
                bestAccelerator = _accelerators[0];
            }

            return new MemoryPlacementDecision
            {
                TargetAccelerator = bestAccelerator ?? throw new InvalidOperationException("No suitable accelerator found for memory allocation"),
                RequiredBytes = requiredBytes,
                Strategy = PlacementStrategy.LeastUsed
            };
        }

        /// <summary>
        /// Suggests memory transfers for optimization.
        /// </summary>
        public Task<List<MemoryTransferSuggestion>> SuggestTransfersAsync()
        {
            var suggestions = new List<MemoryTransferSuggestion>();
            
            // Analyze imbalances and suggest transfers
            var totalUsage = _usageTracker.GetTotalUsageBytes();
            var avgUsagePerAccelerator = totalUsage / _accelerators.Count;
            
            foreach (var accelerator in _accelerators)
            {
                var usage = _usageTracker.GetUsage(accelerator);
                if (usage.TotalAllocatedBytes > avgUsagePerAccelerator * 1.5)
                {
                    // This accelerator is overloaded
                    var targetAccelerator = _usageTracker.FindBestAcceleratorForAllocation(0);
                    if (targetAccelerator != null && targetAccelerator != accelerator)
                    {
                        suggestions.Add(new MemoryTransferSuggestion
                        {
                            SourceAccelerator = accelerator,
                            TargetAccelerator = targetAccelerator,
                            SuggestedBytes = usage.TotalAllocatedBytes - avgUsagePerAccelerator,
                            Reason = "Load balancing"
                        });
                    }
                }
            }
            
            return Task.FromResult(suggestions);
        }

        /// <summary>
        /// Gets the optimal memory placement for the specified parameters.
        /// </summary>
        public MemoryPlacement GetOptimalPlacement<T>(
            long size, 
            MemoryAccessPattern accessPattern, 
            MemoryUsageInfo usageInfo) where T : unmanaged
        {
            var requiredBytes = size * Interop.SizeOf<T>();
            
            // For very small data, prefer zero-copy if available
            if (requiredBytes < 64 * 1024) // 64KB
            {
                return MemoryPlacement.HostAccessible;
            }

            // For large data with sequential access, prefer high bandwidth
            if (accessPattern == MemoryAccessPattern.Sequential && requiredBytes > 1024 * 1024) // 1MB
            {
                return MemoryPlacement.DeviceLocal;
            }

            // For random access, prefer low latency
            if (accessPattern == MemoryAccessPattern.Random)
            {
                return MemoryPlacement.HostPinned;
            }

            // Check memory pressure
            if (usageInfo.AvailableMemory < requiredBytes * 2)
            {
                return MemoryPlacement.HostAccessible;
            }

            // Default to device local memory
            return MemoryPlacement.DeviceLocal;
        }

        /// <summary>
        /// Gets optimization recommendations for memory usage.
        /// </summary>
        public async Task<List<MemoryTransferSuggestion>> GetRecommendations() => await SuggestTransfersAsync().ConfigureAwait(false);

        /// <summary>
        /// Disposes the memory placement optimizer.
        /// </summary>
        public void Dispose() => _accelerators?.Clear();
    }

    /// <summary>
    /// Memory placement decision.
    /// </summary>
    public class MemoryPlacementDecision
    {
        /// <summary>
        /// Gets or sets the target accelerator.
        /// </summary>
        public required Accelerator TargetAccelerator { get; set; }

        /// <summary>
        /// Gets or sets the required bytes.
        /// </summary>
        public long RequiredBytes { get; set; }

        /// <summary>
        /// Gets or sets the placement strategy used.
        /// </summary>
        public PlacementStrategy Strategy { get; set; }
    }

    /// <summary>
    /// Memory transfer suggestion.
    /// </summary>
    public class MemoryTransferSuggestion
    {
        /// <summary>
        /// Gets or sets the source accelerator.
        /// </summary>
        public required Accelerator SourceAccelerator { get; set; }

        /// <summary>
        /// Gets or sets the target accelerator.
        /// </summary>
        public required Accelerator TargetAccelerator { get; set; }

        /// <summary>
        /// Gets or sets the suggested bytes to transfer.
        /// </summary>
        public long SuggestedBytes { get; set; }

        /// <summary>
        /// Gets or sets the reason for the suggestion.
        /// </summary>
        public required string Reason { get; set; }
    }

    /// <summary>
    /// Memory access pattern.
    /// </summary>
    public enum MemoryAccessPattern
    {
        /// <summary>
        /// Unknown or unspecified access pattern.
        /// </summary>
        Unknown,

        /// <summary>
        /// Sequential access pattern.
        /// </summary>
        Sequential,

        /// <summary>
        /// Random access pattern.
        /// </summary>
        Random,

        /// <summary>
        /// Strided access pattern.
        /// </summary>
        Strided,

        /// <summary>
        /// Read-only access.
        /// </summary>
        ReadOnly,

        /// <summary>
        /// Write-only access.
        /// </summary>
        WriteOnly,

        /// <summary>
        /// Read-write access.
        /// </summary>
        ReadWrite,

        /// <summary>
        /// Streaming access pattern.
        /// </summary>
        Streaming,

        /// <summary>
        /// Transpose access pattern.
        /// </summary>
        Transpose,

        /// <summary>
        /// Gather access pattern.
        /// </summary>
        Gather,

        /// <summary>
        /// Scatter access pattern.
        /// </summary>
        Scatter
    }

    /// <summary>
    /// Memory placement strategy.
    /// </summary>
    public enum PlacementStrategy
    {
        /// <summary>
        /// Place on least used device.
        /// </summary>
        LeastUsed,

        /// <summary>
        /// Place for best performance.
        /// </summary>
        Performance,

        /// <summary>
        /// Place near compute.
        /// </summary>
        Locality,

        /// <summary>
        /// Round-robin placement.
        /// </summary>
        RoundRobin
    }

    /// <summary>
    /// Global memory statistics across all accelerators.
    /// </summary>
    public struct GlobalMemoryStats
    {
        public long TotalAllocatedBytes { get; init; }
        public int TotalAllocations { get; init; }
        public long PeakUsageBytes { get; init; }
        public int AcceleratorCount { get; init; }
    }

    /// <summary>
    /// Memory usage information for placement decisions.
    /// </summary>
    public readonly struct MemoryUsageInfo(long totalMemory, long availableMemory)
    {
        public long TotalMemory { get; } = totalMemory;
        public long AvailableMemory { get; } = availableMemory;
    }
}