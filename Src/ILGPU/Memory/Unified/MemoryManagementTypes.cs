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

using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Manages memory allocation and deallocation for accelerators.
    /// </summary>
    public class AcceleratorMemoryManager : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly Dictionary<IntPtr, MemoryAllocation> _allocations = new();
        private long _totalAllocatedBytes;
        private readonly object _syncLock = new();

        /// <summary>
        /// Initializes a new instance of the AcceleratorMemoryManager class.
        /// </summary>
        public AcceleratorMemoryManager(Accelerator accelerator)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        }

        /// <summary>
        /// Gets the associated accelerator.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Gets the total allocated memory in bytes.
        /// </summary>
        public long TotalAllocatedBytes => Interlocked.Read(ref _totalAllocatedBytes);

        /// <summary>
        /// Allocates memory on the accelerator.
        /// </summary>
        public MemoryBuffer<T> Allocate<T>(long length)
            where T : unmanaged
        {
            var buffer = _accelerator.Allocate1D<T>(length);
            
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
    public class MemoryUsageTracker
    {
        private readonly Dictionary<Accelerator, MemoryUsageStats> _usageStats = new();
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
        public Accelerator FindBestAcceleratorForAllocation(long requiredBytes)
        {
            lock (_syncLock)
            {
                return _usageStats
                    .OrderBy(kvp => kvp.Value.TotalAllocatedBytes)
                    .Select(kvp => kvp.Key)
                    .FirstOrDefault();
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
    public class MemoryPlacementOptimizer
    {
        private readonly List<Accelerator> _accelerators;
        private readonly MemoryUsageTracker _usageTracker;

        /// <summary>
        /// Initializes a new instance of the MemoryPlacementOptimizer class.
        /// </summary>
        public MemoryPlacementOptimizer(
            IEnumerable<Accelerator> accelerators,
            MemoryUsageTracker usageTracker)
        {
            _accelerators = accelerators?.ToList() ?? throw new ArgumentNullException(nameof(accelerators));
            _usageTracker = usageTracker ?? throw new ArgumentNullException(nameof(usageTracker));
        }

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
                TargetAccelerator = bestAccelerator,
                RequiredBytes = requiredBytes,
                Strategy = PlacementStrategy.LeastUsed
            };
        }

        /// <summary>
        /// Suggests memory transfers for optimization.
        /// </summary>
        public async Task<List<MemoryTransferSuggestion>> SuggestTransfersAsync()
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
            
            return suggestions;
        }
    }

    /// <summary>
    /// Memory placement decision.
    /// </summary>
    public class MemoryPlacementDecision
    {
        /// <summary>
        /// Gets or sets the target accelerator.
        /// </summary>
        public Accelerator TargetAccelerator { get; set; }

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
        public Accelerator SourceAccelerator { get; set; }

        /// <summary>
        /// Gets or sets the target accelerator.
        /// </summary>
        public Accelerator TargetAccelerator { get; set; }

        /// <summary>
        /// Gets or sets the suggested bytes to transfer.
        /// </summary>
        public long SuggestedBytes { get; set; }

        /// <summary>
        /// Gets or sets the reason for the suggestion.
        /// </summary>
        public string Reason { get; set; }
    }

    /// <summary>
    /// Memory access pattern.
    /// </summary>
    public enum MemoryAccessPattern
    {
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
        ReadWrite
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
}