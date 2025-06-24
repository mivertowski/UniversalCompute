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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ILGPU.Memory.Unified
{
    /// <summary>
    /// Manages universal memory allocation and optimization across all ILGPU accelerator types.
    /// Provides intelligent memory placement, automatic migration, and zero-copy optimizations.
    /// </summary>
    public class UniversalMemoryManager : IDisposable
    {
        private readonly Context _context;
        private readonly ConcurrentDictionary<Accelerator, AcceleratorMemoryManager> _acceleratorManagers;
        private readonly MemoryUsageTracker _usageTracker;
        private readonly MemoryPlacementOptimizer _placementOptimizer;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the UniversalMemoryManager class.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        public UniversalMemoryManager(Context context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _acceleratorManagers = new ConcurrentDictionary<Accelerator, AcceleratorMemoryManager>();
            _usageTracker = new MemoryUsageTracker();
            _placementOptimizer = new MemoryPlacementOptimizer();

            // Initialize accelerator managers for all available accelerators
            foreach (var device in context.Devices)
            {
                try
                {
                    var accelerator = device.CreateAccelerator(context);
                    var manager = new AcceleratorMemoryManager(accelerator);
                    _acceleratorManagers[accelerator] = manager;
                }
                catch
                {
                    // Ignore accelerators that can't be created
                }
            }
        }

        /// <summary>
        /// Gets the available memory placements on the current platform.
        /// </summary>
        public MemoryPlacementInfo[] AvailablePlacements => 
            MemoryPlacementCapabilities.GetAvailablePlacements();

        /// <summary>
        /// Gets the total number of managed accelerators.
        /// </summary>
        public int AcceleratorCount => _acceleratorManagers.Count;

        /// <summary>
        /// Gets the memory usage statistics across all accelerators.
        /// </summary>
        public GlobalMemoryStats GlobalStats => _usageTracker.GetGlobalStats();

        /// <summary>
        /// Allocates a universal buffer with automatic memory placement optimization.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="size">The number of elements to allocate.</param>
        /// <param name="placement">The preferred memory placement strategy.</param>
        /// <param name="accessPattern">The expected access pattern for optimization.</param>
        /// <returns>A universal buffer instance.</returns>
        public IUniversalBuffer<T> AllocateUniversal<T>(
            long size,
            MemoryPlacement placement = MemoryPlacement.Auto,
            MemoryAccessPattern accessPattern = MemoryAccessPattern.Unknown)
            where T : unmanaged
        {
            ThrowIfDisposed();

            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));

            // Optimize placement if auto is requested
            if (placement == MemoryPlacement.Auto)
            {
                placement = _placementOptimizer.GetOptimalPlacement<T>(
                    size, accessPattern, _usageTracker.GetCurrentUsage());
            }

            // Create the appropriate buffer implementation
            return placement switch
            {
                MemoryPlacement.AppleUnified when PlatformDetection.IsAppleSilicon => 
                    new AppleUnifiedBuffer<T>(size, this),
                    
                MemoryPlacement.CudaManaged when PlatformDetection.HasCudaSupport => 
                    new CudaManagedBuffer<T>(size, this),
                    
                MemoryPlacement.IntelShared when PlatformDetection.HasIntelIntegratedGpu => 
                    new IntelSharedBuffer<T>(size, this),
                    
                MemoryPlacement.HostPinned => 
                    new PinnedHostBuffer<T>(size, this),
                    
                _ => new StandardUniversalBuffer<T>(size, placement, this)
            };
        }

        /// <summary>
        /// Allocates a universal buffer and initializes it with data.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="data">The initial data.</param>
        /// <param name="placement">The preferred memory placement strategy.</param>
        /// <param name="accessPattern">The expected access pattern for optimization.</param>
        /// <returns>A universal buffer instance.</returns>
        public IUniversalBuffer<T> AllocateUniversal<T>(
            ReadOnlySpan<T> data,
            MemoryPlacement placement = MemoryPlacement.Auto,
            MemoryAccessPattern accessPattern = MemoryAccessPattern.Unknown)
            where T : unmanaged
        {
            var buffer = AllocateUniversal<T>(data.Length, placement, accessPattern);
            buffer.CopyFromCPU(data);
            return buffer;
        }

        /// <summary>
        /// Creates a universal buffer from an existing ILGPU memory buffer.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="memoryBuffer">The existing memory buffer.</param>
        /// <param name="takeOwnership">Whether to take ownership of the buffer.</param>
        /// <returns>A universal buffer instance.</returns>
        public IUniversalBuffer<T> FromMemoryBuffer<T>(
            MemoryBuffer1D<T, Stride1D.Dense> memoryBuffer,
            bool takeOwnership = false)
            where T : unmanaged
        {
            ThrowIfDisposed();
            return new WrappedMemoryBuffer<T>(memoryBuffer, this, takeOwnership);
        }

        /// <summary>
        /// Gets the memory manager for the specified accelerator.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <returns>The accelerator memory manager.</returns>
        internal AcceleratorMemoryManager GetAcceleratorManager(Accelerator accelerator)
        {
            return _acceleratorManagers.GetOrAdd(accelerator, 
                acc => new AcceleratorMemoryManager(acc));
        }

        /// <summary>
        /// Records memory usage for tracking and optimization.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="sizeInBytes">The allocated size in bytes.</param>
        /// <param name="placement">The memory placement used.</param>
        internal void RecordAllocation(Accelerator accelerator, long sizeInBytes, MemoryPlacement placement)
        {
            _usageTracker.RecordAllocation(accelerator, sizeInBytes, placement);
        }

        /// <summary>
        /// Records memory deallocation for tracking.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="sizeInBytes">The deallocated size in bytes.</param>
        /// <param name="placement">The memory placement used.</param>
        internal void RecordDeallocation(Accelerator accelerator, long sizeInBytes, MemoryPlacement placement)
        {
            _usageTracker.RecordDeallocation(accelerator, sizeInBytes, placement);
        }

        /// <summary>
        /// Gets optimal buffer placement recommendations based on current usage patterns.
        /// </summary>
        /// <returns>Placement recommendations.</returns>
        public MemoryPlacementRecommendations GetPlacementRecommendations()
        {
            ThrowIfDisposed();
            return _placementOptimizer.GetRecommendations(_usageTracker.GetCurrentUsage());
        }

        /// <summary>
        /// Forces garbage collection and cleanup of unused buffers.
        /// </summary>
        public void Cleanup()
        {
            ThrowIfDisposed();
            
            foreach (var manager in _acceleratorManagers.Values)
            {
                manager.Cleanup();
            }
            
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        /// Asynchronously performs memory optimization across all accelerators.
        /// </summary>
        /// <returns>A task representing the optimization operation.</returns>
        public async Task OptimizeAsync()
        {
            ThrowIfDisposed();

            var optimizationTasks = _acceleratorManagers.Values
                .Select(manager => manager.OptimizeAsync())
                .ToArray();

            await Task.WhenAll(optimizationTasks);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(UniversalMemoryManager));
        }

        /// <summary>
        /// Disposes the memory manager and all associated resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            foreach (var manager in _acceleratorManagers.Values)
            {
                manager.Dispose();
            }

            _acceleratorManagers.Clear();
            _usageTracker.Dispose();
            _placementOptimizer.Dispose();

            _disposed = true;
        }
    }

    /// <summary>
    /// Provides placement recommendations based on usage patterns.
    /// </summary>
    public readonly struct MemoryPlacementRecommendations
    {
        /// <summary>
        /// Gets the recommended placement for small buffers (&lt; 64KB).
        /// </summary>
        public MemoryPlacement SmallBuffers { get; }

        /// <summary>
        /// Gets the recommended placement for medium buffers (64KB - 16MB).
        /// </summary>
        public MemoryPlacement MediumBuffers { get; }

        /// <summary>
        /// Gets the recommended placement for large buffers (&gt; 16MB).
        /// </summary>
        public MemoryPlacement LargeBuffers { get; }

        /// <summary>
        /// Gets the recommended placement for frequently accessed buffers.
        /// </summary>
        public MemoryPlacement FrequentlyAccessed { get; }

        /// <summary>
        /// Gets the recommended placement for read-only buffers.
        /// </summary>
        public MemoryPlacement ReadOnly { get; }

        /// <summary>
        /// Gets additional optimization hints.
        /// </summary>
        public string[] OptimizationHints { get; }

        /// <summary>
        /// Initializes a new instance of the MemoryPlacementRecommendations struct.
        /// </summary>
        public MemoryPlacementRecommendations(
            MemoryPlacement smallBuffers,
            MemoryPlacement mediumBuffers,
            MemoryPlacement largeBuffers,
            MemoryPlacement frequentlyAccessed,
            MemoryPlacement readOnly,
            string[] optimizationHints)
        {
            SmallBuffers = smallBuffers;
            MediumBuffers = mediumBuffers;
            LargeBuffers = largeBuffers;
            FrequentlyAccessed = frequentlyAccessed;
            ReadOnly = readOnly;
            OptimizationHints = optimizationHints;
        }
    }

    /// <summary>
    /// Provides global memory usage statistics across all accelerators.
    /// </summary>
    public readonly struct GlobalMemoryStats
    {
        /// <summary>
        /// Gets the total allocated memory across all accelerators in bytes.
        /// </summary>
        public long TotalAllocatedBytes { get; }

        /// <summary>
        /// Gets the number of active buffers across all accelerators.
        /// </summary>
        public int ActiveBufferCount { get; }

        /// <summary>
        /// Gets the total number of memory migrations performed.
        /// </summary>
        public long TotalMigrations { get; }

        /// <summary>
        /// Gets the average memory utilization percentage across all accelerators.
        /// </summary>
        public double AverageUtilizationPercent { get; }

        /// <summary>
        /// Gets per-accelerator memory statistics.
        /// </summary>
        public IReadOnlyDictionary<AcceleratorType, AcceleratorMemoryStats> PerAcceleratorStats { get; }

        /// <summary>
        /// Initializes a new instance of the GlobalMemoryStats struct.
        /// </summary>
        public GlobalMemoryStats(
            long totalAllocatedBytes,
            int activeBufferCount,
            long totalMigrations,
            double averageUtilizationPercent,
            IReadOnlyDictionary<AcceleratorType, AcceleratorMemoryStats> perAcceleratorStats)
        {
            TotalAllocatedBytes = totalAllocatedBytes;
            ActiveBufferCount = activeBufferCount;
            TotalMigrations = totalMigrations;
            AverageUtilizationPercent = averageUtilizationPercent;
            PerAcceleratorStats = perAcceleratorStats;
        }
    }

    /// <summary>
    /// Provides memory statistics for a specific accelerator type.
    /// </summary>
    public readonly struct AcceleratorMemoryStats
    {
        /// <summary>
        /// Gets the allocated memory for this accelerator type in bytes.
        /// </summary>
        public long AllocatedBytes { get; }

        /// <summary>
        /// Gets the number of active buffers for this accelerator type.
        /// </summary>
        public int ActiveBuffers { get; }

        /// <summary>
        /// Gets the memory utilization percentage for this accelerator type.
        /// </summary>
        public double UtilizationPercent { get; }

        /// <summary>
        /// Initializes a new instance of the AcceleratorMemoryStats struct.
        /// </summary>
        public AcceleratorMemoryStats(long allocatedBytes, int activeBuffers, double utilizationPercent)
        {
            AllocatedBytes = allocatedBytes;
            ActiveBuffers = activeBuffers;
            UtilizationPercent = utilizationPercent;
        }
    }
}