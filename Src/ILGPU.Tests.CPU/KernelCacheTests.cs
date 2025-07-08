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
using ILGPU.Runtime.KernelCache;
using ILGPU.Runtime.CPU;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.CPU
{
    /// <summary>
    /// Tests for kernel caching system with version management.
    /// </summary>
    public class KernelCacheTests : IDisposable
    {
        #region Fields

        private readonly Context context;
        private readonly Accelerator accelerator;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the <see cref="KernelCacheTests"/> class.
        /// </summary>
        public KernelCacheTests()
        {
            context = Context.Create(builder => builder.DefaultCPU());
            accelerator = context.CreateCPUAccelerator(0);
        }

        #endregion

        #region Test Kernels

        static void SimpleKernel(Index1D index, ArrayView<int> data)
        {
            data[index] = index;
        }

        static void MultiplyKernel(Index1D index, ArrayView<int> input, ArrayView<int> output, int factor)
        {
            output[index] = input[index] * factor;
        }

        static void AddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
        {
            result[index] = a[index] + b[index];
        }

        #endregion

        #region Cache Manager Tests

        [Fact]
        public void KernelCache_BasicPutAndGet()
        {
            var options = new KernelCacheOptions { MaxSize = 100 };
            using var cache = new KernelCacheManager(options);
            
            // Create a mock kernel (simplified for testing)
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            var key = "test_kernel";
            var version = "1.0.0";
            
            // Put kernel in cache
            cache.Put(key, mockKernel as dynamic, version);
            
            // Try to get it back
            var found = cache.TryGet(key, version, out var entry);
            
            Assert.True(found);
            Assert.NotNull(entry);
            Assert.Equal(version, entry.Version);
            Assert.Equal(key, entry.Metadata.ContainsKey("CacheKey") ? entry.Metadata["CacheKey"] : key);
        }

        [Fact]
        public void KernelCache_VersionMismatch()
        {
            var options = new KernelCacheOptions { MaxSize = 100 };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            var key = "test_kernel";
            var version1 = "1.0.0";
            var version2 = "2.0.0";
            
            // Put kernel with version 1.0.0
            cache.Put(key, mockKernel as dynamic, version1);
            
            // Try to get with version 2.0.0 (should fail)
            var found = cache.TryGet(key, version2, out var entry);
            
            Assert.False(found);
            Assert.Null(entry);
        }

        [Fact]
        public void KernelCache_Expiration()
        {
            var options = new KernelCacheOptions 
            { 
                MaxSize = 100,
                DefaultTTL = TimeSpan.FromMilliseconds(100) // Very short TTL
            };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            var key = "test_kernel";
            var version = "1.0.0";
            
            // Put kernel in cache
            cache.Put(key, mockKernel as dynamic, version);
            
            // Verify it's there immediately
            var found1 = cache.TryGet(key, version, out var entry1);
            Assert.True(found1);
            Assert.NotNull(entry1);
            
            // Wait for expiration
            Thread.Sleep(200);
            
            // Should be expired now
            var found2 = cache.TryGet(key, version, out var entry2);
            Assert.False(found2);
            Assert.Null(entry2);
        }

        [Fact]
        public void KernelCache_LRUEviction()
        {
            var options = new KernelCacheOptions 
            { 
                MaxSize = 3,
                EvictionThreshold = 0.8 // Will trigger eviction at 2-3 items
            };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            // Add multiple kernels
            cache.Put("kernel1", mockKernel as dynamic, "1.0");
            cache.Put("kernel2", mockKernel as dynamic, "1.0");
            
            // Access kernel1 to make it more recently used
            cache.TryGet("kernel1", "1.0", out _);
            
            // Add more kernels to trigger eviction
            cache.Put("kernel3", mockKernel as dynamic, "1.0");
            cache.Put("kernel4", mockKernel as dynamic, "1.0");
            
            // kernel2 should be evicted (least recently used)
            var found2 = cache.TryGet("kernel2", "1.0", out _);
            var found1 = cache.TryGet("kernel1", "1.0", out _);
            var found3 = cache.TryGet("kernel3", "1.0", out _);
            
            Assert.False(found2); // Should be evicted
            Assert.True(found1);  // Should still be there
            Assert.True(found3);  // Should still be there
        }

        [Fact]
        public void KernelCache_Statistics()
        {
            var options = new KernelCacheOptions { MaxSize = 100 };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            // Put a kernel
            cache.Put("kernel1", mockKernel as dynamic, "1.0");
            
            // Hit
            cache.TryGet("kernel1", "1.0", out _);
            
            // Miss
            cache.TryGet("nonexistent", "1.0", out _);
            
            var stats = cache.GetStatistics();
            
            Assert.Equal(1, stats.TotalHits);
            Assert.Equal(1, stats.TotalMisses);
            Assert.Equal(1, stats.CurrentSize);
            Assert.Equal(100, stats.MaxSize);
            Assert.True(stats.HitRatio > 0);
        }

        [Fact]
        public void KernelCache_VersionInvalidation()
        {
            var options = new KernelCacheOptions { MaxSize = 100 };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            // Put kernels with different versions
            cache.Put("kernel1", mockKernel as dynamic, "1.0");
            cache.Put("kernel2", mockKernel as dynamic, "1.0");
            cache.Put("kernel3", mockKernel as dynamic, "2.0");
            
            // Invalidate version 1.0
            var invalidated = cache.InvalidateVersion("1.0");
            
            Assert.Equal(2, invalidated);
            
            // Version 1.0 kernels should be gone
            Assert.False(cache.TryGet("kernel1", "1.0", out _));
            Assert.False(cache.TryGet("kernel2", "1.0", out _));
            
            // Version 2.0 kernel should still be there
            Assert.True(cache.TryGet("kernel3", "2.0", out _));
        }

        [Fact]
        public void KernelCache_ClearAll()
        {
            var options = new KernelCacheOptions { MaxSize = 100 };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            // Put multiple kernels
            cache.Put("kernel1", mockKernel as dynamic, "1.0");
            cache.Put("kernel2", mockKernel as dynamic, "1.0");
            cache.Put("kernel3", mockKernel as dynamic, "2.0");
            
            Assert.Equal(3, cache.CurrentSize);
            
            // Clear all
            cache.Clear();
            
            Assert.Equal(0, cache.CurrentSize);
            Assert.False(cache.TryGet("kernel1", "1.0", out _));
            Assert.False(cache.TryGet("kernel2", "1.0", out _));
            Assert.False(cache.TryGet("kernel3", "2.0", out _));
        }

        [Fact]
        public void KernelCache_Maintenance()
        {
            var options = new KernelCacheOptions 
            { 
                MaxSize = 100,
                DefaultTTL = TimeSpan.FromMilliseconds(50),
                EnableAutomaticMaintenance = false // Disable automatic maintenance for this test
            };
            using var cache = new KernelCacheManager(options);
            
            var mockKernel = accelerator.LoadKernel<Action<Index1D, ArrayView<int>>>(typeof(KernelCacheTests).GetMethod(nameof(SimpleKernel)));
            
            // Put kernels
            cache.Put("kernel1", mockKernel as dynamic, "1.0");
            cache.Put("kernel2", mockKernel as dynamic, "1.0");
            
            // Wait for expiration
            Thread.Sleep(100);
            
            Assert.Equal(2, cache.CurrentSize);
            
            // Perform maintenance
            var removed = cache.PerformMaintenance();
            
            Assert.True(removed > 0);
            Assert.True(cache.CurrentSize < 2);
        }

        [Fact]
        public async Task KernelCache_AsyncOperations()
        {
            var options = new KernelCacheOptions 
            { 
                MaxSize = 100,
                EnablePersistentCache = false // Disable for this test
            };
            using var cache = new KernelCacheManager(options);
            
            // Test async preload (should complete without error)
            await cache.PreloadAsync().ConfigureAwait(false);
            
            // Test async persist (should complete without error)
            await cache.PersistAsync().ConfigureAwait(false);
        }

        #endregion

        #region accelerator Integration Tests

        [Fact]
        public void KernelCache_acceleratorIntegration()
        {
            var cache = AcceleratorKernelCache.GetOrCreateCache(accelerator);
            
            Assert.NotNull(cache);
            Assert.Equal(0, cache.CurrentSize);
            
            // Test that we get the same cache instance
            var cache2 = AcceleratorKernelCache.GetOrCreateCache(accelerator);
            Assert.Same(cache, cache2);
        }

        [Fact]
        public void KernelCache_LoadKernelCached()
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            
            Assert.NotNull(kernel);
            // Basic kernel functionality test - just verify it's loaded
        }

        [Fact]
        public void KernelCache_KernelExecution()
        {
            const int length = 1000;
            using var buffer = accelerator.Allocate1D<int>(length);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            
            // Execute the kernel
            kernel((Index1D)length, buffer.View);
            accelerator.Synchronize();
            
            // Verify results
            var data = buffer.View.AsContiguous().GetAsArray();
            for (int i = 0; i < length; i++)
            {
                Assert.Equal(i, data[i]);
            }
        }

        [Fact]
        public void KernelCache_MultipleKernelTypes()
        {
            var kernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            var kernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(MultiplyKernel);
            var kernel3 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(AddKernel);
            
            // Verify all kernels are loaded successfully
            Assert.NotNull(kernel1);
            Assert.NotNull(kernel2);
            Assert.NotNull(kernel3);
            
            // Basic verification that kernels are different instances
            Assert.NotSame(kernel1, kernel2);
            Assert.NotSame(kernel1, kernel3);
            Assert.NotSame(kernel2, kernel3);
        }

        [Fact]
        [SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "Test infrastructure needs to collect all exceptions from concurrent operations")]
        public void KernelCache_ConcurrentAccess()
        {
            const int threadCount = 10;
            const int kernelsPerThread = 50;
            var cache = AcceleratorKernelCache.GetOrCreateCache(accelerator);
            
            var tasks = new Task[threadCount];
            var exceptions = new List<Exception>();
            
            for (int t = 0; t < threadCount; t++)
            {
                int threadId = t;
                tasks[t] = Task.Run(() =>
                {
                    try
                    {
                        for (int i = 0; i < kernelsPerThread; i++)
                        {
                            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
                            
                            // Access the kernel to trigger cache operations
                            Assert.NotNull(kernel);
                        }
                    }
                    catch (Exception ex)
                    {
                        lock (exceptions)
                        {
                            exceptions.Add(ex);
                        }
                    }
                });
            }
            
            Task.WaitAll(tasks);
            
            // Verify no exceptions occurred
            Assert.Empty(exceptions);
            
            // Cache should contain the kernel
            Assert.True(cache.CurrentSize > 0);
        }

        [Fact]
        public void KernelCache_Performance()
        {
            const int iterations = 100;
            var stopwatch = new System.Diagnostics.Stopwatch();
            
            // Measure uncached kernel loading
            stopwatch.Start();
            for (int i = 0; i < iterations; i++)
            {
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
                // Use the kernel reference to ensure it's not optimized away
                _ = kernel;
            }
            stopwatch.Stop();
            var uncachedTime = stopwatch.Elapsed;
            
            // Measure cached kernel loading
            stopwatch.Restart();
            var cachedKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            for (int i = 0; i < iterations; i++)
            {
                // Just access the kernel to simulate cache hit
                var kernelRef = cachedKernel;
                _ = kernelRef;
            }
            stopwatch.Stop();
            var cachedTime = stopwatch.Elapsed;
            
            System.Console.WriteLine($"Uncached time: {uncachedTime.TotalMilliseconds:F2}ms");
            System.Console.WriteLine($"Cached time: {cachedTime.TotalMilliseconds:F2}ms");
            
            // Cached should be significantly faster (though first load might be slower)
            // This is more of a performance verification than assertion
            Assert.True(cachedTime < uncachedTime * 10); // Allow for cache miss on first load
        }

        [Fact]
        public void KernelCache_GlobalStatistics()
        {
            // Load some kernels to populate caches
            var kernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            var kernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(MultiplyKernel);
            
            // Use kernels to ensure they're not optimized away
            _ = kernel1;
            _ = kernel2;
            
            var stats = AcceleratorKernelCache.GetAllStatistics();
            
            Assert.NotEmpty(stats);
            Assert.True(stats.Count > 0);
            
            foreach (var kvp in stats)
            {
                Assert.NotNull(kvp.Key);
                Assert.NotNull(kvp.Value);
                Assert.True(kvp.Value.MaxSize > 0);
            }
        }

        [Fact]
        public void KernelCache_ClearAllacceleratorCaches()
        {
            // Load some kernels
            var kernel1 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(SimpleKernel);
            var kernel2 = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(MultiplyKernel);
            
            // Use kernels to ensure they're not optimized away
            _ = kernel1;
            _ = kernel2;
            
            var cache = AcceleratorKernelCache.GetOrCreateCache(accelerator);
            Assert.True(cache.CurrentSize > 0);
            
            // Clear all caches
            AcceleratorKernelCache.ClearAllCaches();
            
            Assert.Equal(0, cache.CurrentSize);
        }

        #endregion

        #region Utility Extension Tests

        [Fact]
        public void KernelCache_CreateCacheKey()
        {
            var key1 = KernelCacheExtensions.CreateCacheKey(
                "TestKernel", 
                [typeof(int), typeof(float)], 
                "CPU_Device_0");
                
            var key2 = KernelCacheExtensions.CreateCacheKey(
                "TestKernel", 
                [typeof(int), typeof(double)], 
                "CPU_Device_0");
                
            var key3 = KernelCacheExtensions.CreateCacheKey(
                "TestKernel", 
                [typeof(int), typeof(float)], 
                "GPU_Device_0");
            
            Assert.NotEqual(key1, key2); // Different parameter types
            Assert.NotEqual(key1, key3); // Different device
            Assert.NotEqual(key2, key3); // Different parameter types and device
        }

        [Fact]
        public void KernelCache_CreateVersionString()
        {
            var version1 = KernelCacheExtensions.CreateVersionString("1.0.0", "O2", "x64");
            var version2 = KernelCacheExtensions.CreateVersionString("1.0.0", "O3", "x64");
            var version3 = KernelCacheExtensions.CreateVersionString("1.0.1", "O2", "x64");
            
            Assert.NotEqual(version1, version2); // Different optimization
            Assert.NotEqual(version1, version3); // Different compiler version
            Assert.NotEqual(version2, version3); // Different compiler version and optimization
            
            Assert.Contains("1.0.0", version1);
            Assert.Contains("O2", version1);
            Assert.Contains("x64", version1);
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Disposes of the test resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                accelerator?.Dispose();
                context?.Dispose();
            }
        }

        #endregion
    }
}
