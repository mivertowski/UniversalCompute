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
using ILGPU.Runtime.MemoryPooling;
using ILGPU.Runtime.CPU;
using System;
using System.Threading.Tasks;
using Xunit;

namespace ILGPU.Tests.CPU
{
#pragma warning disable CA1515 // Consider making public types internal
    public class MemoryPoolingTests : IDisposable
#pragma warning restore CA1515 // Consider making public types internal
    {
        private readonly Context context;
        private readonly Accelerator accelerator;
        
        public MemoryPoolingTests()
        {
            context = Context.Create(builder => builder.CPU());
            accelerator = context.CreateCPUAccelerator(0);
        }
        
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

        [Fact]
        public void MemoryPool_Basic_RentAndReturn()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            // Rent a buffer
            var buffer = pool.Rent(1000);
            Assert.NotNull(buffer);
            Assert.True(buffer.Length >= 1000);
            Assert.False(buffer.IsReturned);

            // Return the buffer
            pool.ReturnBuffer(buffer);
            Assert.True(buffer.IsReturned);
        }

        [Fact]
        public void MemoryPool_ReuseBuffer()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            // Rent and return a buffer
            var buffer1 = pool.Rent(1000);
            var originalPtr = buffer1.NativePtr;
            pool.ReturnBuffer(buffer1);

            // Rent again - should get the same buffer back
            var buffer2 = pool.Rent(1000);
            Assert.Equal(originalPtr, buffer2.NativePtr);

            pool.ReturnBuffer(buffer2);
        }

        [Fact]
        public void MemoryPool_Statistics()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            var buffer1 = pool.Rent(1000);
            var buffer2 = pool.Rent(2000);

            var stats = pool.GetStatistics();
            Assert.Equal(2, stats.RentedBuffers);
            Assert.Equal(2, stats.TotalAllocations);
            Assert.True(stats.UsedSizeBytes > 0);

            pool.ReturnBuffer(buffer1);
            pool.ReturnBuffer(buffer2);

            stats = pool.GetStatistics();
            Assert.Equal(0, stats.RentedBuffers);
            Assert.Equal(2, stats.AvailableBuffers);
        }

        [Fact]
        public void MemoryPool_SizeBuckets()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            // Request slightly different sizes - should get bucketed
            var buffer1 = pool.Rent(1000);
            var buffer2 = pool.Rent(1100);

            // Both should be same bucket size (rounded up)
            Assert.Equal(buffer1.ActualLength, buffer2.ActualLength);

            pool.ReturnBuffer(buffer1);
            pool.ReturnBuffer(buffer2);
        }

        [Fact]
        public void MemoryPool_MaxSizeLimit()
        {
            var config = new MemoryPoolConfiguration
            {
                MaxBufferSizeBytes = 1024 * sizeof(int)
            };
            using var pool = new AdaptiveMemoryPool<int>(accelerator, config);

            // This should work
            var buffer1 = pool.Rent(1024);
            Assert.NotNull(buffer1);

            // This should throw
            Assert.Throws<ArgumentException>(() => pool.Rent(2048));

            pool.ReturnBuffer(buffer1);
        }

        [Fact]
        public void MemoryPool_ConfigurationValidation()
        {
            var config = new MemoryPoolConfiguration();
            
            // Valid configuration
            config.Validate(); // Should not throw

            // Invalid configurations
            config.MaxPoolSizeBytes = -1;
            Assert.Throws<ArgumentOutOfRangeException>(() => config.Validate());

            config.MaxPoolSizeBytes = 1000;
            config.MaxBufferSizeBytes = 2000;
            Assert.Throws<ArgumentException>(() => config.Validate());

            config.MaxBufferSizeBytes = 500;
            config.AllocationAlignment = 3; // Not power of 2
            Assert.Throws<ArgumentException>(() => config.Validate());
        }

        [Fact]
        public void MemoryPool_PresetConfigurations()
        {
            // Test all preset configurations
            var presets = new[]
            {
                MemoryPoolConfiguration.CreateHighPerformance(),
                MemoryPoolConfiguration.CreateMemoryEfficient(),
                MemoryPoolConfiguration.CreateDevelopment()
            };

            foreach (var config in presets)
            {
                config.Validate(); // Should not throw
                using var pool = new AdaptiveMemoryPool<int>(accelerator, config);
                
                var buffer = pool.Rent(1000);
                Assert.NotNull(buffer);
                pool.ReturnBuffer(buffer);
            }
        }

        [Fact]
        public void MemoryPool_Trim()
        {
            var config = new MemoryPoolConfiguration
            {
                RetentionPolicy = PoolRetentionPolicy.Aggressive,
                EnableStatistics = true
            };
            using var pool = new AdaptiveMemoryPool<int>(accelerator, config);

            // Create and return several buffers
            for (int i = 0; i < 5; i++)
            {
                var buffer = pool.Rent(1000);
                pool.ReturnBuffer(buffer);
            }

            var statsBefore = pool.GetStatistics();
            Assert.True(statsBefore.AvailableBuffers > 0);

            // Trim should reduce available buffers
            pool.Trim();

            var statsAfter = pool.GetStatistics();
            Assert.True(statsAfter.AvailableBuffers <= statsBefore.AvailableBuffers);
        }

        [Fact]
        public async Task MemoryPool_AsyncRent()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            var buffer = await pool.RentAsync(1000);
            Assert.NotNull(buffer);
            Assert.True(buffer.Length >= 1000);

            pool.ReturnBuffer(buffer);
        }

        [Fact]
        public void MemoryPool_PooledBufferDispose()
        {
            using var pool = new AdaptiveMemoryPool<int>(accelerator);

            IPooledMemoryBuffer<int> buffer;
            using (buffer = pool.Rent(1000))
            {
                Assert.False(buffer.IsReturned);
            }
            
            // Should be automatically returned on dispose
            Assert.True(buffer.IsReturned);
        }

        [Fact]
        public void MemoryPool_FactoryPattern()
        {
            var factory = new DefaultMemoryPoolFactory();

            using var pool = factory.CreatePool<int>(accelerator);
            Assert.NotNull(pool);

            var buffer = pool.Rent(1000);
            Assert.NotNull(buffer);
            
            pool.ReturnBuffer(buffer);
        }
    }
}
