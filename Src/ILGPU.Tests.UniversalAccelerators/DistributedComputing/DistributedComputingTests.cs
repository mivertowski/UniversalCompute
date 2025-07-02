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

using ILGPU.Algorithms.DistributedComputing;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.DistributedComputing
{
    /// <summary>
    /// Tests for distributed computing algorithms.
    /// </summary>
    public class DistributedComputingTests : TestBase
    {
        #region MPI Tests

        [Fact]
        public void TestMPIInitialization()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test MPI initialization (simulated)
            var mpiContext = new MPIContext();
            Assert.NotNull(mpiContext);
            
            // Verify basic properties
            Assert.True(mpiContext.Rank >= 0);
            Assert.True(mpiContext.Size > 0);
        }

        [Fact]
        public void TestMPIAllReduce()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 100;
            var mpiContext = new MPIContext();
            var data = CreateTestData(size);
            
            using var buffer = accelerator!.Allocate1D(data);
            
            // Simulate all-reduce operation
            MPIOperations.AllReduce(buffer.View, MPIOperation.Sum, mpiContext, accelerator!.DefaultStream);
            
            var result = buffer.GetAsArray1D();
            Assert.Equal(size, result.Length);
            
            // In single-node test, result should equal input
            AssertEqual(data, result);
        }

        [Fact]
        public void TestMPIBroadcast()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 50;
            var mpiContext = new MPIContext();
            var data = CreateTestData(size);
            
            using var buffer = accelerator!.Allocate1D(data);
            
            // Broadcast from rank 0
            MPIOperations.Broadcast(buffer.View, 0, mpiContext, accelerator!.DefaultStream);
            
            var result = buffer.GetAsArray1D();
            Assert.Equal(size, result.Length);
            
            // Data should remain unchanged in single-node test
            AssertEqual(data, result);
        }

        [Fact]
        public void TestMPIScatterGather()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int totalSize = 16;
            var mpiContext = new MPIContext();
            var data = CreateSequentialData(totalSize);
            
            using var sendBuffer = accelerator!.Allocate1D(data);
            using var recvBuffer = accelerator!.Allocate1D<float>(totalSize);
            
            // Scatter and then gather
            MPIOperations.Scatter(sendBuffer.View, recvBuffer.View, 0, mpiContext, accelerator!.DefaultStream);
            MPIOperations.Gather(recvBuffer.View, sendBuffer.View, 0, mpiContext, accelerator!.DefaultStream);
            
            var result = sendBuffer.GetAsArray1D();
            AssertEqual(data, result);
        }

        #endregion

        #region Parallel Patterns Tests

        [Fact]
        public void TestMapReduce()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 1000;
            var data = CreateSequentialData(size);
            
            using var inputBuffer = accelerator!.Allocate1D(data);
            using var outputBuffer = accelerator!.Allocate1D<float>(1);
            
            // Map-reduce: square each element and sum
            var mapReduceOp = new MapReduceOperation<float, float>(
                x => x * x,  // Map: square
                (a, b) => a + b,  // Reduce: sum
                0.0f);  // Initial value
            
            ParallelPatterns.MapReduce(inputBuffer.View, outputBuffer.View, mapReduceOp, accelerator!.DefaultStream);
            
            var result = outputBuffer.GetAsArray1D();
            
            // Expected: sum of squares from 0 to 999
            var expected = data.Select(x => x * x).Sum();
            Assert.True(Math.Abs(result[0] - expected) < 1e-5f, 
                $"Expected {expected}, got {result[0]}");
        }

        [Fact]
        public void TestPipeline()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 100;
            var data = CreateTestData(size);
            
            using var stage1Buffer = accelerator!.Allocate1D(data);
            using var stage2Buffer = accelerator!.Allocate1D<float>(size);
            using var stage3Buffer = accelerator!.Allocate1D<float>(size);
            
            // Three-stage pipeline: multiply by 2, add 1, square
            var pipeline = new Pipeline<float>();
            pipeline.AddStage(x => x * 2.0f);
            pipeline.AddStage(x => x + 1.0f);
            pipeline.AddStage(x => x * x);
            
            ParallelPatterns.ExecutePipeline(
                stage1Buffer.View, 
                new[] { stage2Buffer.View, stage3Buffer.View }, 
                pipeline, 
                accelerator!.DefaultStream);
            
            var result = stage3Buffer.GetAsArray1D();
            
            // Verify pipeline transformation
            for (int i = 0; i < size; i++)
            {
                var expected = (data[i] * 2.0f + 1.0f);
                expected *= expected;
                Assert.True(Math.Abs(result[i] - expected) < 1e-5f,
                    $"Pipeline error at index {i}: expected {expected}, got {result[i]}");
            }
        }

        [Fact]
        public void TestWorkStealing()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numTasks = 1000;
            var tasks = new ComputeTask[numTasks];
            
            // Create tasks with varying computational complexity
            for (int i = 0; i < numTasks; i++)
            {
                tasks[i] = new ComputeTask
                {
                    Id = i,
                    Complexity = i % 10 + 1,  // Varying complexity 1-10
                    Data = CreateTestData(i % 100 + 10)
                };
            }
            
            var workStealingScheduler = new WorkStealingScheduler(accelerator!);
            var results = new float[numTasks];
            
            // Execute tasks with work stealing
            var executionTime = MeasureTime(() =>
            {
                workStealingScheduler.Execute(tasks, results);
            });
            
            // Verify all tasks completed
            for (int i = 0; i < numTasks; i++)
            {
                Assert.True(results[i] > 0, $"Task {i} was not executed");
            }
            
            Assert.True(executionTime < 5000, $"Work stealing took {executionTime}ms, expected < 5000ms");
        }

        #endregion

        #region Load Balancing Tests

        [Fact]
        public void TestDynamicLoadBalancing()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numWorkers = 4;
            const int numTasks = 100;
            
            var loadBalancer = new DynamicLoadBalancer(numWorkers);
            var tasks = new WorkItem[numTasks];
            
            // Create tasks with random complexity
            var random = new Random(42);
            for (int i = 0; i < numTasks; i++)
            {
                tasks[i] = new WorkItem
                {
                    Id = i,
                    EstimatedCost = random.Next(1, 10),
                    Data = CreateTestData(random.Next(10, 100))
                };
            }
            
            // Execute with load balancing
            var completedTasks = new bool[numTasks];
            var executionTime = MeasureTime(() =>
            {
                loadBalancer.ExecuteTasks(tasks, completedTasks, accelerator!);
            });
            
            // Verify all tasks completed
            Assert.True(completedTasks.All(completed => completed), 
                "Not all tasks were completed");
            
            // Verify reasonable load distribution
            var workerLoads = loadBalancer.GetWorkerLoads();
            var maxLoad = workerLoads.Max();
            var minLoad = workerLoads.Min();
            var loadImbalance = (maxLoad - minLoad) / maxLoad;
            
            Assert.True(loadImbalance < 0.5, 
                $"Load imbalance too high: {loadImbalance:F2}");
        }

        [Fact]
        public void TestAdaptivePartitioning()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int dataSize = 10000;
            var data = CreateTestData(dataSize);
            
            using var inputBuffer = accelerator!.Allocate1D(data);
            using var outputBuffer = accelerator!.Allocate1D<float>(dataSize);
            
            var partitioner = new AdaptivePartitioner(accelerator!);
            
            // Adaptive computation: complex function with varying cost
            var complexFunction = new Func<float, float>(x => 
            {
                float result = x;
                // Variable complexity based on value
                int iterations = (int)(Math.Abs(x) % 10) + 1;
                for (int i = 0; i < iterations; i++)
                {
                    result = (float)Math.Sin(result) + (float)Math.Cos(result * 0.5f);
                }
                return result;
            });
            
            var executionTime = MeasureTime(() =>
            {
                partitioner.ExecuteAdaptive(
                    inputBuffer.View, 
                    outputBuffer.View, 
                    complexFunction,
                    accelerator!.DefaultStream);
            });
            
            var result = outputBuffer.GetAsArray1D();
            
            // Verify computation completed
            Assert.True(result.All(x => !float.IsNaN(x) && !float.IsInfinity(x)),
                "Adaptive computation produced invalid values");
            
            Assert.True(executionTime < 10000, 
                $"Adaptive partitioning took {executionTime}ms, expected < 10000ms");
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestDistributedPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int largeSize = 100000;
            var data = CreateTestData(largeSize);
            var mpiContext = new MPIContext();
            
            using var buffer = accelerator!.Allocate1D(data);
            
            // Measure collective communication performance
            var allReduceTime = MeasureTime(() =>
            {
                MPIOperations.AllReduce(buffer.View, MPIOperation.Sum, mpiContext, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            var broadcastTime = MeasureTime(() =>
            {
                MPIOperations.Broadcast(buffer.View, 0, mpiContext, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            // Verify reasonable performance
            Assert.True(allReduceTime < 1000, 
                $"AllReduce too slow: {allReduceTime}ms for {largeSize} elements");
            Assert.True(broadcastTime < 500, 
                $"Broadcast too slow: {broadcastTime}ms for {largeSize} elements");
            
            // Verify correctness
            var result = buffer.GetAsArray1D();
            Assert.Equal(largeSize, result.Length);
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestMPIErrorHandling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var mpiContext = new MPIContext();
            
            // Test invalid rank for broadcast
            using var buffer = accelerator!.Allocate1D<float>(10);
            
            Assert.Throws<ArgumentException>(() =>
            {
                MPIOperations.Broadcast(buffer.View, -1, mpiContext, accelerator!.DefaultStream);
            });
            
            Assert.Throws<ArgumentException>(() =>
            {
                MPIOperations.Broadcast(buffer.View, mpiContext.Size, mpiContext, accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestLoadBalancerErrorHandling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test with zero workers
            Assert.Throws<ArgumentException>(() =>
            {
                new DynamicLoadBalancer(0);
            });
            
            // Test with null tasks
            var loadBalancer = new DynamicLoadBalancer(2);
            Assert.Throws<ArgumentNullException>(() =>
            {
                loadBalancer.ExecuteTasks(null!, new bool[0], accelerator!);
            });
        }

        #endregion
    }
}