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

namespace ILGPU.AI.Memory
{
    /// <summary>
    /// Memory optimization strategies for AI workloads.
    /// </summary>
    public enum OptimizationStrategy
    {
        /// <summary>
        /// Optimize for maximum throughput.
        /// </summary>
        Throughput,

        /// <summary>
        /// Optimize for minimum latency.
        /// </summary>
        Latency,

        /// <summary>
        /// Optimize for minimum memory usage.
        /// </summary>
        MemoryEfficient,

        /// <summary>
        /// Optimize for balanced performance.
        /// </summary>
        Balanced,

        /// <summary>
        /// Optimize for training workloads.
        /// </summary>
        Training,

        /// <summary>
        /// Optimize for inference workloads.
        /// </summary>
        Inference
    }

    /// <summary>
    /// Memory usage statistics for optimization analysis.
    /// </summary>
    public struct MemoryStats
    {
        /// <summary>
        /// Total memory allocated in bytes.
        /// </summary>
        public long TotalAllocated;

        /// <summary>
        /// Peak memory usage in bytes.
        /// </summary>
        public long PeakUsage;

        /// <summary>
        /// Current memory usage in bytes.
        /// </summary>
        public long CurrentUsage;

        /// <summary>
        /// Number of memory allocations.
        /// </summary>
        public int AllocationCount;

        /// <summary>
        /// Number of memory deallocations.
        /// </summary>
        public int DeallocationCount;

        /// <summary>
        /// Average allocation size in bytes.
        /// </summary>
        public double AverageAllocationSize;

        /// <summary>
        /// Memory fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented).
        /// </summary>
        public float FragmentationRatio;

        /// <summary>
        /// Cache hit ratio (0.0 = no hits, 1.0 = all hits).
        /// </summary>
        public float CacheHitRatio;
    }

    /// <summary>
    /// Memory optimization recommendation.
    /// </summary>
    public struct OptimizationRecommendation
    {
        /// <summary>
        /// Recommended memory layout.
        /// </summary>
        public AIMemoryLayout Layout;

        /// <summary>
        /// Recommended access pattern.
        /// </summary>
        public AIAccessPattern AccessPattern;

        /// <summary>
        /// Recommended tile size.
        /// </summary>
        public (int Height, int Width) TileSize;

        /// <summary>
        /// Recommended block size.
        /// </summary>
        public int BlockSize;

        /// <summary>
        /// Recommended memory alignment.
        /// </summary>
        public int Alignment;

        /// <summary>
        /// Whether to enable prefetching.
        /// </summary>
        public bool EnablePrefetching;

        /// <summary>
        /// Expected performance improvement ratio.
        /// </summary>
        public float ExpectedImprovement;

        /// <summary>
        /// Confidence level of the recommendation (0.0 to 1.0).
        /// </summary>
        public float Confidence;
    }

    /// <summary>
    /// AI memory optimizer for automatic performance tuning.
    /// </summary>
    public sealed class AIMemoryOptimizer
    {
        private readonly Accelerator _accelerator;
        private readonly Dictionary<string, MemoryStats> _operationStats;
        private readonly Dictionary<string, OptimizationRecommendation> _recommendations;
        private long _totalMemoryBudget;
        private float _memoryUtilizationTarget;

        /// <summary>
        /// Initializes a new AI memory optimizer.
        /// </summary>
        /// <param name="accelerator">The accelerator to optimize for.</param>
        /// <param name="memoryBudget">Total memory budget in bytes.</param>
        public AIMemoryOptimizer(Accelerator accelerator, long memoryBudget = 0)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _totalMemoryBudget = memoryBudget > 0 ? memoryBudget : _accelerator.MemorySize;
            _memoryUtilizationTarget = 0.8f; // Target 80% utilization
            _operationStats = new Dictionary<string, MemoryStats>();
            _recommendations = new Dictionary<string, OptimizationRecommendation>();
        }

        /// <summary>
        /// Gets the current memory utilization target.
        /// </summary>
        public float MemoryUtilizationTarget
        {
            get => _memoryUtilizationTarget;
            set => _memoryUtilizationTarget = Math.Clamp(value, 0.1f, 0.95f);
        }

        /// <summary>
        /// Analyzes memory usage patterns and generates optimization recommendations.
        /// </summary>
        /// <param name="operationType">Type of operation being analyzed.</param>
        /// <param name="workloadSize">Size of the workload in elements.</param>
        /// <param name="dataTypes">Data types used in the operation.</param>
        /// <param name="strategy">Optimization strategy to use.</param>
        /// <returns>Optimization recommendation.</returns>
        public OptimizationRecommendation AnalyzeAndOptimize(
            string operationType,
            long workloadSize,
            Type[] dataTypes,
            OptimizationStrategy strategy = OptimizationStrategy.Balanced)
        {
            if (string.IsNullOrEmpty(operationType))
                throw new ArgumentException("Operation type cannot be null or empty", nameof(operationType));

            // Calculate memory requirements
            var totalMemoryRequired = CalculateMemoryRequirements(workloadSize, dataTypes);
            
            // Analyze access patterns for the operation type
            var accessPattern = DetermineOptimalAccessPattern(operationType, workloadSize);
            
            // Determine optimal layout based on operation and strategy
            var layout = DetermineOptimalLayout(operationType, strategy, workloadSize);
            
            // Calculate optimal tile and block sizes
            var (tileHeight, tileWidth, blockSize) = CalculateOptimalSizes(operationType, workloadSize, totalMemoryRequired);
            
            // Determine memory alignment
            var alignment = CalculateOptimalAlignment(dataTypes, accessPattern);
            
            // Decide on prefetching
            var enablePrefetching = ShouldEnablePrefetching(operationType, accessPattern, strategy);
            
            // Estimate performance improvement
            var expectedImprovement = EstimatePerformanceImprovement(
                operationType, layout, accessPattern, (tileHeight, tileWidth), blockSize);
            
            // Calculate confidence based on available data
            var confidence = CalculateConfidence(operationType, workloadSize);

            var recommendation = new OptimizationRecommendation
            {
                Layout = layout,
                AccessPattern = accessPattern,
                TileSize = (tileHeight, tileWidth),
                BlockSize = blockSize,
                Alignment = alignment,
                EnablePrefetching = enablePrefetching,
                ExpectedImprovement = expectedImprovement,
                Confidence = confidence
            };

            // Cache the recommendation
            _recommendations[operationType] = recommendation;
            
            return recommendation;
        }

        /// <summary>
        /// Creates an optimized memory configuration for the given operation.
        /// </summary>
        /// <param name="operationType">Type of operation.</param>
        /// <param name="strategy">Optimization strategy.</param>
        /// <returns>Optimized memory configuration.</returns>
        public AIMemoryConfig CreateOptimizedConfig(
            string operationType,
            OptimizationStrategy strategy = OptimizationStrategy.Balanced)
        {
            if (_recommendations.TryGetValue(operationType, out var recommendation))
            {
                return new AIMemoryConfig
                {
                    Layout = recommendation.Layout,
                    AccessPattern = recommendation.AccessPattern,
                    TileSize = recommendation.TileSize,
                    BlockSize = recommendation.BlockSize,
                    EnablePrefetching = recommendation.EnablePrefetching,
                    AlignMemory = true,
                    MemoryAlignment = recommendation.Alignment
                };
            }

            // Return default config if no specific recommendation
            return strategy switch
            {
                OptimizationStrategy.Throughput => AIMemoryConfig.ForMatrixOperations(),
                OptimizationStrategy.Latency => AIMemoryConfig.ForConvolution(),
                OptimizationStrategy.Training => AIMemoryConfig.ForMatrixOperations(),
                OptimizationStrategy.Inference => AIMemoryConfig.ForConvolution(),
                _ => AIMemoryConfig.Default
            };
        }

        /// <summary>
        /// Updates memory statistics for performance tracking.
        /// </summary>
        /// <param name="operationType">Type of operation.</param>
        /// <param name="stats">Memory statistics.</param>
        public void UpdateStats(string operationType, MemoryStats stats)
        {
            _operationStats[operationType] = stats;
        }

        /// <summary>
        /// Gets memory statistics for the specified operation.
        /// </summary>
        /// <param name="operationType">Type of operation.</param>
        /// <returns>Memory statistics if available.</returns>
        public MemoryStats? GetStats(string operationType)
        {
            return _operationStats.TryGetValue(operationType, out var stats) ? stats : null;
        }

        /// <summary>
        /// Optimizes memory layout for multiple operations simultaneously.
        /// </summary>
        /// <param name="operations">Operations to optimize.</param>
        /// <param name="priorities">Priority weights for each operation.</param>
        /// <param name="strategy">Overall optimization strategy.</param>
        /// <returns>Global optimization recommendations.</returns>
        public Dictionary<string, OptimizationRecommendation> OptimizeGlobally(
            string[] operations,
            float[] priorities,
            OptimizationStrategy strategy = OptimizationStrategy.Balanced)
        {
            if (operations.Length != priorities.Length)
                throw new ArgumentException("Operations and priorities arrays must have the same length");

            var globalRecommendations = new Dictionary<string, OptimizationRecommendation>();
            
            // Calculate total priority weight
            var totalPriority = priorities.Sum();
            
            // Normalize priorities
            var normalizedPriorities = priorities.Select(p => p / totalPriority).ToArray();
            
            // Generate recommendations considering global constraints
            for (int i = 0; i < operations.Length; i++)
            {
                var operation = operations[i];
                var priority = normalizedPriorities[i];
                
                // Adjust strategy based on priority
                var adjustedStrategy = AdjustStrategyByPriority(strategy, priority);
                
                // Generate recommendation (simplified - would consider memory conflicts)
                var recommendation = AnalyzeAndOptimize(operation, 1000000, new[] { typeof(float) }, adjustedStrategy);
                
                // Adjust recommendation based on global constraints
                recommendation = AdjustForGlobalConstraints(recommendation, priority, _totalMemoryBudget);
                
                globalRecommendations[operation] = recommendation;
            }
            
            return globalRecommendations;
        }

        #region Private Methods

        private long CalculateMemoryRequirements(long workloadSize, Type[] dataTypes)
        {
            long totalSize = 0;
            foreach (var type in dataTypes)
            {
                var elementSize = type == typeof(float) ? 4 :
                                type == typeof(double) ? 8 :
                                type == typeof(half) ? 2 :
                                type == typeof(int) ? 4 :
                                type == typeof(byte) ? 1 : 4;
                
                totalSize += workloadSize * elementSize;
            }
            return totalSize;
        }

        private AIAccessPattern DetermineOptimalAccessPattern(string operationType, long workloadSize)
        {
            return operationType.ToLowerInvariant() switch
            {
                "convolution" => AIAccessPattern.Tiled,
                "matmul" => AIAccessPattern.Blocked,
                "attention" => AIAccessPattern.Strided,
                "elementwise" => AIAccessPattern.Sequential,
                "reduction" => AIAccessPattern.Streaming,
                "sparse" => AIAccessPattern.GatherScatter,
                _ => workloadSize > 1000000 ? AIAccessPattern.Blocked : AIAccessPattern.Sequential
            };
        }

        private AIMemoryLayout DetermineOptimalLayout(string operationType, OptimizationStrategy strategy, long workloadSize)
        {
            // Base layout on operation type
            var baseLayout = operationType.ToLowerInvariant() switch
            {
                "convolution" => AIMemoryLayout.Tiled,
                "matmul" => AIMemoryLayout.Blocked,
                "attention" => AIMemoryLayout.RowMajor,
                "sparse" => AIMemoryLayout.Sparse,
                _ => AIMemoryLayout.RowMajor
            };

            // Adjust based on strategy
            return strategy switch
            {
                OptimizationStrategy.MemoryEfficient => AIMemoryLayout.Packed,
                OptimizationStrategy.Latency when workloadSize < 100000 => AIMemoryLayout.Interleaved,
                OptimizationStrategy.Throughput => AIMemoryLayout.Blocked,
                _ => baseLayout
            };
        }

        private (int Height, int Width) CalculateOptimalTileSize(string operationType, long workloadSize)
        {
            var cacheSize = 64 * 1024; // Assume 64KB L1 cache
            var elementSize = 4; // Float32
            
            var maxTileElements = cacheSize / (3 * elementSize); // Account for input, output, and temporaries
            var tileSize = (int)Math.Sqrt(maxTileElements);
            
            // Clamp to reasonable bounds and make power of 2
            tileSize = Math.Max(8, Math.Min(64, tileSize));
            tileSize = 1 << (int)Math.Log2(tileSize);
            
            return operationType.ToLowerInvariant() switch
            {
                "convolution" => (Math.Min(tileSize, 16), Math.Min(tileSize, 16)),
                "matmul" => (tileSize, tileSize),
                "attention" => (Math.Min(tileSize, 32), Math.Min(tileSize, 64)),
                _ => (tileSize, tileSize)
            };
        }

        private (int tileHeight, int tileWidth, int blockSize) CalculateOptimalSizes(
            string operationType, long workloadSize, long memoryRequired)
        {
            var (tileHeight, tileWidth) = CalculateOptimalTileSize(operationType, workloadSize);
            
            // Calculate block size based on cache hierarchy
            var l2CacheSize = 256 * 1024; // Assume 256KB L2 cache
            var blockSize = Math.Min(1024, (int)(l2CacheSize / (4 * sizeof(float))));
            blockSize = Math.Max(64, blockSize & ~63); // Align to 64-byte boundaries
            
            return (tileHeight, tileWidth, blockSize);
        }

        private int CalculateOptimalAlignment(Type[] dataTypes, AIAccessPattern accessPattern)
        {
            // Base alignment on largest data type
            var maxElementSize = dataTypes.Max(t => 
                t == typeof(double) ? 8 :
                t == typeof(float) || t == typeof(int) ? 4 :
                t == typeof(half) ? 2 : 1);
            
            // Increase alignment for certain access patterns
            var baseAlignment = accessPattern switch
            {
                AIAccessPattern.Sequential => 32,
                AIAccessPattern.Streaming => 64,
                AIAccessPattern.Tiled => 64,
                AIAccessPattern.Blocked => 32,
                _ => 16
            };
            
            return Math.Max(maxElementSize, baseAlignment);
        }

        private bool ShouldEnablePrefetching(string operationType, AIAccessPattern accessPattern, OptimizationStrategy strategy)
        {
            // Disable prefetching for random access or when optimizing for memory efficiency
            if (accessPattern == AIAccessPattern.Random || accessPattern == AIAccessPattern.GatherScatter)
                return false;
            
            if (strategy == OptimizationStrategy.MemoryEfficient)
                return false;
            
            // Enable for most other cases
            return true;
        }

        private float EstimatePerformanceImprovement(
            string operationType, AIMemoryLayout layout, AIAccessPattern accessPattern,
            (int Height, int Width) tileSize, int blockSize)
        {
            // Simplified heuristic based on layout and access pattern compatibility
            var baseImprovement = layout switch
            {
                AIMemoryLayout.Tiled when accessPattern == AIAccessPattern.Tiled => 1.5f,
                AIMemoryLayout.Blocked when accessPattern == AIAccessPattern.Blocked => 1.3f,
                AIMemoryLayout.Interleaved when accessPattern == AIAccessPattern.Sequential => 1.2f,
                AIMemoryLayout.Sparse when accessPattern == AIAccessPattern.GatherScatter => 2.0f,
                _ => 1.1f
            };
            
            // Adjust based on tile/block size optimality
            var sizeOptimality = (tileSize.Height * tileSize.Width) < 1024 ? 1.1f : 1.0f;
            var blockOptimality = (blockSize >= 64 && blockSize <= 256) ? 1.05f : 1.0f;
            
            return baseImprovement * sizeOptimality * blockOptimality;
        }

        private float CalculateConfidence(string operationType, long workloadSize)
        {
            // Higher confidence for well-known operation types and sizes
            var operationConfidence = operationType.ToLowerInvariant() switch
            {
                "convolution" or "matmul" or "attention" => 0.9f,
                "elementwise" or "reduction" => 0.8f,
                _ => 0.6f
            };
            
            // Higher confidence for typical workload sizes
            var sizeConfidence = workloadSize switch
            {
                >= 100000 and <= 10000000 => 0.9f,
                >= 10000 and < 100000 => 0.8f,
                > 10000000 => 0.7f,
                _ => 0.6f
            };
            
            return operationConfidence * sizeConfidence;
        }

        private OptimizationStrategy AdjustStrategyByPriority(OptimizationStrategy baseStrategy, float priority)
        {
            // High priority operations get latency optimization
            if (priority > 0.6f)
                return OptimizationStrategy.Latency;
            
            // Low priority operations get memory efficiency
            if (priority < 0.2f)
                return OptimizationStrategy.MemoryEfficient;
            
            return baseStrategy;
        }

        private OptimizationRecommendation AdjustForGlobalConstraints(
            OptimizationRecommendation recommendation, float priority, long memoryBudget)
        {
            // Reduce tile/block sizes for low priority operations to save memory
            if (priority < 0.3f)
            {
                recommendation.TileSize = (
                    Math.Max(8, recommendation.TileSize.Height / 2),
                    Math.Max(8, recommendation.TileSize.Width / 2)
                );
                recommendation.BlockSize = Math.Max(32, recommendation.BlockSize / 2);
            }
            
            return recommendation;
        }

        #endregion
    }
}