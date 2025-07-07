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

namespace ILGPU.CrossPlatform
{
    /// <summary>
    /// Provides universal access to grid and thread indices across all ILGPU backends.
    /// This class abstracts the differences between CUDA, OpenCL, and other threading models.
    /// </summary>
    public static class Grid
    {
        /// <summary>
        /// Gets the global thread index for the current kernel execution.
        /// This works consistently across all ILGPU backends.
        /// </summary>
        public static Index2D GlobalIndex =>
                // This is automatically translated by the ILGPU compiler
                // to the appropriate backend-specific index calculation
                Group.IdxXY * Group.DimXY + Group.IdxXY;

        /// <summary>
        /// Gets the global thread index as a 1D value.
        /// </summary>
        public static Index1D GlobalIndex1D
        {
            get
            {
                var idx2D = GlobalIndex;
                return idx2D.Y * GridDimension.X + idx2D.X;
            }
        }

        /// <summary>
        /// Gets the global thread index as a 3D value.
        /// </summary>
        public static Index3D GlobalIndex3D
        {
            get
            {
                var idx2D = GlobalIndex;
                return new Index3D(idx2D.X, idx2D.Y, 0);
            }
        }

        /// <summary>
        /// Gets the dimension of the current grid.
        /// </summary>
        public static Index2D GridDimension =>
                // Automatically translated by ILGPU compiler
                Group.DimXY;

        /// <summary>
        /// Gets the total number of threads in the grid.
        /// </summary>
        public static int TotalThreads => GridDimension.Size;

        /// <summary>
        /// Gets the local thread index within the current work group/block.
        /// </summary>
        public static Index2D LocalIndex => Group.IdxXY;

        /// <summary>
        /// Gets the work group/block index.
        /// </summary>
        public static Index2D GroupIndex => ILGPU.Grid.IdxXY;

        /// <summary>
        /// Synchronizes all threads in the current work group/block.
        /// This is automatically translated to the appropriate barrier instruction
        /// for each backend (CUDA __syncthreads(), OpenCL barrier(), etc.).
        /// </summary>
        public static void SynchronizeGroup() => Group.Barrier();

        /// <summary>
        /// Checks if the current thread is the first thread in its work group/block.
        /// </summary>
        public static bool IsFirstThreadInGroup => LocalIndex.X == 0 && LocalIndex.Y == 0;

        /// <summary>
        /// Checks if the current thread is the last thread in its work group/block.
        /// </summary>
        public static bool IsLastThreadInGroup
        {
            get
            {
                var local = LocalIndex;
                var dim = Group.DimXY;
                return local.X == dim.X - 1 && local.Y == dim.Y - 1;
            }
        }

        /// <summary>
        /// Gets the work group/block size.
        /// </summary>
        public static Index2D GroupSize => Group.DimXY;

        /// <summary>
        /// Checks if the current global index is within the specified bounds.
        /// This is useful for bounds checking in kernels with irregular problem sizes.
        /// </summary>
        /// <param name="bounds">The bounds to check against.</param>
        /// <returns>True if the current thread is within bounds.</returns>
        public static bool IsWithinBounds(Index2D bounds)
        {
            var global = GlobalIndex;
            return global.X < bounds.X && global.Y < bounds.Y;
        }

        /// <summary>
        /// Checks if the current global index is within the specified 1D bounds.
        /// </summary>
        /// <param name="bound">The 1D bound to check against.</param>
        /// <returns>True if the current thread is within bounds.</returns>
        public static bool IsWithinBounds(int bound) => GlobalIndex1D < bound;
    }
}