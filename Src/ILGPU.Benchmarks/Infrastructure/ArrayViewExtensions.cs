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
using System.Runtime.CompilerServices;

namespace ILGPU.Benchmarks.Infrastructure;

/// <summary>
/// Extension methods for ILGPU ArrayView types to provide additional functionality.
/// </summary>
public static class ArrayViewExtensions
{
    /// <summary>
    /// Converts a 2D array view to a 1D linear view.
    /// This method provides a flat view of the 2D array data in row-major order.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <param name="view">The 2D array view to convert.</param>
    /// <returns>A 1D array view representing the same data in linear format.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayView<T> AsLinearView<T>(this ArrayView2D<T, Stride2D.DenseX> view)
        where T : unmanaged
    {
        // For DenseX stride, data is stored in row-major order contiguously
        // Use the existing AsContiguous method to get a 1D view
        return view.AsContiguous();
    }

    /// <summary>
    /// Converts a 3D array view to a 1D linear view.
    /// This method provides a flat view of the 3D array data in row-major order.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <param name="view">The 3D array view to convert.</param>
    /// <returns>A 1D array view representing the same data in linear format.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayView<T> AsLinearView<T>(this ArrayView3D<T, Stride3D.DenseXY> view)
        where T : unmanaged
    {
        // For DenseXY stride, data is stored in row-major order contiguously
        // Use the existing AsContiguous method to get a 1D view
        return view.AsContiguous();
    }

    /// <summary>
    /// Converts a general 2D array view to a 1D linear view by copying elements.
    /// This method works with any stride type but may be slower for non-dense strides.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <typeparam name="TStride">The stride type of the 2D array view.</typeparam>
    /// <param name="view">The 2D array view to convert.</param>
    /// <param name="accelerator">The accelerator to use for memory operations.</param>
    /// <returns>A 1D memory buffer containing the linearized data.</returns>
    public static MemoryBuffer1D<T, Stride1D.Dense> ToLinearBuffer<T, TStride>(
        this ArrayView2D<T, TStride> view,
        Accelerator accelerator)
        where T : unmanaged
        where TStride : struct, IStride2D
    {
        var totalElements = view.IntExtent.X * view.IntExtent.Y;
        var linearBuffer = accelerator.Allocate1D<T>(totalElements);
        
        // Use a kernel to copy data in row-major order
        var copyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<T, TStride>, ArrayView<T>>(CopyToLinearKernel);
        
        copyKernel(view.IntExtent, view, linearBuffer.View);
        accelerator.Synchronize();
        
        return linearBuffer;
    }

    /// <summary>
    /// Kernel to copy 2D array data to 1D linear format.
    /// </summary>
    private static void CopyToLinearKernel<T, TStride>(
        Index2D index,
        ArrayView2D<T, TStride> source,
        ArrayView<T> destination)
        where T : unmanaged
        where TStride : struct, IStride2D
    {
        if (index.X >= source.IntExtent.X || index.Y >= source.IntExtent.Y)
        {
            return;
        }

        var linearIndex = index.Y * source.IntExtent.X + index.X;
        if (linearIndex < destination.Length)
        {
            destination[linearIndex] = source[index];
        }
    }

    /// <summary>
    /// Creates a safe sub-view of a 2D array with bounds checking.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <typeparam name="TStride">The stride type of the array view.</typeparam>
    /// <param name="view">The source 2D array view.</param>
    /// <param name="offset">The offset position for the sub-view.</param>
    /// <param name="extent">The extent (dimensions) of the sub-view.</param>
    /// <returns>A sub-view of the original array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayView2D<T, TStride> SafeSubView<T, TStride>(
        this ArrayView2D<T, TStride> view,
        Index2D offset,
        Index2D extent)
        where T : unmanaged
        where TStride : struct, IStride2D
    {
        // Clamp extent to ensure it doesn't exceed bounds
        var maxExtentX = Math.Max(0, view.IntExtent.X - offset.X);
        var maxExtentY = Math.Max(0, view.IntExtent.Y - offset.Y);
        
        var safeExtent = new Index2D(
            Math.Min(extent.X, maxExtentX),
            Math.Min(extent.Y, maxExtentY));
            
        return view.SubView(offset, safeExtent);
    }

    /// <summary>
    /// Calculates the memory size in bytes for an array view.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <param name="view">The array view to calculate size for.</param>
    /// <returns>The memory size in bytes.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long GetMemorySize<T>(this ArrayView<T> view)
        where T : unmanaged
    {
        return view.LengthInBytes;
    }

    /// <summary>
    /// Calculates the memory size in bytes for a 2D array view.
    /// </summary>
    /// <typeparam name="T">The element type of the array view.</typeparam>
    /// <typeparam name="TStride">The stride type of the array view.</typeparam>
    /// <param name="view">The 2D array view to calculate size for.</param>
    /// <returns>The memory size in bytes.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long GetMemorySize<T, TStride>(this ArrayView2D<T, TStride> view)
        where T : unmanaged
        where TStride : struct, IStride2D
    {
        return view.LengthInBytes;
    }
}