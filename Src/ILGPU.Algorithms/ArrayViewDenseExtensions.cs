// ---------------------------------------------------------------------------------------
//                                   ILGPU Algorithms
//                        Copyright (c) 2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: ArrayViewDenseExtensions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace ILGPU.Algorithms
{
    /// <summary>
    /// Extension methods for ArrayView1D with Dense stride.
    /// </summary>
    public static class ArrayViewDenseExtensions
    {
        /// <summary>
        /// Copies the contents of this view into the target view.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source view.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="target">The target view.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyTo<T>(
            this ArrayView1D<T, Stride1D.Dense> source,
            AcceleratorStream stream,
            ArrayView1D<T, Stride1D.Dense> target)
            where T : unmanaged
        {
            source.BaseView.CopyTo(stream, target.BaseView);
        }

        /// <summary>
        /// Copies the contents of the source view into this view.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="target">The target view.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="source">The source view.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyFrom<T>(
            this ArrayView1D<T, Stride1D.Dense> target,
            AcceleratorStream stream,
            ArrayView1D<T, Stride1D.Dense> source)
            where T : unmanaged
        {
            target.BaseView.CopyFrom(stream, source.BaseView);
        }

        /// <summary>
        /// Copies the contents of this view into the target view.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source view.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="target">The target view (non-dense).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyTo<T>(
            this ArrayView1D<T, Stride1D.Dense> source,
            AcceleratorStream stream,
            ArrayView<T> target)
            where T : unmanaged
        {
            source.BaseView.CopyTo(stream, target);
        }

        /// <summary>
        /// Copies the contents of the source view into this view.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="target">The target view.</param>
        /// <param name="stream">The accelerator stream.</param>
        /// <param name="source">The source view (non-dense).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyFrom<T>(
            this ArrayView1D<T, Stride1D.Dense> target,
            AcceleratorStream stream,
            ArrayView<T> source)
            where T : unmanaged
        {
            target.BaseView.CopyFrom(stream, source);
        }
    }
}