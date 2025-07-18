﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2021-2025 ILGPU Project
//                                    www.ilgpu.net
//
// File: LanguageTests.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class LanguageTests(ITestOutputHelper output, TestContext testContext) : TestBase(output, testContext)
#pragma warning restore CA1515 // Consider making public types internal
    {
        internal static void PlainEmitKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit("membar.gl;");
            }
            buffer[index] = index;
        }

        [Fact]
        [KernelMethod(nameof(PlainEmitKernel))]
        public void PlainEmit()
        {
            const int Length = 64;
            var expected = Enumerable.Range(0, Length).ToArray();

            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void OutputEmitKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit("add.s32 %0, %1, %2;", out int result, index.X, 42);
                buffer[index] = result;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(OutputEmitKernel))]
        public void OutputEmit()
        {
            const int Length = 64;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? x + 42 : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void MultipleEmitKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit(
                    "{\n\t" +
                    "   .reg .f64 t1;\n\t" +
                    "   add.f64 t1, %1, %2;\n\t" +
                    "   add.f64 %0, t1, %2;\n\t" +
                    "}",
                    out double result,
                    (double)index.X,
                    42.0);
                buffer[index] = result;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(MultipleEmitKernel))]
        public void MultipleEmit()
        {
            const int Length = 64;
            var expected = Enumerable.Range(0, Length)
                .Select(x => (double)x)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? x + 42.0 + 42.0 : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<double>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void EscapedEmitKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                CudaAsm.Emit("mov.u32 %0, %%laneid;", out int lane);
                buffer[index] = lane;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(EscapedEmitKernel))]
        public void EscapedEmit()
        {
            const int Length = 64;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? x % Accelerator.WarpSize : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void PredicateEmitKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                var isEven = index.X % 2 == 0;
                CudaAsm.Emit(
                    "{\n\t" +
                    "   @%1 mov.u32 %0, %%laneid;\n\t" +
                    "   @!%1 mov.u32 %0, 42;\n\t" +
                    "}",
                    out int lane,
                    isEven);
                buffer[index] = lane;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(PredicateEmitKernel))]
        public void PredicateEmit()
        {
            const int Length = 64;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    if (Accelerator.AcceleratorType == AcceleratorType.Cuda)
                    {
                        return x % 2 == 0 ? x % Accelerator.WarpSize : 42;
                    }
                    else
                    {
                        return x;
                    }
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void Int8EmitKernel(
            Index1D index,
            ArrayView1D<int, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                sbyte truncated = (sbyte)index.X;
                CudaAsm.Emit(
                    "{\n\t" +
                    "   .reg .b32 t1;\n\t" +
                    "   cvt.s32.s8 t1, %1;\n\t" +
                    "   add.s32 %0, t1, 1;\n\t" +
                    "}",
                    out int result,
                    truncated);
                buffer[index] = result;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(Int8EmitKernel))]
        public void Int8Emit()
        {
            const int Length = 512;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? 1 + (sbyte)x : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<int>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }

        internal static void EmitRefUsingOutputParamsKernel(
            Index1D index,
            ArrayView1D<long, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                Input<int> input = index.X;
                Input<int> length = buffer.IntLength;
                Output<long> result = default;

                CudaAsm.EmitRef(
                    "{\n\t" +
                    "   .reg .s64 t0;\n\t" +
                    "   .reg .s64 t2;\n\t" +
                    "   cvt.s64.s32 t0, %0;\n\t" +
                    "   cvt.s64.s32 t2, %2;\n\t" +
                    "   sub.s64 %1, t2, t0;\n\t" +
                    "}",
                    ref input,
                    ref result,
                    ref length);
                buffer[index] = result.Value;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(EmitRefUsingOutputParamsKernel))]
        public void EmitRefUsingOutputParams()
        {
            const int Length = 512;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? (long)(Length - x) : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<long>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }
        internal static void EmitRefUsingRefParamsKernel(
            Index1D index,
            ArrayView1D<long, Stride1D.Dense> buffer)
        {
            if (CudaAsm.IsSupported)
            {
                Ref<long> result = buffer.IntLength;
                Input<int> input = index.X;

                CudaAsm.EmitRef(
                    "{\n\t" +
                    "   .reg .s64 t1;\n\t" +
                    "   cvt.s64.s32 t1, %1;\n\t" +
                    "   sub.s64 %0, %0, t1;\n\t" +
                    "}",
                    ref result,
                    ref input);
                buffer[index] = result.Value;
            }
            else
            {
                buffer[index] = index;
            }
        }

        [Fact]
        [KernelMethod(nameof(EmitRefUsingRefParamsKernel))]
        public void EmitRefUsingRefParams()
        {
            const int Length = 512;
            var expected = Enumerable.Range(0, Length)
                .Select(x =>
                {
                    return Accelerator.AcceleratorType == AcceleratorType.Cuda ? (long)(Length - x) : x;
                })
                .ToArray();

            using var buffer = Accelerator.Allocate1D<long>(Length);
            Execute(Length, buffer.View);
            Verify(buffer.View, expected);
        }
    }
}
