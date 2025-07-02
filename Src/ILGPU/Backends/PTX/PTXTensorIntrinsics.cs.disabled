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

using ILGPU.IR.Intrinsics;
using ILGPU.Runtime.Cuda;
using ILGPU.TensorCores;
using System;

namespace ILGPU.Backends.PTX
{
    /// <summary>
    /// PTX implementation of tensor core intrinsics.
    /// </summary>
    static partial class PTXTensorIntrinsics
    {
        #region Initialization

        /// <summary>
        /// Initializes PTX tensor core intrinsics.
        /// </summary>
        internal static void Initialize(IntrinsicImplementationManager manager)
        {
            // Register tensor core intrinsics based on CUDA architecture
            var intrinsics = new IntrinsicImplementation[]
            {
                // Load operations
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.LoadMatrixA),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70, 
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.LoadMatrixB),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.LoadMatrixC),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),

                // Store operations
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.StoreMatrix),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),

                // MMA operations
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.MatrixMultiplyAccumulate),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.MixedPrecisionMMA),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.BFloat16MixedPrecisionMMA),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_80,
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.TF32MatrixMultiply),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_80,
                    CudaArchitecture.SM_90),
                    
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.INT8MatrixMultiply),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_75,
                    CudaArchitecture.SM_90),

                // Fill operations
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.FillFragment),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90),

                // Capability check
                CreateTensorIntrinsic(
                    nameof(TensorIntrinsics.IsTensorCoreSupported),
                    IntrinsicImplementationMode.GenerateCode,
                    CudaArchitecture.SM_70,
                    CudaArchitecture.SM_90)
            };

            foreach (var intrinsic in intrinsics)
            {
                manager.RegisterIntrinsic(intrinsic);
            }
        }

        /// <summary>
        /// Creates a tensor core intrinsic implementation.
        /// </summary>
        private static PTXIntrinsic CreateTensorIntrinsic(
            string name,
            IntrinsicImplementationMode mode,
            CudaArchitecture minArchitecture,
            CudaArchitecture maxArchitecture) =>
            new(
                typeof(TensorIntrinsics),
                name,
                mode,
                minArchitecture,
                maxArchitecture);

        #endregion

        #region PTX Code Generation

        /// <summary>
        /// Generates PTX code for tensor load operations.
        /// </summary>
        /// <param name="backend">The PTX backend.</param>
        /// <param name="codeGenerator">The code generator.</param>
        /// <param name="value">The intrinsic value.</param>
        public static void GenerateLoadMatrixA(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var ptr = codeGenerator.Load(value[0]);
            var leadingDim = codeGenerator.Load(value[1]);

            // Generate WMMA load.a.sync instruction
            using var command = codeGenerator.BeginCommand(
                "wmma.load.a.sync.aligned.m16n16k16.f16.row");
            command.AppendArgument(target);
            command.AppendArgument(ptr);
            command.AppendArgument(leadingDim);
        }

        /// <summary>
        /// Generates PTX code for tensor load B operations.
        /// </summary>
        public static void GenerateLoadMatrixB(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var ptr = codeGenerator.Load(value[0]);
            var leadingDim = codeGenerator.Load(value[1]);

            using var command = codeGenerator.BeginCommand(
                "wmma.load.b.sync.aligned.m16n16k16.f16.col");
            command.AppendArgument(target);
            command.AppendArgument(ptr);
            command.AppendArgument(leadingDim);
        }

        /// <summary>
        /// Generates PTX code for tensor load C operations.
        /// </summary>
        public static void GenerateLoadMatrixC(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var ptr = codeGenerator.Load(value[0]);
            var leadingDim = codeGenerator.Load(value[1]);

            using var command = codeGenerator.BeginCommand(
                "wmma.load.c.sync.aligned.m16n16k16.f32.row");
            command.AppendArgument(target);
            command.AppendArgument(ptr);
            command.AppendArgument(leadingDim);
        }

        /// <summary>
        /// Generates PTX code for tensor store operations.
        /// </summary>
        public static void GenerateStoreMatrix(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var ptr = codeGenerator.Load(value[0]);
            var fragment = codeGenerator.Load(value[1]);
            var leadingDim = codeGenerator.Load(value[2]);

            using var command = codeGenerator.BeginCommand(
                "wmma.store.d.sync.aligned.m16n16k16.f32.row");
            command.AppendArgument(ptr);
            command.AppendArgument(fragment);
            command.AppendArgument(leadingDim);
        }

        /// <summary>
        /// Generates PTX code for matrix multiply-accumulate operations.
        /// </summary>
        public static void GenerateMatrixMultiplyAccumulate(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fragmentA = codeGenerator.Load(value[0]);
            var fragmentB = codeGenerator.Load(value[1]);
            var fragmentC = codeGenerator.Load(value[2]);

            using var command = codeGenerator.BeginCommand(
                "wmma.mma.sync.aligned.m16n16k16.f32.f16.f16.f32");
            command.AppendArgument(target);
            command.AppendArgument(fragmentA);
            command.AppendArgument(fragmentB);
            command.AppendArgument(fragmentC);
        }

        /// <summary>
        /// Generates PTX code for mixed precision FP16->FP32 MMA.
        /// </summary>
        public static void GenerateMixedPrecisionMMA(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fragmentA = codeGenerator.Load(value[0]);
            var fragmentB = codeGenerator.Load(value[1]);
            var fragmentC = codeGenerator.Load(value[2]);

            using var command = codeGenerator.BeginCommand(
                "wmma.mma.sync.aligned.m16n16k16.f32.f16.f16.f32");
            command.AppendArgument(target);
            command.AppendArgument(fragmentA);
            command.AppendArgument(fragmentB);
            command.AppendArgument(fragmentC);
        }

        /// <summary>
        /// Generates PTX code for BFloat16 mixed precision MMA.
        /// </summary>
        public static void GenerateBFloat16MixedPrecisionMMA(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fragmentA = codeGenerator.Load(value[0]);
            var fragmentB = codeGenerator.Load(value[1]);
            var fragmentC = codeGenerator.Load(value[2]);

            // SM_80+ BFloat16 support
            using var command = codeGenerator.BeginCommand(
                "wmma.mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32");
            command.AppendArgument(target);
            command.AppendArgument(fragmentA);
            command.AppendArgument(fragmentB);
            command.AppendArgument(fragmentC);
        }

        /// <summary>
        /// Generates PTX code for TF32 MMA operations.
        /// </summary>
        public static void GenerateTF32MatrixMultiply(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fragmentA = codeGenerator.Load(value[0]);
            var fragmentB = codeGenerator.Load(value[1]);
            var fragmentC = codeGenerator.Load(value[2]);

            // SM_80+ TensorFloat-32 support
            using var command = codeGenerator.BeginCommand(
                "wmma.mma.sync.aligned.m16n16k8.f32.tf32.tf32.f32");
            command.AppendArgument(target);
            command.AppendArgument(fragmentA);
            command.AppendArgument(fragmentB);
            command.AppendArgument(fragmentC);
        }

        /// <summary>
        /// Generates PTX code for INT8 quantized MMA operations.
        /// </summary>
        public static void GenerateINT8MatrixMultiply(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fragmentA = codeGenerator.Load(value[0]);
            var fragmentB = codeGenerator.Load(value[1]);
            var fragmentC = codeGenerator.Load(value[2]);

            // SM_75+ INT8 quantized support
            using var command = codeGenerator.BeginCommand(
                "wmma.mma.sync.aligned.m16n16k16.s32.s8.s8.s32");
            command.AppendArgument(target);
            command.AppendArgument(fragmentA);
            command.AppendArgument(fragmentB);
            command.AppendArgument(fragmentC);
        }

        /// <summary>
        /// Generates PTX code for fragment fill operations.
        /// </summary>
        public static void GenerateFillFragment(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            var fillValue = codeGenerator.Load(value[0]);

            using var command = codeGenerator.BeginCommand(
                "wmma.fill.sync.aligned.m16n16k16.f32");
            command.AppendArgument(target);
            command.AppendArgument(fillValue);
        }

        /// <summary>
        /// Generates PTX code for tensor core capability check.
        /// </summary>
        public static void GenerateIsTensorCoreSupported(
            PTXBackend backend,
            PTXCodeGenerator codeGenerator,
            PTXIntrinsicValue value)
        {
            var target = codeGenerator.Allocate(value);
            
            // For SM_70+, tensor cores are available
            // This will be compile-time constant based on target architecture
            var arch = backend.Architecture;
            bool supported = arch.Major >= 7;
            
            using var command = codeGenerator.BeginCommand("mov.u32");
            command.AppendArgument(target);
            command.AppendConstant(supported ? 1 : 0);
        }

        #endregion

        #region Architecture-Specific Implementations

        /// <summary>
        /// Gets the appropriate WMMA instruction suffix based on target architecture.
        /// </summary>
        private static string GetWMMAInstructionSuffix(
            CudaArchitecture architecture,
            TensorPrecision inputPrecision,
            TensorPrecision outputPrecision)
        {
            return (architecture.Major, architecture.Minor, inputPrecision, outputPrecision) switch
            {
                // Volta (SM_70, SM_72)
                (7, 0, TensorPrecision.FP16, TensorPrecision.FP16) => "m16n16k16.f16.f16.f16.f16",
                (7, 0, TensorPrecision.FP16, _) => "m16n16k16.f32.f16.f16.f32",
                
                // Turing (SM_75)
                (7, 5, TensorPrecision.FP16, _) => "m16n16k16.f32.f16.f16.f32",
                (7, 5, TensorPrecision.INT8, _) => "m16n16k16.s32.s8.s8.s32",
                
                // Ampere (SM_80, SM_86)
                (8, _, TensorPrecision.FP16, _) => "m16n16k16.f32.f16.f16.f32",
                (8, _, TensorPrecision.BF16, _) => "m16n16k16.f32.bf16.bf16.f32",
                (8, _, TensorPrecision.TF32, _) => "m16n16k8.f32.tf32.tf32.f32",
                (8, _, TensorPrecision.INT8, _) => "m16n16k16.s32.s8.s8.s32",
                
                // Hopper (SM_90+)
                (9, _, TensorPrecision.FP8_E4M3, _) => "m16n16k16.f32.e4m3.e4m3.f32",
                (9, _, TensorPrecision.FP8_E5M2, _) => "m16n16k16.f32.e5m2.e5m2.f32",
                
                _ => "m16n16k16.f32.f16.f16.f32" // Default fallback
            };
        }

        /// <summary>
        /// Gets the memory layout specifier for WMMA instructions.
        /// </summary>
        private static string GetLayoutSpecifier(TensorFragmentLayout layout) =>
            layout switch
            {
                TensorFragmentLayout.RowMajor => "row",
                TensorFragmentLayout.ColMajor => "col",
                _ => "row"
            };

        #endregion
    }
}