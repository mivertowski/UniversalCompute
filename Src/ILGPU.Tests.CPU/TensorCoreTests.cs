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
using ILGPU.Runtime.CPU;
using ILGPU.TensorCores;
using System;
using Xunit;

namespace ILGPU.Tests.CPU
{
    public class TensorCoreTests : IDisposable
    {
        private readonly Context context;
        private readonly Accelerator accelerator;
        
        public TensorCoreTests()
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
        public void TensorFragment_CreateMatrixA_ValidDimensions()
        {
            // Test creating a valid 16x16 matrix A fragment
            var fragment = TensorFragment.CreateMatrixA<Half>(16, 16, TensorPrecision.FP16);
            
            Assert.Equal(TensorFragmentKind.MatrixA, fragment.Kind);
            Assert.Equal(TensorPrecision.FP16, fragment.Precision);
            Assert.Equal(16, fragment.Rows);
            Assert.Equal(16, fragment.Columns);
            Assert.Equal(256, fragment.NumElements);
        }

        [Fact]
        public void TensorFragment_CreateMatrixB_ValidDimensions()
        {
            var fragment = TensorFragment.CreateMatrixB<Half>(16, 16, TensorPrecision.FP16);
            
            Assert.Equal(TensorFragmentKind.MatrixB, fragment.Kind);
            Assert.Equal(16, fragment.Rows);
            Assert.Equal(16, fragment.Columns);
        }

        [Fact]
        public void TensorFragment_CreateAccumulator_ValidDimensions()
        {
            var fragment = TensorFragment.CreateAccumulator<float>(16, 16, TensorPrecision.FP16);
            
            Assert.Equal(TensorFragmentKind.Accumulator, fragment.Kind);
            Assert.Equal(16, fragment.Rows);
            Assert.Equal(16, fragment.Columns);
        }

        [Fact]
        public void TensorFragment_InvalidDimensions_ThrowsException()
        {
            // Test invalid dimensions that aren't supported by tensor cores
            Assert.Throws<ArgumentException>(() => 
                TensorFragment.CreateMatrixA<Half>(15, 15, TensorPrecision.FP16));
            
            Assert.Throws<ArgumentException>(() => 
                TensorFragment.CreateMatrixB<Half>(17, 17, TensorPrecision.FP16));
        }

        [Fact]
        public void TensorIntrinsics_IsTensorCoreSupported_CPU()
        {
            // On CPU, tensor cores should not be supported
            Assert.False(TensorIntrinsics.IsTensorCoreSupported());
        }

        [Fact]
        public void TensorOperations_ValidateDimensions_ValidInput()
        {
            // Test that validation passes for valid tensor core dimensions
            // This is an internal test to ensure our validation logic works
            
            // 16x16x16 should be valid
            var config = TensorOperations.TensorConfig.Default;
            
            // These calls should not throw
            // Note: We can't easily test the internal method, so we test indirectly
            Assert.Equal(16, config.TileSize);
            Assert.Equal(TensorPrecision.FP16, config.Precision);
            Assert.True(config.UseMixedPrecision);
        }

        [Fact]
        public void TensorAcceleratorExtensions_SupportsTensorCores_CPU()
        {
            // CPU accelerator should not support tensor cores
            Assert.False(accelerator.SupportsTensorCores());
        }

        [Fact]
        public void TensorAcceleratorExtensions_GetSupportedPrecisions_CPU()
        {
            // CPU should return empty array for supported tensor precisions
            var precisions = accelerator.GetSupportedTensorPrecisions();
            Assert.Empty(precisions);
        }

        [Fact]
        public void TensorPrecision_EnumValues_AreComplete()
        {
            // Ensure all expected precision modes are available
            var precisions = Enum.GetValues<TensorPrecision>();
            
            Assert.Contains(TensorPrecision.FP16, precisions);
            Assert.Contains(TensorPrecision.BF16, precisions);
            Assert.Contains(TensorPrecision.TF32, precisions);
            Assert.Contains(TensorPrecision.INT8, precisions);
            Assert.Contains(TensorPrecision.FP8_E4M3, precisions);
            Assert.Contains(TensorPrecision.FP8_E5M2, precisions);
        }

        [Fact]
        public void TensorFragmentLayout_EnumValues_AreComplete()
        {
            var layouts = Enum.GetValues<TensorFragmentLayout>();
            
            Assert.Contains(TensorFragmentLayout.RowMajor, layouts);
            Assert.Contains(TensorFragmentLayout.ColMajor, layouts);
        }

        [Fact]
        public void TensorFragmentKind_EnumValues_AreComplete()
        {
            var kinds = Enum.GetValues<TensorFragmentKind>();
            
            Assert.Contains(TensorFragmentKind.MatrixA, kinds);
            Assert.Contains(TensorFragmentKind.MatrixB, kinds);
            Assert.Contains(TensorFragmentKind.Accumulator, kinds);
        }
    }
}
