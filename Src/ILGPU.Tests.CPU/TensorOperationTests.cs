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

using ILGPU.ML;
using ILGPU.Runtime;
using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.CPU
{
    public class TensorOperationTests : TestBase
    {
        public TensorOperationTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorShapeTest(TestConfiguration config)
        {
            // Test tensor shape creation and validation
            var shape1D = new TensorShape(10);
            Assert.Equal(1, shape1D.Rank);
            Assert.Equal(10, shape1D.Size);
            Assert.Equal(10, shape1D[0]);

            var shape2D = new TensorShape(3, 4);
            Assert.Equal(2, shape2D.Rank);
            Assert.Equal(12, shape2D.Size);
            Assert.Equal(3, shape2D[0]);
            Assert.Equal(4, shape2D[1]);

            var shape3D = new TensorShape(2, 3, 4);
            Assert.Equal(3, shape3D.Rank);
            Assert.Equal(24, shape3D.Size);

            // Test equality
            var shape2D_copy = new TensorShape(3, 4);
            Assert.Equal(shape2D, shape2D_copy);
            Assert.True(shape2D == shape2D_copy);
            Assert.False(shape2D != shape2D_copy);

            // Test matrix multiplication compatibility
            var matA = new TensorShape(3, 5);
            var matB = new TensorShape(5, 7);
            var matC = new TensorShape(3, 4);

            Assert.True(matA.IsMatMulCompatible(matB));
            Assert.False(matA.IsMatMulCompatible(matC));

            var resultShape = matA.MatMulResultShape(matB);
            Assert.Equal(new TensorShape(3, 7), resultShape);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorCreationTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(2, 3);
            
            // Test tensor creation
            using var tensor = new Tensor<float>(accelerator, shape);
            Assert.Equal(shape, tensor.Shape);
            Assert.Equal(accelerator, tensor.Accelerator);
            Assert.Equal(6, tensor.View.Length);

            // Test tensor creation with data
            var data = new float[] { 1, 2, 3, 4, 5, 6 };
            using var tensorWithData = new Tensor<float>(accelerator, shape, data);
            
            var result = tensorWithData.ToArray();
            Assert.Equal(data, result);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorFactoryMethodsTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(2, 3);

            // Test zeros tensor
            using var zerosTensor = Tensor.Zeros<float>(accelerator, shape);
            var zerosData = zerosTensor.ToArray();
            Assert.All(zerosData, x => Assert.Equal(0.0f, x));

            // Test ones tensor
            using var onesTensor = Tensor.Ones<float>(accelerator, shape);
            var onesData = onesTensor.ToArray();
            Assert.All(onesData, x => Assert.Equal(1.0f, x));

            // Test from array
            var inputData = new float[] { 1, 2, 3, 4, 5, 6 };
            using var arrayTensor = Tensor.FromArray(accelerator, shape, inputData);
            var arrayData = arrayTensor.ToArray();
            Assert.Equal(inputData, arrayData);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorCopyOperationsTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(3, 2);
            var inputData = new float[] { 1, 2, 3, 4, 5, 6 };

            using var tensor = new Tensor<float>(accelerator, shape);
            
            // Test CPU to tensor copy
            tensor.CopyFromCPU(inputData);
            var result1 = tensor.ToArray();
            Assert.Equal(inputData, result1);

            // Test tensor to CPU copy
            var outputData = new float[6];
            tensor.CopyToCPU(outputData);
            Assert.Equal(inputData, outputData);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorReshapeTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var originalShape = new TensorShape(2, 6);
            var inputData = Enumerable.Range(1, 12).Select(x => (float)x).ToArray();

            using var tensor = new Tensor<float>(accelerator, originalShape, inputData);
            
            // Test reshape to different dimensions but same size
            var newShape = new TensorShape(3, 4);
            using var reshaped = tensor.Reshape(newShape);
            
            Assert.Equal(newShape, reshaped.Shape);
            Assert.Equal(inputData, reshaped.ToArray());

            // Test reshape to 1D
            var flatShape = new TensorShape(12);
            using var flattened = tensor.Reshape(flatShape);
            Assert.Equal(flatShape, flattened.Shape);
            Assert.Equal(inputData, flattened.ToArray());
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorTransposeTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(2, 3);
            var inputData = new float[] { 1, 2, 3, 4, 5, 6 }; // [[1,2,3], [4,5,6]]

            using var tensor = new Tensor<float>(accelerator, shape, inputData);
            using var transposed = tensor.Transpose();
            
            var expectedShape = new TensorShape(3, 2);
            Assert.Equal(expectedShape, transposed.Shape);
            
            var result = transposed.ToArray();
            var expected = new float[] { 1, 4, 2, 5, 3, 6 }; // [[1,4], [2,5], [3,6]]
            Assert.Equal(expected, result);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorElementWiseAdditionTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(2, 3);
            var data1 = new float[] { 1, 2, 3, 4, 5, 6 };
            var data2 = new float[] { 6, 5, 4, 3, 2, 1 };

            using var tensor1 = new Tensor<float>(accelerator, shape, data1);
            using var tensor2 = new Tensor<float>(accelerator, shape, data2);
            using var result = tensor1.Add(tensor2);
            
            var resultData = result.ToArray();
            var expected = new float[] { 7, 7, 7, 7, 7, 7 };
            Assert.Equal(expected, resultData);

            // Test operator overload
            using var result2 = tensor1 + tensor2;
            var resultData2 = result2.ToArray();
            Assert.Equal(expected, resultData2);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorElementWiseMultiplicationTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(2, 2);
            var data1 = new float[] { 2, 3, 4, 5 };
            var data2 = new float[] { 1, 2, 3, 4 };

            using var tensor1 = new Tensor<float>(accelerator, shape, data1);
            using var tensor2 = new Tensor<float>(accelerator, shape, data2);
            using var result = tensor1.Multiply(tensor2);
            
            var resultData = result.ToArray();
            var expected = new float[] { 2, 6, 12, 20 };
            Assert.Equal(expected, resultData);

            // Test operator overload
            using var result2 = tensor1 * tensor2;
            var resultData2 = result2.ToArray();
            Assert.Equal(expected, resultData2);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorReLUTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            var shape = new TensorShape(6);
            var inputData = new float[] { -2, -1, 0, 1, 2, 3 };

            using var tensor = new Tensor<float>(accelerator, shape, inputData);
            using var result = tensor.ReLU();
            
            var resultData = result.ToArray();
            var expected = new float[] { 0, 0, 0, 1, 2, 3 };
            Assert.Equal(expected, resultData);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorMatrixMultiplicationTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            // Test basic matrix multiplication: A(2x3) * B(3x2) = C(2x2)
            var shapeA = new TensorShape(2, 3);
            var shapeB = new TensorShape(3, 2);
            
            var dataA = new float[] { 1, 2, 3, 4, 5, 6 }; // [[1,2,3], [4,5,6]]
            var dataB = new float[] { 7, 8, 9, 10, 11, 12 }; // [[7,8], [9,10], [11,12]]

            using var tensorA = new Tensor<float>(accelerator, shapeA, dataA);
            using var tensorB = new Tensor<float>(accelerator, shapeB, dataB);
            using var result = tensorA.MatMul(tensorB, useTensorCores: false);
            
            var expectedShape = new TensorShape(2, 2);
            Assert.Equal(expectedShape, result.Shape);
            
            var resultData = result.ToArray();
            // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
            //          = [[58, 64], [139, 154]]
            var expected = new float[] { 58, 64, 139, 154 };
            
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(resultData[i] - expected[i]) < 0.001f,
                    $"Matrix multiplication mismatch at index {i}: expected {expected[i]}, got {resultData[i]}");
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorSoftmax2DTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            // Test 2D softmax (batch processing)
            var shape = new TensorShape(2, 3); // 2 samples, 3 features each
            var inputData = new float[] { 1, 2, 3, 4, 5, 6 };

            using var tensor = new Tensor<float>(accelerator, shape, inputData);
            using var result = tensor.Softmax();
            
            Assert.Equal(shape, result.Shape);
            
            var resultData = result.ToArray();
            
            // Verify softmax properties
            // Each row should sum to 1
            var tolerance = 0.001f;
            
            // First row: softmax([1, 2, 3])
            var sum1 = resultData[0] + resultData[1] + resultData[2];
            Assert.True(Math.Abs(sum1 - 1.0f) < tolerance, $"First row sum should be 1.0, got {sum1}");
            
            // Second row: softmax([4, 5, 6])
            var sum2 = resultData[3] + resultData[4] + resultData[5];
            Assert.True(Math.Abs(sum2 - 1.0f) < tolerance, $"Second row sum should be 1.0, got {sum2}");
            
            // Values should be positive
            Assert.All(resultData, x => Assert.True(x > 0, $"Softmax output should be positive, got {x}"));
            
            // Largest input should have largest output in each row
            Assert.True(resultData[2] > resultData[1] && resultData[1] > resultData[0]);
            Assert.True(resultData[5] > resultData[4] && resultData[4] > resultData[3]);
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorSoftmaxNDTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            // Test N-dimensional softmax
            var shape = new TensorShape(2, 2, 3); // 2x2 grid, 3 features each
            var inputData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

            using var tensor = new Tensor<float>(accelerator, shape, inputData);
            using var result = tensor.Softmax();
            
            Assert.Equal(shape, result.Shape);
            
            var resultData = result.ToArray();
            var featureSize = 3;
            var batchSize = 4; // 2 * 2
            
            // Verify each feature vector sums to 1
            var tolerance = 0.001f;
            for (int b = 0; b < batchSize; b++)
            {
                var sum = 0.0f;
                for (int f = 0; f < featureSize; f++)
                {
                    sum += resultData[b * featureSize + f];
                }
                Assert.True(Math.Abs(sum - 1.0f) < tolerance, 
                    $"Batch {b} should sum to 1.0, got {sum}");
            }
        }

        [Theory]
        [MemberData(nameof(TestConfigurations))]
        public void TensorInvalidOperationsTest(TestConfiguration config)
        {
            using var context = new Context();
            using var accelerator = context.CreateCPUAccelerator(0);

            // Test invalid tensor creation
            Assert.Throws<ArgumentException>(() => new TensorShape(0));
            Assert.Throws<ArgumentException>(() => new TensorShape(-1));
            Assert.Throws<ArgumentException>(() => new TensorShape(1, 0));

            var shape = new TensorShape(2, 3);
            using var tensor = new Tensor<float>(accelerator, shape);

            // Test invalid data size
            var wrongSizeData = new float[5]; // Should be 6
            Assert.Throws<ArgumentException>(() => tensor.CopyFromCPU(wrongSizeData));

            // Test invalid reshape
            var invalidShape = new TensorShape(2, 4); // Different total size
            Assert.Throws<ArgumentException>(() => tensor.Reshape(invalidShape));

            // Test incompatible matrix multiplication
            var tensor2 = new Tensor<float>(accelerator, new TensorShape(2, 2));
            Assert.Throws<ArgumentException>(() => tensor.MatMul(tensor2));

            // Test incompatible element-wise operations
            Assert.Throws<ArgumentException>(() => tensor.Add(tensor2));
            Assert.Throws<ArgumentException>(() => tensor.Multiply(tensor2));
        }
    }
}