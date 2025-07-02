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

using ILGPU.Algorithms.SparseMatrix;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.SparseMatrix
{
    /// <summary>
    /// Tests for sparse matrix algorithms.
    /// </summary>
    public class SparseMatrixTests : TestBase
    {
        #region CSR Matrix Tests

        [Fact]
        public void TestCSRMatrixCreation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numRows = 3;
            const int numCols = 3;
            const int nnz = 5;
            
            // Create identity-like sparse matrix
            var rowPtr = new int[] { 0, 1, 3, 5 };
            var colIndices = new int[] { 0, 1, 2, 0, 2 };
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, numRows, numCols, nnz, rowPtr, colIndices, values);
            
            Assert.Equal(numRows, matrix.NumRows);
            Assert.Equal(numCols, matrix.NumCols);
            Assert.Equal(nnz, matrix.NumNonZeros);
            
            // Verify data integrity
            var retrievedRowPtr = matrix.RowPtr.GetAsArray1D();
            var retrievedColIndices = matrix.ColIndices.GetAsArray1D();
            var retrievedValues = matrix.Values.GetAsArray1D();
            
            AssertEqual(rowPtr, retrievedRowPtr);
            AssertEqual(colIndices, retrievedColIndices);
            AssertEqual(values, retrievedValues);
        }

        [Fact]
        public void TestCSRMatrixVectorMultiplication()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create 3x3 sparse matrix:
            // [1 0 2]
            // [0 3 0]
            // [4 0 5]
            const int n = 3;
            var rowPtr = new int[] { 0, 2, 3, 5 };
            var colIndices = new int[] { 0, 2, 1, 0, 2 };
            var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, n, n, values.Length, rowPtr, colIndices, values);
            
            var x = new float[] { 1.0f, 2.0f, 3.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(n);
            
            // Compute y = A * x
            SparseMatrixOperations.SpMV(matrix, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
            
            var result = yBuffer.GetAsArray1D();
            
            // Expected: [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
            var expected = new float[] { 7.0f, 6.0f, 19.0f };
            AssertEqual(expected, result);
        }

        [Fact]
        public void TestCSRMatrixVectorMultiplicationWithScaling()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Simple 2x2 identity matrix
            var rowPtr = new int[] { 0, 1, 2 };
            var colIndices = new int[] { 0, 1 };
            var values = new float[] { 1.0f, 1.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 2, rowPtr, colIndices, values);
            
            var x = new float[] { 2.0f, 3.0f };
            var y = new float[] { 1.0f, 1.0f };
            
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D(y);
            
            // Compute y = 2.0 * A * x + 3.0 * y
            SparseMatrixOperations.SpMV(matrix, xBuffer.View, yBuffer.View, 2.0f, 3.0f);
            
            var result = yBuffer.GetAsArray1D();
            
            // Expected: [2.0 * 2.0 + 3.0 * 1.0, 2.0 * 3.0 + 3.0 * 1.0] = [7.0, 9.0]
            var expected = new float[] { 7.0f, 9.0f };
            AssertEqual(expected, result);
        }

        #endregion

        #region CSC Matrix Tests

        [Fact]
        public void TestCSCMatrixCreation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numRows = 3;
            const int numCols = 3;
            const int nnz = 4;
            
            // Create CSC format matrix
            var colPtr = new int[] { 0, 2, 2, 4 };
            var rowIndices = new int[] { 0, 2, 0, 1 };
            var values = new float[] { 1.0f, 4.0f, 2.0f, 3.0f };
            
            using var matrix = CSCMatrix<float>.Create(
                accelerator!, numRows, numCols, nnz, colPtr, rowIndices, values);
            
            Assert.Equal(numRows, matrix.NumRows);
            Assert.Equal(numCols, matrix.NumCols);
            Assert.Equal(nnz, matrix.NumNonZeros);
            
            // Verify data integrity
            var retrievedColPtr = matrix.ColPtr.GetAsArray1D();
            var retrievedRowIndices = matrix.RowIndices.GetAsArray1D();
            var retrievedValues = matrix.Values.GetAsArray1D();
            
            AssertEqual(colPtr, retrievedColPtr);
            AssertEqual(rowIndices, retrievedRowIndices);
            AssertEqual(values, retrievedValues);
        }

        [Fact]
        public void TestCSCToCSRConversion()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a simple CSC matrix and convert to CSR
            const int n = 2;
            var colPtr = new int[] { 0, 1, 2 };
            var rowIndices = new int[] { 0, 1 };
            var values = new float[] { 5.0f, 7.0f };
            
            using var cscMatrix = CSCMatrix<float>.Create(
                accelerator!, n, n, 2, colPtr, rowIndices, values);
            
            using var csrMatrix = SparseMatrixOperations.ConvertCSCToCSR(cscMatrix, accelerator!.DefaultStream);
            
            // Verify conversion
            Assert.Equal(cscMatrix.NumRows, csrMatrix.NumRows);
            Assert.Equal(cscMatrix.NumCols, csrMatrix.NumCols);
            Assert.Equal(cscMatrix.NumNonZeros, csrMatrix.NumNonZeros);
            
            // Test equivalence via matrix-vector multiplication
            var x = new float[] { 1.0f, 2.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yCSC = accelerator!.Allocate1D<float>(n);
            using var yCSR = accelerator!.Allocate1D<float>(n);
            
            SparseMatrixOperations.SpMV(cscMatrix, xBuffer.View, yCSC.View, 1.0f, 0.0f);
            SparseMatrixOperations.SpMV(csrMatrix, xBuffer.View, yCSR.View, 1.0f, 0.0f);
            
            var resultCSC = yCSC.GetAsArray1D();
            var resultCSR = yCSR.GetAsArray1D();
            
            AssertEqual(resultCSC, resultCSR);
        }

        #endregion

        #region BSR Matrix Tests

        [Fact]
        public void TestBSRMatrixCreation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int numBlockRows = 2;
            const int numBlockCols = 2;
            const int blockSize = 2;
            const int numBlocks = 3;
            
            var blockRowPtr = new int[] { 0, 2, 3 };
            var blockColIndices = new int[] { 0, 1, 0 };
            var values = new float[] 
            { 
                // Block (0,0)
                1.0f, 2.0f, 3.0f, 4.0f,
                // Block (0,1)
                5.0f, 6.0f, 7.0f, 8.0f,
                // Block (1,0)
                9.0f, 10.0f, 11.0f, 12.0f
            };
            
            using var matrix = BSRMatrix<float>.Create(
                accelerator!, numBlockRows, numBlockCols, blockSize, numBlocks,
                blockRowPtr, blockColIndices, values);
            
            Assert.Equal(numBlockRows, matrix.NumBlockRows);
            Assert.Equal(numBlockCols, matrix.NumBlockCols);
            Assert.Equal(blockSize, matrix.BlockSize);
            Assert.Equal(numBlocks, matrix.NumBlocks);
            
            // Verify data integrity
            var retrievedBlockRowPtr = matrix.BlockRowPtr.GetAsArray1D();
            var retrievedBlockColIndices = matrix.BlockColIndices.GetAsArray1D();
            var retrievedValues = matrix.Values.GetAsArray1D();
            
            AssertEqual(blockRowPtr, retrievedBlockRowPtr);
            AssertEqual(blockColIndices, retrievedBlockColIndices);
            AssertEqual(values, retrievedValues);
        }

        [Fact]
        public void TestBSRMatrixVectorMultiplication()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create 2x2 BSR matrix with 2x2 blocks (total 4x4 matrix)
            const int blockSize = 2;
            var blockRowPtr = new int[] { 0, 1, 2 };
            var blockColIndices = new int[] { 0, 1 };
            var values = new float[] 
            { 
                // Block (0,0): [[1, 0], [0, 1]] (identity)
                1.0f, 0.0f, 0.0f, 1.0f,
                // Block (1,1): [[2, 0], [0, 2]] (2 * identity)
                2.0f, 0.0f, 0.0f, 2.0f
            };
            
            using var matrix = BSRMatrix<float>.Create(
                accelerator!, 2, 2, blockSize, 2, blockRowPtr, blockColIndices, values);
            
            var x = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(4);
            
            SparseMatrixOperations.SpMV(matrix, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
            
            var result = yBuffer.GetAsArray1D();
            
            // Expected: [[1,0,0,0],[0,1,0,0],[0,0,2,0],[0,0,0,2]] * [1,2,3,4] = [1,2,6,8]
            var expected = new float[] { 1.0f, 2.0f, 6.0f, 8.0f };
            AssertEqual(expected, result);
        }

        #endregion

        #region Matrix Operations Tests

        [Fact]
        public void TestSparseMatrixAddition()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create two 2x2 sparse matrices
            var rowPtr1 = new int[] { 0, 1, 2 };
            var colIndices1 = new int[] { 0, 1 };
            var values1 = new float[] { 1.0f, 2.0f };
            
            var rowPtr2 = new int[] { 0, 2, 2 };
            var colIndices2 = new int[] { 0, 1 };
            var values2 = new float[] { 3.0f, 4.0f };
            
            using var matrix1 = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 2, rowPtr1, colIndices1, values1);
            using var matrix2 = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 2, rowPtr2, colIndices2, values2);
            
            using var result = SparseMatrixOperations.Add(matrix1, matrix2, accelerator!.DefaultStream);
            
            // Verify result dimensions
            Assert.Equal(2, result.NumRows);
            Assert.Equal(2, result.NumCols);
            
            // Test result via matrix-vector multiplication
            var x = new float[] { 1.0f, 1.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(2);
            
            SparseMatrixOperations.SpMV(result, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
            
            var resultVec = yBuffer.GetAsArray1D();
            
            // Matrix1: [[1,0],[0,2]], Matrix2: [[3,4],[0,0]]
            // Sum: [[4,4],[0,2]], Result with x=[1,1]: [8,2]
            var expected = new float[] { 8.0f, 2.0f };
            AssertEqual(expected, resultVec);
        }

        [Fact]
        public void TestSparseMatrixTranspose()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create 2x3 sparse matrix:
            // [[1, 0, 2],
            //  [0, 3, 0]]
            var rowPtr = new int[] { 0, 2, 3 };
            var colIndices = new int[] { 0, 2, 1 };
            var values = new float[] { 1.0f, 2.0f, 3.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 3, 3, rowPtr, colIndices, values);
            
            using var transpose = SparseMatrixOperations.Transpose(matrix, accelerator!.DefaultStream);
            
            // Verify transposed dimensions
            Assert.Equal(3, transpose.NumRows);
            Assert.Equal(2, transpose.NumCols);
            Assert.Equal(3, transpose.NumNonZeros);
            
            // Test via matrix-vector multiplication
            var x = new float[] { 1.0f, 2.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(3);
            
            SparseMatrixOperations.SpMV(transpose, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
            
            var result = yBuffer.GetAsArray1D();
            
            // Transpose: [[1,0],[0,3],[2,0]], Result with x=[1,2]: [1,6,2]
            var expected = new float[] { 1.0f, 6.0f, 2.0f };
            AssertEqual(expected, result);
        }

        [Fact]
        public void TestSparseMatrixMatrixMultiplication()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create two 2x2 sparse matrices
            // A = [[1,0],[0,2]], B = [[3,1],[0,1]]
            var rowPtrA = new int[] { 0, 1, 2 };
            var colIndicesA = new int[] { 0, 1 };
            var valuesA = new float[] { 1.0f, 2.0f };
            
            var rowPtrB = new int[] { 0, 2, 3 };
            var colIndicesB = new int[] { 0, 1, 1 };
            var valuesB = new float[] { 3.0f, 1.0f, 1.0f };
            
            using var matrixA = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 2, rowPtrA, colIndicesA, valuesA);
            using var matrixB = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 3, rowPtrB, colIndicesB, valuesB);
            
            using var result = SparseMatrixOperations.Multiply(matrixA, matrixB, accelerator!.DefaultStream);
            
            // Verify result dimensions
            Assert.Equal(2, result.NumRows);
            Assert.Equal(2, result.NumCols);
            
            // Test result: A*B should give [[3,1],[0,2]]
            var x = new float[] { 1.0f, 1.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(2);
            
            SparseMatrixOperations.SpMV(result, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
            
            var resultVec = yBuffer.GetAsArray1D();
            var expected = new float[] { 4.0f, 2.0f }; // [3+1, 0+2]
            AssertEqual(expected, resultVec);
        }

        #endregion

        #region Solver Tests

        [Fact]
        public void TestConjugateGradientSolver()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create symmetric positive definite matrix:
            // [[4, 1],
            //  [1, 3]]
            var rowPtr = new int[] { 0, 2, 4 };
            var colIndices = new int[] { 0, 1, 0, 1 };
            var values = new float[] { 4.0f, 1.0f, 1.0f, 3.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 4, rowPtr, colIndices, values);
            
            var rhs = new float[] { 9.0f, 8.0f }; // Right-hand side
            var solution = new float[] { 0.0f, 0.0f }; // Initial guess
            
            using var rhsBuffer = accelerator!.Allocate1D(rhs);
            using var solutionBuffer = accelerator!.Allocate1D(solution);
            
            int iterations = SparseMatrixSolvers.ConjugateGradient(
                matrix, rhsBuffer.View, solutionBuffer.View, 
                tolerance: 1e-6f, maxIterations: 100);
            
            var result = solutionBuffer.GetAsArray1D();
            
            // Expected solution: [2, 1] (since 4*2 + 1*1 = 9, 1*2 + 3*1 = 5)
            // Wait, let me recalculate: we want A*x = b
            // [[4,1],[1,3]] * [x1,x2] = [9,8]
            // 4*x1 + 1*x2 = 9, 1*x1 + 3*x2 = 8
            // From first: x2 = 9 - 4*x1
            // Substitute: x1 + 3*(9-4*x1) = 8 => x1 + 27 - 12*x1 = 8 => -11*x1 = -19 => x1 = 19/11
            // x2 = 9 - 4*19/11 = (99-76)/11 = 23/11
            
            Assert.True(iterations > 0 && iterations <= 100, $"CG should converge, got {iterations} iterations");
            
            // Verify solution by computing residual
            using var residualBuffer = accelerator!.Allocate1D<float>(2);
            SparseMatrixOperations.SpMV(matrix, solutionBuffer.View, residualBuffer.View, 1.0f, 0.0f);
            
            var residual = residualBuffer.GetAsArray1D();
            for (int i = 0; i < 2; i++)
            {
                residual[i] -= rhs[i];
            }
            
            var residualNorm = (float)Math.Sqrt(residual.Sum(x => x * x));
            Assert.True(residualNorm < 1e-5f, $"CG residual too large: {residualNorm}");
        }

        [Fact]
        public void TestBiCGSTABSolver()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a non-symmetric matrix for BiCGSTAB test
            var rowPtr = new int[] { 0, 2, 4 };
            var colIndices = new int[] { 0, 1, 0, 1 };
            var values = new float[] { 3.0f, 2.0f, 1.0f, 4.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 4, rowPtr, colIndices, values);
            
            var rhs = new float[] { 7.0f, 6.0f };
            var solution = new float[] { 0.0f, 0.0f };
            
            using var rhsBuffer = accelerator!.Allocate1D(rhs);
            using var solutionBuffer = accelerator!.Allocate1D(solution);
            
            int iterations = SparseMatrixSolvers.BiCGSTAB(
                matrix, rhsBuffer.View, solutionBuffer.View,
                tolerance: 1e-6f, maxIterations: 100);
            
            Assert.True(iterations > 0 && iterations <= 100, 
                $"BiCGSTAB should converge, got {iterations} iterations");
            
            // Verify solution
            using var residualBuffer = accelerator!.Allocate1D<float>(2);
            SparseMatrixOperations.SpMV(matrix, solutionBuffer.View, residualBuffer.View, 1.0f, 0.0f);
            
            var residual = residualBuffer.GetAsArray1D();
            for (int i = 0; i < 2; i++)
            {
                residual[i] -= rhs[i];
            }
            
            var residualNorm = (float)Math.Sqrt(residual.Sum(x => x * x));
            Assert.True(residualNorm < 1e-5f, $"BiCGSTAB residual too large: {residualNorm}");
        }

        [Fact]
        public void TestJacobiPreconditioner()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create diagonal-dominant matrix
            var rowPtr = new int[] { 0, 2, 4 };
            var colIndices = new int[] { 0, 1, 0, 1 };
            var values = new float[] { 5.0f, 1.0f, 1.0f, 4.0f };
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 4, rowPtr, colIndices, values);
            
            using var preconditioner = SparseMatrixSolvers.CreateJacobiPreconditioner(matrix, accelerator!.DefaultStream);
            
            // Test preconditioner application
            var x = new float[] { 10.0f, 8.0f };
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(2);
            
            SparseMatrixSolvers.ApplyPreconditioner(preconditioner, xBuffer.View, yBuffer.View, accelerator!.DefaultStream);
            
            var result = yBuffer.GetAsArray1D();
            
            // Jacobi preconditioner: M^(-1) = diag(1/A_ii) = [1/5, 1/4]
            var expected = new float[] { 10.0f / 5.0f, 8.0f / 4.0f };
            AssertEqual(expected, result);
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestLargeSparseMatrixPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int n = 10000;
            const int nnzPerRow = 10;
            
            // Generate random sparse matrix
            var random = new Random(42);
            var rowPtr = new int[n + 1];
            var colIndices = new System.Collections.Generic.List<int>();
            var values = new System.Collections.Generic.List<float>();
            
            rowPtr[0] = 0;
            for (int i = 0; i < n; i++)
            {
                // Add diagonal element
                colIndices.Add(i);
                values.Add(random.Next(1, 10));
                
                // Add random off-diagonal elements
                for (int j = 1; j < nnzPerRow; j++)
                {
                    int col = random.Next(n);
                    if (col != i)
                    {
                        colIndices.Add(col);
                        values.Add(random.Next(-5, 5));
                    }
                }
                
                rowPtr[i + 1] = colIndices.Count;
            }
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, n, n, colIndices.Count, 
                rowPtr.ToArray(), colIndices.ToArray(), values.ToArray());
            
            var x = CreateTestData(n);
            using var xBuffer = accelerator!.Allocate1D(x);
            using var yBuffer = accelerator!.Allocate1D<float>(n);
            
            // Measure SpMV performance
            var spMVTime = MeasureTime(() =>
            {
                SparseMatrixOperations.SpMV(matrix, xBuffer.View, yBuffer.View, 1.0f, 0.0f);
                accelerator!.Synchronize();
            });
            
            Assert.True(spMVTime < 1000, $"Large SpMV took {spMVTime}ms, expected < 1000ms");
            
            // Verify result is not all zeros
            var result = yBuffer.GetAsArray1D();
            Assert.True(result.Any(x => Math.Abs(x) > 1e-10), "SpMV result should not be all zeros");
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestInvalidMatrixDimensions()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test matrix creation with invalid dimensions
            Assert.Throws<ArgumentException>(() =>
            {
                CSRMatrix<float>.Create(accelerator!, 0, 2, 1, 
                    new int[] { 0, 1 }, new int[] { 0 }, new float[] { 1.0f });
            });
            
            Assert.Throws<ArgumentException>(() =>
            {
                CSRMatrix<float>.Create(accelerator!, 2, 0, 1,
                    new int[] { 0, 1, 1 }, new int[] { 0 }, new float[] { 1.0f });
            });
        }

        [Fact]
        public void TestIncompatibleMatrixOperations()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create matrices with incompatible dimensions
            using var matrix2x3 = CSRMatrix<float>.Create(
                accelerator!, 2, 3, 1, new int[] { 0, 1, 1 }, new int[] { 0 }, new float[] { 1.0f });
            using var matrix3x2 = CSRMatrix<float>.Create(
                accelerator!, 3, 2, 1, new int[] { 0, 1, 1, 1 }, new int[] { 0 }, new float[] { 1.0f });
            
            // Test incompatible addition
            Assert.Throws<ArgumentException>(() =>
            {
                SparseMatrixOperations.Add(matrix2x3, matrix3x2, accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestSolverNonConvergence()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Create a singular matrix (should not converge)
            var rowPtr = new int[] { 0, 1, 2 };
            var colIndices = new int[] { 0, 0 };
            var values = new float[] { 1.0f, 0.0f }; // Second row is all zeros
            
            using var matrix = CSRMatrix<float>.Create(
                accelerator!, 2, 2, 2, rowPtr, colIndices, values);
            
            var rhs = new float[] { 1.0f, 1.0f };
            var solution = new float[] { 0.0f, 0.0f };
            
            using var rhsBuffer = accelerator!.Allocate1D(rhs);
            using var solutionBuffer = accelerator!.Allocate1D(solution);
            
            // Should hit max iterations for singular system
            int iterations = SparseMatrixSolvers.ConjugateGradient(
                matrix, rhsBuffer.View, solutionBuffer.View,
                tolerance: 1e-6f, maxIterations: 10);
            
            Assert.Equal(10, iterations); // Should hit max iterations
        }

        #endregion
    }
}