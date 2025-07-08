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
using System.Numerics;

namespace ILGPU.Algorithms.SparseMatrix
{
    /// <summary>
    /// GPU-accelerated sparse matrix operations and linear algebra.
    /// </summary>
    public static class SparseMatrixOperations
    {
        #region Sparse Matrix-Vector Operations

        /// <summary>
        /// Sparse matrix-vector multiplication (SpMV): y = alpha * A * x + beta * y
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="matrix">Sparse matrix in CSR format.</param>
        /// <param name="x">Input vector.</param>
        /// <param name="y">Output vector (modified in-place).</param>
        /// <param name="alpha">Scalar multiplier for A*x.</param>
        /// <param name="beta">Scalar multiplier for y.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public static void SpMV<T>(
            CSRMatrix<T> matrix,
            ArrayView<T> x,
            ArrayView<T> y,
            T alpha,
            T beta,
            AcceleratorStream? stream = null)
            where T : unmanaged, INumber<T>
        {
            if (x.Length != matrix.NumCols)
                throw new ArgumentException($"Vector x must have {matrix.NumCols} elements");
            if (y.Length != matrix.NumRows)
                throw new ArgumentException($"Vector y must have {matrix.NumRows} elements");

            var actualStream = stream ?? matrix.Accelerator.DefaultStream;

            var spMVKernel = matrix.Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<T>, ArrayView<T>, 
                ArrayView<T>, T, T>(SpMVKernel);

            spMVKernel(matrix.NumRows,
                matrix.RowPtr.View, matrix.ColIndices.View, matrix.Values.View,
                x, y, alpha, beta);

            actualStream.Synchronize();
        }

        /// <summary>
        /// Sparse matrix-vector multiplication for BSR format.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="matrix">Sparse matrix in BSR format.</param>
        /// <param name="x">Input vector.</param>
        /// <param name="y">Output vector (modified in-place).</param>
        /// <param name="alpha">Scalar multiplier for A*x.</param>
        /// <param name="beta">Scalar multiplier for y.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public static void SpMV<T>(
            BSRMatrix<T> matrix,
            ArrayView<T> x,
            ArrayView<T> y,
            T alpha,
            T beta,
            AcceleratorStream? stream = null)
            where T : unmanaged, INumber<T>
        {
            if (x.Length != matrix.NumCols)
                throw new ArgumentException($"Vector x must have {matrix.NumCols} elements");
            if (y.Length != matrix.NumRows)
                throw new ArgumentException($"Vector y must have {matrix.NumRows} elements");

            var actualStream = stream ?? matrix.Accelerator.DefaultStream;

            var bsrSpMVKernel = matrix.Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<T>, ArrayView<T>,
                ArrayView<T>, T, T, int, int>(BSRSpMVKernel);

            bsrSpMVKernel(matrix.NumBlockRows,
                matrix.RowPtr.View, matrix.ColIndices.View, matrix.Values.View,
                x, y, alpha, beta, matrix.BlockRowSize, matrix.BlockColSize);

            actualStream.Synchronize();
        }

        #endregion

        #region Sparse Matrix-Matrix Operations

        /// <summary>
        /// Sparse matrix-matrix multiplication: C = A * B (both in CSR format).
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="matrixA">Left matrix in CSR format.</param>
        /// <param name="matrixB">Right matrix in CSR format.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Result matrix in CSR format.</returns>
        public static CSRMatrix<T> SpMM<T>(
            CSRMatrix<T> matrixA,
            CSRMatrix<T> matrixB,
            AcceleratorStream? stream = null)
            where T : unmanaged, INumber<T>
        {
            if (matrixA.NumCols != matrixB.NumRows)
                throw new ArgumentException("Matrix dimensions incompatible for multiplication");

            var actualStream = stream ?? matrixA.Accelerator.DefaultStream;

            // First pass: count non-zeros in each row of result
            var resultRowNnz = matrixA.Accelerator.Allocate1D<int>(matrixA.NumRows);
            var countKernel = matrixA.Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, 
                ArrayView<int>>(CountSpMMNonZerosKernel);

            countKernel(matrixA.NumRows,
                matrixA.RowPtr.View, matrixA.ColIndices.View,
                matrixB.RowPtr.View, matrixB.ColIndices.View,
                resultRowNnz.View);

            // Compute row pointers for result matrix
            var resultRowPtr = ComputeRowPointers(matrixA.Accelerator, resultRowNnz.View, actualStream);
            var totalNnz = GetTotalNonZeros(matrixA.Accelerator, resultRowPtr.View, actualStream);

            // Allocate result arrays
            var resultColIndices = matrixA.Accelerator.Allocate1D<int>(totalNnz);
            var resultValues = matrixA.Accelerator.Allocate1D<T>(totalNnz);

            // Second pass: compute actual values
            var computeKernel = matrixA.Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<T>,
                ArrayView<int>, ArrayView<int>, ArrayView<T>,
                ArrayView<int>, ArrayView<int>, ArrayView<T>>(ComputeSpMMValuesKernel);

            computeKernel(matrixA.NumRows,
                matrixA.RowPtr.View, matrixA.ColIndices.View, matrixA.Values.View,
                matrixB.RowPtr.View, matrixB.ColIndices.View, matrixB.Values.View,
                resultRowPtr.View, resultColIndices.View, resultValues.View);

            actualStream.Synchronize();

            // Convert GPU arrays to host arrays for CSRMatrix constructor
            var resultRowPtrHost = new int[matrixA.NumRows + 1];
            var resultColIndicesHost = new int[totalNnz];
            var resultValuesHost = new T[totalNnz];

            resultRowPtr.CopyToCPU(resultRowPtrHost);
            resultColIndices.CopyToCPU(resultColIndicesHost);
            resultValues.CopyToCPU(resultValuesHost);

            // Cleanup temporary buffers
            resultRowNnz.Dispose();
            resultRowPtr.Dispose();
            resultColIndices.Dispose();
            resultValues.Dispose();

            return new CSRMatrix<T>(matrixA.Accelerator, matrixA.NumRows, matrixB.NumCols,
                resultRowPtrHost, resultColIndicesHost, resultValuesHost);
        }

        /// <summary>
        /// Sparse matrix addition: C = alpha * A + beta * B.
        /// </summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="matrixA">First matrix in CSR format.</param>
        /// <param name="matrixB">Second matrix in CSR format.</param>
        /// <param name="alpha">Scalar multiplier for A.</param>
        /// <param name="beta">Scalar multiplier for B.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Result matrix in CSR format.</returns>
        public static CSRMatrix<T> SpAdd<T>(
            CSRMatrix<T> matrixA,
            CSRMatrix<T> matrixB,
            T alpha,
            T beta,
            AcceleratorStream? stream = null)
            where T : unmanaged, INumber<T>
        {
            if (matrixA.NumRows != matrixB.NumRows || matrixA.NumCols != matrixB.NumCols)
                throw new ArgumentException("Matrices must have same dimensions for addition");

            var actualStream = stream ?? matrixA.Accelerator.DefaultStream;

            // Convert both matrices to COO format for easier addition
            var cooA = ConvertCSRToCOO(matrixA, actualStream);
            var cooB = ConvertCSRToCOO(matrixB, actualStream);

            // Merge and add the COO entries
            var mergedCOO = MergeAndAddCOO(matrixA.Accelerator, cooA, cooB, alpha, beta, actualStream);

            // Convert back to CSR format
            return CSRMatrix<T>.FromCOO(matrixA.Accelerator, matrixA.NumRows, matrixA.NumCols,
                mergedCOO.rowIndices, mergedCOO.colIndices, mergedCOO.values);
        }

        #endregion

        #region Iterative Solvers

        /// <summary>
        /// Conjugate Gradient solver for sparse linear systems Ax = b.
        /// </summary>
        /// <param name="matrix">Coefficient matrix (must be symmetric positive definite).</param>
        /// <param name="rhs">Right-hand side vector.</param>
        /// <param name="solution">Solution vector (initial guess, modified in-place).</param>
        /// <param name="tolerance">Convergence tolerance.</param>
        /// <param name="maxIterations">Maximum number of iterations.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Number of iterations taken.</returns>
        public static int ConjugateGradient(
            CSRMatrix<float> matrix,
            ArrayView<float> rhs,
            ArrayView<float> solution,
            float tolerance = 1e-6f,
            int maxIterations = 1000,
            AcceleratorStream? stream = null)
        {
            if (matrix.NumRows != matrix.NumCols)
                throw new ArgumentException("Matrix must be square for CG solver");
            if (rhs.Length != matrix.NumRows)
                throw new ArgumentException("RHS vector size must match matrix rows");
            if (solution.Length != matrix.NumRows)
                throw new ArgumentException("Solution vector size must match matrix rows");

            var actualStream = stream ?? matrix.Accelerator.DefaultStream;
            var n = matrix.NumRows;

            // Allocate working vectors
            var r = matrix.Accelerator.Allocate1D<float>(n);      // residual
            var p = matrix.Accelerator.Allocate1D<float>(n);      // search direction
            var Ap = matrix.Accelerator.Allocate1D<float>(n);     // A * p

            // Initialize: r = b - A*x, p = r
            SpMV(matrix, solution, Ap.View, 1.0f, 0.0f, actualStream);
            ComputeResidual(matrix.Accelerator, rhs, Ap.View, r.View, actualStream);
            r.View.CopyTo(p.View);

            var rsold = DotProduct(matrix.Accelerator, r.View, r.View, actualStream);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Ap = A * p
                SpMV(matrix, p.View, Ap.View, 1.0f, 0.0f, actualStream);

                // alpha = rsold / (p' * Ap)
                var pAp = DotProduct(matrix.Accelerator, p.View, Ap.View, actualStream);
                var alpha = rsold / pAp;

                // x = x + alpha * p
                AXPY(matrix.Accelerator, solution, p.View, alpha, actualStream);

                // r = r - alpha * Ap
                AXPY(matrix.Accelerator, r.View, Ap.View, -alpha, actualStream);

                var rsnew = DotProduct(matrix.Accelerator, r.View, r.View, actualStream);

                if (Math.Sqrt(rsnew) < tolerance)
                {
                    r.Dispose();
                    p.Dispose();
                    Ap.Dispose();
                    return iteration + 1;
                }

                // beta = rsnew / rsold
                var beta = rsnew / rsold;

                // p = r + beta * p
                UpdateSearchDirection(matrix.Accelerator, p.View, r.View, beta, actualStream);

                rsold = rsnew;
            }

            r.Dispose();
            p.Dispose();
            Ap.Dispose();
            return maxIterations;
        }

        /// <summary>
        /// BiCGSTAB solver for sparse linear systems.
        /// </summary>
        /// <param name="matrix">Coefficient matrix.</param>
        /// <param name="rhs">Right-hand side vector.</param>
        /// <param name="solution">Solution vector (initial guess, modified in-place).</param>
        /// <param name="tolerance">Convergence tolerance.</param>
        /// <param name="maxIterations">Maximum number of iterations.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Number of iterations taken.</returns>
        public static int BiCGSTAB(
            CSRMatrix<float> matrix,
            ArrayView<float> rhs,
            ArrayView<float> solution,
            float tolerance = 1e-6f,
            int maxIterations = 1000,
            AcceleratorStream? stream = null)
        {
            if (matrix.NumRows != matrix.NumCols)
                throw new ArgumentException("Matrix must be square for BiCGSTAB solver");

            var actualStream = stream ?? matrix.Accelerator.DefaultStream;
            var n = matrix.NumRows;

            // Allocate working vectors
            var r = matrix.Accelerator.Allocate1D<float>(n);
            var r0 = matrix.Accelerator.Allocate1D<float>(n);
            var p = matrix.Accelerator.Allocate1D<float>(n);
            var v = matrix.Accelerator.Allocate1D<float>(n);
            var s = matrix.Accelerator.Allocate1D<float>(n);
            var t = matrix.Accelerator.Allocate1D<float>(n);

            // Initialize
            SpMV(matrix, solution, v.View, 1.0f, 0.0f, actualStream);
            ComputeResidual(matrix.Accelerator, rhs, v.View, r.View, actualStream);
            r.View.CopyTo(r0.View);
            r.View.CopyTo(p.View);

            var rho = DotProduct(matrix.Accelerator, r0.View, r.View, actualStream);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // v = A * p
                SpMV(matrix, p.View, v.View, 1.0f, 0.0f, actualStream);

                var alpha = rho / DotProduct(matrix.Accelerator, r0.View, v.View, actualStream);

                // s = r - alpha * v
                r.View.CopyTo(s.View);
                AXPY(matrix.Accelerator, s.View, v.View, -alpha, actualStream);

                // Check if s is small enough
                var sNorm = Math.Sqrt(DotProduct(matrix.Accelerator, s.View, s.View, actualStream));
                if (sNorm < tolerance)
                {
                    AXPY(matrix.Accelerator, solution, p.View, alpha, actualStream);
                    break;
                }

                // t = A * s
                SpMV(matrix, s.View, t.View, 1.0f, 0.0f, actualStream);

                var omega = DotProduct(matrix.Accelerator, t.View, s.View, actualStream) /
                           DotProduct(matrix.Accelerator, t.View, t.View, actualStream);

                // x = x + alpha * p + omega * s
                AXPY(matrix.Accelerator, solution, p.View, alpha, actualStream);
                AXPY(matrix.Accelerator, solution, s.View, omega, actualStream);

                // r = s - omega * t
                s.View.CopyTo(r.View);
                AXPY(matrix.Accelerator, r.View, t.View, -omega, actualStream);

                var rNorm = Math.Sqrt(DotProduct(matrix.Accelerator, r.View, r.View, actualStream));
                if (rNorm < tolerance)
                {
                    r.Dispose(); r0.Dispose(); p.Dispose(); v.Dispose(); s.Dispose(); t.Dispose();
                    return iteration + 1;
                }

                var rhoNew = DotProduct(matrix.Accelerator, r0.View, r.View, actualStream);
                var beta = (rhoNew / rho) * (alpha / omega);

                // p = r + beta * (p - omega * v)
                AXPY(matrix.Accelerator, p.View, v.View, -omega, actualStream);
                ScaleAndAdd(matrix.Accelerator, p.View, r.View, beta, actualStream);

                rho = rhoNew;
            }

            r.Dispose(); r0.Dispose(); p.Dispose(); v.Dispose(); s.Dispose(); t.Dispose();
            return maxIterations;
        }

        #endregion

        #region Kernel Implementations

        private static void SpMVKernel<T>(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<T> values,
            ArrayView<T> x,
            ArrayView<T> y,
            T alpha,
            T beta)
            where T : unmanaged, INumber<T>
        {
            if (index >= y.Length) return;

            var row = index.X;
            T sum = T.Zero;

            for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++)
            {
                int col = colIndices[i];
                sum += values[i] * x[col];
            }

            y[row] = alpha * sum + beta * y[row];
        }

        private static void BSRSpMVKernel<T>(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<T> values,
            ArrayView<T> x,
            ArrayView<T> y,
            T alpha,
            T beta,
            int blockRowSize,
            int blockColSize)
            where T : unmanaged, INumber<T>
        {
            if (index >= rowPtr.Length - 1) return;

            int blockRow = index.X;
            
            for (int blockIdx = rowPtr[blockRow]; blockIdx < rowPtr[blockRow + 1]; blockIdx++)
            {
                int blockCol = colIndices[blockIdx];
                
                // Multiply dense block with corresponding vector portion
                for (int i = 0; i < blockRowSize; i++)
                {
                    T sum = T.Zero;
                    for (int j = 0; j < blockColSize; j++)
                    {
                        int valueIdx = blockIdx * blockRowSize * blockColSize + i * blockColSize + j;
                        int vectorIdx = blockCol * blockColSize + j;
                        sum += values[valueIdx] * x[vectorIdx];
                    }
                    
                    int resultIdx = blockRow * blockRowSize + i;
                    if (blockIdx == rowPtr[blockRow]) // First block in row
                        y[resultIdx] = alpha * sum + beta * y[resultIdx];
                    else
                        y[resultIdx] += alpha * sum;
                }
            }
        }

        private static void CountSpMMNonZerosKernel(
            Index1D index,
            ArrayView<int> rowPtrA,
            ArrayView<int> colIndicesA,
            ArrayView<int> rowPtrB,
            ArrayView<int> colIndicesB,
            ArrayView<int> resultRowNnz)
        {
            if (index >= resultRowNnz.Length) return;

            int row = index.X;
            int nnzCount = 0;
            
            // For each non-zero in row of A
            for (int i = rowPtrA[row]; i < rowPtrA[row + 1]; i++)
            {
                int colA = colIndicesA[i];
                
                // Count non-zeros in corresponding row of B
                nnzCount += rowPtrB[colA + 1] - rowPtrB[colA];
            }
            
            resultRowNnz[row] = nnzCount;
        }

        private static void ComputeSpMMValuesKernel<T>(
            Index1D index,
            ArrayView<int> rowPtrA,
            ArrayView<int> colIndicesA,
            ArrayView<T> valuesA,
            ArrayView<int> rowPtrB,
            ArrayView<int> colIndicesB,
            ArrayView<T> valuesB,
            ArrayView<int> resultRowPtr,
            ArrayView<int> resultColIndices,
            ArrayView<T> resultValues)
            where T : unmanaged, INumber<T>
        {
            if (index >= rowPtrA.Length - 1) return;

            int row = index.X;
            int resultIdx = resultRowPtr[row];
            
            // Simplified implementation - would need hash table for efficiency
            for (int i = rowPtrA[row]; i < rowPtrA[row + 1]; i++)
            {
                int colA = colIndicesA[i];
                T valueA = valuesA[i];
                
                for (int j = rowPtrB[colA]; j < rowPtrB[colA + 1]; j++)
                {
                    int colB = colIndicesB[j];
                    T valueB = valuesB[j];
                    
                    resultColIndices[resultIdx] = colB;
                    resultValues[resultIdx] = valueA * valueB;
                    resultIdx++;
                }
            }
        }

        #endregion

        #region Helper Methods

        private static MemoryBuffer1D<int, Stride1D.Dense> ComputeRowPointers(
            Accelerator accelerator, ArrayView<int> rowNnz, AcceleratorStream stream)
        {
            var rowPtr = accelerator.Allocate1D<int>(rowNnz.Length + 1);
            
            // Compute prefix sum on GPU (simplified implementation)
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>>(PrefixSumKernel);
            kernel(new Index1D((int)rowNnz.Length), rowNnz, rowPtr.View);
            
            return rowPtr;
        }

        private static void PrefixSumKernel(Index1D index, ArrayView<int> input, ArrayView<int> output)
        {
            if (index == 0)
            {
                output[0] = 0;
                for (int i = 0; i < input.Length; i++)
                {
                    output[i + 1] = output[i] + input[i];
                }
            }
        }

        private static int GetTotalNonZeros(Accelerator accelerator, ArrayView<int> rowPtr, AcceleratorStream stream)
        {
            var lastElement = new int[1];
            rowPtr.SubView(rowPtr.Length - 1, 1).CopyToCPU(lastElement);
            return lastElement[0];
        }

        private static (int[] rowIndices, int[] colIndices, T[] values) ConvertCSRToCOO<T>(
            CSRMatrix<T> matrix, AcceleratorStream stream) where T : unmanaged, INumber<T>
        {
            var rowIndices = new int[matrix.NumNonZeros];
            var colIndices = new int[matrix.NumNonZeros];
            var values = new T[matrix.NumNonZeros];

            matrix.ColIndices.CopyToCPU(colIndices);
            matrix.Values.CopyToCPU(values);

            var rowPtr = new int[matrix.NumRows + 1];
            matrix.RowPtr.CopyToCPU(rowPtr);

            int idx = 0;
            for (int row = 0; row < matrix.NumRows; row++)
            {
                for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++)
                {
                    rowIndices[idx++] = row;
                }
            }

            return (rowIndices, colIndices, values);
        }

        private static (int[] rowIndices, int[] colIndices, T[] values) MergeAndAddCOO<T>(
            Accelerator accelerator,
            (int[] rowIndices, int[] colIndices, T[] values) cooA,
            (int[] rowIndices, int[] colIndices, T[] values) cooB,
            T alpha, T beta, AcceleratorStream stream) where T : unmanaged, INumber<T>
        {
            // Simplified merge - in practice would use more efficient algorithms
            var totalSize = cooA.values.Length + cooB.values.Length;
            var mergedRows = new int[totalSize];
            var mergedCols = new int[totalSize];
            var mergedVals = new T[totalSize];

            int idx = 0;
            for (int i = 0; i < cooA.values.Length; i++)
            {
                mergedRows[idx] = cooA.rowIndices[i];
                mergedCols[idx] = cooA.colIndices[i];
                mergedVals[idx] = alpha * cooA.values[i];
                idx++;
            }

            for (int i = 0; i < cooB.values.Length; i++)
            {
                mergedRows[idx] = cooB.rowIndices[i];
                mergedCols[idx] = cooB.colIndices[i];
                mergedVals[idx] = beta * cooB.values[i];
                idx++;
            }

            return (mergedRows, mergedCols, mergedVals);
        }

        private static void ComputeResidual(Accelerator accelerator, ArrayView<float> b, ArrayView<float> Ax, ArrayView<float> r, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (index, bVec, AxVec, rVec) =>
                {
                    if (index < rVec.Length)
                        rVec[index] = bVec[index] - AxVec[index];
                });
            kernel(r.IntExtent, b, Ax, r);
        }

        private static float DotProduct(Accelerator accelerator, ArrayView<float> x, ArrayView<float> y, AcceleratorStream stream)
        {
            // This would use ILGPU's reduction operations for efficiency
            // Placeholder implementation
            return 1.0f;
        }

        private static void AXPY(Accelerator accelerator, ArrayView<float> y, ArrayView<float> x, float alpha, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, yVec, xVec, a) =>
                {
                    if (index < yVec.Length)
                        yVec[index] += a * xVec[index];
                });
            kernel(y.IntExtent, y, x, alpha);
        }

        private static void UpdateSearchDirection(Accelerator accelerator, ArrayView<float> p, ArrayView<float> r, float beta, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, pVec, rVec, b) =>
                {
                    if (index < pVec.Length)
                        pVec[index] = rVec[index] + b * pVec[index];
                });
            kernel(p.IntExtent, p, r, beta);
        }

        private static void ScaleAndAdd(Accelerator accelerator, ArrayView<float> y, ArrayView<float> x, float scale, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, yVec, xVec, s) =>
                {
                    if (index < yVec.Length)
                        yVec[index] = xVec[index] + s * yVec[index];
                });
            kernel(y.IntExtent, y, x, scale);
        }

        #endregion
    }
}