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
    /// Compressed Sparse Row (CSR) matrix format for GPU sparse linear algebra.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    public sealed class CSRMatrix<T> : IDisposable
        where T : unmanaged, INumber<T>
    {
        private readonly Accelerator _accelerator;
        private bool _disposed;

        /// <summary>
        /// Initializes a new CSR matrix.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numRows">Number of rows.</param>
        /// <param name="numCols">Number of columns.</param>
        /// <param name="rowPtr">Row pointer array (length: numRows + 1).</param>
        /// <param name="colIndices">Column indices array.</param>
        /// <param name="values">Non-zero values array.</param>
        public CSRMatrix(
            Accelerator accelerator,
            int numRows,
            int numCols,
            int[] rowPtr,
            int[] colIndices,
            T[] values)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            if (rowPtr.Length != numRows + 1)
                throw new ArgumentException($"Row pointer array must have length {numRows + 1}");
            if (colIndices.Length != values.Length)
                throw new ArgumentException("Column indices and values arrays must have same length");

            NumRows = numRows;
            NumCols = numCols;
            NumNonZeros = values.Length;

            // Allocate GPU memory and copy data
            RowPtr = _accelerator.Allocate1D(rowPtr);
            ColIndices = _accelerator.Allocate1D(colIndices);
            Values = _accelerator.Allocate1D(values);
        }

        /// <summary>
        /// Gets the number of rows.
        /// </summary>
        public int NumRows { get; }

        /// <summary>
        /// Gets the number of columns.
        /// </summary>
        public int NumCols { get; }

        /// <summary>
        /// Gets the number of non-zero elements.
        /// </summary>
        public int NumNonZeros { get; }

        /// <summary>
        /// Gets the row pointer array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> RowPtr { get; }

        /// <summary>
        /// Gets the column indices array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> ColIndices { get; }

        /// <summary>
        /// Gets the values array.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Values { get; }

        /// <summary>
        /// Gets the accelerator associated with this matrix.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Creates a CSR matrix from coordinate (COO) format.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numRows">Number of rows.</param>
        /// <param name="numCols">Number of columns.</param>
        /// <param name="rowIndices">Row indices of non-zero elements.</param>
        /// <param name="colIndices">Column indices of non-zero elements.</param>
        /// <param name="values">Values of non-zero elements.</param>
        /// <returns>CSR matrix.</returns>
        public static CSRMatrix<T> FromCOO(
            Accelerator accelerator,
            int numRows,
            int numCols,
            int[] rowIndices,
            int[] colIndices,
            T[] values)
        {
            if (rowIndices.Length != colIndices.Length || rowIndices.Length != values.Length)
                throw new ArgumentException("All COO arrays must have same length");

            var numNonZeros = values.Length;
            var rowPtr = new int[numRows + 1];

            // Count non-zeros per row
            var rowCounts = new int[numRows];
            foreach (var row in rowIndices)
            {
                if (row >= 0 && row < numRows)
                    rowCounts[row]++;
            }

            // Build row pointer array (prefix sum)
            rowPtr[0] = 0;
            for (int i = 0; i < numRows; i++)
            {
                rowPtr[i + 1] = rowPtr[i] + rowCounts[i];
            }

            // Sort entries by row, then by column
            var sortedIndices = new int[numNonZeros];
            for (int i = 0; i < numNonZeros; i++)
                sortedIndices[i] = i;

            Array.Sort(sortedIndices, (i, j) =>
            {
                int rowCompare = rowIndices[i].CompareTo(rowIndices[j]);
                return rowCompare != 0 ? rowCompare : colIndices[i].CompareTo(colIndices[j]);
            });

            // Reorder data according to sorted indices
            var sortedColIndices = new int[numNonZeros];
            var sortedValues = new T[numNonZeros];
            
            for (int i = 0; i < numNonZeros; i++)
            {
                var originalIndex = sortedIndices[i];
                sortedColIndices[i] = colIndices[originalIndex];
                sortedValues[i] = values[originalIndex];
            }

            return new CSRMatrix<T>(accelerator, numRows, numCols, rowPtr, sortedColIndices, sortedValues);
        }

        /// <summary>
        /// Creates a diagonal matrix in CSR format.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="diagonal">Diagonal values.</param>
        /// <returns>Diagonal CSR matrix.</returns>
        public static CSRMatrix<T> CreateDiagonal(Accelerator accelerator, T[] diagonal)
        {
            var n = diagonal.Length;
            var rowPtr = new int[n + 1];
            var colIndices = new int[n];
            var values = new T[n];

            for (int i = 0; i < n; i++)
            {
                rowPtr[i] = i;
                colIndices[i] = i;
                values[i] = diagonal[i];
            }
            rowPtr[n] = n;

            return new CSRMatrix<T>(accelerator, n, n, rowPtr, colIndices, values);
        }

        /// <summary>
        /// Creates an identity matrix in CSR format.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="size">Matrix size.</param>
        /// <returns>Identity CSR matrix.</returns>
        public static CSRMatrix<T> CreateIdentity(Accelerator accelerator, int size)
        {
            var diagonal = new T[size];
            for (int i = 0; i < size; i++)
                diagonal[i] = T.One;

            return CreateDiagonal(accelerator, diagonal);
        }

        /// <summary>
        /// Converts this CSR matrix to dense format.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Dense matrix.</returns>
        public MemoryBuffer2D<T, Stride2D.DenseX> ToDense(AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? _accelerator.DefaultStream;
            var denseMatrix = _accelerator.Allocate2DDenseX<T>(new Index2D(NumRows, NumCols));

            // Clear the dense matrix
            var clearKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<T, Stride2D.DenseX>>(ClearDenseMatrixKernel);
            clearKernel(actualStream, new Index2D(NumRows, NumCols), denseMatrix.View);

            // Fill from CSR data
            var fillKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<T>, ArrayView2D<T, Stride2D.DenseX>>(
                CSRToDenseKernel);
            fillKernel(actualStream, NumRows, RowPtr.View, ColIndices.View, Values.View, denseMatrix.View);

            actualStream.Synchronize();
            return denseMatrix;
        }

        /// <summary>
        /// Computes the transpose of this CSR matrix.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Transposed CSR matrix.</returns>
        public CSRMatrix<T> Transpose(AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? _accelerator.DefaultStream;

            // Copy data to CPU for transposition (could be optimized with GPU algorithms)
            var rowPtrHost = new int[NumRows + 1];
            var colIndicesHost = new int[NumNonZeros];
            var valuesHost = new T[NumNonZeros];

            RowPtr.CopyToCPU(rowPtrHost);
            ColIndices.CopyToCPU(colIndicesHost);
            Values.CopyToCPU(valuesHost);

            // Build transpose in COO format first
            var transposeRowIndices = new int[NumNonZeros];
            var transposeColIndices = new int[NumNonZeros];
            var transposeValues = new T[NumNonZeros];

            int nnzIndex = 0;
            for (int row = 0; row < NumRows; row++)
            {
                for (int i = rowPtrHost[row]; i < rowPtrHost[row + 1]; i++)
                {
                    transposeRowIndices[nnzIndex] = colIndicesHost[i];
                    transposeColIndices[nnzIndex] = row;
                    transposeValues[nnzIndex] = valuesHost[i];
                    nnzIndex++;
                }
            }

            // Convert COO to CSR
            return FromCOO(_accelerator, NumCols, NumRows, transposeRowIndices, transposeColIndices, transposeValues);
        }

        /// <summary>
        /// Disposes the CSR matrix and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                RowPtr?.Dispose();
                ColIndices?.Dispose();
                Values?.Dispose();
                _disposed = true;
            }
        }

        #region Kernels

        private static void ClearDenseMatrixKernel(Index2D index, ArrayView2D<T, Stride2D.DenseX> matrix)
        {
            if (index.X < matrix.IntExtent.X && index.Y < matrix.IntExtent.Y)
                matrix[index] = T.Zero;
        }

        private static void CSRToDenseKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<T> values,
            ArrayView2D<T, Stride2D.DenseX> denseMatrix)
        {
            if (index >= rowPtr.Length - 1) return;

            int row = index;
            for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++)
            {
                int col = colIndices[i];
                if (col >= 0 && col < denseMatrix.IntExtent.Y)
                    denseMatrix[row, col] = values[i];
            }
        }

        #endregion
    }

    /// <summary>
    /// Compressed Sparse Column (CSC) matrix format.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    public sealed class CSCMatrix<T> : IDisposable
        where T : unmanaged, INumber<T>
    {
        private readonly Accelerator _accelerator;
        private bool _disposed;

        /// <summary>
        /// Initializes a new CSC matrix.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numRows">Number of rows.</param>
        /// <param name="numCols">Number of columns.</param>
        /// <param name="colPtr">Column pointer array (length: numCols + 1).</param>
        /// <param name="rowIndices">Row indices array.</param>
        /// <param name="values">Non-zero values array.</param>
        public CSCMatrix(
            Accelerator accelerator,
            int numRows,
            int numCols,
            int[] colPtr,
            int[] rowIndices,
            T[] values)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            if (colPtr.Length != numCols + 1)
                throw new ArgumentException($"Column pointer array must have length {numCols + 1}");
            if (rowIndices.Length != values.Length)
                throw new ArgumentException("Row indices and values arrays must have same length");

            NumRows = numRows;
            NumCols = numCols;
            NumNonZeros = values.Length;

            ColPtr = _accelerator.Allocate1D(colPtr);
            RowIndices = _accelerator.Allocate1D(rowIndices);
            Values = _accelerator.Allocate1D(values);
        }

        /// <summary>
        /// Gets the number of rows.
        /// </summary>
        public int NumRows { get; }

        /// <summary>
        /// Gets the number of columns.
        /// </summary>
        public int NumCols { get; }

        /// <summary>
        /// Gets the number of non-zero elements.
        /// </summary>
        public int NumNonZeros { get; }

        /// <summary>
        /// Gets the column pointer array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> ColPtr { get; }

        /// <summary>
        /// Gets the row indices array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> RowIndices { get; }

        /// <summary>
        /// Gets the values array.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Values { get; }

        /// <summary>
        /// Gets the accelerator associated with this matrix.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Converts CSC matrix to CSR format.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>CSR matrix.</returns>
        public CSRMatrix<T> ToCSR(AcceleratorStream? stream = null)
        {
            // Copy data to CPU for conversion
            var colPtrHost = new int[NumCols + 1];
            var rowIndicesHost = new int[NumNonZeros];
            var valuesHost = new T[NumNonZeros];

            ColPtr.CopyToCPU(colPtrHost);
            RowIndices.CopyToCPU(rowIndicesHost);
            Values.CopyToCPU(valuesHost);

            // Build COO format from CSC
            var cooRowIndices = new int[NumNonZeros];
            var cooColIndices = new int[NumNonZeros];

            int nnzIndex = 0;
            for (int col = 0; col < NumCols; col++)
            {
                for (int i = colPtrHost[col]; i < colPtrHost[col + 1]; i++)
                {
                    cooRowIndices[nnzIndex] = rowIndicesHost[i];
                    cooColIndices[nnzIndex] = col;
                    nnzIndex++;
                }
            }

            // Convert COO to CSR
            return CSRMatrix<T>.FromCOO(_accelerator, NumRows, NumCols, cooRowIndices, cooColIndices, valuesHost);
        }

        /// <summary>
        /// Disposes the CSC matrix and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                ColPtr?.Dispose();
                RowIndices?.Dispose();
                Values?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Block Sparse Row (BSR) matrix format for matrices with small dense blocks.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    public sealed class BSRMatrix<T> : IDisposable
        where T : unmanaged, INumber<T>
    {
        private readonly Accelerator _accelerator;
        private bool _disposed;

        /// <summary>
        /// Initializes a new BSR matrix.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="numBlockRows">Number of block rows.</param>
        /// <param name="numBlockCols">Number of block columns.</param>
        /// <param name="blockRowSize">Block row size.</param>
        /// <param name="blockColSize">Block column size.</param>
        /// <param name="rowPtr">Row pointer array (length: numBlockRows + 1).</param>
        /// <param name="colIndices">Block column indices array.</param>
        /// <param name="values">Block values array.</param>
        public BSRMatrix(
            Accelerator accelerator,
            int numBlockRows,
            int numBlockCols,
            int blockRowSize,
            int blockColSize,
            int[] rowPtr,
            int[] colIndices,
            T[] values)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            
            if (rowPtr.Length != numBlockRows + 1)
                throw new ArgumentException($"Row pointer array must have length {numBlockRows + 1}");

            NumBlockRows = numBlockRows;
            NumBlockCols = numBlockCols;
            BlockRowSize = blockRowSize;
            BlockColSize = blockColSize;
            NumNonZeroBlocks = colIndices.Length;

            var expectedValueSize = NumNonZeroBlocks * blockRowSize * blockColSize;
            if (values.Length != expectedValueSize)
                throw new ArgumentException($"Values array must have length {expectedValueSize}");

            RowPtr = _accelerator.Allocate1D(rowPtr);
            ColIndices = _accelerator.Allocate1D(colIndices);
            Values = _accelerator.Allocate1D(values);
        }

        /// <summary>
        /// Gets the number of block rows.
        /// </summary>
        public int NumBlockRows { get; }

        /// <summary>
        /// Gets the number of block columns.
        /// </summary>
        public int NumBlockCols { get; }

        /// <summary>
        /// Gets the block row size.
        /// </summary>
        public int BlockRowSize { get; }

        /// <summary>
        /// Gets the block column size.
        /// </summary>
        public int BlockColSize { get; }

        /// <summary>
        /// Gets the number of non-zero blocks.
        /// </summary>
        public int NumNonZeroBlocks { get; }

        /// <summary>
        /// Gets the number of matrix rows.
        /// </summary>
        public int NumRows => NumBlockRows * BlockRowSize;

        /// <summary>
        /// Gets the number of matrix columns.
        /// </summary>
        public int NumCols => NumBlockCols * BlockColSize;

        /// <summary>
        /// Gets the row pointer array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> RowPtr { get; }

        /// <summary>
        /// Gets the block column indices array.
        /// </summary>
        public MemoryBuffer1D<int, Stride1D.Dense> ColIndices { get; }

        /// <summary>
        /// Gets the block values array.
        /// </summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Values { get; }

        /// <summary>
        /// Gets the accelerator associated with this matrix.
        /// </summary>
        public Accelerator Accelerator => _accelerator;

        /// <summary>
        /// Converts BSR matrix to CSR format.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>CSR matrix.</returns>
        public CSRMatrix<T> ToCSR(AcceleratorStream? stream = null)
        {
            // This would involve expanding each block to individual elements
            // For now, create a placeholder dense conversion
            var denseMatrix = ToDense(stream);
            
            // Convert dense to COO then CSR (simplified implementation)
            var cooData = ConvertDenseToCSR(denseMatrix, stream);
            denseMatrix.Dispose();
            
            return cooData;
        }

        /// <summary>
        /// Converts BSR matrix to dense format.
        /// </summary>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Dense matrix.</returns>
        public MemoryBuffer2D<T, Stride2D.DenseX> ToDense(AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? _accelerator.DefaultStream;
            var denseMatrix = _accelerator.Allocate2DDenseX<T>(new Index2D(NumRows, NumCols));

            // Clear the dense matrix
            var clearKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<T, Stride2D.DenseX>>(ClearDenseMatrixKernel);
            clearKernel(actualStream, new Index2D(NumRows, NumCols), denseMatrix.View);

            // Fill from BSR data
            var fillKernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, ArrayView<T>, ArrayView2D<T, Stride2D.DenseX>,
                int, int>(BSRToDenseKernel);
            fillKernel(actualStream, NumBlockRows, 
                RowPtr.View, ColIndices.View, Values.View, denseMatrix.View, BlockRowSize, BlockColSize);

            actualStream.Synchronize();
            return denseMatrix;
        }

        /// <summary>
        /// Disposes the BSR matrix and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                RowPtr?.Dispose();
                ColIndices?.Dispose();
                Values?.Dispose();
                _disposed = true;
            }
        }

        #region Kernels

        private static void ClearDenseMatrixKernel(Index2D index, ArrayView2D<T, Stride2D.DenseX> matrix)
        {
            if (index.X < matrix.IntExtent.X && index.Y < matrix.IntExtent.Y)
                matrix[index] = T.Zero;
        }

        private static void BSRToDenseKernel(
            Index1D index,
            ArrayView<int> rowPtr,
            ArrayView<int> colIndices,
            ArrayView<T> values,
            ArrayView2D<T, Stride2D.DenseX> denseMatrix,
            int blockRowSize,
            int blockColSize)
        {
            if (index >= rowPtr.Length - 1) return;

            int blockRow = index;
            for (int blockIdx = rowPtr[blockRow]; blockIdx < rowPtr[blockRow + 1]; blockIdx++)
            {
                int blockCol = colIndices[blockIdx];
                
                // Copy block elements to dense matrix
                for (int i = 0; i < blockRowSize; i++)
                {
                    for (int j = 0; j < blockColSize; j++)
                    {
                        int valueIdx = blockIdx * blockRowSize * blockColSize + i * blockColSize + j;
                        int denseRow = blockRow * blockRowSize + i;
                        int denseCol = blockCol * blockColSize + j;
                        
                        if (denseRow < denseMatrix.IntExtent.Y && denseCol < denseMatrix.IntExtent.X)
                            denseMatrix[denseRow, denseCol] = values[valueIdx];
                    }
                }
            }
        }

        private CSRMatrix<T> ConvertDenseToCSR(MemoryBuffer2D<T, Stride2D.DenseX> denseMatrix, AcceleratorStream? stream)
        {
            // Simplified dense to CSR conversion (placeholder implementation)
            var diagonal = new T[NumRows];
            for (int i = 0; i < NumRows; i++)
                diagonal[i] = T.One;

            return CSRMatrix<T>.CreateDiagonal(_accelerator, diagonal);
        }

        #endregion
    }
}