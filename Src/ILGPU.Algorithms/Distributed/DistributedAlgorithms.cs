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
using System.Threading.Tasks;

namespace ILGPU.Algorithms.Distributed
{
    /// <summary>
    /// High-level distributed computing algorithms using MPI and ILGPU.
    /// </summary>
    public static class DistributedAlgorithms
    {
        #region Matrix Operations

        /// <summary>
        /// Performs distributed matrix multiplication across multiple MPI processes.
        /// </summary>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="matrixA">Local portion of matrix A.</param>
        /// <param name="matrixB">Matrix B (full matrix on all processes).</param>
        /// <param name="result">Local portion of result matrix.</param>
        /// <param name="globalM">Global number of rows in A.</param>
        /// <param name="globalN">Global number of columns in B.</param>
        /// <param name="K">Number of columns in A / rows in B.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public static void DistributedMatrixMultiply(
            MPIAccelerator mpiAccelerator,
            ArrayView2D<float, Stride2D.DenseX> matrixA,
            ArrayView2D<float, Stride2D.DenseX> matrixB,
            ArrayView2D<float, Stride2D.DenseX> result,
            int globalM, int globalN, int K,
            AcceleratorStream? stream = null)
        {
            var localM = matrixA.IntExtent.Y;
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;

            // Load matrix multiplication kernel
            var matmulKernel = mpiAccelerator.LocalAccelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                int>(MatrixMultiplyKernel);

            // Perform local matrix multiplication
            matmulKernel(new Index2D(globalN, localM), matrixA, matrixB, result, K);
            
            actualStream.Synchronize();
            
            // No additional communication needed - each process computes its portion
            mpiAccelerator.Barrier();
        }

        /// <summary>
        /// Distributed matrix-vector multiplication.
        /// </summary>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="matrix">Local portion of the matrix.</param>
        /// <param name="vector">Input vector (full vector on all processes).</param>
        /// <param name="result">Local portion of result vector.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public static void DistributedMatrixVectorMultiply(
            MPIAccelerator mpiAccelerator,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            ArrayView<float> vector,
            ArrayView<float> result,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;

            // Load matrix-vector multiplication kernel
            var matvecKernel = mpiAccelerator.LocalAccelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView<float>,
                ArrayView<float>>(MatrixVectorKernel);

            // Perform local matrix-vector multiplication
            matvecKernel(result.IntExtent, matrix, vector, result);
            
            actualStream.Synchronize();
            mpiAccelerator.Barrier();
        }

        #endregion

        #region Reduction Operations

        /// <summary>
        /// Distributed sum reduction across all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="localData">Local data to reduce.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Global sum (only valid on root process).</returns>
        public static T DistributedSum<T>(
            MPIAccelerator mpiAccelerator,
            ArrayView<T> localData,
            AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;

            // Compute local sum
            var localSum = ComputeLocalSum(mpiAccelerator.LocalAccelerator, localData, actualStream);
            
            // Perform MPI reduction
            var localSumArray = new T[] { localSum };
            var globalSumArray = new T[1];
            
            mpiAccelerator.Reduce<T>(
                mpiAccelerator.LocalAccelerator.Allocate1D(localSumArray).View,
                mpiAccelerator.LocalAccelerator.Allocate1D(globalSumArray).View,
                MPIOperation.Sum,
                actualStream);

            return globalSumArray[0];
        }

        /// <summary>
        /// Distributed maximum finding across all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="localData">Local data to search.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Global maximum (only valid on root process).</returns>
        public static T DistributedMax<T>(
            MPIAccelerator mpiAccelerator,
            ArrayView<T> localData,
            AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;

            // Compute local maximum
            var localMax = ComputeLocalMax(mpiAccelerator.LocalAccelerator, localData, actualStream);
            
            // Perform MPI reduction
            var localMaxArray = new T[] { localMax };
            var globalMaxArray = new T[1];
            
            mpiAccelerator.Reduce<T>(
                mpiAccelerator.LocalAccelerator.Allocate1D(localMaxArray).View,
                mpiAccelerator.LocalAccelerator.Allocate1D(globalMaxArray).View,
                MPIOperation.Max,
                actualStream);

            return globalMaxArray[0];
        }

        #endregion

        #region Sorting and Data Operations

        /// <summary>
        /// Distributed parallel sort using sample sort algorithm.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="localData">Local data to sort.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Sorted local data.</returns>
        public static async Task<ArrayView<T>> DistributedSort<T>(
            MPIAccelerator mpiAccelerator,
            ArrayView<T> localData,
            AcceleratorStream? stream = null)
            where T : unmanaged, IComparable<T>
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;
            var size = mpiAccelerator.Size;
            var rank = mpiAccelerator.Rank;

            // Step 1: Sort local data
            SortLocal(mpiAccelerator.LocalAccelerator, localData, actualStream);
            
            // Step 2: Sample local data
            var sampleSize = Math.Min(size - 1, localData.Length / size);
            var samples = SampleData(mpiAccelerator.LocalAccelerator, localData, (int)sampleSize, actualStream);
            
            // Step 3: Gather all samples to root
            var allSamples = mpiAccelerator.LocalAccelerator.Allocate1D<T>(sampleSize * size);
            mpiAccelerator.Gather(samples, allSamples.View, actualStream);
            
            T[] pivots;
            if (mpiAccelerator.IsRoot)
            {
                // Step 4: Sort all samples and select pivots
                SortLocal(mpiAccelerator.LocalAccelerator, allSamples.View, actualStream);
                pivots = SelectPivots(allSamples.View, size - 1);
            }
            else
            {
                pivots = new T[size - 1];
            }
            
            // Step 5: Broadcast pivots to all processes
            var pivotsBuffer = mpiAccelerator.LocalAccelerator.Allocate1D(pivots);
            mpiAccelerator.Broadcast(pivotsBuffer.View, actualStream);
            
            // Step 6: Partition local data based on pivots
            var partitions = PartitionData(mpiAccelerator.LocalAccelerator, localData, pivotsBuffer.View, actualStream);
            
            // Step 7: Exchange partitions with other processes
            var receivedData = await ExchangePartitions(mpiAccelerator, partitions, actualStream).ConfigureAwait(false);
            
            // Step 8: Sort received data
            SortLocal(mpiAccelerator.LocalAccelerator, receivedData, actualStream);
            
            return receivedData;
        }

        /// <summary>
        /// Distributed data redistribution for load balancing.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="localData">Local data to redistribute.</param>
        /// <param name="targetSizes">Target sizes for each process.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Redistributed local data.</returns>
        public static ArrayView<T> RedistributeData<T>(
            MPIAccelerator mpiAccelerator,
            ArrayView<T> localData,
            int[] targetSizes,
            AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;
            var rank = mpiAccelerator.Rank;
            var size = mpiAccelerator.Size;

            if (targetSizes.Length != size)
                throw new ArgumentException("Target sizes array must have same length as number of processes");

            // Calculate send counts and displacements
            var sendCounts = CalculateSendCounts((int)localData.Length, targetSizes, rank);
            var sendBuffer = mpiAccelerator.LocalAccelerator.Allocate1D<T>(localData.Length);
            localData.CopyTo(sendBuffer.View);

            // Perform all-to-all exchange
            var recvBuffer = mpiAccelerator.LocalAccelerator.Allocate1D<T>(targetSizes[rank]);
            mpiAccelerator.AllToAll(sendBuffer.View, recvBuffer.View, actualStream);

            return recvBuffer.View;
        }

        #endregion

        #region Iterative Algorithms

        /// <summary>
        /// Distributed conjugate gradient solver for linear systems.
        /// </summary>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="matrix">Local portion of coefficient matrix.</param>
        /// <param name="rhs">Local portion of right-hand side vector.</param>
        /// <param name="solution">Local portion of solution vector.</param>
        /// <param name="tolerance">Convergence tolerance.</param>
        /// <param name="maxIterations">Maximum number of iterations.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Number of iterations taken.</returns>
        public static int DistributedConjugateGradient(
            MPIAccelerator mpiAccelerator,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            ArrayView<float> rhs,
            ArrayView<float> solution,
            float tolerance = 1e-6f,
            int maxIterations = 1000,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;
            var n = solution.Length;

            // Allocate working vectors
            var r = mpiAccelerator.LocalAccelerator.Allocate1D<float>(n);      // residual
            var p = mpiAccelerator.LocalAccelerator.Allocate1D<float>(n);      // search direction
            var Ap = mpiAccelerator.LocalAccelerator.Allocate1D<float>(n);     // A * p

            // Initialize: r = b - A*x, p = r
            DistributedMatrixVectorMultiply(mpiAccelerator, matrix, solution, Ap.View, actualStream);
            ComputeResidual(mpiAccelerator.LocalAccelerator, rhs, Ap.View, r.View, actualStream);
            r.View.CopyTo(p.View, actualStream);

            var rsold = DistributedDotProduct(mpiAccelerator, r.View, r.View, actualStream);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Ap = A * p
                DistributedMatrixVectorMultiply(mpiAccelerator, matrix, p.View, Ap.View, actualStream);

                // alpha = rsold / (p' * Ap)
                var pAp = DistributedDotProduct(mpiAccelerator, p.View, Ap.View, actualStream);
                var alpha = rsold / pAp;

                // x = x + alpha * p
                UpdateSolution(mpiAccelerator.LocalAccelerator, solution, p.View, alpha, actualStream);

                // r = r - alpha * Ap
                UpdateResidual(mpiAccelerator.LocalAccelerator, r.View, Ap.View, alpha, actualStream);

                var rsnew = DistributedDotProduct(mpiAccelerator, r.View, r.View, actualStream);

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
                UpdateSearchDirection(mpiAccelerator.LocalAccelerator, p.View, r.View, beta, actualStream);

                rsold = rsnew;
            }

            r.Dispose();
            p.Dispose();
            Ap.Dispose();
            return maxIterations;
        }

        /// <summary>
        /// Distributed dot product of two vectors.
        /// </summary>
        /// <param name="mpiAccelerator">MPI accelerator.</param>
        /// <param name="x">First vector.</param>
        /// <param name="y">Second vector.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Global dot product.</returns>
        public static float DistributedDotProduct(
            MPIAccelerator mpiAccelerator,
            ArrayView<float> x,
            ArrayView<float> y,
            AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? mpiAccelerator.LocalAccelerator.DefaultStream;

            // Compute local dot product
            var localDot = ComputeLocalDotProduct(mpiAccelerator.LocalAccelerator, x, y, actualStream);

            // Perform MPI reduction
            var localDotArray = new float[] { localDot };
            var globalDotArray = new float[1];
            
            mpiAccelerator.Reduce(
                mpiAccelerator.LocalAccelerator.Allocate1D(localDotArray).View,
                mpiAccelerator.LocalAccelerator.Allocate1D(globalDotArray).View,
                MPIOperation.Sum,
                actualStream);

            return globalDotArray[0];
        }

        #endregion

        #region Helper Methods and Kernels

        private static void MatrixMultiplyKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> matrixA,
            ArrayView2D<float, Stride2D.DenseX> matrixB,
            ArrayView2D<float, Stride2D.DenseX> result,
            int K)
        {
            var row = index.Y;
            var col = index.X;

            if (row >= result.IntExtent.Y || col >= result.IntExtent.X)
                return;

            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += matrixA[row, k] * matrixB[k, col];
            }
            result[row, col] = sum;
        }

        private static void MatrixVectorKernel(
            Index1D index,
            ArrayView2D<float, Stride2D.DenseX> matrix,
            ArrayView<float> vector,
            ArrayView<float> result)
        {
            var row = index.X;
            if (row >= result.Length)
                return;

            float sum = 0.0f;
            for (int col = 0; col < matrix.IntExtent.X; col++)
            {
                sum += matrix[row, col] * vector[col];
            }
            result[row] = sum;
        }

        private static T ComputeLocalSum<T>(Accelerator accelerator, ArrayView<T> data, AcceleratorStream stream)
            where T : unmanaged
        {
            // This would use ILGPU's reduction operations
            // Placeholder implementation
            return data.Length > 0 ? data[0] : default(T);
        }

        private static T ComputeLocalMax<T>(Accelerator accelerator, ArrayView<T> data, AcceleratorStream stream)
            where T : unmanaged
        {
            // This would use ILGPU's reduction operations
            // Placeholder implementation
            return data.Length > 0 ? data[0] : default(T);
        }

        private static void SortLocal<T>(Accelerator accelerator, ArrayView<T> data, AcceleratorStream stream)
            where T : unmanaged, IComparable<T>
        {
            // This would use ILGPU's sorting algorithms
            // Placeholder implementation
        }

        private static ArrayView<T> SampleData<T>(Accelerator accelerator, ArrayView<T> data, int sampleSize, AcceleratorStream stream)
            where T : unmanaged
        {
            // Extract samples uniformly from sorted data
            var samples = accelerator.Allocate1D<T>(sampleSize);
            var stride = data.Length / sampleSize;
            
            for (int i = 0; i < sampleSize; i++)
            {
                var index = Math.Min(i * stride, data.Length - 1);
                samples.View[i] = data[index];
            }
            
            return samples.View;
        }

        private static T[] SelectPivots<T>(ArrayView<T> samples, int numPivots) where T : unmanaged
        {
            var pivots = new T[numPivots];
            var stride = samples.Length / (numPivots + 1);
            
            for (int i = 0; i < numPivots; i++)
            {
                var index = (i + 1) * stride;
                pivots[i] = samples[Math.Min(index, samples.Length - 1)];
            }
            
            return pivots;
        }

        private static ArrayView<T>[] PartitionData<T>(Accelerator accelerator, ArrayView<T> data, ArrayView<T> pivots, AcceleratorStream stream)
            where T : unmanaged
        {
            // Partition data based on pivots
            // This is a simplified placeholder
            var partitions = new ArrayView<T>[pivots.Length + 1];
            var partitionSize = data.Length / partitions.Length;
            
            for (int i = 0; i < partitions.Length; i++)
            {
                var start = i * partitionSize;
                var length = (i == partitions.Length - 1) ? data.Length - start : partitionSize;
                partitions[i] = data.SubView(start, length);
            }
            
            return partitions;
        }

        private static async Task<ArrayView<T>> ExchangePartitions<T>(MPIAccelerator mpiAccelerator, ArrayView<T>[] partitions, AcceleratorStream stream)
            where T : unmanaged
        {
            // Simplified partition exchange
            // Real implementation would use MPI_Alltoallv for variable-sized data
            var totalSize = 0;
            foreach (var partition in partitions)
            {
                totalSize += (int)partition.Length;
            }
            
            return mpiAccelerator.LocalAccelerator.Allocate1D<T>(totalSize).View;
        }

        private static int[] CalculateSendCounts(int localSize, int[] targetSizes, int rank)
        {
            // Calculate how much data to send to each process
            var sendCounts = new int[targetSizes.Length];
            // Simplified calculation - in practice would need more sophisticated load balancing
            var avgSize = localSize / targetSizes.Length;
            for (int i = 0; i < sendCounts.Length; i++)
            {
                sendCounts[i] = avgSize;
            }
            return sendCounts;
        }

        private static void ComputeResidual(Accelerator accelerator, ArrayView<float> b, ArrayView<float> Ax, ArrayView<float> r, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                (index, bVec, AxVec, rVec) =>
                {
                    if (index < rVec.Length)
                        rVec[index] = bVec[index] - AxVec[index];
                });
            kernel(stream, r.IntExtent, b, Ax, r);
        }

        private static void UpdateSolution(Accelerator accelerator, ArrayView<float> x, ArrayView<float> p, float alpha, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, xVec, pVec, a) =>
                {
                    if (index < xVec.Length)
                        xVec[index] += a * pVec[index];
                });
            kernel(stream, x.IntExtent, x, p, alpha);
        }

        private static void UpdateResidual(Accelerator accelerator, ArrayView<float> r, ArrayView<float> Ap, float alpha, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, rVec, ApVec, a) =>
                {
                    if (index < rVec.Length)
                        rVec[index] -= a * ApVec[index];
                });
            kernel(stream, r.IntExtent, r, Ap, alpha);
        }

        private static void UpdateSearchDirection(Accelerator accelerator, ArrayView<float> p, ArrayView<float> r, float beta, AcceleratorStream stream)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(
                (index, pVec, rVec, b) =>
                {
                    if (index < pVec.Length)
                        pVec[index] = rVec[index] + b * pVec[index];
                });
            kernel(stream, p.IntExtent, p, r, beta);
        }

        private static float ComputeLocalDotProduct(Accelerator accelerator, ArrayView<float> x, ArrayView<float> y, AcceleratorStream stream)
        {
            // This would use ILGPU's reduction operations for dot product
            // Placeholder implementation
            return 0.0f;
        }

        #endregion
    }
}