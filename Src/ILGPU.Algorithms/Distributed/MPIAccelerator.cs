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
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ILGPU.Algorithms.Distributed
{
    /// <summary>
    /// MPI-enabled accelerator for distributed GPU computing across multiple nodes.
    /// </summary>
    public sealed class MPIAccelerator : IDisposable
    {
        private readonly Accelerator _localAccelerator;
        private readonly MPICommunicator _communicator;
        private readonly int _rank;
        private readonly int _size;
        private bool _disposed;

        /// <summary>
        /// Initializes a new MPI accelerator.
        /// </summary>
        /// <param name="localAccelerator">The local accelerator on this node.</param>
        /// <param name="communicator">MPI communicator.</param>
        public MPIAccelerator(Accelerator localAccelerator, MPICommunicator communicator)
        {
            _localAccelerator = localAccelerator ?? throw new ArgumentNullException(nameof(localAccelerator));
            _communicator = communicator ?? throw new ArgumentNullException(nameof(communicator));
            
            _rank = _communicator.Rank;
            _size = _communicator.Size;
        }

        /// <summary>
        /// Gets the local accelerator for this node.
        /// </summary>
        public Accelerator LocalAccelerator => _localAccelerator;

        /// <summary>
        /// Gets the MPI communicator.
        /// </summary>
        public MPICommunicator Communicator => _communicator;

        /// <summary>
        /// Gets the MPI rank of this node.
        /// </summary>
        public int Rank => _rank;

        /// <summary>
        /// Gets the total number of MPI processes.
        /// </summary>
        public int Size => _size;

        /// <summary>
        /// Gets whether this is the root process (rank 0).
        /// </summary>
        public bool IsRoot => _rank == 0;

        #region Data Distribution

        /// <summary>
        /// Distributes data from the root process to all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="rootData">Data on root process (ignored on non-root).</param>
        /// <param name="localBuffer">Buffer to receive local portion of data.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Scatter<T>(ArrayView<T> rootData, ArrayView<T> localBuffer, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            if (localBuffer.Length * _size != rootData.Length && IsRoot)
                throw new ArgumentException("Root data size must equal local buffer size times number of processes");

            var actualStream = stream ?? _localAccelerator.DefaultStream;
            
            if (IsRoot)
            {
                // Root process distributes data
                var elementsPerProcess = localBuffer.Length;
                
                // Copy local portion directly
                var localPortion = rootData.SubView(0, elementsPerProcess);
                localBuffer.CopyFrom(localPortion, actualStream);
                
                // Send portions to other processes
                for (int destRank = 1; destRank < _size; destRank++)
                {
                    var offset = destRank * elementsPerProcess;
                    var portion = rootData.SubView(offset, elementsPerProcess);
                    
                    // Copy to CPU buffer for MPI sending
                    var hostBuffer = new T[elementsPerProcess];
                    portion.CopyToCPU(hostBuffer);
                    
                    _communicator.Send(hostBuffer, destRank, tag: 0);
                }
            }
            else
            {
                // Non-root processes receive data
                var hostBuffer = new T[localBuffer.Length];
                _communicator.Receive(hostBuffer, 0, tag: 0);
                
                // Copy to GPU
                localBuffer.CopyFromCPU(hostBuffer);
            }
            
            actualStream.Synchronize();
        }

        /// <summary>
        /// Gathers data from all processes to the root process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="localData">Local data from this process.</param>
        /// <param name="rootBuffer">Buffer on root to receive all data (ignored on non-root).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Gather<T>(ArrayView<T> localData, ArrayView<T> rootBuffer, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            if (IsRoot && rootBuffer.Length != localData.Length * _size)
                throw new ArgumentException("Root buffer size must equal local data size times number of processes");

            var actualStream = stream ?? _localAccelerator.DefaultStream;
            
            if (IsRoot)
            {
                // Root process gathers data
                var elementsPerProcess = localData.Length;
                
                // Copy local portion directly
                var localPortion = rootBuffer.SubView(0, elementsPerProcess);
                localPortion.CopyFrom(localData, actualStream);
                actualStream.Synchronize();
                
                // Receive portions from other processes
                for (int srcRank = 1; srcRank < _size; srcRank++)
                {
                    var hostBuffer = new T[elementsPerProcess];
                    _communicator.Receive(hostBuffer, srcRank, tag: 1);
                    
                    var offset = srcRank * elementsPerProcess;
                    var portion = rootBuffer.SubView(offset, elementsPerProcess);
                    portion.CopyFromCPU(hostBuffer);
                }
            }
            else
            {
                // Non-root processes send data
                actualStream.Synchronize();
                var hostBuffer = new T[localData.Length];
                localData.CopyToCPU(hostBuffer);
                
                _communicator.Send(hostBuffer, 0, tag: 1);
            }
        }

        /// <summary>
        /// Broadcasts data from root process to all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data buffer (source on root, destination on others).</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Broadcast<T>(ArrayView<T> data, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            
            if (IsRoot)
            {
                // Root process sends data to all others
                actualStream.Synchronize();
                var hostBuffer = new T[data.Length];
                data.CopyToCPU(hostBuffer);
                
                for (int destRank = 1; destRank < _size; destRank++)
                {
                    _communicator.Send(hostBuffer, destRank, tag: 2);
                }
            }
            else
            {
                // Non-root processes receive data
                var hostBuffer = new T[data.Length];
                _communicator.Receive(hostBuffer, 0, tag: 2);
                
                data.CopyFromCPU(hostBuffer);
            }
            
            actualStream.Synchronize();
        }

        #endregion

        #region Collective Operations

        /// <summary>
        /// Performs an all-reduce operation across all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="localData">Local data to reduce.</param>
        /// <param name="result">Buffer to store the reduction result.</param>
        /// <param name="operation">Reduction operation.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void AllReduce<T>(ArrayView<T> localData, ArrayView<T> result, MPIOperation operation, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            if (localData.Length != result.Length)
                throw new ArgumentException("Local data and result must have the same length");

            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
            
            // Copy local data to CPU for MPI operations
            var hostData = new T[localData.Length];
            localData.CopyToCPU(hostData);
            
            // Perform MPI all-reduce
            var reduced = _communicator.AllReduce(hostData, operation);
            
            // Copy result back to GPU
            result.CopyFromCPU(reduced);
            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs a reduction operation to the root process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="localData">Local data to reduce.</param>
        /// <param name="result">Buffer to store the reduction result (ignored on non-root).</param>
        /// <param name="operation">Reduction operation.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Reduce<T>(ArrayView<T> localData, ArrayView<T> result, MPIOperation operation, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
            
            // Copy local data to CPU for MPI operations
            var hostData = new T[localData.Length];
            localData.CopyToCPU(hostData);
            
            if (IsRoot)
            {
                if (localData.Length != result.Length)
                    throw new ArgumentException("Local data and result must have the same length");
                
                // Perform MPI reduce to root
                var reduced = _communicator.Reduce(hostData, operation, 0);
                
                // Copy result back to GPU
                result.CopyFromCPU(reduced);
            }
            else
            {
                // Non-root processes just participate in reduction
                _communicator.Reduce(hostData, operation, 0);
            }
            
            actualStream.Synchronize();
        }

        /// <summary>
        /// Exchanges data between all pairs of processes (all-to-all).
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="sendData">Data to send (size must be multiple of process count).</param>
        /// <param name="recvData">Buffer to receive data.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void AllToAll<T>(ArrayView<T> sendData, ArrayView<T> recvData, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            if (sendData.Length != recvData.Length)
                throw new ArgumentException("Send and receive buffers must have the same length");
            
            if (sendData.Length % _size != 0)
                throw new ArgumentException("Data size must be multiple of process count");

            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
            
            var elementsPerProcess = sendData.Length / _size;
            var hostSendBuffer = new T[sendData.Length];
            var hostRecvBuffer = new T[recvData.Length];
            
            sendData.CopyToCPU(hostSendBuffer);
            
            // Perform all-to-all exchange
            _communicator.AllToAll(hostSendBuffer, hostRecvBuffer, elementsPerProcess);
            
            recvData.CopyFromCPU(hostRecvBuffer);
            actualStream.Synchronize();
        }

        #endregion

        #region Distributed Kernels

        /// <summary>
        /// Launches a kernel across all MPI processes with data distribution.
        /// </summary>
        /// <typeparam name="TIndex">Index type.</typeparam>
        /// <param name="kernel">Kernel to execute.</param>
        /// <param name="globalExtent">Global extent across all processes.</param>
        /// <param name="parameters">Kernel parameters.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void LaunchDistributedKernel<TIndex>(
            Action<AcceleratorStream, TIndex, KernelParameters> kernel,
            TIndex globalExtent,
            KernelParameters parameters,
            AcceleratorStream? stream = null)
            where TIndex : struct, IIndex
        {
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            
            // Calculate local extent for this process
            var localExtent = CalculateLocalExtent(globalExtent);
            
            // Launch kernel with local extent
            kernel(actualStream, localExtent, parameters);
            
            // Synchronize all processes
            _communicator.Barrier();
        }

        /// <summary>
        /// Calculates the local extent for this process from a global extent.
        /// </summary>
        private TIndex CalculateLocalExtent<TIndex>(TIndex globalExtent)
            where TIndex : struct, IIndex
        {
            if (globalExtent is Index1D index1D)
            {
                var totalSize = index1D.Size;
                var localSize = (totalSize + _size - 1) / _size; // Ceiling division
                var startOffset = _rank * localSize;
                var endOffset = Math.Min(startOffset + localSize, totalSize);
                var actualLocalSize = Math.Max(0, endOffset - startOffset);
                
                return (TIndex)(object)new Index1D(actualLocalSize);
            }
            else if (globalExtent is Index2D index2D)
            {
                // Distribute along the Y dimension
                var totalY = index2D.Y;
                var localY = (totalY + _size - 1) / _size;
                var startY = _rank * localY;
                var endY = Math.Min(startY + localY, totalY);
                var actualLocalY = Math.Max(0, endY - startY);
                
                return (TIndex)(object)new Index2D(index2D.X, actualLocalY);
            }
            else if (globalExtent is Index3D index3D)
            {
                // Distribute along the Z dimension
                var totalZ = index3D.Z;
                var localZ = (totalZ + _size - 1) / _size;
                var startZ = _rank * localZ;
                var endZ = Math.Min(startZ + localZ, totalZ);
                var actualLocalZ = Math.Max(0, endZ - startZ);
                
                return (TIndex)(object)new Index3D(index3D.X, index3D.Y, actualLocalZ);
            }
            
            throw new NotSupportedException($"Index type {typeof(TIndex)} not supported for distributed execution");
        }

        #endregion

        #region Communication Utilities

        /// <summary>
        /// Synchronizes all MPI processes.
        /// </summary>
        public void Barrier()
        {
            _communicator.Barrier();
        }

        /// <summary>
        /// Sends data to another process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data to send.</param>
        /// <param name="destRank">Destination process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Send<T>(ArrayView<T> data, int destRank, int tag = 0, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
            
            var hostBuffer = new T[data.Length];
            data.CopyToCPU(hostBuffer);
            
            _communicator.Send(hostBuffer, destRank, tag);
        }

        /// <summary>
        /// Receives data from another process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Buffer to receive data.</param>
        /// <param name="srcRank">Source process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        public void Receive<T>(ArrayView<T> data, int srcRank, int tag = 0, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var hostBuffer = new T[data.Length];
            _communicator.Receive(hostBuffer, srcRank, tag);
            
            data.CopyFromCPU(hostBuffer);
            
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
        }

        /// <summary>
        /// Performs non-blocking send.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data to send.</param>
        /// <param name="destRank">Destination process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Request handle for completion checking.</returns>
        public async Task<MPIRequest> SendAsync<T>(ArrayView<T> data, int destRank, int tag = 0, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var actualStream = stream ?? _localAccelerator.DefaultStream;
            actualStream.Synchronize();
            
            var hostBuffer = new T[data.Length];
            data.CopyToCPU(hostBuffer);
            
            return await _communicator.SendAsync(hostBuffer, destRank, tag);
        }

        /// <summary>
        /// Performs non-blocking receive.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Buffer to receive data.</param>
        /// <param name="srcRank">Source process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Request handle for completion checking.</returns>
        public async Task<MPIRequest> ReceiveAsync<T>(ArrayView<T> data, int srcRank, int tag = 0, AcceleratorStream? stream = null)
            where T : unmanaged
        {
            var request = await _communicator.ReceiveAsync<T>(data.Length, srcRank, tag);
            
            // When request completes, copy data to GPU
            _ = Task.Run(async () =>
            {
                await request.Wait();
                var hostBuffer = request.GetData<T>();
                data.CopyFromCPU(hostBuffer);
                
                var actualStream = stream ?? _localAccelerator.DefaultStream;
                actualStream.Synchronize();
            });
            
            return request;
        }

        #endregion

        /// <summary>
        /// Disposes the MPI accelerator.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _communicator?.Dispose();
                _disposed = true;
            }
        }
    }
}