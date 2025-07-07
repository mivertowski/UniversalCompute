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

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace ILGPU.Algorithms.Distributed
{
    /// <summary>
    /// MPI communication operations.
    /// </summary>
    public enum MPIOperation
    {
        /// <summary>
        /// Addition operation.
        /// </summary>
        Sum,

        /// <summary>
        /// Multiplication operation.
        /// </summary>
        Product,

        /// <summary>
        /// Maximum operation.
        /// </summary>
        Max,

        /// <summary>
        /// Minimum operation.
        /// </summary>
        Min,

        /// <summary>
        /// Logical AND operation.
        /// </summary>
        LogicalAnd,

        /// <summary>
        /// Logical OR operation.
        /// </summary>
        LogicalOr,

        /// <summary>
        /// Bitwise AND operation.
        /// </summary>
        BitwiseAnd,

        /// <summary>
        /// Bitwise OR operation.
        /// </summary>
        BitwiseOr,

        /// <summary>
        /// Bitwise XOR operation.
        /// </summary>
        BitwiseXor
    }

    /// <summary>
    /// MPI request handle for non-blocking operations.
    /// </summary>
    public sealed class MPIRequest : IDisposable
    {
        private IntPtr _request;
        private object? _data;
        private bool _disposed;

        internal MPIRequest(IntPtr request)
        {
            _request = request;
        }

        /// <summary>
        /// Gets whether the request has completed.
        /// </summary>
        public bool IsCompleted => MPINative.TestRequest(_request);

        /// <summary>
        /// Waits for the request to complete.
        /// </summary>
        public async Task Wait()
        {
            await Task.Run(() =>
            {
                while (!IsCompleted)
                {
                    System.Threading.Thread.Sleep(1);
                }
            }).ConfigureAwait(false);
        }

        /// <summary>
        /// Gets the data associated with this request.
        /// </summary>
        internal T[] GetData<T>() where T : unmanaged
        {
            return (T[])_data!;
        }

        /// <summary>
        /// Sets the data associated with this request.
        /// </summary>
        internal void SetData<T>(T[] data) where T : unmanaged
        {
            _data = data;
        }

        /// <summary>
        /// Disposes the MPI request.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_request != IntPtr.Zero)
                {
                    MPINative.FreeRequest(_request);
                    _request = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// MPI communicator for inter-process communication.
    /// </summary>
    public sealed class MPICommunicator : IDisposable
    {
        private IntPtr _communicator;
        private bool _disposed;

        /// <summary>
        /// Initializes a new MPI communicator.
        /// </summary>
        private MPICommunicator(IntPtr communicator, int rank, int size)
        {
            _communicator = communicator;
            Rank = rank;
            Size = size;
        }

        /// <summary>
        /// Gets the rank of this process.
        /// </summary>
        public int Rank { get; }

        /// <summary>
        /// Gets the total number of processes.
        /// </summary>
        public int Size { get; }

        /// <summary>
        /// Creates the world communicator (all processes).
        /// </summary>
        /// <returns>MPI communicator for all processes.</returns>
        public static MPICommunicator CreateWorld()
        {
            if (!MPINative.IsInitialized())
            {
                MPINative.Initialize();
            }

            var worldComm = MPINative.GetWorldCommunicator();
            var rank = MPINative.GetRank(worldComm);
            var size = MPINative.GetSize(worldComm);
            
            return new MPICommunicator(worldComm, rank, size);
        }

        /// <summary>
        /// Creates a sub-communicator with specified processes.
        /// </summary>
        /// <param name="ranks">Ranks to include in the sub-communicator.</param>
        /// <returns>MPI sub-communicator.</returns>
        public MPICommunicator CreateSubCommunicator(int[] ranks)
        {
            var subComm = MPINative.CreateSubCommunicator(_communicator, ranks);
            var rank = MPINative.GetRank(subComm);
            var size = MPINative.GetSize(subComm);
            
            return new MPICommunicator(subComm, rank, size);
        }

        #region Point-to-Point Communication

        /// <summary>
        /// Sends data to another process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data to send.</param>
        /// <param name="destRank">Destination process rank.</param>
        /// <param name="tag">Message tag.</param>
        public void Send<T>(T[] data, int destRank, int tag = 0) where T : unmanaged
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                var ptr = handle.AddrOfPinnedObject();
                var size = Marshal.SizeOf<T>() * data.Length;
                MPINative.Send(ptr, size, GetMPIDataType<T>(), destRank, tag, _communicator);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Receives data from another process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Buffer to receive data.</param>
        /// <param name="srcRank">Source process rank.</param>
        /// <param name="tag">Message tag.</param>
        public void Receive<T>(T[] data, int srcRank, int tag = 0) where T : unmanaged
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                var ptr = handle.AddrOfPinnedObject();
                var size = Marshal.SizeOf<T>() * data.Length;
                MPINative.Receive(ptr, size, GetMPIDataType<T>(), srcRank, tag, _communicator);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Performs non-blocking send.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data to send.</param>
        /// <param name="destRank">Destination process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <returns>Request handle.</returns>
        public async Task<MPIRequest> SendAsync<T>(T[] data, int destRank, int tag = 0) where T : unmanaged
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            var ptr = handle.AddrOfPinnedObject();
            var size = Marshal.SizeOf<T>() * data.Length;
            
            var requestPtr = MPINative.SendAsync(ptr, size, GetMPIDataType<T>(), destRank, tag, _communicator);
            var request = new MPIRequest(requestPtr);
            
            // Clean up handle when request completes
            _ = Task.Run(async () =>
            {
                await request.Wait().ConfigureAwait(false);
                handle.Free();
            });
            
            return request;
        }

        /// <summary>
        /// Performs non-blocking receive.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="count">Number of elements to receive.</param>
        /// <param name="srcRank">Source process rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <returns>Request handle.</returns>
        public async Task<MPIRequest> ReceiveAsync<T>(int count, int srcRank, int tag = 0) where T : unmanaged
        {
            var data = new T[count];
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            var ptr = handle.AddrOfPinnedObject();
            var size = Marshal.SizeOf<T>() * count;
            
            var requestPtr = MPINative.ReceiveAsync(ptr, size, GetMPIDataType<T>(), srcRank, tag, _communicator);
            var request = new MPIRequest(requestPtr);
            request.SetData(data);
            
            // Clean up handle when request completes
            _ = Task.Run(async () =>
            {
                await request.Wait().ConfigureAwait(false);
                handle.Free();
            });
            
            return request;
        }

        #endregion

        #region Collective Operations

        /// <summary>
        /// Synchronizes all processes in the communicator.
        /// </summary>
        public void Barrier()
        {
            MPINative.Barrier(_communicator);
        }

        /// <summary>
        /// Broadcasts data from one process to all others.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Data buffer.</param>
        /// <param name="rootRank">Root process rank.</param>
        public void Broadcast<T>(T[] data, int rootRank = 0) where T : unmanaged
        {
            var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                var ptr = handle.AddrOfPinnedObject();
                var size = Marshal.SizeOf<T>() * data.Length;
                MPINative.Broadcast(ptr, size, GetMPIDataType<T>(), rootRank, _communicator);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Performs an all-reduce operation.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Input data.</param>
        /// <param name="operation">Reduction operation.</param>
        /// <returns>Reduced result.</returns>
        public T[] AllReduce<T>(T[] data, MPIOperation operation) where T : unmanaged
        {
            var result = new T[data.Length];
            
            var sendHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            var recvHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            
            try
            {
                var sendPtr = sendHandle.AddrOfPinnedObject();
                var recvPtr = recvHandle.AddrOfPinnedObject();
                var size = Marshal.SizeOf<T>() * data.Length;
                
                MPINative.AllReduce(sendPtr, recvPtr, size, GetMPIDataType<T>(), 
                    GetMPIOperation(operation), _communicator);
            }
            finally
            {
                sendHandle.Free();
                recvHandle.Free();
            }
            
            return result;
        }

        /// <summary>
        /// Performs a reduce operation to the root process.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="data">Input data.</param>
        /// <param name="operation">Reduction operation.</param>
        /// <param name="rootRank">Root process rank.</param>
        /// <returns>Reduced result (only valid on root).</returns>
        public T[] Reduce<T>(T[] data, MPIOperation operation, int rootRank = 0) where T : unmanaged
        {
            var result = new T[data.Length];
            
            var sendHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            var recvHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            
            try
            {
                var sendPtr = sendHandle.AddrOfPinnedObject();
                var recvPtr = recvHandle.AddrOfPinnedObject();
                var size = Marshal.SizeOf<T>() * data.Length;
                
                MPINative.Reduce(sendPtr, recvPtr, size, GetMPIDataType<T>(), 
                    GetMPIOperation(operation), rootRank, _communicator);
            }
            finally
            {
                sendHandle.Free();
                recvHandle.Free();
            }
            
            return result;
        }

        /// <summary>
        /// Performs all-to-all exchange.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="sendData">Data to send.</param>
        /// <param name="recvData">Buffer to receive data.</param>
        /// <param name="elementsPerProcess">Number of elements per process.</param>
        public void AllToAll<T>(T[] sendData, T[] recvData, int elementsPerProcess) where T : unmanaged
        {
            var sendHandle = GCHandle.Alloc(sendData, GCHandleType.Pinned);
            var recvHandle = GCHandle.Alloc(recvData, GCHandleType.Pinned);
            
            try
            {
                var sendPtr = sendHandle.AddrOfPinnedObject();
                var recvPtr = recvHandle.AddrOfPinnedObject();
                var elementSize = Marshal.SizeOf<T>();
                
                MPINative.AllToAll(sendPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(),
                    recvPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(), _communicator);
            }
            finally
            {
                sendHandle.Free();
                recvHandle.Free();
            }
        }

        /// <summary>
        /// Scatters data from root to all processes.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="sendData">Data to scatter (only used on root).</param>
        /// <param name="recvData">Buffer to receive data.</param>
        /// <param name="elementsPerProcess">Number of elements per process.</param>
        /// <param name="rootRank">Root process rank.</param>
        public void Scatter<T>(T[]? sendData, T[] recvData, int elementsPerProcess, int rootRank = 0) where T : unmanaged
        {
            var sendPtr = IntPtr.Zero;
            GCHandle? sendHandle = null;
            
            if (Rank == rootRank && sendData != null)
            {
                sendHandle = GCHandle.Alloc(sendData, GCHandleType.Pinned);
                sendPtr = sendHandle.Value.AddrOfPinnedObject();
            }
            
            var recvHandle = GCHandle.Alloc(recvData, GCHandleType.Pinned);
            
            try
            {
                var recvPtr = recvHandle.AddrOfPinnedObject();
                var elementSize = Marshal.SizeOf<T>();
                
                MPINative.Scatter(sendPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(),
                    recvPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(), rootRank, _communicator);
            }
            finally
            {
                sendHandle?.Free();
                recvHandle.Free();
            }
        }

        /// <summary>
        /// Gathers data from all processes to root.
        /// </summary>
        /// <typeparam name="T">Data type.</typeparam>
        /// <param name="sendData">Data to send.</param>
        /// <param name="recvData">Buffer to receive data (only used on root).</param>
        /// <param name="elementsPerProcess">Number of elements per process.</param>
        /// <param name="rootRank">Root process rank.</param>
        public void Gather<T>(T[] sendData, T[]? recvData, int elementsPerProcess, int rootRank = 0) where T : unmanaged
        {
            var recvPtr = IntPtr.Zero;
            GCHandle? recvHandle = null;
            
            if (Rank == rootRank && recvData != null)
            {
                recvHandle = GCHandle.Alloc(recvData, GCHandleType.Pinned);
                recvPtr = recvHandle.Value.AddrOfPinnedObject();
            }
            
            var sendHandle = GCHandle.Alloc(sendData, GCHandleType.Pinned);
            
            try
            {
                var sendPtr = sendHandle.AddrOfPinnedObject();
                var elementSize = Marshal.SizeOf<T>();
                
                MPINative.Gather(sendPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(),
                    recvPtr, elementsPerProcess * elementSize, GetMPIDataType<T>(), rootRank, _communicator);
            }
            finally
            {
                sendHandle.Free();
                recvHandle?.Free();
            }
        }

        #endregion

        #region Utility Methods

        private static int GetMPIDataType<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(int)) return MPINative.MPI_INT;
            if (typeof(T) == typeof(float)) return MPINative.MPI_FLOAT;
            if (typeof(T) == typeof(double)) return MPINative.MPI_DOUBLE;
            if (typeof(T) == typeof(byte)) return MPINative.MPI_BYTE;
            if (typeof(T) == typeof(long)) return MPINative.MPI_LONG;
            if (typeof(T) == typeof(short)) return MPINative.MPI_SHORT;
            
            throw new NotSupportedException($"Type {typeof(T)} is not supported for MPI operations");
        }

        private static int GetMPIOperation(MPIOperation operation)
        {
            return operation switch
            {
                MPIOperation.Sum => MPINative.MPI_SUM,
                MPIOperation.Product => MPINative.MPI_PROD,
                MPIOperation.Max => MPINative.MPI_MAX,
                MPIOperation.Min => MPINative.MPI_MIN,
                MPIOperation.LogicalAnd => MPINative.MPI_LAND,
                MPIOperation.LogicalOr => MPINative.MPI_LOR,
                MPIOperation.BitwiseAnd => MPINative.MPI_BAND,
                MPIOperation.BitwiseOr => MPINative.MPI_BOR,
                MPIOperation.BitwiseXor => MPINative.MPI_BXOR,
                _ => throw new NotSupportedException($"Operation {operation} is not supported")
            };
        }

        #endregion

        /// <summary>
        /// Disposes the MPI communicator.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_communicator != IntPtr.Zero)
                {
                    MPINative.FreeCommunicator(_communicator);
                    _communicator = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}