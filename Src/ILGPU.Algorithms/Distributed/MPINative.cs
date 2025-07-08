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

namespace ILGPU.Algorithms.Distributed
{
    /// <summary>
    /// Native MPI bindings for inter-process communication.
    /// </summary>
    internal static class MPINative
    {
        private const string LibraryName = "mpi";

        #region MPI Constants

        public const int MPI_SUCCESS = 0;
        public const int MPI_ANY_SOURCE = -1;
        public const int MPI_ANY_TAG = -1;

        // MPI Data Types
        public const int MPI_BYTE = 1;
        public const int MPI_INT = 2;
        public const int MPI_FLOAT = 3;
        public const int MPI_DOUBLE = 4;
        public const int MPI_LONG = 5;
        public const int MPI_SHORT = 6;

        // MPI Operations
        public const int MPI_SUM = 1;
        public const int MPI_PROD = 2;
        public const int MPI_MAX = 3;
        public const int MPI_MIN = 4;
        public const int MPI_LAND = 5; // Logical AND
        public const int MPI_LOR = 6;  // Logical OR
        public const int MPI_BAND = 7; // Bitwise AND
        public const int MPI_BOR = 8;  // Bitwise OR
        public const int MPI_BXOR = 9; // Bitwise XOR

        #endregion

        #region Initialization and Finalization

        /// <summary>
        /// Initializes MPI.
        /// </summary>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Init(IntPtr argc, IntPtr argv);

        /// <summary>
        /// Finalizes MPI.
        /// </summary>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Finalize();

        /// <summary>
        /// Checks if MPI has been initialized.
        /// </summary>
        /// <param name="flag">Output flag indicating initialization status.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Initialized(out int flag);

        #endregion

        #region Communicator Operations

        /// <summary>
        /// Gets the rank of the calling process in the communicator.
        /// </summary>
        /// <param name="comm">Communicator handle.</param>
        /// <param name="rank">Output rank.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Comm_rank(IntPtr comm, out int rank);

        /// <summary>
        /// Gets the size of the communicator.
        /// </summary>
        /// <param name="comm">Communicator handle.</param>
        /// <param name="size">Output size.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Comm_size(IntPtr comm, out int size);

        /// <summary>
        /// Creates a sub-communicator.
        /// </summary>
        /// <param name="comm">Parent communicator.</param>
        /// <param name="group">Group handle.</param>
        /// <param name="newcomm">Output new communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Comm_create(IntPtr comm, IntPtr group, out IntPtr newcomm);

        /// <summary>
        /// Frees a communicator.
        /// </summary>
        /// <param name="comm">Communicator to free.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Comm_free(ref IntPtr comm);

        #endregion

        #region Point-to-Point Communication

        /// <summary>
        /// Sends data to another process.
        /// </summary>
        /// <param name="buf">Data buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="dest">Destination rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Send(IntPtr buf, int count, int datatype, int dest, int tag, IntPtr comm);

        /// <summary>
        /// Receives data from another process.
        /// </summary>
        /// <param name="buf">Data buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="source">Source rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="comm">Communicator.</param>
        /// <param name="status">Status information.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Recv(IntPtr buf, int count, int datatype, int source, int tag, IntPtr comm, IntPtr status);

        /// <summary>
        /// Non-blocking send.
        /// </summary>
        /// <param name="buf">Data buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="dest">Destination rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="comm">Communicator.</param>
        /// <param name="request">Request handle.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Isend(IntPtr buf, int count, int datatype, int dest, int tag, IntPtr comm, out IntPtr request);

        /// <summary>
        /// Non-blocking receive.
        /// </summary>
        /// <param name="buf">Data buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="source">Source rank.</param>
        /// <param name="tag">Message tag.</param>
        /// <param name="comm">Communicator.</param>
        /// <param name="request">Request handle.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Irecv(IntPtr buf, int count, int datatype, int source, int tag, IntPtr comm, out IntPtr request);

        /// <summary>
        /// Tests if a request has completed.
        /// </summary>
        /// <param name="request">Request handle.</param>
        /// <param name="flag">Completion flag.</param>
        /// <param name="status">Status information.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Test(ref IntPtr request, out int flag, IntPtr status);

        /// <summary>
        /// Waits for a request to complete.
        /// </summary>
        /// <param name="request">Request handle.</param>
        /// <param name="status">Status information.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Wait(ref IntPtr request, IntPtr status);

        #endregion

        #region Collective Operations

        /// <summary>
        /// Barrier synchronization.
        /// </summary>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Barrier(IntPtr comm);

        /// <summary>
        /// Broadcasts data from one process to all others.
        /// </summary>
        /// <param name="buffer">Data buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="root">Root process rank.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Bcast(IntPtr buffer, int count, int datatype, int root, IntPtr comm);

        /// <summary>
        /// All-reduce operation.
        /// </summary>
        /// <param name="sendbuf">Send buffer.</param>
        /// <param name="recvbuf">Receive buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="op">Operation.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Allreduce(IntPtr sendbuf, IntPtr recvbuf, int count, int datatype, int op, IntPtr comm);

        /// <summary>
        /// Reduce operation.
        /// </summary>
        /// <param name="sendbuf">Send buffer.</param>
        /// <param name="recvbuf">Receive buffer.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="datatype">Data type.</param>
        /// <param name="op">Operation.</param>
        /// <param name="root">Root process rank.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Reduce(IntPtr sendbuf, IntPtr recvbuf, int count, int datatype, int op, int root, IntPtr comm);

        /// <summary>
        /// Scatter operation.
        /// </summary>
        /// <param name="sendbuf">Send buffer.</param>
        /// <param name="sendcount">Send count per process.</param>
        /// <param name="sendtype">Send data type.</param>
        /// <param name="recvbuf">Receive buffer.</param>
        /// <param name="recvcount">Receive count.</param>
        /// <param name="recvtype">Receive data type.</param>
        /// <param name="root">Root process rank.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Scatter(IntPtr sendbuf, int sendcount, int sendtype, IntPtr recvbuf, int recvcount, int recvtype, int root, IntPtr comm);

        /// <summary>
        /// Gather operation.
        /// </summary>
        /// <param name="sendbuf">Send buffer.</param>
        /// <param name="sendcount">Send count.</param>
        /// <param name="sendtype">Send data type.</param>
        /// <param name="recvbuf">Receive buffer.</param>
        /// <param name="recvcount">Receive count per process.</param>
        /// <param name="recvtype">Receive data type.</param>
        /// <param name="root">Root process rank.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Gather(IntPtr sendbuf, int sendcount, int sendtype, IntPtr recvbuf, int recvcount, int recvtype, int root, IntPtr comm);

        /// <summary>
        /// All-to-all operation.
        /// </summary>
        /// <param name="sendbuf">Send buffer.</param>
        /// <param name="sendcount">Send count per process.</param>
        /// <param name="sendtype">Send data type.</param>
        /// <param name="recvbuf">Receive buffer.</param>
        /// <param name="recvcount">Receive count per process.</param>
        /// <param name="recvtype">Receive data type.</param>
        /// <param name="comm">Communicator.</param>
        /// <returns>Error code.</returns>
        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int MPI_Alltoall(IntPtr sendbuf, int sendcount, int sendtype, IntPtr recvbuf, int recvcount, int recvtype, IntPtr comm);

        #endregion

        #region Public Interface

        private static bool _initialized;
        private static IntPtr _worldCommunicator = IntPtr.Zero;

        /// <summary>
        /// Checks if MPI has been initialized.
        /// </summary>
        /// <returns>True if MPI is initialized.</returns>
        public static bool IsInitialized()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // On Windows, we might use Microsoft MPI or Intel MPI
                return _initialized;
            }
            else
            {
                // On Linux/Unix, try to check MPI status
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    var result = MPI_Initialized(out int flag);
                    if (result != 0) return false; // Handle MPI error
                    return flag != 0;
                }
                catch
                {
                    return false;
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
        }

        /// <summary>
        /// Initializes MPI.
        /// </summary>
        public static void Initialize()
        {
            if (_initialized) return;

            try
            {
                var result = MPI_Init(IntPtr.Zero, IntPtr.Zero);
                if (result == MPI_SUCCESS)
                {
                    _initialized = true;
                    // Get world communicator (typically has value 0x94000000 or similar)
                    _worldCommunicator = GetWorldCommunicatorHandle();
                }
            }
            catch (DllNotFoundException)
            {
                throw new InvalidOperationException("MPI library not found. Please install an MPI implementation (OpenMPI, MPICH, Intel MPI, or Microsoft MPI).");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to initialize MPI: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Gets the world communicator handle.
        /// </summary>
        /// <returns>World communicator handle.</returns>
        public static IntPtr GetWorldCommunicator()
        {
            if (!_initialized)
                throw new InvalidOperationException("MPI not initialized");
            
            return _worldCommunicator;
        }

        /// <summary>
        /// Gets the platform-specific world communicator handle.
        /// </summary>
        private static IntPtr GetWorldCommunicatorHandle()
        {
            // This is a placeholder - in real MPI implementations,
            // MPI_COMM_WORLD is a predefined constant or handle
            // that varies by MPI implementation
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return new IntPtr(0x94000000); // Microsoft MPI
            }
            else
            {
                return new IntPtr(0x44000000); // OpenMPI/MPICH
            }
        }

        /// <summary>
        /// Gets the rank of a process in a communicator.
        /// </summary>
        /// <param name="comm">Communicator handle.</param>
        /// <returns>Process rank.</returns>
        public static int GetRank(IntPtr comm)
        {
            var result = MPI_Comm_rank(comm, out int rank);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Comm_rank failed with error {result}");
            return rank;
        }

        /// <summary>
        /// Gets the size of a communicator.
        /// </summary>
        /// <param name="comm">Communicator handle.</param>
        /// <returns>Communicator size.</returns>
        public static int GetSize(IntPtr comm)
        {
            var result = MPI_Comm_size(comm, out int size);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Comm_size failed with error {result}");
            return size;
        }

        /// <summary>
        /// Creates a sub-communicator with specified ranks.
        /// </summary>
        /// <param name="parentComm">Parent communicator.</param>
        /// <param name="ranks">Ranks to include.</param>
        /// <returns>Sub-communicator handle.</returns>
        public static IntPtr CreateSubCommunicator(IntPtr parentComm, int[] ranks)
        {
            // This is a simplified version - real implementation would need
            // to create groups and then communicators from groups
            throw new NotImplementedException("Sub-communicator creation not yet implemented");
        }

        /// <summary>
        /// Frees a communicator.
        /// </summary>
        /// <param name="comm">Communicator to free.</param>
        public static void FreeCommunicator(IntPtr comm)
        {
            if (comm != _worldCommunicator) // Don't free world communicator
            {
                var result = MPI_Comm_free(ref comm);
                if (result != 0)
                {
                    // Log MPI error but don't throw in cleanup
                    System.Diagnostics.Debug.WriteLine($"MPI_Comm_free failed with error code: {result}");
                }
            }
        }

        /// <summary>
        /// Sends data to another process.
        /// </summary>
        public static void Send(IntPtr data, int size, int datatype, int dest, int tag, IntPtr comm)
        {
            var result = MPI_Send(data, size, datatype, dest, tag, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Send failed with error {result}");
        }

        /// <summary>
        /// Receives data from another process.
        /// </summary>
        public static void Receive(IntPtr data, int size, int datatype, int source, int tag, IntPtr comm)
        {
            var result = MPI_Recv(data, size, datatype, source, tag, comm, IntPtr.Zero);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Recv failed with error {result}");
        }

        /// <summary>
        /// Non-blocking send.
        /// </summary>
        public static IntPtr SendAsync(IntPtr data, int size, int datatype, int dest, int tag, IntPtr comm)
        {
            var result = MPI_Isend(data, size, datatype, dest, tag, comm, out IntPtr request);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Isend failed with error {result}");
            return request;
        }

        /// <summary>
        /// Non-blocking receive.
        /// </summary>
        public static IntPtr ReceiveAsync(IntPtr data, int size, int datatype, int source, int tag, IntPtr comm)
        {
            var result = MPI_Irecv(data, size, datatype, source, tag, comm, out IntPtr request);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Irecv failed with error {result}");
            return request;
        }

        /// <summary>
        /// Tests if a request has completed.
        /// </summary>
        public static bool TestRequest(IntPtr request)
        {
            var requestPtr = request;
            var result = MPI_Test(ref requestPtr, out int flag, IntPtr.Zero);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Test failed with error {result}");
            return flag != 0;
        }

        /// <summary>
        /// Frees a request handle.
        /// </summary>
        public static void FreeRequest(IntPtr request)
        {
            // Request handles are typically freed automatically when they complete
            // This is a placeholder for cleanup if needed
        }

        /// <summary>
        /// Barrier synchronization.
        /// </summary>
        public static void Barrier(IntPtr comm)
        {
            var result = MPI_Barrier(comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Barrier failed with error {result}");
        }

        /// <summary>
        /// Broadcasts data.
        /// </summary>
        public static void Broadcast(IntPtr data, int size, int datatype, int root, IntPtr comm)
        {
            var result = MPI_Bcast(data, size, datatype, root, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Bcast failed with error {result}");
        }

        /// <summary>
        /// All-reduce operation.
        /// </summary>
        public static void AllReduce(IntPtr sendbuf, IntPtr recvbuf, int size, int datatype, int op, IntPtr comm)
        {
            var result = MPI_Allreduce(sendbuf, recvbuf, size, datatype, op, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Allreduce failed with error {result}");
        }

        /// <summary>
        /// Reduce operation.
        /// </summary>
        public static void Reduce(IntPtr sendbuf, IntPtr recvbuf, int size, int datatype, int op, int root, IntPtr comm)
        {
            var result = MPI_Reduce(sendbuf, recvbuf, size, datatype, op, root, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Reduce failed with error {result}");
        }

        /// <summary>
        /// Scatter operation.
        /// </summary>
        public static void Scatter(IntPtr sendbuf, int sendsize, int sendtype, IntPtr recvbuf, int recvsize, int recvtype, int root, IntPtr comm)
        {
            var result = MPI_Scatter(sendbuf, sendsize, sendtype, recvbuf, recvsize, recvtype, root, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Scatter failed with error {result}");
        }

        /// <summary>
        /// Gather operation.
        /// </summary>
        public static void Gather(IntPtr sendbuf, int sendsize, int sendtype, IntPtr recvbuf, int recvsize, int recvtype, int root, IntPtr comm)
        {
            var result = MPI_Gather(sendbuf, sendsize, sendtype, recvbuf, recvsize, recvtype, root, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Gather failed with error {result}");
        }

        /// <summary>
        /// All-to-all operation.
        /// </summary>
        public static void AllToAll(IntPtr sendbuf, int sendsize, int sendtype, IntPtr recvbuf, int recvsize, int recvtype, IntPtr comm)
        {
            var result = MPI_Alltoall(sendbuf, sendsize, sendtype, recvbuf, recvsize, recvtype, comm);
            if (result != MPI_SUCCESS)
                throw new InvalidOperationException($"MPI_Alltoall failed with error {result}");
        }

        /// <summary>
        /// Shuts down MPI (finalizes MPI context).
        /// </summary>
        public static void Shutdown()
        {
            if (_initialized)
            {
                var result = MPI_Finalize();
                if (result != 0)
                {
                    // Log MPI error but don't throw in shutdown
                    System.Diagnostics.Debug.WriteLine($"MPI_Finalize failed with error code: {result}");
                }
                _initialized = false;
                _worldCommunicator = IntPtr.Zero;
            }
        }

        #endregion
    }
}