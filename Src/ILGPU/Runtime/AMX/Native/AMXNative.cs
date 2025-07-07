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

namespace ILGPU.Runtime.AMX.Native
{
    /// <summary>
    /// Native Intel AMX (Advanced Matrix Extensions) bindings.
    /// </summary>
    /// <remarks>
    /// Intel AMX provides hardware acceleration for AI workloads through:
    /// - Tile-based matrix operations (TMUL)
    /// - 8K bytes of tile memory per tile
    /// - Support for BF16, INT8, and FP32 data types
    /// - Hardware matrix multiplication acceleration
    /// 
    /// Requirements:
    /// - Intel 4th Gen Xeon Scalable processors (Sapphire Rapids) or newer
    /// - Intel Core 12th Gen (Alder Lake) or newer with AMX support
    /// - Operating system AMX context switching support
    /// - Intel AMX runtime libraries
    /// </remarks>
    internal static partial class AMXNative
    {
        #region Constants

#if WINDOWS
        private const string AMXLibrary = "amx"; // Intel AMX runtime
        private const string TileLibrary = "tilecfg"; // Tile configuration library
        private const string OneDNNLibrary = "dnnl"; // Intel oneDNN with AMX support
#else
        private const string AMXLibrary = "libamx.so.1";
        private const string TileLibrary = "libtilecfg.so.1";
        private const string OneDNNLibrary = "libdnnl.so.3";
#endif

        #endregion

        #region AMX Tile Configuration

        /// <summary>
        /// AMX tile configuration structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct AMXTileConfig
        {
            public byte palette_id;          // Tile palette ID (0 or 1)
            public byte start_row;           // Starting row (usually 0)
            public byte reserved1;           // Reserved byte
            public byte reserved2;           // Reserved byte
            
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] colsb;            // Bytes per row for each tile (16 tiles max)
            
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public byte[] rows;             // Number of rows for each tile
        }

        /// <summary>
        /// Configures AMX tile usage.
        /// </summary>
        /// <param name="config">Tile configuration.</param>
        /// <returns>Status code.</returns>
        [DllImport(TileLibrary, EntryPoint = "ldtilecfg", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ConfigureTiles(ref AMXTileConfig config);

        /// <summary>
        /// Releases AMX tile configuration.
        /// </summary>
        /// <returns>Status code.</returns>
        [DllImport(TileLibrary, EntryPoint = "tilerelease", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReleaseTiles();

        /// <summary>
        /// Loads data into an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        /// <param name="src">Source data pointer.</param>
        /// <param name="stride">Stride in bytes.</param>
        [DllImport(AMXLibrary, EntryPoint = "tileloadd", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileLoad(byte tile, IntPtr src, long stride);

        /// <summary>
        /// Stores data from an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        /// <param name="dst">Destination data pointer.</param>
        /// <param name="stride">Stride in bytes.</param>
        [DllImport(AMXLibrary, EntryPoint = "tilestored", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileStore(byte tile, IntPtr dst, long stride);

        /// <summary>
        /// Zeros an AMX tile.
        /// </summary>
        /// <param name="tile">Tile number (0-7).</param>
        [DllImport(AMXLibrary, EntryPoint = "tilezero", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileZero(byte tile);

        #endregion

        #region AMX Matrix Operations

        /// <summary>
        /// Performs matrix multiplication using AMX TMUL (Tile Matrix Multiply).
        /// </summary>
        /// <param name="tileA">Source tile A (0-7).</param>
        /// <param name="tileB">Source tile B (0-7).</param>
        /// <param name="tileC">Destination tile C (0-7).</param>
        [DllImport(AMXLibrary, EntryPoint = "tdpbf16ps", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileMatMulBF16(byte tileA, byte tileB, byte tileC);

        /// <summary>
        /// Performs INT8 matrix multiplication using AMX.
        /// </summary>
        /// <param name="tileA">Source tile A (0-7).</param>
        /// <param name="tileB">Source tile B (0-7).</param>
        /// <param name="tileC">Destination tile C (0-7).</param>
        [DllImport(AMXLibrary, EntryPoint = "tdpbssd", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileMatMulINT8(byte tileA, byte tileB, byte tileC);

        /// <summary>
        /// Performs UINT8 matrix multiplication using AMX.
        /// </summary>
        /// <param name="tileA">Source tile A (0-7).</param>
        /// <param name="tileB">Source tile B (0-7).</param>
        /// <param name="tileC">Destination tile C (0-7).</param>
        [DllImport(AMXLibrary, EntryPoint = "tdpbusd", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileMatMulUINT8(byte tileA, byte tileB, byte tileC);

        /// <summary>
        /// Performs mixed precision UINT8/INT8 matrix multiplication.
        /// </summary>
        /// <param name="tileA">Source tile A (0-7).</param>
        /// <param name="tileB">Source tile B (0-7).</param>
        /// <param name="tileC">Destination tile C (0-7).</param>
        [DllImport(AMXLibrary, EntryPoint = "tdpbuud", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void TileMatMulMixed(byte tileA, byte tileB, byte tileC);

        #endregion

        #region High-Level AMX Operations

        /// <summary>
        /// Checks if Intel AMX is supported on this system.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        internal static bool IsAMXSupported()
        {
            try
            {
                // Check CPUID for AMX support
                return CheckAMXCPUID();
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Executes matrix multiplication using Intel AMX with hardware acceleration.
        /// </summary>
        /// <param name="a">Matrix A data pointer.</param>
        /// <param name="b">Matrix B data pointer.</param>
        /// <param name="c">Matrix C result pointer.</param>
        /// <param name="m">Rows in A and C.</param>
        /// <param name="k">Columns in A, rows in B.</param>
        /// <param name="n">Columns in B and C.</param>
        /// <param name="dataType">Data type for computation.</param>
        internal static unsafe void ExecuteAMXMatMul(
            void* a, void* b, void* c,
            int m, int k, int n,
            AMXDataType dataType)
        {
            try
            {
                // Configure tiles for the matrix operation
                var config = CreateTileConfig(m, k, n, dataType);
                var result = ConfigureTiles(ref config);
                if (result != 0)
                    throw new InvalidOperationException($"Failed to configure AMX tiles: {result}");

                // Perform the matrix multiplication using appropriate data type
                switch (dataType)
                {
                    case AMXDataType.BF16:
                        ExecuteAMXMatMulBF16(a, b, c, m, k, n);
                        break;
                    case AMXDataType.INT8:
                        ExecuteAMXMatMulINT8(a, b, c, m, k, n);
                        break;
                    case AMXDataType.UINT8:
                        ExecuteAMXMatMulUINT8(a, b, c, m, k, n);
                        break;
                    case AMXDataType.Mixed:
                        ExecuteAMXMatMulMixed(a, b, c, m, k, n);
                        break;
                    default:
                        throw new NotSupportedException($"Data type {dataType} not supported for AMX");
                }

                // Release tile configuration
                ReleaseTiles();
            }
            catch (DllNotFoundException)
            {
                // Fall back to CPU implementation if AMX is not available
                ExecuteCPUMatMulFallback(a, b, c, m, k, n, dataType);
            }
            catch (EntryPointNotFoundException)
            {
                // Fall back to CPU implementation if AMX functions are not found
                ExecuteCPUMatMulFallback(a, b, c, m, k, n, dataType);
            }
        }

        /// <summary>
        /// Executes BF16 matrix multiplication using AMX tiles.
        /// </summary>
        private static unsafe void ExecuteAMXMatMulBF16(
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            // Tile layout: A=tile0, B=tile1, C=tile2
            const byte tileA = 0;
            const byte tileB = 1;
            const byte tileC = 2;

            // Zero the result tile
            TileZero(tileC);

            // Process matrix in tile-sized chunks
            const int tileSize = 16; // 16x16 BF16 elements per tile
            
            for (int mi = 0; mi < m; mi += tileSize)
            {
                for (int ni = 0; ni < n; ni += tileSize)
                {
                    for (int ki = 0; ki < k; ki += tileSize)
                    {
                        // Load A tile
                        var aOffset = new IntPtr((byte*)a + (mi * k + ki) * sizeof(ushort));
                        TileLoad(tileA, aOffset, k * sizeof(ushort));

                        // Load B tile
                        var bOffset = new IntPtr((byte*)b + (ki * n + ni) * sizeof(ushort));
                        TileLoad(tileB, bOffset, n * sizeof(ushort));

                        // Perform tile matrix multiplication
                        TileMatMulBF16(tileA, tileB, tileC);
                    }

                    // Store result tile
                    var cOffset = new IntPtr((byte*)c + (mi * n + ni) * sizeof(float));
                    TileStore(tileC, cOffset, n * sizeof(float));
                }
            }
        }

        /// <summary>
        /// Executes INT8 matrix multiplication using AMX tiles.
        /// </summary>
        private static unsafe void ExecuteAMXMatMulINT8(
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            const byte tileA = 0;
            const byte tileB = 1;
            const byte tileC = 2;

            TileZero(tileC);

            const int tileSize = 16; // 16x16 INT8 elements per tile
            
            for (int mi = 0; mi < m; mi += tileSize)
            {
                for (int ni = 0; ni < n; ni += tileSize)
                {
                    for (int ki = 0; ki < k; ki += tileSize)
                    {
                        var aOffset = new IntPtr((byte*)a + (mi * k + ki));
                        TileLoad(tileA, aOffset, k);

                        var bOffset = new IntPtr((byte*)b + (ki * n + ni));
                        TileLoad(tileB, bOffset, n);

                        TileMatMulINT8(tileA, tileB, tileC);
                    }

                    var cOffset = new IntPtr((byte*)c + (mi * n + ni) * sizeof(int));
                    TileStore(tileC, cOffset, n * sizeof(int));
                }
            }
        }

        /// <summary>
        /// Executes UINT8 matrix multiplication using AMX tiles.
        /// </summary>
        private static unsafe void ExecuteAMXMatMulUINT8(
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            const byte tileA = 0;
            const byte tileB = 1;
            const byte tileC = 2;

            TileZero(tileC);

            const int tileSize = 16;
            
            for (int mi = 0; mi < m; mi += tileSize)
            {
                for (int ni = 0; ni < n; ni += tileSize)
                {
                    for (int ki = 0; ki < k; ki += tileSize)
                    {
                        var aOffset = new IntPtr((byte*)a + (mi * k + ki));
                        TileLoad(tileA, aOffset, k);

                        var bOffset = new IntPtr((byte*)b + (ki * n + ni));
                        TileLoad(tileB, bOffset, n);

                        TileMatMulUINT8(tileA, tileB, tileC);
                    }

                    var cOffset = new IntPtr((byte*)c + (mi * n + ni) * sizeof(int));
                    TileStore(tileC, cOffset, n * sizeof(int));
                }
            }
        }

        /// <summary>
        /// Executes mixed precision matrix multiplication using AMX tiles.
        /// </summary>
        private static unsafe void ExecuteAMXMatMulMixed(
            void* a, void* b, void* c,
            int m, int k, int n)
        {
            const byte tileA = 0;
            const byte tileB = 1;
            const byte tileC = 2;

            TileZero(tileC);

            const int tileSize = 16;
            
            for (int mi = 0; mi < m; mi += tileSize)
            {
                for (int ni = 0; ni < n; ni += tileSize)
                {
                    for (int ki = 0; ki < k; ki += tileSize)
                    {
                        var aOffset = new IntPtr((byte*)a + (mi * k + ki));
                        TileLoad(tileA, aOffset, k);

                        var bOffset = new IntPtr((byte*)b + (ki * n + ni));
                        TileLoad(tileB, bOffset, n);

                        TileMatMulMixed(tileA, tileB, tileC);
                    }

                    var cOffset = new IntPtr((byte*)c + (mi * n + ni) * sizeof(int));
                    TileStore(tileC, cOffset, n * sizeof(int));
                }
            }
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Creates tile configuration for matrix operations.
        /// </summary>
        /// <param name="m">Matrix rows.</param>
        /// <param name="k">Matrix inner dimension.</param>
        /// <param name="n">Matrix columns.</param>
        /// <param name="dataType">Data type for computation.</param>
        /// <returns>Tile configuration.</returns>
        private static AMXTileConfig CreateTileConfig(int m, int k, int n, AMXDataType dataType)
        {
            var config = new AMXTileConfig
            {
                palette_id = 1, // Use palette 1 for matrix operations
                start_row = 0,
                reserved1 = 0,
                reserved2 = 0,
                colsb = new byte[16],
                rows = new byte[16]
            };

            // Configure tiles based on data type
            switch (dataType)
            {
                case AMXDataType.BF16:
                    // BF16: 16 elements per row = 32 bytes
                    config.colsb[0] = 32; // Tile A
                    config.colsb[1] = 32; // Tile B
                    config.colsb[2] = 64; // Tile C (FP32 output)
                    config.rows[0] = Math.Min((byte)16, (byte)m);
                    config.rows[1] = Math.Min((byte)16, (byte)k);
                    config.rows[2] = Math.Min((byte)16, (byte)m);
                    break;
                    
                case AMXDataType.INT8:
                case AMXDataType.UINT8:
                case AMXDataType.Mixed:
                    // INT8/UINT8: 16 elements per row = 16 bytes
                    config.colsb[0] = 16; // Tile A
                    config.colsb[1] = 16; // Tile B
                    config.colsb[2] = 64; // Tile C (INT32 output)
                    config.rows[0] = Math.Min((byte)16, (byte)m);
                    config.rows[1] = Math.Min((byte)16, (byte)k);
                    config.rows[2] = Math.Min((byte)16, (byte)m);
                    break;
                    
                default:
                    throw new NotSupportedException($"Data type {dataType} not supported");
            }

            return config;
        }

        /// <summary>
        /// Checks CPUID for AMX support.
        /// </summary>
        /// <returns>True if AMX is supported.</returns>
        [DllImport(AMXLibrary, EntryPoint = "check_amx_support", CallingConvention = CallingConvention.Cdecl)]
        private static extern bool CheckAMXCPUID();

        /// <summary>
        /// CPU fallback for matrix multiplication.
        /// </summary>
        private static unsafe void ExecuteCPUMatMulFallback(
            void* a, void* b, void* c,
            int m, int k, int n,
            AMXDataType dataType)
        {
            // Basic CPU implementation for fallback
            switch (dataType)
            {
                case AMXDataType.BF16:
                    ExecuteCPUMatMulBF16(a, b, c, m, k, n);
                    break;
                case AMXDataType.INT8:
                    ExecuteCPUMatMulINT8(a, b, c, m, k, n);
                    break;
                case AMXDataType.UINT8:
                    ExecuteCPUMatMulUINT8(a, b, c, m, k, n);
                    break;
                case AMXDataType.Mixed:
                    ExecuteCPUMatMulMixed(a, b, c, m, k, n);
                    break;
            }
        }

        private static unsafe void ExecuteCPUMatMulBF16(void* a, void* b, void* c, int m, int k, int n)
        {
            var aPtr = (ushort*)a; // BF16 as ushort
            var bPtr = (ushort*)b;
            var cPtr = (float*)c;   // Output as FP32

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0.0f;
                    for (int ki = 0; ki < k; ki++)
                    {
                        // Convert BF16 to FP32 for computation
                        float aVal = BF16ToFloat(aPtr[i * k + ki]);
                        float bVal = BF16ToFloat(bPtr[ki * n + j]);
                        sum += aVal * bVal;
                    }
                    cPtr[i * n + j] = sum;
                }
            }
        }

        private static unsafe void ExecuteCPUMatMulINT8(void* a, void* b, void* c, int m, int k, int n)
        {
            var aPtr = (sbyte*)a;
            var bPtr = (sbyte*)b;
            var cPtr = (int*)c;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int sum = 0;
                    for (int ki = 0; ki < k; ki++)
                        sum += aPtr[i * k + ki] * bPtr[ki * n + j];
                    cPtr[i * n + j] = sum;
                }
            }
        }

        private static unsafe void ExecuteCPUMatMulUINT8(void* a, void* b, void* c, int m, int k, int n)
        {
            var aPtr = (byte*)a;
            var bPtr = (byte*)b;
            var cPtr = (int*)c;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int sum = 0;
                    for (int ki = 0; ki < k; ki++)
                        sum += aPtr[i * k + ki] * bPtr[ki * n + j];
                    cPtr[i * n + j] = sum;
                }
            }
        }

        private static unsafe void ExecuteCPUMatMulMixed(void* a, void* b, void* c, int m, int k, int n)
        {
            var aPtr = (byte*)a;    // UINT8
            var bPtr = (sbyte*)b;   // INT8
            var cPtr = (int*)c;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int sum = 0;
                    for (int ki = 0; ki < k; ki++)
                        sum += aPtr[i * k + ki] * bPtr[ki * n + j];
                    cPtr[i * n + j] = sum;
                }
            }
        }

        /// <summary>
        /// Converts BF16 to FP32.
        /// </summary>
        private static float BF16ToFloat(ushort bf16)
        {
            // BF16 is the upper 16 bits of FP32
            uint fp32Bits = (uint)bf16 << 16;
            return BitConverter.Int32BitsToSingle((int)fp32Bits);
        }

        #endregion
    }

    #region AMX Enums and Structures

    /// <summary>
    /// AMX data types supported for matrix operations.
    /// </summary>
    internal enum AMXDataType
    {
        BF16 = 0,    // Brain Floating Point 16-bit
        INT8 = 1,    // Signed 8-bit integer
        UINT8 = 2,   // Unsigned 8-bit integer
        Mixed = 3    // Mixed UINT8/INT8 precision
    }

    /// <summary>
    /// AMX exception for error handling.
    /// </summary>
    public class AMXException : Exception
    {
        public AMXException() { }
        public AMXException(string message) : base(message) { }
        public AMXException(string message, Exception innerException) : base(message, innerException) { }
    }

    #endregion
}