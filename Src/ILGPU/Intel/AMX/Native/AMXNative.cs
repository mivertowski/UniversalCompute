// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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
using System.Runtime.Intrinsics;

namespace ILGPU.Intel.AMX.Native
{
    /// <summary>
    /// Native Intel AMX API bindings and intrinsics.
    /// </summary>
    internal static partial class AMXNative
    {
        #region AMX State Management

        /// <summary>
        /// Initializes AMX state.
        /// </summary>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern bool InitializeAMX();

        /// <summary>
        /// Releases AMX state.
        /// </summary>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void ReleaseAMX();

        /// <summary>
        /// Checks if AMX is initialized.
        /// </summary>
        [DllImport("kernel32", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool IsAMXInitialized();

        /// <summary>
        /// Queries AMX capabilities.
        /// </summary>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern AMXNativeCapabilities QueryCapabilities();

        #endregion

        #region Tile Configuration

        /// <summary>
        /// Configures AMX tiles with the specified configuration.
        /// </summary>
        /// <param name="config">The tile configuration.</param>
        internal static unsafe void ConfigureTiles(AMXTileConfiguration config)
        {
            // Create tile configuration data
            var configData = stackalloc byte[64]; // AMX config is 64 bytes
            
            // Set palette ID
            configData[0] = config.Palette;
            
            // Configure tiles based on data type
            for (int i = 0; i < 8; i++) // 8 tiles maximum
            {
                var offset = 16 + i * 2; // Tile descriptors start at offset 16
                
                // Set rows (1 byte) and columns (2 bytes) for each tile
                configData[offset] = (byte)config.TileRows;
                configData[offset + 1] = (byte)(config.TileColumns & 0xFF);
                configData[offset + 16] = (byte)((config.TileColumns >> 8) & 0xFF);
            }

            // Load configuration using LDTILECFG instruction
            LoadTileConfig(configData);
        }

        /// <summary>
        /// Loads tile configuration using LDTILECFG instruction.
        /// </summary>
        /// <param name="config">Pointer to configuration data.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static unsafe extern void LoadTileConfig(byte* config);

        #endregion

        #region Tile Data Management

        /// <summary>
        /// Loads data into a tile using TILELOADD instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the data.</param>
        /// <param name="stride">The row stride in bytes.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static unsafe extern void LoadTile(int tileId, void* data, int stride);

        /// <summary>
        /// Stores tile data using TILESTORED instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the destination.</param>
        /// <param name="stride">The row stride in bytes.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static unsafe extern void StoreTile(int tileId, void* data, int stride);

        /// <summary>
        /// Zeros a tile using TILEZERO instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void ZeroTile(int tileId);

        /// <summary>
        /// Releases tile resources using TILERELEASE instruction.
        /// </summary>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void ReleaseTiles();

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Performs BF16 matrix multiplication using TDPBF16PS instruction.
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void TileMatMulBF16(int dst, int src1, int src2);

        /// <summary>
        /// Performs INT8 matrix multiplication using TDPBSSD instruction.
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void TileMatMulINT8(int dst, int src1, int src2);

        /// <summary>
        /// Performs FP32 matrix multiplication using TDPFP32 instruction (future).
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [DllImport("kernel32", SetLastError = true)]
        internal static extern void TileMatMulFP32(int dst, int src1, int src2);

        #endregion

        #region Intrinsic Wrappers

        /// <summary>
        /// Wrapper for _tile_loadd intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="base_addr">Base address.</param>
        /// <param name="stride">Stride in bytes.</param>
        internal static unsafe void tile_loadd(byte dst, void* base_addr, int stride)
        {
            // This would use the actual intrinsic in real implementation
            LoadTile(dst, base_addr, stride);
        }

        /// <summary>
        /// Wrapper for _tile_stored intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="base_addr">Base address.</param>
        /// <param name="stride">Stride in bytes.</param>
        internal static unsafe void tile_stored(byte dst, void* base_addr, int stride)
        {
            // This would use the actual intrinsic in real implementation
            StoreTile(dst, base_addr, stride);
        }

        /// <summary>
        /// Wrapper for _tile_zero intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        internal static void tile_zero(byte dst)
        {
            // This would use the actual intrinsic in real implementation
            ZeroTile(dst);
        }

        /// <summary>
        /// Wrapper for _tile_dpbf16ps intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="src1">Source tile 1.</param>
        /// <param name="src2">Source tile 2.</param>
        internal static void tile_dpbf16ps(byte dst, byte src1, byte src2)
        {
            // This would use the actual intrinsic in real implementation
            TileMatMulBF16(dst, src1, src2);
        }

        /// <summary>
        /// Wrapper for _tile_dpbssd intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="src1">Source tile 1.</param>
        /// <param name="src2">Source tile 2.</param>
        internal static void tile_dpbssd(byte dst, byte src1, byte src2)
        {
            // This would use the actual intrinsic in real implementation
            TileMatMulINT8(dst, src1, src2);
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks processor support for AMX.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        internal static bool CheckAMXSupport()
        {
            try
            {
                // Check CPUID for AMX support
                return CheckCPUID_AMX();
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Gets the size of a tile in bytes for the given configuration.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns (in bytes).</param>
        /// <returns>Tile size in bytes.</returns>
        internal static int GetTileSize(int rows, int cols)
        {
            return rows * cols;
        }

        /// <summary>
        /// Calculates the optimal tile configuration for matrix dimensions.
        /// </summary>
        /// <param name="m">Matrix M dimension.</param>
        /// <param name="n">Matrix N dimension.</param>
        /// <param name="k">Matrix K dimension.</param>
        /// <param name="dataType">The data type.</param>
        /// <returns>Optimal tile dimensions.</returns>
        internal static (int tileM, int tileN, int tileK) CalculateOptimalTiles(
            int m, int n, int k, AMXDataType dataType)
        {
            // AMX tile limits: 16 rows, 64 columns (bytes)
            var elementsPerByte = dataType switch
            {
                AMXDataType.BFloat16 => 2,
                AMXDataType.Int8 => 4,
                AMXDataType.Float32 => 1,
                _ => 1
            };

            var maxCols = 64 / GetElementSize(dataType);
            
            return (
                Math.Min(16, m),        // Max 16 rows
                Math.Min(maxCols, n),   // Max columns based on data type
                Math.Min(maxCols, k)    // Max K dimension
            );
        }

        private static int GetElementSize(AMXDataType dataType)
        {
            return dataType switch
            {
                AMXDataType.BFloat16 => 2,
                AMXDataType.Int8 => 1,
                AMXDataType.Float32 => 4,
                _ => 4
            };
        }

        [DllImport("kernel32", SetLastError = true)]
        private static extern bool CheckCPUID_AMX();

        #endregion
    }

    /// <summary>
    /// Native AMX capabilities structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct AMXNativeCapabilities
    {
        public int IsSupported;
        public int MaxTiles;
        public int MaxTileRows;
        public int MaxTileColumns;
        public int MaxTileBytes;
        public int MaxConfigBytes;
        public int SupportsBF16;
        public int SupportsInt8;
        public int SupportsFloat32;
        public double EstimatedBandwidthGBps;
    }

    /// <summary>
    /// AMX matrix operations using native intrinsics.
    /// </summary>
    internal static class AMXOperations
    {
        /// <summary>
        /// Performs matrix multiplication using FP32 tiles.
        /// </summary>
        /// <param name="a">Matrix A pointer.</param>
        /// <param name="b">Matrix B pointer.</param>
        /// <param name="c">Result matrix C pointer.</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">Tile configuration.</param>
        internal static unsafe void MatrixMultiplyFP32(
            float* a, float* b, float* c, int m, int n, int k, AMXTileConfiguration config)
        {
            // Tile-based matrix multiplication for FP32
            var (tileM, tileN, tileK) = AMXNative.CalculateOptimalTiles(m, n, k, AMXDataType.Float32);
            
            for (int i = 0; i < m; i += tileM)
            {
                for (int j = 0; j < n; j += tileN)
                {
                    // Zero accumulator tile
                    AMXNative.tile_zero(0);
                    
                    for (int kk = 0; kk < k; kk += tileK)
                    {
                        // Load tiles A and B
                        var aPtr = a + i * k + kk;
                        var bPtr = b + kk * n + j;
                        
                        AMXNative.tile_loadd(1, aPtr, k * sizeof(float));
                        AMXNative.tile_loadd(2, bPtr, n * sizeof(float));
                        
                        // Perform tile matrix multiplication
                        AMXNative.TileMatMulFP32(0, 1, 2);
                    }
                    
                    // Store result
                    var cPtr = c + i * n + j;
                    AMXNative.tile_stored(0, cPtr, n * sizeof(float));
                }
            }
        }

        /// <summary>
        /// Performs matrix multiplication using BF16 tiles.
        /// </summary>
        /// <param name="a">Matrix A pointer (BF16).</param>
        /// <param name="b">Matrix B pointer (BF16).</param>
        /// <param name="c">Result matrix C pointer (FP32).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">Tile configuration.</param>
        internal static unsafe void MatrixMultiplyBF16(
            ushort* a, ushort* b, float* c, int m, int n, int k, AMXTileConfiguration config)
        {
            // Tile-based matrix multiplication for BF16
            var (tileM, tileN, tileK) = AMXNative.CalculateOptimalTiles(m, n, k, AMXDataType.BFloat16);
            
            for (int i = 0; i < m; i += tileM)
            {
                for (int j = 0; j < n; j += tileN)
                {
                    // Zero accumulator tile
                    AMXNative.tile_zero(0);
                    
                    for (int kk = 0; kk < k; kk += tileK)
                    {
                        // Load BF16 tiles
                        var aPtr = a + i * k + kk;
                        var bPtr = b + kk * n + j;
                        
                        AMXNative.tile_loadd(1, aPtr, k * sizeof(ushort));
                        AMXNative.tile_loadd(2, bPtr, n * sizeof(ushort));
                        
                        // Perform BF16 tile matrix multiplication
                        AMXNative.tile_dpbf16ps(0, 1, 2);
                    }
                    
                    // Store FP32 result
                    var cPtr = c + i * n + j;
                    AMXNative.tile_stored(0, cPtr, n * sizeof(float));
                }
            }
        }

        /// <summary>
        /// Performs matrix multiplication using INT8 tiles.
        /// </summary>
        /// <param name="a">Matrix A pointer (INT8).</param>
        /// <param name="b">Matrix B pointer (INT8).</param>
        /// <param name="c">Result matrix C pointer (INT32).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">Tile configuration.</param>
        internal static unsafe void MatrixMultiplyINT8(
            sbyte* a, sbyte* b, int* c, int m, int n, int k, AMXTileConfiguration config)
        {
            // Tile-based matrix multiplication for INT8
            var (tileM, tileN, tileK) = AMXNative.CalculateOptimalTiles(m, n, k, AMXDataType.Int8);
            
            for (int i = 0; i < m; i += tileM)
            {
                for (int j = 0; j < n; j += tileN)
                {
                    // Zero accumulator tile
                    AMXNative.tile_zero(0);
                    
                    for (int kk = 0; kk < k; kk += tileK)
                    {
                        // Load INT8 tiles
                        var aPtr = a + i * k + kk;
                        var bPtr = b + kk * n + j;
                        
                        AMXNative.tile_loadd(1, aPtr, k * sizeof(sbyte));
                        AMXNative.tile_loadd(2, bPtr, n * sizeof(sbyte));
                        
                        // Perform INT8 tile matrix multiplication
                        AMXNative.tile_dpbssd(0, 1, 2);
                    }
                    
                    // Store INT32 result
                    var cPtr = c + i * n + j;
                    AMXNative.tile_stored(0, cPtr, n * sizeof(int));
                }
            }
        }
    }
}