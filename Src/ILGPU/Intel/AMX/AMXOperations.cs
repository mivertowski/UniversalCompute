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

using ILGPU.Intel.AMX.Native;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Low-level AMX matrix operations.
    /// </summary>
    public static class AMXOperations
    {
        #region Matrix Multiplication

        /// <summary>
        /// Performs FP32 matrix multiplication using AMX tiles.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B.</param>
        /// <param name="c">Matrix C (output).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void MatrixMultiplyFP32(
            float* a, float* b, float* c,
            int m, int n, int k,
            AMXTileConfiguration config)
        {
            if (!AMXNative.IsAMXSupported())
                throw new NotSupportedException("AMX is not supported on this processor");

            // Configure tiles
            AMXNative.LoadTileConfig((byte*)&config);

            const int tileSize = 16; // Standard AMX tile size
            
            // Process blocks
            for (int i = 0; i < m; i += tileSize)
            {
                for (int j = 0; j < n; j += tileSize)
                {
                    // Zero the output tile
                    AMXNative.tile_zero(2);

                    for (int l = 0; l < k; l += tileSize)
                    {
                        // Load A tile
                        AMXNative.tile_loadd(0, a + i * k + l, k * sizeof(float));
                        
                        // Load B tile
                        AMXNative.tile_loadd(1, b + l * n + j, n * sizeof(float));
                        
                        // Perform tile multiplication
                        AMXNative.TileMatMulFP32(2, 0, 1);
                    }

                    // Store result tile
                    AMXNative.tile_stored(2, c + i * n + j, n * sizeof(float));
                }
            }

            // Release tiles
            AMXNative.ReleaseTiles();
        }

        /// <summary>
        /// Performs BF16 matrix multiplication using AMX tiles.
        /// </summary>
        /// <param name="a">Matrix A in BF16 format.</param>
        /// <param name="b">Matrix B in BF16 format.</param>
        /// <param name="c">Matrix C in FP32 format (output).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void MatrixMultiplyBF16(
            ushort* a, ushort* b, float* c,
            int m, int n, int k,
            AMXTileConfiguration config)
        {
            if (!AMXNative.IsAMXSupported())
                throw new NotSupportedException("AMX is not supported on this processor");

            // Configure tiles
            AMXNative.LoadTileConfig((byte*)&config);

            const int tileSize = 16; // Standard AMX tile size
            
            // Process blocks
            for (int i = 0; i < m; i += tileSize)
            {
                for (int j = 0; j < n; j += tileSize)
                {
                    // Zero the output tile
                    AMXNative.tile_zero(2);

                    for (int l = 0; l < k; l += tileSize)
                    {
                        // Load A tile
                        AMXNative.tile_loadd(0, a + i * k + l, k * sizeof(ushort));
                        
                        // Load B tile
                        AMXNative.tile_loadd(1, b + l * n + j, n * sizeof(ushort));
                        
                        // Perform tile multiplication
                        AMXNative.tile_dpbf16ps(2, 0, 1);
                    }

                    // Store result tile
                    AMXNative.tile_stored(2, c + i * n + j, n * sizeof(float));
                }
            }

            // Release tiles
            AMXNative.ReleaseTiles();
        }

        /// <summary>
        /// Performs INT8 matrix multiplication using AMX tiles.
        /// </summary>
        /// <param name="a">Matrix A in INT8 format.</param>
        /// <param name="b">Matrix B in INT8 format.</param>
        /// <param name="c">Matrix C in INT32 format (output).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B and C.</param>
        /// <param name="k">Number of columns in A and rows in B.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void MatrixMultiplyINT8(
            sbyte* a, sbyte* b, int* c,
            int m, int n, int k,
            AMXTileConfiguration config)
        {
            if (!AMXNative.IsAMXSupported())
                throw new NotSupportedException("AMX is not supported on this processor");

            // Configure tiles
            AMXNative.LoadTileConfig((byte*)&config);

            const int tileSize = 16; // Standard AMX tile size
            
            // Process blocks
            for (int i = 0; i < m; i += tileSize)
            {
                for (int j = 0; j < n; j += tileSize)
                {
                    // Zero the output tile
                    AMXNative.tile_zero(2);

                    for (int l = 0; l < k; l += tileSize)
                    {
                        // Load A tile
                        AMXNative.tile_loadd(0, a + i * k + l, k * sizeof(sbyte));
                        
                        // Load B tile
                        AMXNative.tile_loadd(1, b + l * n + j, n * sizeof(sbyte));
                        
                        // Perform tile multiplication
                        AMXNative.tile_dpbssd(2, 0, 1);
                    }

                    // Store result tile
                    AMXNative.tile_stored(2, c + i * n + j, n * sizeof(int));
                }
            }

            // Release tiles
            AMXNative.ReleaseTiles();
        }

        /// <summary>
        /// Performs FP32 matrix multiplication with transposed B matrix using AMX tiles.
        /// </summary>
        /// <param name="a">Matrix A.</param>
        /// <param name="b">Matrix B (will be transposed).</param>
        /// <param name="c">Matrix C (output).</param>
        /// <param name="m">Number of rows in A and C.</param>
        /// <param name="n">Number of columns in B^T and C.</param>
        /// <param name="k">Number of columns in A and columns in B.</param>
        /// <param name="config">AMX tile configuration.</param>
        public static unsafe void MatrixMultiplyTransposedFP32(
            float* a, float* b, float* c,
            int m, int n, int k,
            AMXTileConfiguration config)
        {
            if (!AMXNative.IsAMXSupported())
                throw new NotSupportedException("AMX is not supported on this processor");

            // Configure tiles
            AMXNative.LoadTileConfig((byte*)&config);

            const int tileSize = 16; // Standard AMX tile size
            
            // Process blocks
            for (int i = 0; i < m; i += tileSize)
            {
                for (int j = 0; j < n; j += tileSize)
                {
                    // Zero the output tile
                    AMXNative.tile_zero(2);

                    for (int l = 0; l < k; l += tileSize)
                    {
                        // Load A tile
                        AMXNative.tile_loadd(0, a + i * k + l, k * sizeof(float));
                        
                        // Load B tile (transposed access pattern)
                        // B is accessed as B[j][l] instead of B[l][j]
                        AMXNative.tile_loadd(1, b + j * k + l, k * sizeof(float));
                        
                        // Perform tile multiplication
                        AMXNative.TileMatMulFP32(2, 0, 1);
                    }

                    // Store result tile
                    AMXNative.tile_stored(2, c + i * n + j, n * sizeof(float));
                }
            }

            // Release tiles
            AMXNative.ReleaseTiles();
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Creates a default AMX tile configuration for the given data type.
        /// </summary>
        /// <param name="dataType">The data type for matrix operations.</param>
        /// <param name="rows">Number of rows per tile.</param>
        /// <param name="cols">Number of columns per tile.</param>
        /// <returns>The AMX tile configuration.</returns>
        public static AMXTileConfiguration CreateConfiguration(
            AMXDataType dataType,
            int rows = 16,
            int cols = 16)
        {
            return new AMXTileConfiguration
            {
                DataType = dataType,
                TileRows = rows,
                TileColumns = cols,
                Tiles = new AMXTileDescriptor[8] // AMX supports up to 8 tiles
            };
        }

        #endregion
    }
}