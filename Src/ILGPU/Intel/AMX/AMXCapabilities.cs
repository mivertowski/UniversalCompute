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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Represents the capabilities of Intel Advanced Matrix Extensions (AMX).
    /// </summary>
    /// <remarks>
    /// Initializes a new instance of the AMXCapabilities struct.
    /// </remarks>
    /// <param name="isSupported">Whether AMX is supported.</param>
    /// <param name="maxTiles">Maximum number of tiles.</param>
    /// <param name="maxTileRows">Maximum tile rows.</param>
    /// <param name="maxTileColumns">Maximum tile columns.</param>
    /// <param name="maxTileBytes">Maximum tile size in bytes.</param>
    /// <param name="maxConfigBytes">Maximum configuration size in bytes.</param>
    /// <param name="supportsBF16">Whether BF16 is supported.</param>
    /// <param name="supportsInt8">Whether INT8 is supported.</param>
    /// <param name="supportsFloat32">Whether Float32 is supported.</param>
    /// <param name="estimatedBandwidthGBps">Estimated bandwidth in GB/s.</param>
    public readonly struct AMXCapabilities(
        bool isSupported,
        int maxTiles,
        int maxTileRows,
        int maxTileColumns,
        int maxTileBytes,
        int maxConfigBytes,
        bool supportsBF16,
        bool supportsInt8,
        bool supportsFloat32,
        double estimatedBandwidthGBps)
    {

        /// <summary>
        /// Gets whether AMX is supported on this processor.
        /// </summary>
        public bool IsSupported { get; } = isSupported;

        /// <summary>
        /// Gets the maximum number of tiles (typically 8).
        /// </summary>
        public int MaxTiles { get; } = maxTiles;

        /// <summary>
        /// Gets the maximum number of rows per tile (typically 16).
        /// </summary>
        public int MaxTileRows { get; } = maxTileRows;

        /// <summary>
        /// Gets the maximum number of columns per tile (typically 64 bytes).
        /// </summary>
        public int MaxTileColumns { get; } = maxTileColumns;

        /// <summary>
        /// Gets the maximum tile size in bytes (typically 1024).
        /// </summary>
        public int MaxTileBytes { get; } = maxTileBytes;

        /// <summary>
        /// Gets the maximum configuration size in bytes (typically 64).
        /// </summary>
        public int MaxConfigBytes { get; } = maxConfigBytes;

        /// <summary>
        /// Gets whether BFloat16 operations are supported.
        /// </summary>
        public bool SupportsBF16 { get; } = supportsBF16;

        /// <summary>
        /// Gets whether INT8 operations are supported.
        /// </summary>
        public bool SupportsInt8 { get; } = supportsInt8;

        /// <summary>
        /// Gets whether Float32 operations are supported.
        /// </summary>
        public bool SupportsFloat32 { get; } = supportsFloat32;

        /// <summary>
        /// Gets the estimated memory bandwidth in GB/s.
        /// </summary>
        public double EstimatedBandwidthGBps { get; } = estimatedBandwidthGBps;

        /// <summary>
        /// Gets the estimated peak performance in GOPS for the given data type.
        /// </summary>
        /// <param name="dataType">The data type.</param>
        /// <returns>Estimated GOPS performance.</returns>
        public static double GetEstimatedPerformance(AMXDataType dataType) =>
            // Rough estimates based on Sapphire Rapids specifications
            dataType switch
            {
                AMXDataType.BFloat16 => 512.0, // ~512 GOPS for BF16
                AMXDataType.Int8 => 1024.0,    // ~1024 GOPS for INT8
                AMXDataType.Float32 => 256.0,  // ~256 GOPS for FP32
                _ => 100.0
            };

        /// <summary>
        /// Gets the optimal tile configuration for matrix dimensions.
        /// </summary>
        /// <param name="m">Matrix M dimension.</param>
        /// <param name="n">Matrix N dimension.</param>
        /// <param name="k">Matrix K dimension.</param>
        /// <param name="dataType">The data type.</param>
        /// <returns>Optimal tile configuration.</returns>
        public (int tileM, int tileN, int tileK) GetOptimalTileSize(int m, int n, int k, AMXDataType dataType)
        {
            var elementsPerByte = dataType switch
            {
                AMXDataType.BFloat16 => 2, // 2 BF16 elements per 4-byte tile element
                AMXDataType.Int8 => 4,     // 4 INT8 elements per 4-byte tile element
                AMXDataType.Float32 => 1,  // 1 FP32 element per 4-byte tile element
                _ => 1
            };

            var maxK = MaxTileColumns / 4 * elementsPerByte; // 64 bytes / 4 = 16, then scale by elements per byte
            var maxM = Math.Min(MaxTileRows, m);
            var maxN = Math.Min(MaxTileColumns / 4 * elementsPerByte, n);
            var optimalK = Math.Min(maxK, k);

            return (maxM, maxN, optimalK);
        }

        /// <summary>
        /// Checks if AMX is supported on the current processor.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        public static bool IsAMXSupported()
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Check for AMX support via CPUID
                return X86Base.IsSupported && CheckAMXSupport();
            }
            catch
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Queries AMX capabilities from the current processor.
        /// </summary>
        /// <returns>AMX capabilities structure.</returns>
        public static AMXCapabilities Query()
        {
            if (!IsAMXSupported())
            {
                return new AMXCapabilities(
                    false, 0, 0, 0, 0, 0, false, false, false, 0.0);
            }

#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                var nativeCapabilities = AMXNative.QueryCapabilities();
                return MapFromNative(nativeCapabilities);
            }
            catch
            {
                return new AMXCapabilities(
                    false, 0, 0, 0, 0, 0, false, false, false, 0.0);
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        private static bool CheckAMXSupport()
        {
            // Check CPUID leaf 7, sub-leaf 0, EDX bit 24 for AMX-TILE
            // and bit 25 for AMX-INT8, bit 22 for AMX-BF16
            unsafe
            {
                var cpuidResult = new int[4];
                fixed (int* ptr = cpuidResult)
                {
                    // This would use native CPUID instruction
                    // For now, return false as a safe default
                    return false;
                }
            }
        }

        private static AMXCapabilities MapFromNative(AMXNativeCapabilities native) => new(
                native.IsSupported != 0,
                native.MaxTiles,
                native.MaxTileRows,
                native.MaxTileColumns,
                native.MaxTileBytes,
                native.MaxConfigBytes,
                native.SupportsBF16 != 0,
                native.SupportsInt8 != 0,
                native.SupportsFloat32 != 0,
                native.EstimatedBandwidthGBps
            );

        /// <summary>
        /// Returns a string representation of the AMX capabilities.
        /// </summary>
        /// <returns>A string describing the AMX capabilities.</returns>
        public override string ToString() => !IsSupported
                ? "Intel AMX: Not Supported"
                : $"Intel AMX: {MaxTiles} tiles, {MaxTileRows}x{MaxTileColumns} max, " +
                   $"BF16={SupportsBF16}, INT8={SupportsInt8}, FP32={SupportsFloat32}, " +
                   $"Bandwidth={EstimatedBandwidthGBps:F1} GB/s";
    }

    /// <summary>
    /// AMX data types for tile operations.
    /// </summary>
    public enum AMXDataType
    {
        /// <summary>
        /// BFloat16 data type (16-bit brain floating point).
        /// </summary>
        BFloat16,

        /// <summary>
        /// INT8 data type (8-bit signed integer).
        /// </summary>
        Int8,

        /// <summary>
        /// Float32 data type (32-bit floating point).
        /// </summary>
        Float32
    }

    /// <summary>
    /// Represents an Intel AMX tile descriptor with layout and configuration information.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AMXTileDescriptor
    {
        /// <summary>
        /// Gets or sets the number of rows in the tile (max 16).
        /// </summary>
        public byte Rows { get; set; }

        /// <summary>
        /// Gets or sets the number of bytes per row (max 64).
        /// </summary>
        public ushort BytesPerRow { get; set; }

        /// <summary>
        /// Gets or sets the data type for this tile.
        /// </summary>
        public AMXDataType DataType { get; set; }

        /// <summary>
        /// Gets or sets the tile ID (0-7).
        /// </summary>
        public byte TileId { get; set; }

        /// <summary>
        /// Gets or sets whether this tile is active.
        /// </summary>
        public bool IsActive { get; set; }

        /// <summary>
        /// Gets the size of the tile in bytes.
        /// </summary>
        public readonly int TileSize => Rows * BytesPerRow;

        /// <summary>
        /// Gets the number of elements per row based on the data type.
        /// </summary>
        public readonly int ElementsPerRow => DataType switch
        {
            AMXDataType.BFloat16 => BytesPerRow / 2,
            AMXDataType.Int8 => BytesPerRow,
            AMXDataType.Float32 => BytesPerRow / 4,
            _ => BytesPerRow / 4
        };

        /// <summary>
        /// Creates a new tile descriptor with the specified parameters.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="rows">Number of rows (max 16).</param>
        /// <param name="bytesPerRow">Bytes per row (max 64).</param>
        /// <param name="dataType">Data type for the tile.</param>
        /// <param name="isActive">Whether the tile is active.</param>
        public AMXTileDescriptor(byte tileId, byte rows, ushort bytesPerRow, AMXDataType dataType, bool isActive = true)
        {
            TileId = tileId;
            Rows = rows;
            BytesPerRow = bytesPerRow;
            DataType = dataType;
            IsActive = isActive;
        }

        /// <summary>
        /// Creates a default tile descriptor for matrix operations.
        /// </summary>
        /// <param name="tileId">The tile ID.</param>
        /// <param name="dataType">The data type.</param>
        /// <returns>A default tile descriptor.</returns>
        public static AMXTileDescriptor CreateDefault(byte tileId, AMXDataType dataType)
        {
            var bytesPerRow = dataType switch
            {
                AMXDataType.BFloat16 => (ushort)32, // 16 BF16 elements * 2 bytes
                AMXDataType.Int8 => (ushort)64,     // 64 INT8 elements * 1 byte
                AMXDataType.Float32 => (ushort)64,  // 16 FP32 elements * 4 bytes
                _ => (ushort)64
            };

            return new AMXTileDescriptor(tileId, 16, bytesPerRow, dataType, true);
        }

        /// <summary>
        /// Validates the tile descriptor parameters.
        /// </summary>
        /// <returns>True if valid; otherwise, false.</returns>
        public readonly bool IsValid()
        {
            return TileId < 8 && Rows > 0 && Rows <= 16 && BytesPerRow > 0 && BytesPerRow <= 64;
        }
    }

    /// <summary>
    /// Native AMX tile configuration structure for direct memory layout.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct AMXNativeTileConfiguration
    {
        /// <summary>
        /// Palette ID (1 byte).
        /// </summary>
        public byte Palette;
        
        /// <summary>
        /// Reserved bytes (15 bytes).
        /// </summary>
        public unsafe fixed byte Reserved[15];
        
        /// <summary>
        /// Tile rows configuration (8 bytes, one per tile).
        /// </summary>
        public unsafe fixed byte TileRows[8];
        
        /// <summary>
        /// Tile columns configuration (16 bytes, two per tile).
        /// </summary>
        public unsafe fixed byte TileColumns[16];
        
        /// <summary>
        /// Additional reserved bytes (24 bytes).
        /// </summary>
        public unsafe fixed byte Reserved2[24];
    }

    /// <summary>
    /// AMX tile configuration for matrix operations.
    /// </summary>
    public sealed class AMXTileConfiguration
    {
        /// <summary>
        /// Gets or sets the data type for tile operations.
        /// </summary>
        public AMXDataType DataType { get; set; }

        /// <summary>
        /// Gets or sets the number of rows in tiles.
        /// </summary>
        public int TileRows { get; set; }

        /// <summary>
        /// Gets or sets the number of columns in tiles (in bytes).
        /// </summary>
        public int TileColumns { get; set; }

        /// <summary>
        /// Gets or sets the palette configuration.
        /// </summary>
        public byte Palette { get; set; }

        /// <summary>
        /// Gets the tile descriptors for all 8 AMX tiles.
        /// </summary>
        public IList<AMXTileDescriptor> Tiles { get; } = new AMXTileDescriptor[8];

        /// <summary>
        /// Converts this configuration to a native configuration structure.
        /// </summary>
        /// <returns>Native configuration structure.</returns>
        public unsafe AMXNativeTileConfiguration ToNative()
        {
            var nativeConfig = new AMXNativeTileConfiguration
            {
                Palette = Palette
            };

            // Configure tiles
            for (int i = 0; i < 8; i++)
            {
                var tile = Tiles[i];
                nativeConfig.TileRows[i] = tile.IsActive ? tile.Rows : (byte)0;
                nativeConfig.TileColumns[i * 2] = tile.IsActive ? (byte)(tile.BytesPerRow & 0xFF) : (byte)0;
                nativeConfig.TileColumns[i * 2 + 1] = tile.IsActive ? (byte)((tile.BytesPerRow >> 8) & 0xFF) : (byte)0;
            }

            return nativeConfig;
        }

        /// <summary>
        /// Creates a default tile configuration for the given capabilities.
        /// </summary>
        /// <param name="capabilities">The AMX capabilities.</param>
        /// <returns>A default tile configuration.</returns>
        public static AMXTileConfiguration CreateDefault(AMXCapabilities capabilities)
        {
            var config = new AMXTileConfiguration
            {
                DataType = capabilities.SupportsBF16 ? AMXDataType.BFloat16 : AMXDataType.Float32,
                TileRows = capabilities.MaxTileRows,
                TileColumns = capabilities.MaxTileColumns,
                Palette = 1 // Default palette
            };

            // Initialize all 8 tiles with default descriptors
            for (byte i = 0; i < 8; i++)
            {
                config.Tiles[i] = AMXTileDescriptor.CreateDefault(i, config.DataType);
            }

            return config;
        }

        /// <summary>
        /// Creates a tile configuration optimized for the given data type.
        /// </summary>
        /// <param name="dataType">The data type.</param>
        /// <param name="capabilities">The AMX capabilities.</param>
        /// <returns>An optimized tile configuration.</returns>
        public static AMXTileConfiguration CreateForDataType(AMXDataType dataType, AMXCapabilities capabilities)
        {
            var config = new AMXTileConfiguration
            {
                DataType = dataType,
                TileRows = capabilities.MaxTileRows,
                TileColumns = capabilities.MaxTileColumns,
                Palette = dataType switch
                {
                    AMXDataType.BFloat16 => 1,
                    AMXDataType.Int8 => 2,
                    AMXDataType.Float32 => 3,
                    _ => 1
                }
            };

            // Initialize all 8 tiles with descriptors for the specified data type
            for (byte i = 0; i < 8; i++)
            {
                config.Tiles[i] = AMXTileDescriptor.CreateDefault(i, dataType);
            }

            return config;
        }

        /// <summary>
        /// Creates a new configuration with the specified data type.
        /// </summary>
        /// <param name="dataType">The new data type.</param>
        /// <returns>A new tile configuration.</returns>
        public AMXTileConfiguration WithDataType(AMXDataType dataType)
        {
            var config = new AMXTileConfiguration
            {
                DataType = dataType,
                TileRows = TileRows,
                TileColumns = TileColumns,
                Palette = dataType switch
                {
                    AMXDataType.BFloat16 => 1,
                    AMXDataType.Int8 => 2,
                    AMXDataType.Float32 => 3,
                    _ => Palette
                }
            };

            // Initialize all 8 tiles with descriptors for the new data type
            for (byte i = 0; i < 8; i++)
            {
                config.Tiles[i] = AMXTileDescriptor.CreateDefault(i, dataType);
            }

            return config;
        }
    }
}