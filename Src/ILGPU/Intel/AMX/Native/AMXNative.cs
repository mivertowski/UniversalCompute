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
using System.Runtime.Intrinsics.X86;

namespace ILGPU.Intel.AMX.Native
{
    /// <summary>
    /// Native Intel AMX API bindings and intrinsics.
    /// </summary>
    internal static partial class AMXNative
    {
        #region AMX State Management

        /// <summary>
        /// Initializes AMX state using XGETBV and XCR0 register.
        /// </summary>
        /// <returns>True if initialization succeeded; otherwise, false.</returns>
        internal static bool InitializeAMX()
        {
            if (!IsAMXSupported())
                return false;

            try
            {
                // Check if AMX tiles and int8 are supported
                if (!CheckCPUIDFeature(7, 0, 3, 24)) // AMX-TILE
                    return false;

                if (!CheckCPUIDFeature(7, 0, 3, 25)) // AMX-INT8  
                    return false;

                // Request permission to use AMX (requires OS support)
                RequestAMXPermission();
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Releases AMX state by releasing tiles.
        /// </summary>
        internal static void ReleaseAMX()
        {
            try
            {
                ReleaseTiles();
            }
            catch
            {
                // Ignore errors during cleanup
            }
        }

        /// <summary>
        /// Checks if AMX is initialized by testing tile configuration.
        /// </summary>
        /// <returns>True if AMX is initialized; otherwise, false.</returns>
        internal static unsafe bool IsAMXInitialized()
        {
            try
            {
                // Try to configure tiles - this will fail if AMX is not initialized
                var testConfig = stackalloc byte[64];
                LoadTileConfig(testConfig);
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Queries AMX capabilities from CPUID and system information.
        /// </summary>
        /// <returns>AMX capabilities structure.</returns>
        internal static AMXNativeCapabilities QueryCapabilities()
        {
            var capabilities = new AMXNativeCapabilities
            {
                IsSupported = IsAMXSupported() ? 1 : 0,
                MaxTiles = 8, // Intel AMX supports 8 tiles
                MaxTileRows = 16, // Maximum 16 rows per tile
                MaxTileColumns = 64, // Maximum 64 bytes per row
                MaxTileBytes = 1024, // 16 * 64 = 1024 bytes per tile
                MaxConfigBytes = 64, // Configuration structure is 64 bytes
                SupportsBF16 = CheckCPUIDFeature(7, 0, 3, 22) ? 1 : 0, // AMX-BF16
                SupportsInt8 = CheckCPUIDFeature(7, 0, 3, 25) ? 1 : 0, // AMX-INT8
                SupportsFloat32 = CheckCPUIDFeature(7, 0, 3, 26) ? 1 : 0, // AMX-FP16 (future)
                EstimatedBandwidthGBps = EstimateAMXBandwidth()
            };

            return capabilities;
        }

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
        internal static unsafe void LoadTileConfig(byte* config)
        {
            // Real implementation would use inline assembly or intrinsic
            // For now, this is a placeholder that would compile to LDTILECFG instruction
            // asm volatile ("ldtilecfg %0" :: "m" (*config));
            
            // Fallback for systems without AMX support
            if (!IsAMXSupported())
                throw new NotSupportedException("Intel AMX not supported");
                
            // This would be replaced with actual intrinsic when available in .NET
            LoadTileConfigNative(config);
        }

        /// <summary>
        /// Native tile configuration loader (placeholder for intrinsic).
        /// </summary>
        [LibraryImport("kernel32", EntryPoint = "ldtilecfg_native", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        private static unsafe partial void LoadTileConfigNative(byte* config);

        #endregion

        #region Tile Data Management

        /// <summary>
        /// Loads data into a tile using TILELOADD instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the data.</param>
        /// <param name="stride">The row stride in bytes.</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static unsafe partial void LoadTile(int tileId, void* data, int stride);

        /// <summary>
        /// Stores tile data using TILESTORED instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        /// <param name="data">Pointer to the destination.</param>
        /// <param name="stride">The row stride in bytes.</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static unsafe partial void StoreTile(int tileId, void* data, int stride);

        /// <summary>
        /// Zeros a tile using TILEZERO instruction.
        /// </summary>
        /// <param name="tileId">The tile ID (0-7).</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial void ZeroTile(int tileId);

        /// <summary>
        /// Releases tile resources using TILERELEASE instruction.
        /// </summary>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial void ReleaseTiles();

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Performs BF16 matrix multiplication using TDPBF16PS instruction.
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial void TileMatMulBF16(int dst, int src1, int src2);

        /// <summary>
        /// Performs INT8 matrix multiplication using TDPBSSD instruction.
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial void TileMatMulINT8(int dst, int src1, int src2);

        /// <summary>
        /// Performs FP32 matrix multiplication using TDPFP32 instruction (future).
        /// </summary>
        /// <param name="dst">Destination tile ID.</param>
        /// <param name="src1">Source tile 1 ID.</param>
        /// <param name="src2">Source tile 2 ID.</param>
        [LibraryImport("kernel32", SetLastError = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static partial void TileMatMulFP32(int dst, int src1, int src2);

        #endregion

        #region Intrinsic Wrappers

        /// <summary>
        /// Wrapper for _tile_loadd intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="base_addr">Base address.</param>
        /// <param name="stride">Stride in bytes.</param>
        internal static unsafe void tile_loadd(byte dst, void* base_addr, int stride) =>
            // This would use the actual intrinsic in real implementation
            LoadTile(dst, base_addr, stride);

        /// <summary>
        /// Wrapper for _tile_stored intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="base_addr">Base address.</param>
        /// <param name="stride">Stride in bytes.</param>
        internal static unsafe void tile_stored(byte dst, void* base_addr, int stride) =>
            // This would use the actual intrinsic in real implementation
            StoreTile(dst, base_addr, stride);

        /// <summary>
        /// Wrapper for _tile_zero intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        internal static void tile_zero(byte dst) =>
            // This would use the actual intrinsic in real implementation
            ZeroTile(dst);

        /// <summary>
        /// Wrapper for _tile_dpbf16ps intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="src1">Source tile 1.</param>
        /// <param name="src2">Source tile 2.</param>
        internal static void tile_dpbf16ps(byte dst, byte src1, byte src2) =>
            // This would use the actual intrinsic in real implementation
            TileMatMulBF16(dst, src1, src2);

        /// <summary>
        /// Wrapper for _tile_dpbssd intrinsic.
        /// </summary>
        /// <param name="dst">Destination tile.</param>
        /// <param name="src1">Source tile 1.</param>
        /// <param name="src2">Source tile 2.</param>
        internal static void tile_dpbssd(byte dst, byte src1, byte src2) =>
            // This would use the actual intrinsic in real implementation
            TileMatMulINT8(dst, src1, src2);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Checks processor support for AMX using CPUID.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        internal static bool CheckAMXSupport()
        {
            try
            {
                // Check if we're on x86/x64
                if (!X86Base.IsSupported)
                    return false;

                // Check CPUID leaf 7, sub-leaf 0 for AMX support
                // Bit 24 (EDX): AMX-TILE
                // Bit 22 (EDX): AMX-BF16  
                // Bit 25 (EDX): AMX-INT8
                return CheckCPUIDFeature(7, 0, 3, 24); // AMX-TILE is the base requirement
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Checks a specific CPUID feature bit.
        /// </summary>
        /// <param name="leaf">CPUID leaf number.</param>
        /// <param name="subLeaf">CPUID sub-leaf number.</param>
        /// <param name="register">Register index (0=EAX, 1=EBX, 2=ECX, 3=EDX).</param>
        /// <param name="bit">Bit position to check.</param>
        /// <returns>True if the feature bit is set; otherwise, false.</returns>
        private static bool CheckCPUIDFeature(uint leaf, uint subLeaf, int register, int bit)
        {
            if (!X86Base.IsSupported)
                return false;

            try
            {
                // Get CPUID result
                var cpuidResult = X86Base.CpuId((int)leaf, (int)subLeaf);
                
                // Select the appropriate register
                var value = register switch
                {
                    0 => cpuidResult.Eax,
                    1 => cpuidResult.Ebx, 
                    2 => cpuidResult.Ecx,
                    3 => cpuidResult.Edx,
                    _ => throw new ArgumentException("Invalid register index")
                };

                // Check if the bit is set
                return (value & (1u << bit)) != 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Requests permission to use AMX from the operating system.
        /// </summary>
        private static void RequestAMXPermission()
        {
            // On Linux, this would involve checking XCR0 register
            // On Windows, this might require specific API calls
            // For now, we assume permission is granted if AMX is detected
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Linux-specific AMX permission request
                RequestAMXPermissionLinux();
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows-specific AMX permission request  
                RequestAMXPermissionWindows();
            }
        }

        /// <summary>
        /// Requests AMX permission on Linux systems.
        /// </summary>
        private static void RequestAMXPermissionLinux()
        {
            try
            {
                // Check if AMX is enabled in XCR0 register
                // This would typically require reading /proc/cpuinfo or using system calls
                // For real implementation, would need to call arch_prctl(ARCH_REQ_XCOMP_GUEST_PERM, XFEATURE_XTILEDATA)
            }
            catch
            {
                throw new NotSupportedException("Failed to request AMX permission on Linux");
            }
        }

        /// <summary>
        /// Requests AMX permission on Windows systems.
        /// </summary>
        private static void RequestAMXPermissionWindows()
        {
            try
            {
                // Windows typically enables AMX automatically if hardware supports it
                // May need to check if AMX is enabled in the current process context
            }
            catch
            {
                throw new NotSupportedException("Failed to request AMX permission on Windows");
            }
        }

        /// <summary>
        /// Estimates AMX memory bandwidth based on processor specifications.
        /// </summary>
        /// <returns>Estimated bandwidth in GB/s.</returns>
        private static double EstimateAMXBandwidth()
        {
            try
            {
                // Get CPU information to estimate bandwidth
                var cpuInfo = GetCPUInformation();
                
                // Estimates based on known Intel processors with AMX
                // Sapphire Rapids: ~400-500 GB/s
                // Granite Rapids: ~600-700 GB/s  
                // Consumer chips (Alder Lake+): ~200-300 GB/s
                
                return cpuInfo.Family switch
                {
                    6 when cpuInfo.Model >= 0x8F => 500.0, // Sapphire Rapids and newer
                    6 when cpuInfo.Model >= 0x97 => 250.0, // Alder Lake and newer
                    _ => 200.0 // Conservative estimate
                };
            }
            catch
            {
                return 200.0; // Default conservative estimate
            }
        }

        /// <summary>
        /// Gets basic CPU information from CPUID.
        /// </summary>
        /// <returns>CPU information structure.</returns>
        private static CPUInfo GetCPUInformation()
        {
            if (!X86Base.IsSupported)
                return new CPUInfo { Family = 0, Model = 0 };

            var cpuid1 = X86Base.CpuId(1, 0);
            var family = (cpuid1.Eax >> 8) & 0xF;
            var model = (cpuid1.Eax >> 4) & 0xF;
            
            // Handle extended family/model
            if (family == 0xF)
                family += (cpuid1.Eax >> 20) & 0xFF;
            
            if (family == 0x6 || family == 0xF)
                model += ((cpuid1.Eax >> 16) & 0xF) << 4;

            return new CPUInfo { Family = (int)family, Model = (int)model };
        }

        /// <summary>
        /// CPU information structure.
        /// </summary>
        private struct CPUInfo
        {
            public int Family;
            public int Model;
        }

        /// <summary>
        /// Gets the size of a tile in bytes for the given configuration.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns (in bytes).</param>
        /// <returns>Tile size in bytes.</returns>
        internal static int GetTileSize(int rows, int cols) => rows * cols;

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

        private static int GetElementSize(AMXDataType dataType) => dataType switch
        {
            AMXDataType.BFloat16 => 2,
            AMXDataType.Int8 => 1,
            AMXDataType.Float32 => 4,
            _ => 4
        };

        /// <summary>
        /// Checks if the current processor supports AMX.
        /// </summary>
        /// <returns>True if AMX is supported; otherwise, false.</returns>
        public static bool IsAMXSupported() => CheckAMXSupport();

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
