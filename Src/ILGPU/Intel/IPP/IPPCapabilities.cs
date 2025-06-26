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

// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.

using System;
using System.Runtime.InteropServices;
using ILGPU.Intel.IPP.Native;

namespace ILGPU.Intel.IPP
{
    /// <summary>
    /// Provides capabilities detection and information for Intel Integrated Performance Primitives (IPP).
    /// </summary>
    public static class IPPCapabilities
    {
        private static IPPInfo? _cachedInfo;
        private static readonly object _lockObject = new object();

        /// <summary>
        /// Detects if Intel IPP is available on the current system.
        /// </summary>
        /// <returns>True if IPP is available, false otherwise.</returns>
        public static bool DetectIPP()
        {
            try
            {
                // Try to call a basic IPP function
                var status = IPPNative.ippInit();
                return status == IPPNative.IppStatus.ippStsNoErr || 
                       status == IPPNative.IppStatus.ippStsNotSupportedModeErr; // Library exists but may not be fully supported
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
        /// Queries comprehensive IPP capabilities and information.
        /// </summary>
        /// <returns>IPP capabilities information.</returns>
        public static IPPInfo Query()
        {
            lock (_lockObject)
            {
                if (_cachedInfo.HasValue)
                    return _cachedInfo.Value;

                var info = new IPPInfo();

                try
                {
                    // Initialize IPP
                    var initStatus = IPPNative.ippInit();
                    info.IsAvailable = initStatus == IPPNative.IppStatus.ippStsNoErr;

                    if (info.IsAvailable)
                    {
                        // Get CPU type and features
                        info.CpuType = IPPNative.ippGetCpuType();
                        info.CpuFeatures = IPPNative.ippGetCpuFeatures();

                        // Determine supported instruction sets
                        info.SupportsSSE = info.CpuType >= IPPNative.IppCpuType.ippCpuSSE;
                        info.SupportsSSE2 = info.CpuType >= IPPNative.IppCpuType.ippCpuSSE2;
                        info.SupportsSSE3 = info.CpuType >= IPPNative.IppCpuType.ippCpuSSE3;
                        info.SupportsSSE41 = info.CpuType >= IPPNative.IppCpuType.ippCpuSSE41;
                        info.SupportsSSE42 = info.CpuType >= IPPNative.IppCpuType.ippCpuSSE42;
                        info.SupportsAVX = info.CpuType >= IPPNative.IppCpuType.ippCpuAVX;
                        info.SupportsAVX2 = info.CpuType >= IPPNative.IppCpuType.ippCpuAVX2;
                        info.SupportsAVX512 = info.CpuType >= IPPNative.IppCpuType.ippCpuAVX512F;

                        // Get version information
                        var versionPtr = IPPNative.ippGetLibVersion();
                        if (versionPtr != IntPtr.Zero)
                        {
                            info.Version = Marshal.PtrToStringAnsi(versionPtr) ?? "Unknown";
                        }

                        // Determine optimal FFT sizes based on CPU capabilities
                        info.OptimalFFTSizes = DetermineOptimalFFTSizes(info);

                        // Estimate performance characteristics
                        info.EstimatedPerformance = EstimatePerformance(info);
                    }
                }
                catch
                {
                    info.IsAvailable = false;
                }

                _cachedInfo = info;
                return info;
            }
        }

        /// <summary>
        /// Gets the optimal FFT algorithm hint based on requirements.
        /// </summary>
        /// <param name="prioritizeSpeed">True to prioritize speed over accuracy.</param>
        /// <returns>Algorithm hint for IPP FFT functions.</returns>
        public static IPPNative.IppHintAlgorithm GetOptimalHint(bool prioritizeSpeed = true)
        {
            var info = Query();
            
            if (!info.IsAvailable)
                return IPPNative.IppHintAlgorithm.ippAlgHintNone;

            // Use fast algorithms for modern CPUs with AVX support
            if (info.SupportsAVX2 && prioritizeSpeed)
                return IPPNative.IppHintAlgorithm.ippAlgHintFast;

            // Use accurate algorithms for older CPUs or when accuracy is prioritized
            return IPPNative.IppHintAlgorithm.ippAlgHintAccurate;
        }

        /// <summary>
        /// Determines if the current platform supports IPP FFT operations.
        /// </summary>
        /// <returns>True if FFT operations are supported.</returns>
        public static bool SupportsFFT()
        {
            var info = Query();
            return info.IsAvailable && info.SupportsSSE2; // SSE2 is minimum requirement for good FFT performance
        }

        /// <summary>
        /// Gets recommended thread count for IPP operations based on system capabilities.
        /// </summary>
        /// <returns>Recommended thread count.</returns>
        public static int GetRecommendedThreadCount()
        {
            var info = Query();
            
            if (!info.IsAvailable)
                return 1;

            // IPP performs best with thread count matching physical cores
            var coreCount = Environment.ProcessorCount;
            
            // For AVX512 systems, can handle more threads efficiently
            if (info.SupportsAVX512)
                return Math.Min(coreCount, 16);
                
            // For AVX2 systems, optimal thread count is typically physical cores
            if (info.SupportsAVX2)
                return Math.Min(coreCount, 8);
                
            // For older systems, conservative thread count
            return Math.Min(coreCount / 2, 4);
        }

        private static int[] DetermineOptimalFFTSizes(IPPInfo info)
        {
            if (info.SupportsAVX512)
            {
                // AVX512 systems can handle larger transforms efficiently
                return new[] { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
            }
            else if (info.SupportsAVX2)
            {
                // AVX2 systems optimal for medium to large transforms
                return new[] { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
            }
            else if (info.SupportsSSE42)
            {
                // SSE4.2 systems good for small to medium transforms
                return new[] { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
            }
            else
            {
                // Older systems limited to smaller transforms
                return new[] { 8, 16, 32, 64, 128, 256, 512, 1024 };
            }
        }

        private static IPPPerformanceInfo EstimatePerformance(IPPInfo info)
        {
            var perf = new IPPPerformanceInfo();

            if (info.SupportsAVX512)
            {
                perf.RelativePerformance = 1.0; // Reference performance
                perf.EstimatedGFLOPS = 100.0; // High-end estimate
                perf.OptimalDataAlignment = 64; // AVX512 alignment
            }
            else if (info.SupportsAVX2)
            {
                perf.RelativePerformance = 0.7;
                perf.EstimatedGFLOPS = 50.0;
                perf.OptimalDataAlignment = 32; // AVX2 alignment
            }
            else if (info.SupportsSSE42)
            {
                perf.RelativePerformance = 0.4;
                perf.EstimatedGFLOPS = 20.0;
                perf.OptimalDataAlignment = 16; // SSE alignment
            }
            else
            {
                perf.RelativePerformance = 0.2;
                perf.EstimatedGFLOPS = 5.0;
                perf.OptimalDataAlignment = 8; // Basic alignment
            }

            return perf;
        }
    }

    /// <summary>
    /// Information about Intel IPP capabilities.
    /// </summary>
    public struct IPPInfo
    {
        /// <summary>
        /// Whether IPP is available and functional.
        /// </summary>
        public bool IsAvailable { get; set; }

        /// <summary>
        /// IPP library version string.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Detected CPU type.
        /// </summary>
        public IPPNative.IppCpuType CpuType { get; set; }

        /// <summary>
        /// CPU feature flags.
        /// </summary>
        public ulong CpuFeatures { get; set; }

        /// <summary>
        /// Whether SSE instruction set is supported.
        /// </summary>
        public bool SupportsSSE { get; set; }

        /// <summary>
        /// Whether SSE2 instruction set is supported.
        /// </summary>
        public bool SupportsSSE2 { get; set; }

        /// <summary>
        /// Whether SSE3 instruction set is supported.
        /// </summary>
        public bool SupportsSSE3 { get; set; }

        /// <summary>
        /// Whether SSE4.1 instruction set is supported.
        /// </summary>
        public bool SupportsSSE41 { get; set; }

        /// <summary>
        /// Whether SSE4.2 instruction set is supported.
        /// </summary>
        public bool SupportsSSE42 { get; set; }

        /// <summary>
        /// Whether AVX instruction set is supported.
        /// </summary>
        public bool SupportsAVX { get; set; }

        /// <summary>
        /// Whether AVX2 instruction set is supported.
        /// </summary>
        public bool SupportsAVX2 { get; set; }

        /// <summary>
        /// Whether AVX-512 instruction set is supported.
        /// </summary>
        public bool SupportsAVX512 { get; set; }

        /// <summary>
        /// Array of optimal FFT sizes for this CPU.
        /// </summary>
        public int[] OptimalFFTSizes { get; set; }

        /// <summary>
        /// Performance characteristics of this IPP installation.
        /// </summary>
        public IPPPerformanceInfo EstimatedPerformance { get; set; }
    }

    /// <summary>
    /// Performance information for IPP operations.
    /// </summary>
    public struct IPPPerformanceInfo
    {
        /// <summary>
        /// Relative performance compared to reference (0.0 to 1.0).
        /// </summary>
        public double RelativePerformance { get; set; }

        /// <summary>
        /// Estimated GFLOPS for FFT operations.
        /// </summary>
        public double EstimatedGFLOPS { get; set; }

        /// <summary>
        /// Optimal data alignment in bytes.
        /// </summary>
        public int OptimalDataAlignment { get; set; }
    }
}