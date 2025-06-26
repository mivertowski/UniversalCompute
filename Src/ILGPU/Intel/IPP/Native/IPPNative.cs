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

namespace ILGPU.Intel.IPP.Native
{
    /// <summary>
    /// Native bindings for Intel Integrated Performance Primitives (IPP) FFT functions.
    /// Provides high-performance CPU-based FFT operations optimized for Intel processors.
    /// </summary>
    public static class IPPNative
    {
        #region Library Constants

        private const string IPPLibrary = "ipp";
        private const string IPPSLibrary = "ipps";
        private const string IPPCCLibrary = "ippcc";

        #endregion

        #region FFT Status Codes

        /// <summary>
        /// IPP status codes for FFT operations.
        /// </summary>
        public enum IppStatus : int
        {
            ippStsNoErr = 0,
            ippStsNullPtrErr = -2,
            ippStsSizeErr = -6,
            ippStsBadArgErr = -5,
            ippStsMemAllocErr = -4,
            ippStsNotSupportedModeErr = -9999,
            ippStsCpuNotSupportedErr = -9998
        }

        #endregion

        #region FFT Specification Structures

        /// <summary>
        /// FFT specification handle for complex-to-complex transforms.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct IppsFFTSpec_C_32fc
        {
            public IntPtr Handle;
        }

        /// <summary>
        /// FFT specification handle for real-to-complex transforms.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct IppsFFTSpec_R_32f
        {
            public IntPtr Handle;
        }

        /// <summary>
        /// Complex number structure for 32-bit float.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Ipp32fc
        {
            public float re;
            public float im;

            public Ipp32fc(float real, float imaginary)
            {
                re = real;
                im = imaginary;
            }
        }

        #endregion

        #region Complex-to-Complex FFT Functions

        /// <summary>
        /// Gets the size of the FFT specification structure and work buffer.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTGetSize_C_32fc(
            int order,
            int flag,
            IppHintAlgorithm hint,
            out int pSpecSize,
            out int pSpecBufferSize,
            out int pBufferSize);

        /// <summary>
        /// Initializes the FFT specification structure.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTInit_C_32fc(
            out IntPtr ppFFTSpec,
            int order,
            int flag,
            IppHintAlgorithm hint,
            IntPtr pSpec,
            IntPtr pSpecBuffer);

        /// <summary>
        /// Performs forward complex-to-complex FFT.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTFwd_CToC_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        /// <summary>
        /// Performs inverse complex-to-complex FFT.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTInv_CToC_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        #endregion

        #region Real-to-Complex FFT Functions

        /// <summary>
        /// Gets the size of the real FFT specification structure and work buffer.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTGetSize_R_32f(
            int order,
            int flag,
            IppHintAlgorithm hint,
            out int pSpecSize,
            out int pSpecBufferSize,
            out int pBufferSize);

        /// <summary>
        /// Initializes the real FFT specification structure.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTInit_R_32f(
            out IntPtr ppFFTSpec,
            int order,
            int flag,
            IppHintAlgorithm hint,
            IntPtr pSpec,
            IntPtr pSpecBuffer);

        /// <summary>
        /// Performs forward real-to-complex FFT.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTFwd_RToPack_32f(
            IntPtr pSrc,
            IntPtr pDst,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        /// <summary>
        /// Performs inverse complex-to-real FFT.
        /// </summary>
        [DllImport(IPPSLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippsFFTInv_PackToR_32f(
            IntPtr pSrc,
            IntPtr pDst,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        #endregion

        #region 2D FFT Functions

        /// <summary>
        /// Gets the size of 2D FFT specification structure and work buffer.
        /// </summary>
        [DllImport(IPPCCLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippiFFTGetSize_C_32fc(
            IppiSize roiSize,
            int flag,
            IppHintAlgorithm hint,
            out int pSpecSize,
            out int pInitBufSize,
            out int pWorkBufSize);

        /// <summary>
        /// Initializes 2D FFT specification structure.
        /// </summary>
        [DllImport(IPPCCLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippiFFTInit_C_32fc(
            out IntPtr ppFFTSpec,
            IppiSize roiSize,
            int flag,
            IppHintAlgorithm hint,
            IntPtr pSpec,
            IntPtr pInitBuf);

        /// <summary>
        /// Performs forward 2D complex-to-complex FFT.
        /// </summary>
        [DllImport(IPPCCLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippiFFTFwd_CToC_32fc_C1R(
            IntPtr pSrc,
            int srcStep,
            IntPtr pDst,
            int dstStep,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        /// <summary>
        /// Performs inverse 2D complex-to-complex FFT.
        /// </summary>
        [DllImport(IPPCCLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippiFFTInv_CToC_32fc_C1R(
            IntPtr pSrc,
            int srcStep,
            IntPtr pDst,
            int dstStep,
            IntPtr pFFTSpec,
            IntPtr pBuffer);

        #endregion

        #region Utility Structures and Enums

        /// <summary>
        /// Algorithm hint for FFT optimization.
        /// </summary>
        public enum IppHintAlgorithm : int
        {
            ippAlgHintNone = 0,
            ippAlgHintFast = 1,
            ippAlgHintAccurate = 2
        }

        /// <summary>
        /// Size structure for 2D operations.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct IppiSize
        {
            public int width;
            public int height;

            public IppiSize(int w, int h)
            {
                width = w;
                height = h;
            }
        }

        #endregion

        #region Memory Management

        /// <summary>
        /// Allocates aligned memory for IPP operations.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ippsMalloc_32f(int len);

        /// <summary>
        /// Allocates aligned memory for complex IPP operations.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ippsMalloc_32fc(int len);

        /// <summary>
        /// Frees memory allocated by IPP.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ippsFree(IntPtr ptr);

        #endregion

        #region CPU Feature Detection

        /// <summary>
        /// Gets CPU features supported by IPP.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong ippGetCpuFeatures();

        /// <summary>
        /// Gets the current CPU type.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppCpuType ippGetCpuType();

        /// <summary>
        /// CPU types supported by IPP.
        /// </summary>
        public enum IppCpuType : int
        {
            ippCpuUnknown = 0x00,
            ippCpuPP = 0x01,
            ippCpuPMX = 0x02,
            ippCpuPPR = 0x03,
            ippCpuPII = 0x04,
            ippCpuPIII = 0x05,
            ippCpuP4 = 0x06,
            ippCpuP4HT = 0x07,
            ippCpuP4HT2 = 0x08,
            ippCpuCentrino = 0x09,
            ippCpuDS = 0x0a,
            ippCpuDC = 0x0b,
            ippCpuEM64T = 0x0c,
            ippCpuC2D = 0x0d,
            ippCpuC2Q = 0x0e,
            ippCpuPenryn = 0x0f,
            ippCpuBonnell = 0x10,
            ippCpuNehalem = 0x11,
            ippCpuNext = 0x12,
            ippCpuSSE = 0x20,
            ippCpuSSE2 = 0x21,
            ippCpuSSE3 = 0x22,
            ippCpuSSSE3 = 0x23,
            ippCpuSSE41 = 0x24,
            ippCpuSSE42 = 0x25,
            ippCpuAVX = 0x26,
            ippCpuAVX2 = 0x27,
            ippCpuAVX512F = 0x28,
            ippCpuAVX512CD = 0x29,
            ippCpuAVX512ER = 0x2a,
            ippCpuAVX512PF = 0x2b,
            ippCpuAVX512BW = 0x2c,
            ippCpuAVX512DQ = 0x2d,
            ippCpuAVX512VL = 0x2e,
            ippCpuAVX512VBMI = 0x2f
        }

        #endregion

        #region Initialization and Cleanup

        /// <summary>
        /// Initializes IPP library.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IppStatus ippInit();

        /// <summary>
        /// Gets IPP library version information.
        /// </summary>
        [DllImport(IPPLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ippGetLibVersion();

        #endregion
    }
}