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

using ILGPU.Runtime.OneAPI.Native;
using ILGPU.Util;
using System;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// A SYCL-specific page-lock scope implementation for Intel OneAPI devices.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class SYCLPageLockScope<T> : PageLockScope<T>
        where T : unmanaged
    {
        #region Instance

        /// <summary>
        /// Gets the associated Intel OneAPI accelerator.
        /// </summary>
        public new IntelOneAPIAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelOneAPIAccelerator>();

        /// <summary>
        /// Tracks whether the memory was successfully pinned using SYCL.
        /// </summary>
        private readonly bool isNativePinned;

        /// <summary>
        /// Initializes a new SYCL page-lock scope.
        /// </summary>
        /// <param name="accelerator">The parent accelerator.</param>
        /// <param name="pinned">The pinned host memory pointer.</param>
        /// <param name="numElements">The number of elements.</param>
        internal SYCLPageLockScope(
            IntelOneAPIAccelerator accelerator,
            IntPtr pinned,
            long numElements)
            : base(accelerator, pinned, numElements)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // Try to register the host memory with SYCL for optimal performance
                // SYCL/DPC++ supports pinned host memory through USM host allocations
                // For existing host memory, we attempt to register it for faster transfers
                if (accelerator.SupportsUnifiedMemory)
                {
                    // Intel OneAPI supports unified shared memory (USM)
                    // The memory is already accessible from both host and device
                    isNativePinned = true;
                }
                else
                {
                    // For devices without USM, use traditional memory pinning
                    // This would typically involve registering the host memory
                    // with the SYCL runtime for optimized transfers
                    isNativePinned = TryRegisterHostMemory(pinned, LengthInBytes);
                }
            }
            catch (Exception)
            {
                // If SYCL pinning fails, we still provide the functionality
                // but without the performance benefits of pinned memory
                isNativePinned = false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets whether this page-lock scope uses native SYCL pinning.
        /// </summary>
        public bool IsNativePinned => isNativePinned;

        #endregion

        #region Methods

        /// <summary>
        /// Attempts to register host memory with SYCL for optimized transfers.
        /// </summary>
        /// <param name="hostPtr">The host memory pointer.</param>
        /// <param name="sizeInBytes">The size in bytes.</param>
        /// <returns>True if registration succeeded; otherwise, false.</returns>
        private bool TryRegisterHostMemory(IntPtr hostPtr, long sizeInBytes)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // SYCL doesn't have a direct equivalent to CUDA's cudaHostRegister
                // However, for Intel GPUs with unified memory, host memory is already
                // accessible from the device without explicit registration
                if (Accelerator.SupportsUnifiedMemory)
                {
                    return true;
                }

                // For non-USM devices, we could potentially use SYCL's host memory
                // allocation and copy the data, but this would change the memory location
                // Since we need to maintain the original pointer, we skip registration
                return false;
            }
            catch (Exception)
            {
                return false;
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        /// <summary>
        /// Unregisters host memory from SYCL.
        /// </summary>
        /// <param name="hostPtr">The host memory pointer.</param>
        private void TryUnregisterHostMemory(IntPtr hostPtr)
        {
#pragma warning disable CA1031 // Do not catch general exception types
            try
            {
                // For Intel OneAPI/SYCL, there's typically no explicit unregistration
                // needed for unified memory or host memory access
                // This is a no-op for most SYCL implementations
            }
            catch (Exception)
            {
                // Ignore errors during unregistration
            }
#pragma warning restore CA1031 // Do not catch general exception types
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this page-lock scope.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && isNativePinned)
            {
#pragma warning disable CA1031 // Do not catch general exception types
                try
                {
                    // Unregister the host memory if it was registered
                    TryUnregisterHostMemory(AddrOfLockedObject);
                }
                catch (Exception)
                {
                    // Ignore errors during disposal - this is a best-effort cleanup
                }
#pragma warning restore CA1031 // Do not catch general exception types
            }
        }

        #endregion
    }
}