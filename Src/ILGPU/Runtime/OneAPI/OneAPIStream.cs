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
using System;

namespace ILGPU.Runtime.OneAPI
{
    /// <summary>
    /// An Intel OneAPI/SYCL stream for asynchronous kernel execution and memory operations.
    /// </summary>
    public sealed class OneAPIStream : AcceleratorStream
    {
        #region Instance

        /// <summary>
        /// The native SYCL queue handle.
        /// </summary>
        internal IntPtr NativeQueue { get; private set; }

        /// <summary>
        /// The associated OneAPI accelerator.
        /// </summary>
        public new IntelOneAPIAccelerator Accelerator => base.Accelerator.AsNotNullCast<IntelOneAPIAccelerator>();

        /// <summary>
        /// Initializes a new OneAPI stream.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        internal OneAPIStream(IntelOneAPIAccelerator accelerator)
            : base(accelerator)
        {
            try
            {
                // Create SYCL queue
                var result = SYCLNative.CreateQueue(
                    accelerator.ContextHandle, 
                    accelerator.DeviceHandle, 
                    SYCLQueueProperties.None, 
                    out var queue);
                SYCLException.ThrowIfFailed(result);
                NativeQueue = queue;
            }
            catch (DllNotFoundException)
            {
                // SYCL not available - use dummy handle
                NativeQueue = new IntPtr(-1);
            }
            catch (EntryPointNotFoundException)
            {
                // SYCL functions not found - use dummy handle
                NativeQueue = new IntPtr(-1);
            }
        }

        #endregion

        #region Stream Operations

        /// <summary>
        /// Synchronizes this stream and waits for all operations to complete.
        /// </summary>
        public override void Synchronize()
        {
            if (NativeQueue == new IntPtr(-1))
                return; // SYCL not available

            try
            {
                var result = SYCLNative.QueueWait(NativeQueue);
                SYCLException.ThrowIfFailed(result);
            }
            catch (DllNotFoundException)
            {
                // SYCL not available - no operation needed
            }
            catch (EntryPointNotFoundException)
            {
                // SYCL functions not found - no operation needed
            }
        }

        /// <summary>
        /// Adds a profiling marker to this stream.
        /// </summary>
        /// <returns>The created profiling marker.</returns>
        protected override ProfilingMarker AddProfilingMarkerInternal()
        {
            using var binding = Accelerator.BindScoped();
            return new OneAPIProfilingMarker(Accelerator, NativeQueue);
        }

        #endregion

        #region Disposal

        /// <summary>
        /// Disposes this OneAPI stream.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (disposing && NativeQueue != IntPtr.Zero && NativeQueue != new IntPtr(-1))
            {
                try
                {
                    SYCLNative.ReleaseQueue(NativeQueue);
                }
                catch
                {
                    // Ignore errors during disposal
                }
                finally
                {
                    NativeQueue = IntPtr.Zero;
                }
            }
        }

        #endregion
    }

    /// <summary>
    /// Intel OneAPI profiling marker implementation.
    /// </summary>
    internal sealed class OneAPIProfilingMarker : ProfilingMarker
    {
        private readonly IntPtr _queue;
        private readonly DateTime _timestamp;

        /// <summary>
        /// Initializes a new OneAPI profiling marker.
        /// </summary>
        /// <param name="accelerator">The accelerator.</param>
        /// <param name="queue">The SYCL queue.</param>
        internal OneAPIProfilingMarker(Accelerator accelerator, IntPtr queue)
            : base(accelerator)
        {
            _queue = queue;
            _timestamp = DateTime.UtcNow;
        }

        /// <summary>
        /// Synchronizes this profiling marker.
        /// </summary>
        public override void Synchronize()
        {
            if (_queue != new IntPtr(-1))
            {
                try
                {
                    SYCLNative.QueueWait(_queue);
                }
                catch
                {
                    // Ignore errors during profiling
                }
            }
        }

        /// <summary>
        /// Measures the elapsed time from another marker.
        /// </summary>
        /// <param name="marker">The starting marker.</param>
        /// <returns>The elapsed time.</returns>
        public override TimeSpan MeasureFrom(ProfilingMarker marker)
        {
            if (marker is OneAPIProfilingMarker oneAPIMarker)
                return _timestamp - oneAPIMarker._timestamp;
            throw new ArgumentException("Marker must be a OneAPI profiling marker", nameof(marker));
        }

        /// <summary>
        /// Disposes this profiling marker.
        /// </summary>
        /// <param name="disposing">True if disposing.</param>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            // No resources to dispose for timestamp markers
        }
    }
}