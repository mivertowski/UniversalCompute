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

namespace ILGPU.Intel.NPU
{
    /// <summary>
    /// Intel NPU specific context extensions.
    /// </summary>
    public static class NPUContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables the Intel NPU accelerator if supported.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder IntelNPU(this Context.Builder builder)
        {
            IntelNPUDevice.RegisterDevices(builder.DeviceRegistry);
            return builder;
        }

        /// <summary>
        /// Enables the Intel NPU accelerator with a predicate.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <param name="predicate">The predicate to include the NPU device.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder IntelNPU(
            this Context.Builder builder,
            Predicate<IntelNPUDevice> predicate)
        {
            if (predicate(IntelNPUDevice.Default))
                builder.DeviceRegistry.Register(IntelNPUDevice.Default);
            return builder;
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets all registered Intel NPU accelerators from the given context.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>All registered Intel NPU accelerators.</returns>
        public static Context.DeviceCollection<IntelNPUDevice> GetIntelNPUDevices(
            this Context context) =>
            context.GetDevices<IntelNPUDevice>();

        /// <summary>
        /// Creates a new Intel NPU accelerator using the first available NPU device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created Intel NPU accelerator.</returns>
        public static IntelNPUAccelerator CreateIntelNPUAccelerator(this Context context)
        {
            var devices = context.GetIntelNPUDevices();
            return devices.Count < 1
                ? throw new NotSupportedException("No Intel NPU device available")
                : devices[0].CreateAccelerator(context) as IntelNPUAccelerator
                ?? throw new InvalidOperationException("Failed to create Intel NPU accelerator");
        }

        #endregion
    }
}