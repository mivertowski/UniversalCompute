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
// Change License: Apache License, Version 2.0using System;

namespace ILGPU.Intel.AMX
{
    /// <summary>
    /// Intel AMX specific context extensions.
    /// </summary>
    public static class AMXContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables the Intel AMX accelerator if supported.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder AMX(this Context.Builder builder)
        {
            AMXDevice.RegisterDevices(builder.DeviceRegistry);
            return builder;
        }

        /// <summary>
        /// Enables the Intel AMX accelerator with a predicate.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <param name="predicate">The predicate to include the AMX device.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder AMX(
            this Context.Builder builder,
            Predicate<AMXDevice> predicate)
        {
            if (predicate(AMXDevice.Default))
                builder.DeviceRegistry.Register(AMXDevice.Default);
            return builder;
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets all registered AMX accelerators from the given context.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>All registered AMX accelerators.</returns>
        public static Context.DeviceCollection<AMXDevice> GetAMXDevices(
            this Context context) =>
            context.GetDevices<AMXDevice>();

        /// <summary>
        /// Creates a new AMX accelerator using the first available AMX device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created AMX accelerator.</returns>
        public static AMXAccelerator CreateAMXAccelerator(this Context context)
        {
            var devices = context.GetAMXDevices();
            if (devices.Count < 1)
                throw new NotSupportedException("No AMX device available");
            return devices[0].CreateAccelerator(context) as AMXAccelerator
                ?? throw new InvalidOperationException("Failed to create AMX accelerator");
        }

        #endregion
    }
}