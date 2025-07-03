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

namespace ILGPU.Apple.NeuralEngine
{
    /// <summary>
    /// Apple Neural Engine specific context extensions.
    /// </summary>
    public static class ANEContextExtensions
    {
        #region Builder

        /// <summary>
        /// Enables the Apple Neural Engine accelerator if supported.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder AppleNeuralEngine(this Context.Builder builder)
        {
            AppleNeuralEngineDevice.RegisterDevices(builder.DeviceRegistry);
            return builder;
        }

        /// <summary>
        /// Enables the Apple Neural Engine accelerator with a predicate.
        /// </summary>
        /// <param name="builder">The builder instance.</param>
        /// <param name="predicate">The predicate to include the ANE device.</param>
        /// <returns>The updated builder instance.</returns>
        public static Context.Builder AppleNeuralEngine(
            this Context.Builder builder,
            Predicate<AppleNeuralEngineDevice> predicate)
        {
            if (predicate(AppleNeuralEngineDevice.Default))
                builder.DeviceRegistry.Register(AppleNeuralEngineDevice.Default);
            return builder;
        }

        #endregion

        #region Context

        /// <summary>
        /// Gets all registered Apple Neural Engine accelerators from the given context.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>All registered Apple Neural Engine accelerators.</returns>
        public static Context.DeviceCollection<AppleNeuralEngineDevice> GetAppleNeuralEngineDevices(
            this Context context) =>
            context.GetDevices<AppleNeuralEngineDevice>();

        /// <summary>
        /// Creates a new Apple Neural Engine accelerator using the first available ANE device.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <returns>The created Apple Neural Engine accelerator.</returns>
        public static AppleNeuralEngineAccelerator CreateAppleNeuralEngineAccelerator(this Context context)
        {
            var devices = context.GetAppleNeuralEngineDevices();
            if (devices.Count < 1)
                throw new NotSupportedException("No Apple Neural Engine device available");
            var accelerator = devices[0].CreateAccelerator(context);
            if (accelerator is AppleNeuralEngineAccelerator aneAccelerator)
                return aneAccelerator;
            throw new InvalidOperationException("Failed to create Apple Neural Engine accelerator");
        }

        #endregion
    }
}