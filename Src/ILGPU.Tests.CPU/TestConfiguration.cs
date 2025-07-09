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

using ILGPU.Runtime;
using System.Collections.Generic;

namespace ILGPU.Tests.CPU
{
    /// <summary>
    /// Test configuration for CPU-specific tests.
    /// </summary>
#pragma warning disable CA1515 // Consider making public types internal
    public class TestConfiguration
#pragma warning restore CA1515 // Consider making public types internal
    {
        /// <summary>
        /// Initializes a new test configuration.
        /// </summary>
        /// <param name="name">Configuration name.</param>
        /// <param name="acceleratorType">Type of accelerator to test.</param>
        public TestConfiguration(string name, AcceleratorType acceleratorType)
        {
            Name = name;
            AcceleratorType = acceleratorType;
        }

        /// <summary>
        /// Gets the configuration name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public AcceleratorType AcceleratorType { get; }

        /// <summary>
        /// Returns the configuration name.
        /// </summary>
        public override string ToString() => Name;
    }

    /// <summary>
    /// Base class for CPU tests with test configuration support.
    /// </summary>
#pragma warning disable CA1515 // Consider making public types internal
    public abstract class TestBase : ILGPU.Tests.TestBase
#pragma warning restore CA1515 // Consider making public types internal
    {
        /// <summary>
        /// Initializes a new test base.
        /// </summary>
        /// <param name="output">Test output helper.</param>
        /// <param name="testContext">Test context.</param>
        protected TestBase(Xunit.Abstractions.ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        /// <summary>
        /// Gets the available test configurations for CPU tests.
        /// </summary>
        public static IEnumerable<object[]> TestConfigurations =>
            new[]
            {
                new object[] { new TestConfiguration("CPU", AcceleratorType.CPU) }
            };
    }
}