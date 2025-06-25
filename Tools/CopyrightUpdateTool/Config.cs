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

using System.Collections.Immutable;

namespace CopyrightUpdateTool
{
    /// <summary>
    /// License type enumeration for dual licensing support.
    /// </summary>
    public enum LicenseType
    {
        /// <summary>
        /// University of Illinois Open Source License (original ILGPU).
        /// </summary>
        UniversityOfIllinois,
        
        /// <summary>
        /// Business Source License 1.1 with Apache 2.0 future license (new files).
        /// </summary>
        BusinessSourceLicense
    }

    static class Config
    {
        public const int LineLength = 90;

        // Original ILGPU Project License Configuration
        public const string OriginalCopyrightText = "Copyright (c)";
        public const string OriginalCopyrightOwner = "ILGPU Project";
        public const string OriginalCopyrightOwnerWebsite = "www.ilgpu.net";

        public static readonly ImmutableArray<string> OriginalCopyrightLicenseText =
            ImmutableArray.Create(
                "This file is part of ILGPU and is distributed under the University of Illinois Open",
                "Source License. See LICENSE.txt for details."
            );

        // New BSL 1.1 License Configuration
        public const string NewCopyrightOwner = "Michael Ivertowski, Ernst & Young Ltd. Switzerland";
        public const string NewLicenseUrl = "https://github.com/mivertowsi/ILGPU/blob/main/LICENSE";
        
        public static readonly ImmutableArray<string> BusinessSourceLicenseText =
            ImmutableArray.Create(
                "Licensed under the Business Source License 1.1 (the \"License\");",
                "you may not use this file except in compliance with the License.",
                "You may obtain a copy of the License at",
                "",
                "    https://github.com/mivertowsi/ILGPU/blob/main/LICENSE",
                "",
                "Unless required by applicable law or agreed to in writing, software",
                "distributed under the License is distributed on an \"AS IS\" BASIS,",
                "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
                "See the License for the specific language governing permissions and",
                "limitations under the License.",
                "",
                "NOTICE: This software is NOT licensed for commercial or production use.",
                "Change Date: 2029-06-24",
                "Change License: Apache License, Version 2.0"
            );

        // Backward compatibility properties for other parsers
        public static string CopyrightText => OriginalCopyrightText;
        public static string CopyrightOwner => OriginalCopyrightOwner;
        public static string CopyrightOwnerWebsite => OriginalCopyrightOwnerWebsite;
        public static ImmutableArray<string> CopyrightLicenseText => OriginalCopyrightLicenseText;

        /// <summary>
        /// Determines the appropriate license type based on file location and content.
        /// </summary>
        public static LicenseType GetLicenseType(string filePath, string? existingContent = null)
        {
            // Check if file already has BSL license
            if (!string.IsNullOrEmpty(existingContent) && 
                existingContent.Contains("Business Source License"))
            {
                return LicenseType.BusinessSourceLicense;
            }

            // New files in specific directories get BSL license
            if (filePath.Contains("ML\\Integration") ||
                filePath.Contains("Numerics") ||
                filePath.Contains("Memory\\Unified") ||
                filePath.Contains("Runtime\\Scheduling") ||
                filePath.Contains("TensorCores") ||
                filePath.Contains("SIMD"))
            {
                return LicenseType.BusinessSourceLicense;
            }

            // Default to original license for existing files
            return LicenseType.UniversityOfIllinois;
        }
    }
}
