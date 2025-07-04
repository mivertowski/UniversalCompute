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

using ILGPU.Runtime.ROCm.Native;
using ILGPU.Util;
using System;
using System.Runtime.Serialization;

namespace ILGPU.Runtime.ROCm
{
    /// <summary>
    /// Represents a ROCm-specific runtime exception.
    /// </summary>
    [Serializable]
    public class ROCmException : AcceleratorException
    {
        #region Constants

        /// <summary>
        /// The error code property name.
        /// </summary>
        public const string ErrorCodePropertyName = "ErrorCode";

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        public ROCmException()
            : this(HipError.ErrorUnknown)
        {
        }

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        /// <param name="errorCode">The ROCm error code.</param>
        internal ROCmException(HipError errorCode)
            : base(GetErrorString(errorCode))
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        /// <param name="message">The error message.</param>
        public ROCmException(string message)
            : this(message, HipError.ErrorUnknown)
        {
        }

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        /// <param name="message">The error message.</param>
        /// <param name="errorCode">The ROCm error code.</param>
        internal ROCmException(string message, HipError errorCode)
            : base(message)
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        /// <param name="message">The error message.</param>
        /// <param name="innerException">The inner exception.</param>
        public ROCmException(string message, Exception innerException)
            : base(message, innerException)
        {
            ErrorCode = HipError.ErrorUnknown;
        }

        /// <summary>
        /// Constructs a new ROCm exception.
        /// </summary>
        /// <param name="info">The serialization info.</param>
        /// <param name="context">The streaming context.</param>
        protected ROCmException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
            ErrorCode = (HipError)info.GetInt32(ErrorCodePropertyName);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the associated ROCm error code.
        /// </summary>
        internal HipError ErrorCode { get; }

        /// <summary>
        /// Gets the accelerator type.
        /// </summary>
        public override AcceleratorType AcceleratorType => AcceleratorType.ROCm;

        #endregion

        #region Methods

        /// <summary>
        /// Gets the object data for serialization.
        /// </summary>
        /// <param name="info">The serialization info.</param>
        /// <param name="context">The streaming context.</param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue(ErrorCodePropertyName, (int)ErrorCode);
        }

        /// <summary>
        /// Throws a ROCm exception if the given error code represents an error.
        /// </summary>
        /// <param name="errorCode">The error code to check.</param>
        internal static void ThrowIfFailed(HipError errorCode)
        {
            if (errorCode != HipError.Success)
                throw new ROCmException(errorCode);
        }

        /// <summary>
        /// Checks the given error code and throws an exception in case of an error.
        /// </summary>
        /// <param name="errorCode">The error code to check.</param>
        /// <returns>True if the operation was successful.</returns>
        internal static bool VerifyResult(HipError errorCode)
        {
            if (errorCode == HipError.Success)
                return true;
            
            throw new ROCmException(errorCode);
        }

        /// <summary>
        /// Gets an error string for the given error code.
        /// </summary>
        /// <param name="errorCode">The error code.</param>
        /// <returns>The error string.</returns>
        internal static string GetErrorString(HipError errorCode) => errorCode switch
        {
            HipError.Success => "No error",
            HipError.ErrorInvalidValue => "Invalid value",
            HipError.ErrorOutOfMemory => "Out of memory",
            HipError.ErrorNotInitialized => "ROCm not initialized",
            HipError.ErrorDeinitialized => "ROCm deinitialized",
            HipError.ErrorNoDevice => "No ROCm device available",
            HipError.ErrorInvalidDevice => "Invalid device",
            HipError.ErrorInvalidImage => "Invalid image",
            HipError.ErrorInvalidContext => "Invalid context",
            HipError.ErrorMapFailed => "Memory mapping failed",
            HipError.ErrorUnmapFailed => "Memory unmapping failed",
            HipError.ErrorArrayIsMapped => "Array is mapped",
            HipError.ErrorAlreadyMapped => "Already mapped",
            HipError.ErrorNoBinaryForGpu => "No binary for GPU",
            HipError.ErrorAlreadyAcquired => "Already acquired",
            HipError.ErrorNotMapped => "Not mapped",
            HipError.ErrorNotMappedAsArray => "Not mapped as array",
            HipError.ErrorNotMappedAsPointer => "Not mapped as pointer",
            HipError.ErrorEccUncorrectable => "ECC uncorrectable error",
            HipError.ErrorUnsupportedLimit => "Unsupported limit",
            HipError.ErrorContextAlreadyInUse => "Context already in use",
            HipError.ErrorPeerAccessUnsupported => "Peer access unsupported",
            HipError.ErrorInvalidPtx => "Invalid PTX",
            HipError.ErrorInvalidGraphicsContext => "Invalid graphics context",
            HipError.ErrorNvlinkUncorrectable => "NVLink uncorrectable error",
            HipError.ErrorJitCompilerNotFound => "JIT compiler not found",
            HipError.ErrorInvalidSource => "Invalid source",
            HipError.ErrorFileNotFound => "File not found",
            HipError.ErrorSharedObjectSymbolNotFound => "Shared object symbol not found",
            HipError.ErrorSharedObjectInitFailed => "Shared object initialization failed",
            HipError.ErrorOperatingSystem => "Operating system error",
            HipError.ErrorInvalidHandle => "Invalid handle",
            HipError.ErrorNotFound => "Not found",
            HipError.ErrorNotReady => "Not ready",
            HipError.ErrorIllegalAddress => "Illegal address",
            HipError.ErrorLaunchOutOfResources => "Launch out of resources",
            HipError.ErrorLaunchTimeOut => "Launch timeout",
            HipError.ErrorPeerAccessAlreadyEnabled => "Peer access already enabled",
            HipError.ErrorPeerAccessNotEnabled => "Peer access not enabled",
            HipError.ErrorSetOnActiveProcess => "Set on active process",
            HipError.ErrorAssert => "Assertion failed",
            HipError.ErrorHostMemoryAlreadyRegistered => "Host memory already registered",
            HipError.ErrorHostMemoryNotRegistered => "Host memory not registered",
            HipError.ErrorLaunchFailure => "Launch failure",
            HipError.ErrorCooperativeLaunchTooLarge => "Cooperative launch too large",
            HipError.ErrorNotSupported => "Not supported",
            HipError.ErrorUnknown => "Unknown error",
            _ => $"Unknown ROCm error code: {(int)errorCode}"
        };

        #endregion
    }
}