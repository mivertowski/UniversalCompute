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
using System;

namespace ILGPU.Algorithms.Cryptography
{
    /// <summary>
    /// Cryptographic hash algorithm types.
    /// </summary>
    public enum HashAlgorithm
    {
        /// <summary>SHA-256 hash algorithm</summary>
        SHA256,
        /// <summary>SHA-512 hash algorithm</summary>
        SHA512,
        /// <summary>MD5 hash algorithm (insecure, for compatibility only)</summary>
        MD5,
        /// <summary>BLAKE2b hash algorithm</summary>
        BLAKE2b,
        /// <summary>Keccak-256 (used in Ethereum)</summary>
        Keccak256
    }

    /// <summary>
    /// Symmetric encryption algorithm types.
    /// </summary>
    public enum SymmetricAlgorithm
    {
        /// <summary>AES-128 encryption</summary>
        AES128,
        /// <summary>AES-192 encryption</summary>
        AES192,
        /// <summary>AES-256 encryption</summary>
        AES256,
        /// <summary>ChaCha20 stream cipher</summary>
        ChaCha20,
        /// <summary>Salsa20 stream cipher</summary>
        Salsa20
    }

    /// <summary>
    /// Block cipher modes of operation.
    /// </summary>
    public enum CipherMode
    {
        /// <summary>Electronic Codebook (ECB) mode</summary>
        ECB,
        /// <summary>Cipher Block Chaining (CBC) mode</summary>
        CBC,
        /// <summary>Cipher Feedback (CFB) mode</summary>
        CFB,
        /// <summary>Output Feedback (OFB) mode</summary>
        OFB,
        /// <summary>Counter (CTR) mode</summary>
        CTR,
        /// <summary>Galois/Counter Mode (GCM)</summary>
        GCM
    }

    /// <summary>
    /// Cryptographic random number generator algorithms.
    /// </summary>
    public enum RandomAlgorithm
    {
        /// <summary>Linear Congruential Generator (for testing only)</summary>
        LCG,
        /// <summary>Mersenne Twister</summary>
        MersenneTwister,
        /// <summary>ChaCha20-based CSPRNG</summary>
        ChaCha20RNG,
        /// <summary>AES-based CSPRNG</summary>
        AESRNG
    }

    /// <summary>
    /// 256-bit hash result structure.
    /// </summary>
    public struct Hash256
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong Word0, Word1, Word2, Word3;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="Hash256"/> struct.
        /// </summary>
        /// <param name="w0">The w0.</param>
        /// <param name="w1">The w1.</param>
        /// <param name="w2">The w2.</param>
        /// <param name="w3">The w3.</param>
        public Hash256(ulong w0, ulong w1, ulong w2, ulong w3)
        {
            Word0 = w0; Word1 = w1; Word2 = w2; Word3 = w3;
        }

        /// <summary>
        /// Converts hash to byte array.
        /// </summary>
        public byte[] ToBytes()
        {
            var bytes = new byte[32];
            var words = new ulong[] { Word0, Word1, Word2, Word3 };
            
            for (int i = 0; i < 4; i++)
            {
                var word = words[i];
                for (int j = 0; j < 8; j++)
                {
                    bytes[i * 8 + j] = (byte)(word >> (j * 8));
                }
            }
            
            return bytes;
        }

        /// <summary>
        /// Creates hash from byte array.
        /// </summary>
        public static Hash256 FromBytes(byte[] bytes)
        {
            if (bytes.Length != 32) throw new ArgumentException("Hash must be 32 bytes");
            
            var words = new ulong[4];
            for (int i = 0; i < 4; i++)
            {
                ulong word = 0;
                for (int j = 0; j < 8; j++)
                {
                    word |= ((ulong)bytes[i * 8 + j]) << (j * 8);
                }
                words[i] = word;
            }
            
            return new Hash256(words[0], words[1], words[2], words[3]);
        }
    }

    /// <summary>
    /// 512-bit hash result structure.
    /// </summary>
    public struct Hash512
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong Word0, Word1, Word2, Word3, Word4, Word5, Word6, Word7;
#pragma warning restore CA1051 // Do not declare visible instance fields


        /// <summary>
        /// Initializes a new instance of the <see cref="Hash512"/> struct.
        /// </summary>
        /// <param name="w0">The w0.</param>
        /// <param name="w1">The w1.</param>
        /// <param name="w2">The w2.</param>
        /// <param name="w3">The w3.</param>
        /// <param name="w4">The w4.</param>
        /// <param name="w5">The w5.</param>
        /// <param name="w6">The w6.</param>
        /// <param name="w7">The w7.</param>
        public Hash512(ulong w0, ulong w1, ulong w2, ulong w3, ulong w4, ulong w5, ulong w6, ulong w7)
        {
            Word0 = w0; Word1 = w1; Word2 = w2; Word3 = w3;
            Word4 = w4; Word5 = w5; Word6 = w6; Word7 = w7;
        }

        /// <summary>
        /// Converts hash to byte array.
        /// </summary>
        public byte[] ToBytes()
        {
            var bytes = new byte[64];
            var words = new ulong[] { Word0, Word1, Word2, Word3, Word4, Word5, Word6, Word7 };
            
            for (int i = 0; i < 8; i++)
            {
                var word = words[i];
                for (int j = 0; j < 8; j++)
                {
                    bytes[i * 8 + j] = (byte)(word >> (j * 8));
                }
            }
            
            return bytes;
        }

        /// <summary>
        /// Creates hash from byte array.
        /// </summary>
        public static Hash512 FromBytes(byte[] bytes)
        {
            if (bytes.Length != 64) throw new ArgumentException("Hash must be 64 bytes");
            
            var words = new ulong[8];
            for (int i = 0; i < 8; i++)
            {
                ulong word = 0;
                for (int j = 0; j < 8; j++)
                {
                    word |= ((ulong)bytes[i * 8 + j]) << (j * 8);
                }
                words[i] = word;
            }
            
            return new Hash512(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
        }
    }

    /// <summary>
    /// AES encryption key structure.
    /// </summary>
    public struct AESKey
    {
        /// <summary>Key size in bits</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public int KeySize;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Round keys for encryption</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public uint[] RoundKeys;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Number of rounds</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public int Rounds;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="AESKey"/> struct.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <exception cref="System.ArgumentException">AES key must be 128, 192, or 256 bits</exception>
        public AESKey(byte[] key)
        {
            if (key.Length != 16 && key.Length != 24 && key.Length != 32)
                throw new ArgumentException("AES key must be 128, 192, or 256 bits");

            KeySize = key.Length * 8;
            Rounds = KeySize switch
            {
                128 => 10,
                192 => 12,
                256 => 14,
                _ => throw new ArgumentException("Invalid key size")
            };

            // Simplified key expansion (real implementation would be more complex)
            RoundKeys = new uint[(Rounds + 1) * 4];
            
            // Copy initial key
            for (int i = 0; i < key.Length / 4; i++)
            {
                RoundKeys[i] = BitConverter.ToUInt32(key, i * 4);
            }

            // Generate round keys (simplified)
            for (int i = key.Length / 4; i < RoundKeys.Length; i++)
            {
                RoundKeys[i] = RoundKeys[i - 1] ^ RoundKeys[i - key.Length / 4];
            }
        }
    }

    /// <summary>
    /// Elliptic curve point for ECC operations.
    /// </summary>
    public struct ECPoint
    {
        /// <summary>X coordinate</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] X;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Y coordinate</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] Y;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Whether the point is at infinity</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public bool IsInfinity;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="ECPoint"/> struct.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="isInfinity">if set to <c>true</c> [is infinity].</param>
        /// <exception cref="System.ArgumentNullException">
        /// x
        /// or
        /// y
        /// </exception>
        public ECPoint(ulong[] x, ulong[] y, bool isInfinity = false)
        {
            X = x ?? throw new ArgumentNullException(nameof(x));
            Y = y ?? throw new ArgumentNullException(nameof(y));
            IsInfinity = isInfinity;
        }

        /// <summary>
        /// Point at infinity constructor.
        /// </summary>
        public static ECPoint Infinity => new ECPoint(new ulong[4], new ulong[4], true);
    }

    /// <summary>
    /// Elliptic curve parameters (secp256k1 example).
    /// </summary>
    public struct ECCurve
    {
        /// <summary>Prime modulus</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] P;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Curve parameter A</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] A;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Curve parameter B</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] B;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Generator point</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ECPoint G;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Order of the generator</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] N;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Creates secp256k1 curve parameters.
        /// </summary>
        public static ECCurve Secp256k1()
        {
            return new ECCurve
            {
                // Simplified secp256k1 parameters
                P = new ulong[] { 0xFFFFFFFFFFFFC2F, 0xFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFF },
                A = new ulong[] { 0, 0, 0, 0 }, // a = 0 for secp256k1
                B = new ulong[] { 7, 0, 0, 0 }, // b = 7 for secp256k1
                G = new ECPoint(
                    new ulong[] { 0x79BE667EF9DCBBAC, 0x55A06295CE870B07, 0x029BFCDB2DCE28D9, 0x59F2815B16F81798 },
                    new ulong[] { 0x483ADA7726A3C465, 0x5DA4FBFC0E1108A8, 0xFD17B448A6855419, 0x9C47D08FFB10D4B8 }
                ),
                N = new ulong[] { 0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF }
            };
        }
    }

    /// <summary>
    /// Big integer structure for cryptographic operations.
    /// </summary>
    public struct BigInteger256
    {
        /// <summary>
        /// 
        /// </summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong Word0, Word1, Word2, Word3;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="BigInteger256"/> struct.
        /// </summary>
        /// <param name="w0">The w0.</param>
        /// <param name="w1">The w1.</param>
        /// <param name="w2">The w2.</param>
        /// <param name="w3">The w3.</param>
        public BigInteger256(ulong w0, ulong w1, ulong w2, ulong w3)
        {
            Word0 = w0; Word1 = w1; Word2 = w2; Word3 = w3;
        }

        /// <summary>
        /// Creates BigInteger from byte array (little-endian).
        /// </summary>
        public static BigInteger256 FromBytes(byte[] bytes)
        {
            if (bytes.Length > 32) throw new ArgumentException("Too many bytes for 256-bit integer");
            
            var padded = new byte[32];
            Array.Copy(bytes, padded, bytes.Length);
            
            return new BigInteger256(
                BitConverter.ToUInt64(padded, 0),
                BitConverter.ToUInt64(padded, 8),
                BitConverter.ToUInt64(padded, 16),
                BitConverter.ToUInt64(padded, 24)
            );
        }

        /// <summary>
        /// Converts to byte array (little-endian).
        /// </summary>
        public byte[] ToBytes()
        {
            var bytes = new byte[32];
            BitConverter.GetBytes(Word0).CopyTo(bytes, 0);
            BitConverter.GetBytes(Word1).CopyTo(bytes, 8);
            BitConverter.GetBytes(Word2).CopyTo(bytes, 16);
            BitConverter.GetBytes(Word3).CopyTo(bytes, 24);
            return bytes;
        }

        /// <summary>
        /// Zero constant.
        /// </summary>
        public static BigInteger256 Zero => new BigInteger256(0, 0, 0, 0);

        /// <summary>
        /// One constant.
        /// </summary>
        public static BigInteger256 One => new BigInteger256(1, 0, 0, 0);
    }

    /// <summary>
    /// Cryptographic operation result.
    /// </summary>
    /// <typeparam name="T">Result data type.</typeparam>
    public sealed class CryptoResult<T> : IDisposable where T : unmanaged
    {
        /// <summary>Result data buffer</summary>
        public MemoryBuffer1D<T, Stride1D.Dense> Data { get; }
        /// <summary>Operation status</summary>
        public bool Success { get; }
        /// <summary>Error message if operation failed</summary>
        public string? ErrorMessage { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="CryptoResult{T}"/> class.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="success">if set to <c>true</c> [success].</param>
        /// <param name="error">The error.</param>
        /// <exception cref="System.ArgumentNullException">data</exception>
        public CryptoResult(MemoryBuffer1D<T, Stride1D.Dense> data, bool success, string? error = null)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Success = success;
            ErrorMessage = error;
        }

        /// <summary>
        /// Disposes the result and releases GPU memory.
        /// </summary>
        public void Dispose()
        {
            Data?.Dispose();
        }
    }

    /// <summary>
    /// Cryptographic random number generator state.
    /// </summary>
    public struct CryptoRNGState
    {
        /// <summary>Internal state words</summary>

#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong State0, State1, State2, State3;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>Counter for stream ciphers</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong Counter;
#pragma warning restore CA1051 // Do not declare visible instance fields
        /// <summary>Key for keyed PRNGs</summary>
#pragma warning disable CA1051 // Do not declare visible instance fields
        public ulong[] Key;
#pragma warning restore CA1051 // Do not declare visible instance fields

        /// <summary>
        /// Initializes a new instance of the <see cref="CryptoRNGState"/> struct.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <exception cref="System.ArgumentException">Seed must have at least 4 elements</exception>
        public CryptoRNGState(ulong[] seed)
        {
            if (seed.Length < 4) throw new ArgumentException("Seed must have at least 4 elements");
            
            State0 = seed[0];
            State1 = seed[1];
            State2 = seed[2];
            State3 = seed[3];
            Counter = 0;
            Key = seed.Length > 4 ? seed[4..] : new ulong[4];
        }
    }

    /// <summary>
    /// Cryptographic constants and utilities.
    /// </summary>
    public static class CryptoConstants
    {
        /// <summary>SHA-256 initial hash values</summary>
#pragma warning disable CA1707 // Identifiers should not contain underscores
        public static readonly uint[] SHA256_H = {
#pragma warning restore CA1707 // Identifiers should not contain underscores
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        /// <summary>SHA-256 round constants</summary>
#pragma warning disable CA1707 // Identifiers should not contain underscores
        public static readonly uint[] SHA256_K = {
#pragma warning restore CA1707 // Identifiers should not contain underscores
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        /// <summary>AES S-box</summary>
#pragma warning disable CA1707 // Identifiers should not contain underscores
        public static readonly byte[] AES_SBOX = {
#pragma warning restore CA1707 // Identifiers should not contain underscores
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        };

        /// <summary>ChaCha20 constants</summary>
#pragma warning disable CA1707 // Identifiers should not contain underscores
        public static readonly uint[] CHACHA20_CONSTANTS = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };
#pragma warning restore CA1707 // Identifiers should not contain underscores
    }
}
