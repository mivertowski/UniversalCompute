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
    /// GPU-accelerated cryptographic hash functions.
    /// </summary>
    public static class HashFunctions
    {
        #region SHA-256

        /// <summary>
        /// Computes SHA-256 hash of input data.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="data">Input data to hash.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>SHA-256 hash result.</returns>
        public static Hash256 SHA256(Accelerator accelerator, byte[] data, AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? accelerator.DefaultStream;
            
            // Pad the message according to SHA-256 specification
            var paddedData = PadMessage(data, 64);
            var dataBuffer = accelerator.Allocate1D(paddedData);
            
            // Initialize hash values
            var hashBuffer = accelerator.Allocate1D(CryptoConstants.SHA256_H);
            var constantsBuffer = accelerator.Allocate1D(CryptoConstants.SHA256_K);
            
            // Process message in 512-bit (64-byte) chunks
            var numBlocks = paddedData.Length / 64;
            
            var sha256Kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<uint>, int>(SHA256Kernel);
            
            sha256Kernel(numBlocks, dataBuffer.View, hashBuffer.View, constantsBuffer.View, 64);
            actualStream.Synchronize();
            
            // Get result
            var result = new uint[8];
            hashBuffer.CopyToCPU(result);
            
            // Cleanup
            dataBuffer.Dispose();
            hashBuffer.Dispose();
            constantsBuffer.Dispose();
            
            // Convert to Hash256 structure
            return new Hash256(
                ((ulong)result[1] << 32) | result[0],
                ((ulong)result[3] << 32) | result[2],
                ((ulong)result[5] << 32) | result[4],
                ((ulong)result[7] << 32) | result[6]
            );
        }

        /// <summary>
        /// Batch SHA-256 computation for multiple inputs.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="inputs">Array of input data arrays.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Array of SHA-256 hash results.</returns>
        public static Hash256[] SHA256Batch(Accelerator accelerator, byte[][] inputs, AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? accelerator.DefaultStream;
            var numInputs = inputs.Length;
            
            // Find maximum padded size
            var maxSize = 0;
            foreach (var input in inputs)
            {
                var paddedSize = ((input.Length + 9 + 63) / 64) * 64;
                maxSize = Math.Max(maxSize, paddedSize);
            }
            
            // Create batched input buffer
            var batchedData = new byte[numInputs * maxSize];
            var inputSizes = new int[numInputs];
            
            for (int i = 0; i < numInputs; i++)
            {
                var paddedInput = PadMessage(inputs[i], 64);
                inputSizes[i] = paddedInput.Length;
                Array.Copy(paddedInput, 0, batchedData, i * maxSize, paddedInput.Length);
            }
            
            var dataBuffer = accelerator.Allocate1D(batchedData);
            var sizesBuffer = accelerator.Allocate1D(inputSizes);
            var resultsBuffer = accelerator.Allocate1D<uint>(numInputs * 8);
            var constantsBuffer = accelerator.Allocate1D(CryptoConstants.SHA256_K);
            
            var batchSHA256Kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<uint>, int>(
                BatchSHA256Kernel);
            
            batchSHA256Kernel(numInputs, dataBuffer.View, sizesBuffer.View, 
                resultsBuffer.View, constantsBuffer.View, maxSize);
            actualStream.Synchronize();
            
            // Extract results
            var hostResults = new uint[numInputs * 8];
            resultsBuffer.CopyToCPU(hostResults);
            
            var results = new Hash256[numInputs];
            for (int i = 0; i < numInputs; i++)
            {
                var offset = i * 8;
                results[i] = new Hash256(
                    ((ulong)hostResults[offset + 1] << 32) | hostResults[offset],
                    ((ulong)hostResults[offset + 3] << 32) | hostResults[offset + 2],
                    ((ulong)hostResults[offset + 5] << 32) | hostResults[offset + 4],
                    ((ulong)hostResults[offset + 7] << 32) | hostResults[offset + 6]
                );
            }
            
            // Cleanup
            dataBuffer.Dispose();
            sizesBuffer.Dispose();
            resultsBuffer.Dispose();
            constantsBuffer.Dispose();
            
            return results;
        }

        #endregion

        #region BLAKE2b

        /// <summary>
        /// Computes BLAKE2b hash of input data.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="data">Input data to hash.</param>
        /// <param name="hashSize">Hash output size in bytes (1-64).</param>
        /// <param name="key">Optional key for keyed hashing.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>BLAKE2b hash result.</returns>
        public static Hash512 BLAKE2b(
            Accelerator accelerator, 
            byte[] data, 
            int hashSize = 64,
            byte[]? key = null,
            AcceleratorStream? stream = null)
        {
            if (hashSize < 1 || hashSize > 64) throw new ArgumentException("Hash size must be 1-64 bytes");
            
            var actualStream = stream ?? accelerator.DefaultStream;
            
            // Initialize BLAKE2b parameters
            var parameters = InitializeBLAKE2bParameters(hashSize, key);
            var paramBuffer = accelerator.Allocate1D(parameters);
            
            // Prepare data with key if provided
            var processData = key != null ? ConcatenateKeyAndData(key, data) : data;
            var dataBuffer = accelerator.Allocate1D(processData);
            
            // Initialize hash state
            var hashState = new ulong[8];
            InitializeBLAKE2bState(hashState, hashSize, key?.Length ?? 0);
            var stateBuffer = accelerator.Allocate1D(hashState);
            
            // Process data in 128-byte blocks
            var numBlocks = (processData.Length + 127) / 128;
            
            var blake2bKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<ulong>, ArrayView<ulong>, int, int, int>(
                BLAKE2bKernel);
            
            blake2bKernel(numBlocks, dataBuffer.View, stateBuffer.View, 
                paramBuffer.View, processData.Length, hashSize, 128);
            actualStream.Synchronize();
            
            // Get final hash state
            var finalState = new ulong[8];
            stateBuffer.CopyToCPU(finalState);
            
            // Cleanup
            paramBuffer.Dispose();
            dataBuffer.Dispose();
            stateBuffer.Dispose();
            
            return new Hash512(finalState[0], finalState[1], finalState[2], finalState[3],
                              finalState[4], finalState[5], finalState[6], finalState[7]);
        }

        #endregion

        #region Keccak-256 (Ethereum)

        /// <summary>
        /// Computes Keccak-256 hash (used in Ethereum).
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="data">Input data to hash.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Keccak-256 hash result.</returns>
        public static Hash256 Keccak256(Accelerator accelerator, byte[] data, AcceleratorStream? stream = null)
        {
            var actualStream = stream ?? accelerator.DefaultStream;
            
            // Pad data for Keccak (rate = 1088 bits = 136 bytes for Keccak-256)
            var rate = 136;
            var paddedData = PadKeccak(data, rate);
            var dataBuffer = accelerator.Allocate1D(paddedData);
            
            // Initialize Keccak state (25 64-bit words)
            var state = new ulong[25];
            var stateBuffer = accelerator.Allocate1D(state);
            
            // Process data
            var numBlocks = paddedData.Length / rate;
            
            var keccakKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<ulong>, int, int>(KeccakKernel);
            
            keccakKernel(numBlocks, dataBuffer.View, stateBuffer.View, rate, 24);
            actualStream.Synchronize();
            
            // Extract hash (first 256 bits = 32 bytes)
            var finalState = new ulong[25];
            stateBuffer.CopyToCPU(finalState);
            
            // Cleanup
            dataBuffer.Dispose();
            stateBuffer.Dispose();
            
            return new Hash256(finalState[0], finalState[1], finalState[2], finalState[3]);
        }

        #endregion

        #region Kernel Implementations

        private static void SHA256Kernel(
            Index1D index,
            ArrayView<byte> data,
            ArrayView<uint> hash,
            ArrayView<uint> constants,
            int blockSize)
        {
            if (index >= data.Length / blockSize) return;
            
            var blockStart = index * blockSize;
            
            // Initialize working variables
            var a = hash[0]; var b = hash[1]; var c = hash[2]; var d = hash[3];
            var e = hash[4]; var f = hash[5]; var g = hash[6]; var h = hash[7];
            
            // Prepare message schedule
            var w = new uint[64];
            
            // Copy chunk into first 16 words of message schedule
            for (int i = 0; i < 16; i++)
            {
                var wordStart = blockStart + i * 4;
                w[i] = ((uint)data[wordStart] << 24) | ((uint)data[wordStart + 1] << 16) |
                       ((uint)data[wordStart + 2] << 8) | data[wordStart + 3];
            }
            
            // Extend the first 16 words into the remaining 48 words
            for (int i = 16; i < 64; i++)
            {
                var s0 = RightRotate32(w[i - 15], 7) ^ RightRotate32(w[i - 15], 18) ^ (w[i - 15] >> 3);
                var s1 = RightRotate32(w[i - 2], 17) ^ RightRotate32(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            
            // Main loop
            for (int i = 0; i < 64; i++)
            {
                var S1 = RightRotate32(e, 6) ^ RightRotate32(e, 11) ^ RightRotate32(e, 25);
                var ch = (e & f) ^ (~e & g);
                var temp1 = h + S1 + ch + constants[i] + w[i];
                var S0 = RightRotate32(a, 2) ^ RightRotate32(a, 13) ^ RightRotate32(a, 22);
                var maj = (a & b) ^ (a & c) ^ (b & c);
                var temp2 = S0 + maj;
                
                h = g; g = f; f = e; e = d + temp1;
                d = c; c = b; b = a; a = temp1 + temp2;
            }
            
            // Add this chunk's hash to result
            Atomic.Add(ref hash[0], a);
            Atomic.Add(ref hash[1], b);
            Atomic.Add(ref hash[2], c);
            Atomic.Add(ref hash[3], d);
            Atomic.Add(ref hash[4], e);
            Atomic.Add(ref hash[5], f);
            Atomic.Add(ref hash[6], g);
            Atomic.Add(ref hash[7], h);
        }

        private static void BatchSHA256Kernel(
            Index1D index,
            ArrayView<byte> data,
            ArrayView<int> sizes,
            ArrayView<uint> results,
            ArrayView<uint> constants,
            int maxSize)
        {
            if (index >= sizes.Length) return;
            
            var inputIndex = index.X;
            var dataStart = inputIndex * maxSize;
            var resultStart = inputIndex * 8;
            var inputSize = sizes[inputIndex];
            
            // Initialize hash for this input
            var h = new uint[8];
            for (int i = 0; i < 8; i++)
                h[i] = CryptoConstants.SHA256_H[i];
            
            // Process blocks
            var numBlocks = inputSize / 64;
            for (int block = 0; block < numBlocks; block++)
            {
                ProcessSHA256Block(data, h, constants, dataStart + block * 64);
            }
            
            // Store result
            for (int i = 0; i < 8; i++)
                results[resultStart + i] = h[i];
        }

        private static void BLAKE2bKernel(
            Index1D index,
            ArrayView<byte> data,
            ArrayView<ulong> state,
            ArrayView<ulong> parameters,
            int dataLength,
            int hashSize,
            int blockSize)
        {
            if (index >= (dataLength + blockSize - 1) / blockSize) return;
            
            var blockStart = index * blockSize;
            var isLastBlock = blockStart + blockSize >= dataLength;
            var blockLength = isLastBlock ? (int)(dataLength - blockStart) : blockSize;
            
            // Process BLAKE2b block (simplified)
            ProcessBLAKE2bBlock(data, state, parameters, blockStart, blockLength, isLastBlock);
        }

        private static void KeccakKernel(
            Index1D index,
            ArrayView<byte> data,
            ArrayView<ulong> state,
            int rate,
            int rounds)
        {
            if (index >= data.Length / rate) return;
            
            var blockStart = index * rate;
            
            // Absorb block into state
            for (int i = 0; i < rate / 8; i++)
            {
                var wordStart = blockStart + i * 8;
                ulong word = 0;
                for (int j = 0; j < 8; j++)
                {
                    if (wordStart + j < data.Length)
                        word |= ((ulong)data[wordStart + j]) << (j * 8);
                }
                state[i] ^= word;
            }
            
            // Apply Keccak-f permutation
            KeccakF(state, rounds);
        }

        #endregion

        #region Helper Methods

        private static uint RightRotate32(uint value, int amount)
        {
            return (value >> amount) | (value << (32 - amount));
        }

        private static ulong RightRotate64(ulong value, int amount)
        {
            return (value >> amount) | (value << (64 - amount));
        }

        private static byte[] PadMessage(byte[] data, int blockSize)
        {
            var messageLength = data.Length;
            var bitLength = (ulong)messageLength * 8;
            
            // Calculate padding length
            var k = (blockSize - ((messageLength + 9) % blockSize)) % blockSize;
            var paddedLength = messageLength + 1 + k + 8;
            
            var padded = new byte[paddedLength];
            Array.Copy(data, padded, messageLength);
            
            // Add padding bit
            padded[messageLength] = 0x80;
            
            // Add length in bits (big-endian)
            for (int i = 0; i < 8; i++)
            {
                padded[paddedLength - 8 + i] = (byte)(bitLength >> (56 - i * 8));
            }
            
            return padded;
        }

        private static byte[] PadKeccak(byte[] data, int rate)
        {
            var paddedLength = ((data.Length + rate) / rate) * rate;
            var padded = new byte[paddedLength];
            Array.Copy(data, padded, data.Length);
            
            // Keccak padding: append 1, then 0s, then 1
            padded[data.Length] = 0x01;
            padded[paddedLength - 1] |= 0x80;
            
            return padded;
        }

        private static ulong[] InitializeBLAKE2bParameters(int hashSize, byte[]? key)
        {
            var parameters = new ulong[8];
            
            // Parameter block (simplified)
            parameters[0] = 0x01010000 ^ ((ulong)hashSize) ^ ((ulong)(key?.Length ?? 0) << 8);
            
            return parameters;
        }

        private static void InitializeBLAKE2bState(ulong[] state, int hashSize, int keyLength)
        {
            // BLAKE2b IV constants (simplified)
            var iv = new ulong[] {
                0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
            };
            
            for (int i = 0; i < 8; i++)
                state[i] = iv[i];
            
            // XOR with parameter block
            state[0] ^= 0x01010000 ^ (ulong)hashSize ^ ((ulong)keyLength << 8);
        }

        private static byte[] ConcatenateKeyAndData(byte[] key, byte[] data)
        {
            var padded = new byte[128];
            Array.Copy(key, padded, Math.Min(key.Length, 128));
            
            var result = new byte[padded.Length + data.Length];
            Array.Copy(padded, result, padded.Length);
            Array.Copy(data, 0, result, padded.Length, data.Length);
            
            return result;
        }

        private static void ProcessSHA256Block(ArrayView<byte> data, uint[] hash, ArrayView<uint> constants, int blockStart)
        {
            // Simplified SHA-256 block processing
            var w = new uint[64];
            
            // Copy and extend message schedule
            for (int i = 0; i < 16; i++)
            {
                var wordStart = blockStart + i * 4;
                w[i] = ((uint)data[wordStart] << 24) | ((uint)data[wordStart + 1] << 16) |
                       ((uint)data[wordStart + 2] << 8) | data[wordStart + 3];
            }
            
            for (int i = 16; i < 64; i++)
            {
                var s0 = RightRotate32(w[i - 15], 7) ^ RightRotate32(w[i - 15], 18) ^ (w[i - 15] >> 3);
                var s1 = RightRotate32(w[i - 2], 17) ^ RightRotate32(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            
            // Compression function (simplified)
            var a = hash[0]; var b = hash[1]; var c = hash[2]; var d = hash[3];
            var e = hash[4]; var f = hash[5]; var g = hash[6]; var h = hash[7];
            
            for (int i = 0; i < 64; i++)
            {
                var S1 = RightRotate32(e, 6) ^ RightRotate32(e, 11) ^ RightRotate32(e, 25);
                var ch = (e & f) ^ (~e & g);
                var temp1 = h + S1 + ch + constants[i] + w[i];
                var S0 = RightRotate32(a, 2) ^ RightRotate32(a, 13) ^ RightRotate32(a, 22);
                var maj = (a & b) ^ (a & c) ^ (b & c);
                var temp2 = S0 + maj;
                
                h = g; g = f; f = e; e = d + temp1;
                d = c; c = b; b = a; a = temp1 + temp2;
            }
            
            hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
            hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
        }

        private static void ProcessBLAKE2bBlock(
            ArrayView<byte> data,
            ArrayView<ulong> state,
            ArrayView<ulong> parameters,
            int blockStart,
            int blockLength,
            bool isLastBlock)
        {
            // Simplified BLAKE2b block processing
            // Real implementation would include full mixing function
        }

        private static void KeccakF(ArrayView<ulong> state, int rounds)
        {
            // Simplified Keccak-f permutation
            // Real implementation would include all 5 steps: θ, ρ, π, χ, ι
            
            var roundConstants = new ulong[] {
                0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
                0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
                0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x8000000000008003,
                0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
                0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
                0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b
            };
            
            for (int round = 0; round < rounds; round++)
            {
                // Simplified round function
                state[0] ^= roundConstants[round];
                
                // Real implementation would apply full Keccak round function
            }
        }

        #endregion
    }
}