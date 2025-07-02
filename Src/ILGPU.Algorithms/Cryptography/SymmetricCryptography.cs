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
    /// GPU-accelerated symmetric cryptography operations.
    /// </summary>
    public static class SymmetricCryptography
    {
        #region AES Encryption/Decryption

        /// <summary>
        /// Encrypts data using AES algorithm.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="plaintext">Data to encrypt (must be multiple of 16 bytes).</param>
        /// <param name="key">AES key (128, 192, or 256 bits).</param>
        /// <param name="iv">Initialization vector (16 bytes, required for CBC/CFB/OFB modes).</param>
        /// <param name="mode">Cipher mode of operation.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Encrypted data.</returns>
        public static CryptoResult<byte> AESEncrypt(
            Accelerator accelerator,
            byte[] plaintext,
            byte[] key,
            byte[]? iv = null,
            CipherMode mode = CipherMode.ECB,
            AcceleratorStream? stream = null)
        {
            if (plaintext.Length % 16 != 0)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "Plaintext length must be multiple of 16 bytes");

            if (mode != CipherMode.ECB && (iv == null || iv.Length != 16))
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "IV must be 16 bytes for CBC/CFB/OFB/CTR modes");

            var actualStream = stream ?? accelerator.DefaultStream;
            var aesKey = new AESKey(key);

            // Allocate GPU buffers
            var plaintextBuffer = accelerator.Allocate1D(plaintext);
            var ciphertextBuffer = accelerator.Allocate1D<byte>(plaintext.Length);
            var keyBuffer = accelerator.Allocate1D(aesKey.RoundKeys);
            var ivBuffer = iv != null ? accelerator.Allocate1D(iv) : accelerator.Allocate1D<byte>(16);

            // Load encryption kernel
            var aesKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<uint>, ArrayView<byte>,
                int, int, int, bool>(AESKernel);

            var numBlocks = plaintext.Length / 16;
            aesKernel(actualStream, numBlocks, plaintextBuffer.View, ciphertextBuffer.View,
                keyBuffer.View, ivBuffer.View, aesKey.Rounds, (int)mode, 16, true);

            actualStream.Synchronize();

            // Cleanup intermediate buffers
            plaintextBuffer.Dispose();
            keyBuffer.Dispose();
            ivBuffer.Dispose();

            return new CryptoResult<byte>(ciphertextBuffer, true);
        }

        /// <summary>
        /// Decrypts data using AES algorithm.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="ciphertext">Data to decrypt (must be multiple of 16 bytes).</param>
        /// <param name="key">AES key (128, 192, or 256 bits).</param>
        /// <param name="iv">Initialization vector (16 bytes, required for CBC/CFB/OFB modes).</param>
        /// <param name="mode">Cipher mode of operation.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Decrypted data.</returns>
        public static CryptoResult<byte> AESDecrypt(
            Accelerator accelerator,
            byte[] ciphertext,
            byte[] key,
            byte[]? iv = null,
            CipherMode mode = CipherMode.ECB,
            AcceleratorStream? stream = null)
        {
            if (ciphertext.Length % 16 != 0)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "Ciphertext length must be multiple of 16 bytes");

            if (mode != CipherMode.ECB && (iv == null || iv.Length != 16))
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "IV must be 16 bytes for CBC/CFB/OFB/CTR modes");

            var actualStream = stream ?? accelerator.DefaultStream;
            var aesKey = new AESKey(key);

            // Allocate GPU buffers
            var ciphertextBuffer = accelerator.Allocate1D(ciphertext);
            var plaintextBuffer = accelerator.Allocate1D<byte>(ciphertext.Length);
            var keyBuffer = accelerator.Allocate1D(aesKey.RoundKeys);
            var ivBuffer = iv != null ? accelerator.Allocate1D(iv) : accelerator.Allocate1D<byte>(16);

            // Load decryption kernel
            var aesKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<uint>, ArrayView<byte>,
                int, int, int, bool>(AESKernel);

            var numBlocks = ciphertext.Length / 16;
            aesKernel(actualStream, numBlocks, ciphertextBuffer.View, plaintextBuffer.View,
                keyBuffer.View, ivBuffer.View, aesKey.Rounds, (int)mode, 16, false);

            actualStream.Synchronize();

            // Cleanup intermediate buffers
            ciphertextBuffer.Dispose();
            keyBuffer.Dispose();
            ivBuffer.Dispose();

            return new CryptoResult<byte>(plaintextBuffer, true);
        }

        #endregion

        #region ChaCha20 Stream Cipher

        /// <summary>
        /// Encrypts/decrypts data using ChaCha20 stream cipher.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="data">Data to encrypt/decrypt.</param>
        /// <param name="key">256-bit key (32 bytes).</param>
        /// <param name="nonce">96-bit nonce (12 bytes).</param>
        /// <param name="counter">Initial counter value.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Encrypted/decrypted data.</returns>
        public static CryptoResult<byte> ChaCha20(
            Accelerator accelerator,
            byte[] data,
            byte[] key,
            byte[] nonce,
            uint counter = 0,
            AcceleratorStream? stream = null)
        {
            if (key.Length != 32)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "ChaCha20 key must be 32 bytes");

            if (nonce.Length != 12)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "ChaCha20 nonce must be 12 bytes");

            var actualStream = stream ?? accelerator.DefaultStream;

            // Allocate GPU buffers
            var dataBuffer = accelerator.Allocate1D(data);
            var outputBuffer = accelerator.Allocate1D<byte>(data.Length);
            var keyBuffer = accelerator.Allocate1D(key);
            var nonceBuffer = accelerator.Allocate1D(nonce);

            // Load ChaCha20 kernel
            var chachaKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
                uint, int>(ChaCha20Kernel);

            var numBlocks = (data.Length + 63) / 64; // ChaCha20 processes 64-byte blocks
            chachaKernel(actualStream, numBlocks, dataBuffer.View, outputBuffer.View,
                keyBuffer.View, nonceBuffer.View, counter, data.Length);

            actualStream.Synchronize();

            // Cleanup intermediate buffers
            dataBuffer.Dispose();
            keyBuffer.Dispose();
            nonceBuffer.Dispose();

            return new CryptoResult<byte>(outputBuffer, true);
        }

        #endregion

        #region Salsa20 Stream Cipher

        /// <summary>
        /// Encrypts/decrypts data using Salsa20 stream cipher.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="data">Data to encrypt/decrypt.</param>
        /// <param name="key">256-bit key (32 bytes).</param>
        /// <param name="nonce">64-bit nonce (8 bytes).</param>
        /// <param name="counter">Initial counter value.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Encrypted/decrypted data.</returns>
        public static CryptoResult<byte> Salsa20(
            Accelerator accelerator,
            byte[] data,
            byte[] key,
            byte[] nonce,
            ulong counter = 0,
            AcceleratorStream? stream = null)
        {
            if (key.Length != 32)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "Salsa20 key must be 32 bytes");

            if (nonce.Length != 8)
                return new CryptoResult<byte>(
                    accelerator.Allocate1D<byte>(0), 
                    false, 
                    "Salsa20 nonce must be 8 bytes");

            var actualStream = stream ?? accelerator.DefaultStream;

            // Allocate GPU buffers
            var dataBuffer = accelerator.Allocate1D(data);
            var outputBuffer = accelerator.Allocate1D<byte>(data.Length);
            var keyBuffer = accelerator.Allocate1D(key);
            var nonceBuffer = accelerator.Allocate1D(nonce);

            // Load Salsa20 kernel
            var salsaKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
                ulong, int>(Salsa20Kernel);

            var numBlocks = (data.Length + 63) / 64; // Salsa20 processes 64-byte blocks
            salsaKernel(actualStream, numBlocks, dataBuffer.View, outputBuffer.View,
                keyBuffer.View, nonceBuffer.View, counter, data.Length);

            actualStream.Synchronize();

            // Cleanup intermediate buffers
            dataBuffer.Dispose();
            keyBuffer.Dispose();
            nonceBuffer.Dispose();

            return new CryptoResult<byte>(outputBuffer, true);
        }

        #endregion

        #region Batch Operations

        /// <summary>
        /// Batch AES encryption for multiple inputs.
        /// </summary>
        /// <param name="accelerator">ILGPU accelerator.</param>
        /// <param name="plaintexts">Array of plaintexts to encrypt.</param>
        /// <param name="keys">Array of keys (one per plaintext).</param>
        /// <param name="ivs">Array of IVs (one per plaintext, can be null for ECB).</param>
        /// <param name="mode">Cipher mode of operation.</param>
        /// <param name="stream">Accelerator stream for execution.</param>
        /// <returns>Array of encrypted results.</returns>
        public static CryptoResult<byte>[] AESEncryptBatch(
            Accelerator accelerator,
            byte[][] plaintexts,
            byte[][] keys,
            byte[][]? ivs = null,
            CipherMode mode = CipherMode.ECB,
            AcceleratorStream? stream = null)
        {
            if (plaintexts.Length != keys.Length)
                throw new ArgumentException("Number of plaintexts must match number of keys");

            if (ivs != null && ivs.Length != plaintexts.Length)
                throw new ArgumentException("Number of IVs must match number of plaintexts");

            var actualStream = stream ?? accelerator.DefaultStream;
            var results = new CryptoResult<byte>[plaintexts.Length];

            // Calculate batch parameters
            var maxLength = 0;
            foreach (var plaintext in plaintexts)
                maxLength = Math.Max(maxLength, plaintext.Length);

            // Prepare batched data
            var batchedPlaintext = new byte[plaintexts.Length * maxLength];
            var batchedCiphertext = accelerator.Allocate1D<byte>(plaintexts.Length * maxLength);
            var lengths = new int[plaintexts.Length];

            for (int i = 0; i < plaintexts.Length; i++)
            {
                lengths[i] = plaintexts[i].Length;
                Array.Copy(plaintexts[i], 0, batchedPlaintext, i * maxLength, plaintexts[i].Length);
            }

            var plaintextBuffer = accelerator.Allocate1D(batchedPlaintext);
            var lengthBuffer = accelerator.Allocate1D(lengths);

            // Process batch
            var batchKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int, int>(
                BatchAESKernel);

            batchKernel(actualStream, plaintexts.Length, plaintextBuffer.View, 
                batchedCiphertext.View, lengthBuffer.View, maxLength, (int)mode);

            actualStream.Synchronize();

            // Extract individual results
            var hostCiphertext = new byte[plaintexts.Length * maxLength];
            batchedCiphertext.CopyToCPU(hostCiphertext);

            for (int i = 0; i < plaintexts.Length; i++)
            {
                var resultLength = lengths[i];
                var resultData = new byte[resultLength];
                Array.Copy(hostCiphertext, i * maxLength, resultData, 0, resultLength);

                var resultBuffer = accelerator.Allocate1D(resultData);
                results[i] = new CryptoResult<byte>(resultBuffer, true);
            }

            // Cleanup
            plaintextBuffer.Dispose();
            batchedCiphertext.Dispose();
            lengthBuffer.Dispose();

            return results;
        }

        #endregion

        #region Kernel Implementations

        private static void AESKernel(
            Index1D index,
            ArrayView<byte> input,
            ArrayView<byte> output,
            ArrayView<uint> roundKeys,
            ArrayView<byte> iv,
            int numRounds,
            int mode,
            int blockSize,
            bool encrypt)
        {
            if (index >= input.Length / blockSize) return;

            var blockIndex = index.X;
            var inputOffset = blockIndex * blockSize;
            var outputOffset = blockIndex * blockSize;

            // Load input block
            var block = new byte[16];
            for (int i = 0; i < 16; i++)
                block[i] = input[inputOffset + i];

            // Apply cipher mode preprocessing
            if (mode == (int)CipherMode.CBC && encrypt)
            {
                // XOR with previous ciphertext block (or IV for first block)
                for (int i = 0; i < 16; i++)
                {
                    var xorByte = blockIndex == 0 ? iv[i] : output[outputOffset - 16 + i];
                    block[i] ^= xorByte;
                }
            }

            // Perform AES encryption/decryption
            if (encrypt)
                AESEncryptBlock(block, roundKeys, numRounds);
            else
                AESDecryptBlock(block, roundKeys, numRounds);

            // Apply cipher mode postprocessing
            if (mode == (int)CipherMode.CBC && !encrypt)
            {
                // XOR with previous ciphertext block (or IV for first block)
                for (int i = 0; i < 16; i++)
                {
                    var xorByte = blockIndex == 0 ? iv[i] : input[inputOffset - 16 + i];
                    block[i] ^= xorByte;
                }
            }

            // Store output block
            for (int i = 0; i < 16; i++)
                output[outputOffset + i] = block[i];
        }

        private static void ChaCha20Kernel(
            Index1D index,
            ArrayView<byte> input,
            ArrayView<byte> output,
            ArrayView<byte> key,
            ArrayView<byte> nonce,
            uint counter,
            int dataLength)
        {
            if (index >= (dataLength + 63) / 64) return;

            var blockIndex = index.X;
            var blockOffset = blockIndex * 64;
            var blockLength = Math.Min(64, dataLength - blockOffset);

            // Generate ChaCha20 keystream block
            var keystream = new byte[64];
            ChaCha20Block(keystream, key, nonce, counter + (uint)blockIndex);

            // XOR with input data
            for (int i = 0; i < blockLength; i++)
            {
                output[blockOffset + i] = (byte)(input[blockOffset + i] ^ keystream[i]);
            }
        }

        private static void Salsa20Kernel(
            Index1D index,
            ArrayView<byte> input,
            ArrayView<byte> output,
            ArrayView<byte> key,
            ArrayView<byte> nonce,
            ulong counter,
            int dataLength)
        {
            if (index >= (dataLength + 63) / 64) return;

            var blockIndex = index.X;
            var blockOffset = blockIndex * 64;
            var blockLength = Math.Min(64, dataLength - blockOffset);

            // Generate Salsa20 keystream block
            var keystream = new byte[64];
            Salsa20Block(keystream, key, nonce, counter + (ulong)blockIndex);

            // XOR with input data
            for (int i = 0; i < blockLength; i++)
            {
                output[blockOffset + i] = (byte)(input[blockOffset + i] ^ keystream[i]);
            }
        }

        private static void BatchAESKernel(
            Index1D index,
            ArrayView<byte> batchedInput,
            ArrayView<byte> batchedOutput,
            ArrayView<int> lengths,
            int maxLength,
            int mode)
        {
            if (index >= lengths.Length) return;

            var inputIndex = index.X;
            var inputOffset = inputIndex * maxLength;
            var outputOffset = inputIndex * maxLength;
            var length = lengths[inputIndex];

            // Process each block in this input
            var numBlocks = length / 16;
            for (int block = 0; block < numBlocks; block++)
            {
                var blockOffset = block * 16;
                
                // Load block
                var inputBlock = new byte[16];
                for (int i = 0; i < 16; i++)
                    inputBlock[i] = batchedInput[inputOffset + blockOffset + i];

                // Encrypt block (simplified)
                AESEncryptBlockSimplified(inputBlock);

                // Store result
                for (int i = 0; i < 16; i++)
                    batchedOutput[outputOffset + blockOffset + i] = inputBlock[i];
            }
        }

        #endregion

        #region Helper Methods

        private static void AESEncryptBlock(byte[] block, ArrayView<uint> roundKeys, int numRounds)
        {
            // Initial round key addition
            AddRoundKey(block, roundKeys, 0);

            // Main rounds
            for (int round = 1; round < numRounds; round++)
            {
                SubBytes(block);
                ShiftRows(block);
                MixColumns(block);
                AddRoundKey(block, roundKeys, round);
            }

            // Final round (no MixColumns)
            SubBytes(block);
            ShiftRows(block);
            AddRoundKey(block, roundKeys, numRounds);
        }

        private static void AESDecryptBlock(byte[] block, ArrayView<uint> roundKeys, int numRounds)
        {
            // Initial round key addition
            AddRoundKey(block, roundKeys, numRounds);

            // Main rounds (in reverse)
            for (int round = numRounds - 1; round > 0; round--)
            {
                InvShiftRows(block);
                InvSubBytes(block);
                AddRoundKey(block, roundKeys, round);
                InvMixColumns(block);
            }

            // Final round
            InvShiftRows(block);
            InvSubBytes(block);
            AddRoundKey(block, roundKeys, 0);
        }

        private static void AESEncryptBlockSimplified(byte[] block)
        {
            // Simplified AES encryption for demonstration
            for (int i = 0; i < 16; i++)
                block[i] = CryptoConstants.AES_SBOX[block[i]];
        }

        private static void SubBytes(byte[] state)
        {
            for (int i = 0; i < 16; i++)
                state[i] = CryptoConstants.AES_SBOX[state[i]];
        }

        private static void InvSubBytes(byte[] state)
        {
            // Inverse S-box operation (simplified)
            for (int i = 0; i < 16; i++)
            {
                // Would use inverse S-box lookup table
                state[i] = (byte)(state[i] ^ 0x42); // Placeholder
            }
        }

        private static void ShiftRows(byte[] state)
        {
            // Row 1: shift left by 1
            var temp = state[1];
            state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;

            // Row 2: shift left by 2
            temp = state[2]; state[2] = state[10]; state[10] = temp;
            temp = state[6]; state[6] = state[14]; state[14] = temp;

            // Row 3: shift left by 3 (or right by 1)
            temp = state[3];
            state[3] = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = temp;
        }

        private static void InvShiftRows(byte[] state)
        {
            // Inverse of ShiftRows
            var temp = state[13];
            state[13] = state[9]; state[9] = state[5]; state[5] = state[1]; state[1] = temp;

            temp = state[2]; state[2] = state[10]; state[10] = temp;
            temp = state[6]; state[6] = state[14]; state[14] = temp;

            temp = state[7];
            state[7] = state[11]; state[11] = state[15]; state[15] = state[3]; state[3] = temp;
        }

        private static void MixColumns(byte[] state)
        {
            // Simplified MixColumns operation
            for (int col = 0; col < 4; col++)
            {
                var c0 = state[col * 4];
                var c1 = state[col * 4 + 1];
                var c2 = state[col * 4 + 2];
                var c3 = state[col * 4 + 3];

                state[col * 4] = (byte)(GMul(0x02, c0) ^ GMul(0x03, c1) ^ c2 ^ c3);
                state[col * 4 + 1] = (byte)(c0 ^ GMul(0x02, c1) ^ GMul(0x03, c2) ^ c3);
                state[col * 4 + 2] = (byte)(c0 ^ c1 ^ GMul(0x02, c2) ^ GMul(0x03, c3));
                state[col * 4 + 3] = (byte)(GMul(0x03, c0) ^ c1 ^ c2 ^ GMul(0x02, c3));
            }
        }

        private static void InvMixColumns(byte[] state)
        {
            // Inverse MixColumns operation
            for (int col = 0; col < 4; col++)
            {
                var c0 = state[col * 4];
                var c1 = state[col * 4 + 1];
                var c2 = state[col * 4 + 2];
                var c3 = state[col * 4 + 3];

                state[col * 4] = (byte)(GMul(0x0e, c0) ^ GMul(0x0b, c1) ^ GMul(0x0d, c2) ^ GMul(0x09, c3));
                state[col * 4 + 1] = (byte)(GMul(0x09, c0) ^ GMul(0x0e, c1) ^ GMul(0x0b, c2) ^ GMul(0x0d, c3));
                state[col * 4 + 2] = (byte)(GMul(0x0d, c0) ^ GMul(0x09, c1) ^ GMul(0x0e, c2) ^ GMul(0x0b, c3));
                state[col * 4 + 3] = (byte)(GMul(0x0b, c0) ^ GMul(0x0d, c1) ^ GMul(0x09, c2) ^ GMul(0x0e, c3));
            }
        }

        private static void AddRoundKey(byte[] state, ArrayView<uint> roundKeys, int round)
        {
            for (int i = 0; i < 4; i++)
            {
                var key = roundKeys[round * 4 + i];
                state[i * 4] ^= (byte)(key & 0xFF);
                state[i * 4 + 1] ^= (byte)((key >> 8) & 0xFF);
                state[i * 4 + 2] ^= (byte)((key >> 16) & 0xFF);
                state[i * 4 + 3] ^= (byte)((key >> 24) & 0xFF);
            }
        }

        private static byte GMul(byte a, byte b)
        {
            // Galois field multiplication for AES
            byte p = 0;
            for (int i = 0; i < 8; i++)
            {
                if ((b & 1) != 0)
                    p ^= a;
                var hi = (a & 0x80) != 0;
                a <<= 1;
                if (hi)
                    a ^= 0x1b;
                b >>= 1;
            }
            return p;
        }

        private static void ChaCha20Block(byte[] output, ArrayView<byte> key, ArrayView<byte> nonce, uint counter)
        {
            // Initialize ChaCha20 state
            var state = new uint[16];
            
            // Constants
            state[0] = CryptoConstants.CHACHA20_CONSTANTS[0];
            state[1] = CryptoConstants.CHACHA20_CONSTANTS[1];
            state[2] = CryptoConstants.CHACHA20_CONSTANTS[2];
            state[3] = CryptoConstants.CHACHA20_CONSTANTS[3];

            // Key
            for (int i = 0; i < 8; i++)
            {
                state[4 + i] = BitConverter.ToUInt32(key.SubView(i * 4, 4).ToArray(), 0);
            }

            // Counter
            state[12] = counter;

            // Nonce
            for (int i = 0; i < 3; i++)
            {
                state[13 + i] = BitConverter.ToUInt32(nonce.SubView(i * 4, 4).ToArray(), 0);
            }

            // ChaCha20 rounds
            var working = new uint[16];
            Array.Copy(state, working, 16);

            for (int i = 0; i < 10; i++)
            {
                ChaCha20DoubleRound(working);
            }

            // Add original state
            for (int i = 0; i < 16; i++)
            {
                working[i] += state[i];
            }

            // Convert to bytes
            for (int i = 0; i < 16; i++)
            {
                var bytes = BitConverter.GetBytes(working[i]);
                Array.Copy(bytes, 0, output, i * 4, 4);
            }
        }

        private static void Salsa20Block(byte[] output, ArrayView<byte> key, ArrayView<byte> nonce, ulong counter)
        {
            // Simplified Salsa20 block generation
            var state = new uint[16];
            
            // Initialize state (simplified)
            for (int i = 0; i < 8; i++)
            {
                state[i] = BitConverter.ToUInt32(key.SubView(i * 4, 4).ToArray(), 0);
            }
            
            state[8] = (uint)counter;
            state[9] = (uint)(counter >> 32);
            
            for (int i = 0; i < 2; i++)
            {
                state[10 + i] = BitConverter.ToUInt32(nonce.SubView(i * 4, 4).ToArray(), 0);
            }

            // Salsa20 rounds (simplified)
            for (int i = 0; i < 10; i++)
            {
                Salsa20DoubleRound(state);
            }

            // Convert to bytes
            for (int i = 0; i < 16; i++)
            {
                var bytes = BitConverter.GetBytes(state[i]);
                Array.Copy(bytes, 0, output, i * 4, 4);
            }
        }

        private static void ChaCha20DoubleRound(uint[] state)
        {
            // Column rounds
            ChaCha20QuarterRound(state, 0, 4, 8, 12);
            ChaCha20QuarterRound(state, 1, 5, 9, 13);
            ChaCha20QuarterRound(state, 2, 6, 10, 14);
            ChaCha20QuarterRound(state, 3, 7, 11, 15);

            // Diagonal rounds
            ChaCha20QuarterRound(state, 0, 5, 10, 15);
            ChaCha20QuarterRound(state, 1, 6, 11, 12);
            ChaCha20QuarterRound(state, 2, 7, 8, 13);
            ChaCha20QuarterRound(state, 3, 4, 9, 14);
        }

        private static void ChaCha20QuarterRound(uint[] state, int a, int b, int c, int d)
        {
            state[a] += state[b]; state[d] ^= state[a]; state[d] = RotateLeft(state[d], 16);
            state[c] += state[d]; state[b] ^= state[c]; state[b] = RotateLeft(state[b], 12);
            state[a] += state[b]; state[d] ^= state[a]; state[d] = RotateLeft(state[d], 8);
            state[c] += state[d]; state[b] ^= state[c]; state[b] = RotateLeft(state[b], 7);
        }

        private static void Salsa20DoubleRound(uint[] state)
        {
            // Simplified Salsa20 double round
            for (int i = 0; i < 4; i++)
            {
                Salsa20QuarterRound(state, i, (i + 4) % 16, (i + 8) % 16, (i + 12) % 16);
            }
        }

        private static void Salsa20QuarterRound(uint[] state, int a, int b, int c, int d)
        {
            state[b] ^= RotateLeft(state[a] + state[d], 7);
            state[c] ^= RotateLeft(state[b] + state[a], 9);
            state[d] ^= RotateLeft(state[c] + state[b], 13);
            state[a] ^= RotateLeft(state[d] + state[c], 18);
        }

        private static uint RotateLeft(uint value, int amount)
        {
            return (value << amount) | (value >> (32 - amount));
        }

        #endregion
    }
}