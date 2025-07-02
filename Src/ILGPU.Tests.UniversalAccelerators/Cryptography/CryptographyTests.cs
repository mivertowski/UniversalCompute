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

using ILGPU.Algorithms.Cryptography;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System;
using System.Linq;
using System.Text;
using Xunit;

namespace ILGPU.Tests.UniversalAccelerators.Cryptography
{
    /// <summary>
    /// Tests for cryptographic algorithms.
    /// </summary>
    public class CryptographyTests : TestBase
    {
        #region Hash Function Tests

        [Fact]
        public void TestSHA256HashFunction()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test vector: "abc" -> known SHA-256 hash
            var input = Encoding.UTF8.GetBytes("abc");
            var expectedHash = new byte[]
            {
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
            };
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32); // SHA-256 output size
            
            HashFunctions.SHA256(inputBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var computedHash = hashBuffer.GetAsArray1D();
            
            Assert.Equal(expectedHash.Length, computedHash.Length);
            for (int i = 0; i < expectedHash.Length; i++)
            {
                Assert.Equal(expectedHash[i], computedHash[i]);
            }
        }

        [Fact]
        public void TestSHA256EmptyInput()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            // Test vector: empty string -> known SHA-256 hash
            var input = new byte[0];
            var expectedHash = new byte[]
            {
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
            };
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            
            HashFunctions.SHA256(inputBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var computedHash = hashBuffer.GetAsArray1D();
            
            for (int i = 0; i < expectedHash.Length; i++)
            {
                Assert.Equal(expectedHash[i], computedHash[i]);
            }
        }

        [Fact]
        public void TestBLAKE2bHashFunction()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var input = Encoding.UTF8.GetBytes("test");
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var hashBuffer = accelerator!.Allocate1D<byte>(64); // BLAKE2b-512 output size
            
            HashFunctions.BLAKE2b(inputBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var computedHash = hashBuffer.GetAsArray1D();
            
            // Verify hash properties
            Assert.Equal(64, computedHash.Length);
            Assert.True(computedHash.Any(b => b != 0), "Hash should not be all zeros");
            
            // Test determinism - same input should produce same hash
            using var hashBuffer2 = accelerator!.Allocate1D<byte>(64);
            HashFunctions.BLAKE2b(inputBuffer.View, hashBuffer2.View, accelerator!.DefaultStream);
            
            var computedHash2 = hashBuffer2.GetAsArray1D();
            Assert.Equal(computedHash, computedHash2);
        }

        [Fact]
        public void TestKeccak256HashFunction()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var input = Encoding.UTF8.GetBytes("hello world");
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32); // Keccak-256 output size
            
            HashFunctions.Keccak256(inputBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
            
            var computedHash = hashBuffer.GetAsArray1D();
            
            // Verify hash properties
            Assert.Equal(32, computedHash.Length);
            Assert.True(computedHash.Any(b => b != 0), "Hash should not be all zeros");
            
            // Test avalanche effect - small change in input should drastically change output
            var input2 = Encoding.UTF8.GetBytes("hello worlD"); // Changed last character
            using var inputBuffer2 = accelerator!.Allocate1D(input2);
            using var hashBuffer2 = accelerator!.Allocate1D<byte>(32);
            
            HashFunctions.Keccak256(inputBuffer2.View, hashBuffer2.View, accelerator!.DefaultStream);
            
            var computedHash2 = hashBuffer2.GetAsArray1D();
            
            // Hashes should be significantly different
            int differences = 0;
            for (int i = 0; i < 32; i++)
            {
                if (computedHash[i] != computedHash2[i])
                    differences++;
            }
            
            Assert.True(differences > 10, $"Avalanche effect insufficient: only {differences} bytes different");
        }

        [Fact]
        public void TestBatchHashingPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int batchSize = 1000;
            const int inputSize = 64;
            
            // Create batch of random inputs
            var batchInput = new byte[batchSize * inputSize];
            var random = new Random(42);
            random.NextBytes(batchInput);
            
            using var inputBuffer = accelerator!.Allocate1D(batchInput);
            using var hashBuffer = accelerator!.Allocate1D<byte>(batchSize * 32); // SHA-256 outputs
            
            // Measure batch hashing performance
            var hashTime = MeasureTime(() =>
            {
                HashFunctions.BatchSHA256(inputBuffer.View, hashBuffer.View, batchSize, inputSize, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            Assert.True(hashTime < 2000, $"Batch hashing took {hashTime}ms, expected < 2000ms");
            
            var hashes = hashBuffer.GetAsArray1D();
            
            // Verify all hashes are non-zero
            for (int i = 0; i < batchSize; i++)
            {
                var hashStart = i * 32;
                var hasNonZero = false;
                for (int j = 0; j < 32; j++)
                {
                    if (hashes[hashStart + j] != 0)
                    {
                        hasNonZero = true;
                        break;
                    }
                }
                Assert.True(hasNonZero, $"Hash {i} should not be all zeros");
            }
        }

        #endregion

        #region Symmetric Encryption Tests

        [Fact]
        public void TestAESEncryptionDecryption()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = Encoding.UTF8.GetBytes("This is a test message for AES encryption!");
            var key = new byte[32]; // AES-256 key
            var iv = new byte[16];  // AES block size
            
            // Generate random key and IV
            var random = new Random(42);
            random.NextBytes(key);
            random.NextBytes(iv);
            
            // Pad plaintext to block size
            var paddedSize = ((plaintext.Length + 15) / 16) * 16;
            var paddedPlaintext = new byte[paddedSize];
            Array.Copy(plaintext, paddedPlaintext, plaintext.Length);
            
            using var plaintextBuffer = accelerator!.Allocate1D(paddedPlaintext);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var ivBuffer = accelerator!.Allocate1D(iv);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(paddedSize);
            using var decryptedBuffer = accelerator!.Allocate1D<byte>(paddedSize);
            
            // Encrypt
            SymmetricCryptography.AESEncrypt(
                plaintextBuffer.View, ciphertextBuffer.View,
                keyBuffer.View, ivBuffer.View,
                AESMode.CBC, accelerator!.DefaultStream);
            
            // Decrypt
            SymmetricCryptography.AESDecrypt(
                ciphertextBuffer.View, decryptedBuffer.View,
                keyBuffer.View, ivBuffer.View,
                AESMode.CBC, accelerator!.DefaultStream);
            
            var ciphertext = ciphertextBuffer.GetAsArray1D();
            var decrypted = decryptedBuffer.GetAsArray1D();
            
            // Verify encryption changed the data
            Assert.False(paddedPlaintext.SequenceEqual(ciphertext), "Ciphertext should differ from plaintext");
            
            // Verify decryption restored original data
            Assert.True(paddedPlaintext.SequenceEqual(decrypted), "Decryption should restore original plaintext");
        }

        [Fact]
        public void TestAESGCMAuthentication()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = Encoding.UTF8.GetBytes("Authenticated encryption test");
            var key = new byte[32];
            var nonce = new byte[12]; // GCM nonce
            var aad = Encoding.UTF8.GetBytes("Additional authenticated data");
            
            var random = new Random(42);
            random.NextBytes(key);
            random.NextBytes(nonce);
            
            using var plaintextBuffer = accelerator!.Allocate1D(plaintext);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var nonceBuffer = accelerator!.Allocate1D(nonce);
            using var aadBuffer = accelerator!.Allocate1D(aad);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(plaintext.Length);
            using var tagBuffer = accelerator!.Allocate1D<byte>(16); // GCM tag
            
            // Encrypt with authentication
            SymmetricCryptography.AESGCMEncrypt(
                plaintextBuffer.View, ciphertextBuffer.View,
                keyBuffer.View, nonceBuffer.View, aadBuffer.View,
                tagBuffer.View, accelerator!.DefaultStream);
            
            var tag = tagBuffer.GetAsArray1D();
            Assert.True(tag.Any(b => b != 0), "Authentication tag should not be all zeros");
            
            // Verify with correct tag
            using var verificationBuffer = accelerator!.Allocate1D<bool>(1);
            SymmetricCryptography.AESGCMVerify(
                ciphertextBuffer.View, keyBuffer.View, nonceBuffer.View,
                aadBuffer.View, tagBuffer.View, verificationBuffer.View,
                accelerator!.DefaultStream);
            
            var isValid = verificationBuffer.GetAsArray1D();
            Assert.True(isValid[0], "GCM authentication should succeed with correct tag");
            
            // Test with tampered ciphertext
            var tamperedCiphertext = ciphertextBuffer.GetAsArray1D();
            tamperedCiphertext[0] ^= 1; // Flip one bit
            using var tamperedBuffer = accelerator!.Allocate1D(tamperedCiphertext);
            
            SymmetricCryptography.AESGCMVerify(
                tamperedBuffer.View, keyBuffer.View, nonceBuffer.View,
                aadBuffer.View, tagBuffer.View, verificationBuffer.View,
                accelerator!.DefaultStream);
            
            isValid = verificationBuffer.GetAsArray1D();
            Assert.False(isValid[0], "GCM authentication should fail with tampered ciphertext");
        }

        [Fact]
        public void TestChaCha20Encryption()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = Encoding.UTF8.GetBytes("ChaCha20 stream cipher test message");
            var key = new byte[32]; // ChaCha20 key
            var nonce = new byte[12]; // ChaCha20 nonce
            
            var random = new Random(42);
            random.NextBytes(key);
            random.NextBytes(nonce);
            
            using var plaintextBuffer = accelerator!.Allocate1D(plaintext);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var nonceBuffer = accelerator!.Allocate1D(nonce);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(plaintext.Length);
            using var decryptedBuffer = accelerator!.Allocate1D<byte>(plaintext.Length);
            
            // Encrypt
            SymmetricCryptography.ChaCha20Encrypt(
                plaintextBuffer.View, ciphertextBuffer.View,
                keyBuffer.View, nonceBuffer.View, 0, // counter = 0
                accelerator!.DefaultStream);
            
            // Decrypt (ChaCha20 is symmetric)
            SymmetricCryptography.ChaCha20Encrypt(
                ciphertextBuffer.View, decryptedBuffer.View,
                keyBuffer.View, nonceBuffer.View, 0,
                accelerator!.DefaultStream);
            
            var ciphertext = ciphertextBuffer.GetAsArray1D();
            var decrypted = decryptedBuffer.GetAsArray1D();
            
            // Verify encryption/decryption
            Assert.False(plaintext.SequenceEqual(ciphertext), "Ciphertext should differ from plaintext");
            Assert.True(plaintext.SequenceEqual(decrypted), "Decryption should restore original plaintext");
        }

        [Fact]
        public void TestSalsa20StreamCipher()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = CreateTestData(1000).Select(f => (byte)(f * 255)).ToArray();
            var key = new byte[32];
            var nonce = new byte[8]; // Salsa20 nonce
            
            var random = new Random(42);
            random.NextBytes(key);
            random.NextBytes(nonce);
            
            using var plaintextBuffer = accelerator!.Allocate1D(plaintext);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var nonceBuffer = accelerator!.Allocate1D(nonce);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(plaintext.Length);
            
            // Generate keystream and encrypt
            SymmetricCryptography.Salsa20Encrypt(
                plaintextBuffer.View, ciphertextBuffer.View,
                keyBuffer.View, nonceBuffer.View, 0,
                accelerator!.DefaultStream);
            
            var ciphertext = ciphertextBuffer.GetAsArray1D();
            
            // Verify encryption changed the data
            Assert.False(plaintext.SequenceEqual(ciphertext), "Ciphertext should differ from plaintext");
            
            // Test keystream properties - different positions should produce different values
            var keystreamSample1 = new byte[64];
            var keystreamSample2 = new byte[64];
            
            using var sample1Buffer = accelerator!.Allocate1D<byte>(64);
            using var sample2Buffer = accelerator!.Allocate1D<byte>(64);
            
            SymmetricCryptography.Salsa20Keystream(
                keyBuffer.View, nonceBuffer.View, 0, sample1Buffer.View, accelerator!.DefaultStream);
            SymmetricCryptography.Salsa20Keystream(
                keyBuffer.View, nonceBuffer.View, 1, sample2Buffer.View, accelerator!.DefaultStream);
            
            keystreamSample1 = sample1Buffer.GetAsArray1D();
            keystreamSample2 = sample2Buffer.GetAsArray1D();
            
            Assert.False(keystreamSample1.SequenceEqual(keystreamSample2), 
                "Different counter values should produce different keystreams");
        }

        #endregion

        #region Key Derivation Tests

        [Fact]
        public void TestPBKDF2KeyDerivation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var password = Encoding.UTF8.GetBytes("password123");
            var salt = Encoding.UTF8.GetBytes("salt1234");
            const int iterations = 1000;
            const int keyLength = 32;
            
            using var passwordBuffer = accelerator!.Allocate1D(password);
            using var saltBuffer = accelerator!.Allocate1D(salt);
            using var keyBuffer = accelerator!.Allocate1D<byte>(keyLength);
            
            SymmetricCryptography.PBKDF2(
                passwordBuffer.View, saltBuffer.View,
                iterations, keyBuffer.View, accelerator!.DefaultStream);
            
            var derivedKey = keyBuffer.GetAsArray1D();
            
            // Verify key properties
            Assert.Equal(keyLength, derivedKey.Length);
            Assert.True(derivedKey.Any(b => b != 0), "Derived key should not be all zeros");
            
            // Test determinism - same inputs should produce same key
            using var keyBuffer2 = accelerator!.Allocate1D<byte>(keyLength);
            SymmetricCryptography.PBKDF2(
                passwordBuffer.View, saltBuffer.View,
                iterations, keyBuffer2.View, accelerator!.DefaultStream);
            
            var derivedKey2 = keyBuffer2.GetAsArray1D();
            Assert.True(derivedKey.SequenceEqual(derivedKey2), "PBKDF2 should be deterministic");
            
            // Test salt dependency - different salt should produce different key
            var salt2 = Encoding.UTF8.GetBytes("salt5678");
            using var saltBuffer2 = accelerator!.Allocate1D(salt2);
            using var keyBuffer3 = accelerator!.Allocate1D<byte>(keyLength);
            
            SymmetricCryptography.PBKDF2(
                passwordBuffer.View, saltBuffer2.View,
                iterations, keyBuffer3.View, accelerator!.DefaultStream);
            
            var derivedKey3 = keyBuffer3.GetAsArray1D();
            Assert.False(derivedKey.SequenceEqual(derivedKey3), "Different salts should produce different keys");
        }

        [Fact]
        public void TestScryptKeyDerivation()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var password = Encoding.UTF8.GetBytes("testpassword");
            var salt = Encoding.UTF8.GetBytes("testsalt");
            const int N = 16; // Small N for testing
            const int r = 1;
            const int p = 1;
            const int keyLength = 32;
            
            using var passwordBuffer = accelerator!.Allocate1D(password);
            using var saltBuffer = accelerator!.Allocate1D(salt);
            using var keyBuffer = accelerator!.Allocate1D<byte>(keyLength);
            
            SymmetricCryptography.Scrypt(
                passwordBuffer.View, saltBuffer.View,
                N, r, p, keyBuffer.View, accelerator!.DefaultStream);
            
            var derivedKey = keyBuffer.GetAsArray1D();
            
            // Verify key properties
            Assert.Equal(keyLength, derivedKey.Length);
            Assert.True(derivedKey.Any(b => b != 0), "Scrypt derived key should not be all zeros");
            
            // Test parameter dependency - different N should produce different key
            using var keyBuffer2 = accelerator!.Allocate1D<byte>(keyLength);
            SymmetricCryptography.Scrypt(
                passwordBuffer.View, saltBuffer.View,
                N * 2, r, p, keyBuffer2.View, accelerator!.DefaultStream);
            
            var derivedKey2 = keyBuffer2.GetAsArray1D();
            Assert.False(derivedKey.SequenceEqual(derivedKey2), "Different Scrypt parameters should produce different keys");
        }

        #endregion

        #region Random Number Generation Tests

        [Fact]
        public void TestCryptographicRandomGeneration()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 1000;
            using var randomBuffer = accelerator!.Allocate1D<byte>(size);
            
            // Generate cryptographically secure random bytes
            SymmetricCryptography.GenerateSecureRandom(randomBuffer.View, accelerator!.DefaultStream);
            
            var randomBytes = randomBuffer.GetAsArray1D();
            
            // Basic randomness tests
            Assert.True(randomBytes.Any(b => b != 0), "Random bytes should not be all zeros");
            Assert.True(randomBytes.Any(b => b != 255), "Random bytes should not be all 255s");
            
            // Test uniform distribution (rough check)
            var histogram = new int[256];
            foreach (var b in randomBytes)
                histogram[b]++;
            
            var expectedFreq = size / 256.0;
            var outliers = histogram.Count(freq => Math.Abs(freq - expectedFreq) > expectedFreq * 0.5);
            
            Assert.True(outliers < 50, $"Too many outliers in random distribution: {outliers}");
        }

        [Fact]
        public void TestChaCha20RandomNumberGenerator()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int size = 512;
            var seed = new byte[32];
            var random = new Random(42);
            random.NextBytes(seed);
            
            using var seedBuffer = accelerator!.Allocate1D(seed);
            using var randomBuffer = accelerator!.Allocate1D<byte>(size);
            
            SymmetricCryptography.ChaCha20Random(seedBuffer.View, randomBuffer.View, accelerator!.DefaultStream);
            
            var randomBytes = randomBuffer.GetAsArray1D();
            
            // Verify randomness properties
            Assert.True(randomBytes.Distinct().Count() > size * 0.9, "Random bytes should have high uniqueness");
            
            // Test reproducibility with same seed
            using var randomBuffer2 = accelerator!.Allocate1D<byte>(size);
            SymmetricCryptography.ChaCha20Random(seedBuffer.View, randomBuffer2.View, accelerator!.DefaultStream);
            
            var randomBytes2 = randomBuffer2.GetAsArray1D();
            Assert.True(randomBytes.SequenceEqual(randomBytes2), "Same seed should produce same random sequence");
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void TestCryptographicPerformance()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int dataSize = 1024 * 1024; // 1MB
            var data = new byte[dataSize];
            var random = new Random(42);
            random.NextBytes(data);
            
            using var dataBuffer = accelerator!.Allocate1D(data);
            using var hashBuffer = accelerator!.Allocate1D<byte>(32);
            
            // Measure SHA-256 throughput
            var hashTime = MeasureTime(() =>
            {
                HashFunctions.SHA256(dataBuffer.View, hashBuffer.View, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            var throughputMBps = (dataSize / (1024.0 * 1024.0)) / (hashTime / 1000.0);
            Assert.True(throughputMBps > 10, $"SHA-256 throughput too low: {throughputMBps:F2} MB/s");
            
            // Measure AES encryption throughput
            var key = new byte[32];
            var iv = new byte[16];
            random.NextBytes(key);
            random.NextBytes(iv);
            
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var ivBuffer = accelerator!.Allocate1D(iv);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(dataSize);
            
            var encryptTime = MeasureTime(() =>
            {
                SymmetricCryptography.AESEncrypt(
                    dataBuffer.View, ciphertextBuffer.View,
                    keyBuffer.View, ivBuffer.View,
                    AESMode.CBC, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            var encryptThroughputMBps = (dataSize / (1024.0 * 1024.0)) / (encryptTime / 1000.0);
            Assert.True(encryptThroughputMBps > 5, $"AES encryption throughput too low: {encryptThroughputMBps:F2} MB/s");
        }

        [Fact]
        public void TestParallelCryptographicOperations()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            const int batchSize = 100;
            const int messageSize = 64;
            
            var batchData = new byte[batchSize * messageSize];
            var random = new Random(42);
            random.NextBytes(batchData);
            
            using var dataBuffer = accelerator!.Allocate1D(batchData);
            using var hashBuffer = accelerator!.Allocate1D<byte>(batchSize * 32);
            
            // Measure parallel hashing performance
            var parallelTime = MeasureTime(() =>
            {
                HashFunctions.BatchSHA256(dataBuffer.View, hashBuffer.View, batchSize, messageSize, accelerator!.DefaultStream);
                accelerator!.Synchronize();
            });
            
            Assert.True(parallelTime < 1000, $"Parallel hashing took {parallelTime}ms, expected < 1000ms");
            
            // Verify all hashes are computed
            var hashes = hashBuffer.GetAsArray1D();
            for (int i = 0; i < batchSize; i++)
            {
                var hashStart = i * 32;
                var hasNonZero = false;
                for (int j = 0; j < 32; j++)
                {
                    if (hashes[hashStart + j] != 0)
                    {
                        hasNonZero = true;
                        break;
                    }
                }
                Assert.True(hasNonZero, $"Batch hash {i} should not be all zeros");
            }
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void TestInvalidKeySize()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = new byte[16];
            var invalidKey = new byte[15]; // Invalid AES key size
            var iv = new byte[16];
            
            using var plaintextBuffer = accelerator!.Allocate1D(plaintext);
            using var keyBuffer = accelerator!.Allocate1D(invalidKey);
            using var ivBuffer = accelerator!.Allocate1D(iv);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(16);
            
            Assert.Throws<ArgumentException>(() =>
            {
                SymmetricCryptography.AESEncrypt(
                    plaintextBuffer.View, ciphertextBuffer.View,
                    keyBuffer.View, ivBuffer.View,
                    AESMode.CBC, accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestInvalidIVSize()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var plaintext = new byte[16];
            var key = new byte[32];
            var invalidIV = new byte[8]; // Invalid AES IV size
            
            using var plaintextBuffer = accelerator!.Allocate1D(plaintext);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var ivBuffer = accelerator!.Allocate1D(invalidIV);
            using var ciphertextBuffer = accelerator!.Allocate1D<byte>(16);
            
            Assert.Throws<ArgumentException>(() =>
            {
                SymmetricCryptography.AESEncrypt(
                    plaintextBuffer.View, ciphertextBuffer.View,
                    keyBuffer.View, ivBuffer.View,
                    AESMode.CBC, accelerator!.DefaultStream);
            });
        }

        [Fact]
        public void TestMismatchedBufferSizes()
        {
            using var accelerator = CreateAcceleratorIfAvailable<CPUAccelerator>();
            SkipIfNotAvailable(accelerator);

            var input = new byte[32];
            var key = new byte[32];
            var iv = new byte[16];
            var outputTooSmall = new byte[16]; // Smaller than input
            
            using var inputBuffer = accelerator!.Allocate1D(input);
            using var keyBuffer = accelerator!.Allocate1D(key);
            using var ivBuffer = accelerator!.Allocate1D(iv);
            using var outputBuffer = accelerator!.Allocate1D(outputTooSmall);
            
            Assert.Throws<ArgumentException>(() =>
            {
                SymmetricCryptography.AESEncrypt(
                    inputBuffer.View, outputBuffer.View,
                    keyBuffer.View, ivBuffer.View,
                    AESMode.CBC, accelerator!.DefaultStream);
            });
        }

        #endregion
    }
}