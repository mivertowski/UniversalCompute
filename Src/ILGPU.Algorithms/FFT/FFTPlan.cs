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
using System.Numerics;

namespace ILGPU.Algorithms.FFT
{
    /// <summary>
    /// FFT plan containing pre-computed twiddle factors and configuration.
    /// </summary>
    public sealed class FFTPlan : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly FFTConfiguration _config;
        private MemoryBuffer1D<Complex, Stride1D.Dense>? _twiddleFactors;
        private MemoryBuffer1D<int, Stride1D.Dense>? _bitReversalTable;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the FFTPlan class.
        /// </summary>
        private FFTPlan(Accelerator accelerator, FFTConfiguration config)
        {
            _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            // Pre-compute values
            LogN = (int)Math.Log2(_config.Size);
            IsPowerOfTwo = (_config.Size & (_config.Size - 1)) == 0;
            
            // Initialize buffers
            InitializeTwiddleFactors();
            InitializeBitReversalTable();
        }

        /// <summary>
        /// Gets the FFT configuration.
        /// </summary>
        public FFTConfiguration Configuration => _config;

        /// <summary>
        /// Gets the log base 2 of the FFT size.
        /// </summary>
        public int LogN { get; }

        /// <summary>
        /// Gets whether the FFT size is a power of two.
        /// </summary>
        public bool IsPowerOfTwo { get; }

        /// <summary>
        /// Gets the twiddle factors buffer.
        /// </summary>
        public ArrayView<Complex> TwiddleFactors => _twiddleFactors?.View ?? throw new ObjectDisposedException(nameof(FFTPlan));

        /// <summary>
        /// Gets the bit reversal table.
        /// </summary>
        public ArrayView<int> BitReversalTable => _bitReversalTable?.View ?? throw new ObjectDisposedException(nameof(FFTPlan));

        /// <summary>
        /// Creates an FFT plan for the specified configuration.
        /// </summary>
        /// <param name="accelerator">The accelerator to use.</param>
        /// <param name="config">The FFT configuration.</param>
        /// <returns>FFT plan.</returns>
        public static FFTPlan Create(Accelerator accelerator, FFTConfiguration config)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (config == null)
                throw new ArgumentNullException(nameof(config));
            
            // Validate configuration
            if (config.Size <= 0)
                throw new ArgumentException("FFT size must be positive", nameof(config));
            
            // Select algorithm if auto
            if (config.Algorithm == FFTAlgorithm.Auto)
            {
                config.Algorithm = SelectOptimalAlgorithm(config.Size);
            }
            
            // Validate algorithm compatibility
            ValidateAlgorithmCompatibility(config);
            
            return new FFTPlan(accelerator, config);
        }

        /// <summary>
        /// Selects the optimal FFT algorithm for the given size.
        /// </summary>
        private static FFTAlgorithm SelectOptimalAlgorithm(int size)
        {
            // Check if size is power of 2
            if ((size & (size - 1)) == 0)
            {
                // Power of 2: prefer radix-2 or radix-4
                if ((size & 3) == 0) // Divisible by 4
                    return FFTAlgorithm.Radix4;
                else
                    return FFTAlgorithm.Radix2;
            }
            else
            {
                // Not power of 2: use mixed-radix or Bluestein
                if (IsPrime(size))
                    return FFTAlgorithm.Bluestein;
                else
                    return FFTAlgorithm.MixedRadix;
            }
        }

        /// <summary>
        /// Validates that the selected algorithm is compatible with the FFT size.
        /// </summary>
        private static void ValidateAlgorithmCompatibility(FFTConfiguration config)
        {
            switch (config.Algorithm)
            {
                case FFTAlgorithm.Radix2:
                    if ((config.Size & (config.Size - 1)) != 0)
                        throw new ArgumentException($"Radix-2 algorithm requires power-of-2 size, but got {config.Size}");
                    break;
                    
                case FFTAlgorithm.Radix4:
                    if ((config.Size & (config.Size - 1)) != 0 || (config.Size & 3) != 0)
                        throw new ArgumentException($"Radix-4 algorithm requires power-of-4 size, but got {config.Size}");
                    break;
                    
                case FFTAlgorithm.Bluestein:
                    // Bluestein works for any size
                    break;
                    
                case FFTAlgorithm.MixedRadix:
                    // Mixed-radix works for composite sizes
                    if (IsPrime(config.Size))
                        throw new ArgumentException($"Mixed-radix algorithm doesn't work efficiently for prime size {config.Size}");
                    break;
            }
        }

        /// <summary>
        /// Checks if a number is prime.
        /// </summary>
        private static bool IsPrime(int n)
        {
            if (n <= 1) return false;
            if (n <= 3) return true;
            if (n % 2 == 0 || n % 3 == 0) return false;
            
            for (int i = 5; i * i <= n; i += 6)
            {
                if (n % i == 0 || n % (i + 2) == 0)
                    return false;
            }
            
            return true;
        }

        /// <summary>
        /// Initializes the twiddle factors for FFT computation.
        /// </summary>
        private void InitializeTwiddleFactors()
        {
            var size = _config.Size;
            var twiddleCount = CalculateTwiddleFactorCount();
            
            // Allocate buffer for twiddle factors
            _twiddleFactors = _accelerator.Allocate1D<Complex>(twiddleCount);
            
            // Generate twiddle factors on CPU
            var twiddleArray = new Complex[twiddleCount];
            
            switch (_config.Algorithm)
            {
                case FFTAlgorithm.Radix2:
                    GenerateRadix2TwiddleFactors(twiddleArray);
                    break;
                    
                case FFTAlgorithm.Radix4:
                    GenerateRadix4TwiddleFactors(twiddleArray);
                    break;
                    
                case FFTAlgorithm.MixedRadix:
                    GenerateMixedRadixTwiddleFactors(twiddleArray);
                    break;
                    
                case FFTAlgorithm.Bluestein:
                    GenerateBluesteinTwiddleFactors(twiddleArray);
                    break;
            }
            
            // Copy to GPU
            _twiddleFactors.CopyFromCPU(twiddleArray);
        }

        /// <summary>
        /// Calculates the number of twiddle factors needed.
        /// </summary>
        private int CalculateTwiddleFactorCount()
        {
            return _config.Algorithm switch
            {
                FFTAlgorithm.Radix2 => _config.Size / 2,
                FFTAlgorithm.Radix4 => 3 * _config.Size / 4,
                FFTAlgorithm.MixedRadix => _config.Size, // Conservative estimate
                FFTAlgorithm.Bluestein => 2 * _config.Size, // For convolution
                _ => _config.Size
            };
        }

        /// <summary>
        /// Generates twiddle factors for radix-2 FFT.
        /// </summary>
        private void GenerateRadix2TwiddleFactors(Complex[] twiddle)
        {
            var n = _config.Size;
            var halfN = n / 2;
            
            for (int k = 0; k < halfN; k++)
            {
                double angle = -2.0 * Math.PI * k / n;
                twiddle[k] = Complex.FromPolarCoordinates(1.0, angle);
            }
        }

        /// <summary>
        /// Generates twiddle factors for radix-4 FFT.
        /// </summary>
        private void GenerateRadix4TwiddleFactors(Complex[] twiddle)
        {
            var n = _config.Size;
            var quarterN = n / 4;
            
            // Generate twiddle factors for each of the three non-trivial outputs
            for (int stage = 0; stage < 3; stage++)
            {
                for (int k = 0; k < quarterN; k++)
                {
                    double angle = -2.0 * Math.PI * k * (stage + 1) / n;
                    twiddle[stage * quarterN + k] = Complex.FromPolarCoordinates(1.0, angle);
                }
            }
        }

        /// <summary>
        /// Generates twiddle factors for mixed-radix FFT.
        /// </summary>
        private void GenerateMixedRadixTwiddleFactors(Complex[] twiddle)
        {
            // For mixed-radix, we need factors for different stages
            // This is a simplified version - real implementation would factor N
            var n = _config.Size;
            
            for (int k = 0; k < n; k++)
            {
                double angle = -2.0 * Math.PI * k / n;
                twiddle[k] = Complex.FromPolarCoordinates(1.0, angle);
            }
        }

        /// <summary>
        /// Generates twiddle factors for Bluestein's algorithm.
        /// </summary>
        private void GenerateBluesteinTwiddleFactors(Complex[] twiddle)
        {
            var n = _config.Size;
            
            // Bluestein's algorithm uses chirp Z-transform
            // Generate chirp sequence
            for (int k = 0; k < n; k++)
            {
                double angle = -Math.PI * k * k / n;
                twiddle[k] = Complex.FromPolarCoordinates(1.0, angle);
            }
            
            // Generate inverse chirp for convolution
            for (int k = 0; k < n; k++)
            {
                double angle = Math.PI * k * k / n;
                twiddle[n + k] = Complex.FromPolarCoordinates(1.0, angle);
            }
        }

        /// <summary>
        /// Initializes the bit reversal table for FFT computation.
        /// </summary>
        private void InitializeBitReversalTable()
        {
            if (!IsPowerOfTwo)
            {
                // Bit reversal only applies to power-of-2 sizes
                return;
            }
            
            var size = _config.Size;
            _bitReversalTable = _accelerator.Allocate1D<int>(size);
            
            // Generate bit reversal table on CPU
            var table = new int[size];
            var logN = LogN;
            
            for (int i = 0; i < size; i++)
            {
                int reversed = 0;
                int temp = i;
                
                for (int j = 0; j < logN; j++)
                {
                    reversed = (reversed << 1) | (temp & 1);
                    temp >>= 1;
                }
                
                table[i] = reversed;
            }
            
            // Copy to GPU
            _bitReversalTable.CopyFromCPU(table);
        }

        /// <summary>
        /// Gets the prime factors of the FFT size (for mixed-radix).
        /// </summary>
        public int[] GetPrimeFactors()
        {
            var factors = new System.Collections.Generic.List<int>();
            var n = _config.Size;
            
            // Factor out powers of 2
            while (n % 2 == 0)
            {
                factors.Add(2);
                n /= 2;
            }
            
            // Factor out odd primes
            for (int i = 3; i * i <= n; i += 2)
            {
                while (n % i == 0)
                {
                    factors.Add(i);
                    n /= i;
                }
            }
            
            // If n is still greater than 1, it's prime
            if (n > 1)
                factors.Add(n);
            
            return factors.ToArray();
        }

        /// <summary>
        /// Gets the optimal radix for the current configuration.
        /// </summary>
        public int GetOptimalRadix()
        {
            return _config.Algorithm switch
            {
                FFTAlgorithm.Radix2 => 2,
                FFTAlgorithm.Radix4 => 4,
                FFTAlgorithm.MixedRadix => GetPrimeFactors()[0], // Use first factor
                _ => 2 // Default
            };
        }

        /// <summary>
        /// Disposes the FFT plan and releases resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _twiddleFactors?.Dispose();
                _bitReversalTable?.Dispose();
                _disposed = true;
            }
        }
    }
}