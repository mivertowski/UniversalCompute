using System;
using System.Reflection;
using ILGPU.Benchmarks.Benchmarks;
using BenchmarkDotNet.Attributes;

namespace ILGPU.Benchmarks
{
    public static class BenchmarkTest
    {
        public static void AnalyzeBenchmarkClasses()
        {
            var types = new[]
            {
                typeof(TensorCoreBenchmarks),
                typeof(SimdVectorBenchmarks),
                typeof(MixedPrecisionBenchmarks),
                typeof(BFloat16Benchmarks),
                typeof(PlatformIntrinsicsBenchmarks),
                typeof(MatrixVectorBenchmarks),
                typeof(CpuGpuComparisonBenchmarks),
                typeof(MemoryBenchmarks),
                typeof(ScalabilityBenchmarks),
                typeof(AIPerformancePrimitivesBenchmarks)
            };

            foreach (var type in types)
            {
                Console.WriteLine($"\n=== {type.Name} ===");
                try
                {
                    var methods = type.GetMethods(BindingFlags.Public | BindingFlags.Instance);
                    var benchmarkMethods = 0;
                    var voidMethods = 0;
                    var returnMethods = 0;

                    foreach (var method in methods)
                    {
                        if (method.GetCustomAttribute<BenchmarkAttribute>() != null)
                        {
                            benchmarkMethods++;
                            if (method.ReturnType == typeof(void))
                            {
                                voidMethods++;
                                Console.WriteLine($"  VOID: {method.Name}");
                            }
                            else
                            {
                                returnMethods++;
                                Console.WriteLine($"  RETURN {method.ReturnType.Name}: {method.Name}");
                            }
                        }
                    }
                    
                    Console.WriteLine($"  Total: {benchmarkMethods}, Void: {voidMethods}, Return: {returnMethods}");
                    
                    // Try to instantiate
                    var instance = Activator.CreateInstance(type);
                    Console.WriteLine($"  ✅ Can instantiate");
                    
                    // Check for GlobalSetup
                    var setupMethod = type.GetMethods().FirstOrDefault(m => m.GetCustomAttribute<GlobalSetupAttribute>() != null);
                    if (setupMethod != null)
                    {
                        Console.WriteLine($"  ✅ Has GlobalSetup: {setupMethod.Name}");
                        try
                        {
                            setupMethod.Invoke(instance, null);
                            Console.WriteLine($"  ✅ Setup succeeded");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"  ❌ Setup failed: {ex.InnerException?.Message ?? ex.Message}");
                        }
                    }
                    
                    if (instance is IDisposable disposable)
                    {
                        disposable.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ❌ Error: {ex.Message}");
                }
            }
        }
    }
}