```

BenchmarkDotNet v0.14.0, Ubuntu 22.04.5 LTS (Jammy Jellyfish) WSL
Intel Core Ultra 7 165H, 1 CPU, 22 logical and 11 physical cores
.NET SDK 9.0.203
  [Host] : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2

IterationCount=10  WarmupCount=5  

```
| Method                    | MatrixSize | BatchSize | Channels | Mean | Error |
|-------------------------- |----------- |---------- |--------- |-----:|------:|
| GEMM_MatrixMultiplication | 128        | 64        | 32       |   NA |    NA |

Benchmarks with issues:
  AIPerformancePrimitivesBenchmarks.GEMM_MatrixMultiplication: Job-EQXTBS(IterationCount=10, WarmupCount=5) [MatrixSize=128, BatchSize=64, Channels=32]
