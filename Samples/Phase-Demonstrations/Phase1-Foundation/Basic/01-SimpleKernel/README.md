# Simple Kernel - Your First GPU Programming Experience

## 🎯 **Overview**

This sample introduces the fundamental concepts of GPU programming with ILGPU. You'll learn how to write your first GPU kernel, manage memory, and execute parallel computations across different accelerator backends.

## 🧠 **What You'll Learn**

### **Core Concepts**
- ✅ **ILGPU Context Creation** - Setting up the GPU programming environment
- ✅ **Accelerator Selection** - Choosing between CPU, CUDA, and OpenCL backends
- ✅ **Kernel Writing** - Creating functions that run on the GPU
- ✅ **Memory Management** - Allocating and transferring data between CPU and GPU
- ✅ **Parallel Execution** - Understanding how threads process data in parallel

### **Programming Patterns**
- ✅ **Thread Indexing** - How each GPU thread knows which data to process
- ✅ **Bounds Checking** - Defensive programming for GPU kernels
- ✅ **Resource Cleanup** - Proper disposal of GPU resources
- ✅ **Error Handling** - Graceful fallback and error management

## 💻 **Sample Features**

### **1. Basic Kernel (`AddConstantKernel`)**
```csharp
static void AddConstantKernel(Index1D index, ArrayView<int> dataView, int constant)
{
    dataView[index] = dataView[index] + constant;
}
```
- Adds a constant value to each array element
- Demonstrates basic parallel processing
- Shows thread indexing fundamentals

### **2. Advanced Kernel (`SafeMultiplyKernel`)**
```csharp
static void SafeMultiplyKernel(Index1D index, ArrayView<float> input, ArrayView<float> output, float multiplier)
{
    if (index < input.Length && index < output.Length)
    {
        float value = input[index];
        output[index] = value > 0.0f ? value * multiplier : 0.0f;
    }
}
```
- Demonstrates bounds checking
- Shows conditional GPU computation
- Uses separate input/output buffers

### **3. Memory Management Demo**
- Multiple buffer allocation
- Memory usage monitoring
- Proper resource disposal

## 🚀 **Running the Sample**

### **Prerequisites**
```bash
# Ensure you have .NET 9.0 installed
dotnet --version

# Build the sample
dotnet build
```

### **Execution**
```bash
dotnet run
```

### **Expected Output**
```
🚀 ILGPU Phase 1: Simple Kernel Demonstration
===============================================
✅ ILGPU context created successfully
🎯 CUDA GPU detected - using GPU acceleration
✅ Using accelerator: NVIDIA RTX 2000 Ada Generation Laptop GPU (Cuda)
   Memory Size: 8192 MB
   Warp Size: 32

📋 Basic Kernel Demonstration
-----------------------------
✅ Kernel compiled successfully
✅ Allocated 4096 bytes on GPU
✅ Data copied to GPU
✅ Kernel executed: added 42 to each element
✅ Results copied back to CPU
✅ Verification passed! First 10 results: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

🔬 Advanced Kernel Demonstration
--------------------------------
✅ Processed 2048 elements
   Positive results: 1024
   Zero results (from negative inputs): 1024
   Sample results: [0.00, 2.68, 0.00, 1.83, 0.00]

💾 Memory Management Demonstration
----------------------------------
Memory before allocation: 8192 MB available
   Allocated buffer 1: 1 MB - 8191 MB remaining
   Allocated buffer 2: 2 MB - 8189 MB remaining
   Allocated buffer 3: 3 MB - 8186 MB remaining
   Allocated buffer 4: 4 MB - 8182 MB remaining
   Allocated buffer 5: 5 MB - 8177 MB remaining
✅ Successfully allocated 5 buffers
✅ All buffers properly disposed
Memory after cleanup: 8192 MB available

🎉 Phase 1 Simple Kernel demonstration completed successfully!
```

## 🔍 **Key Technical Details**

### **Thread Execution Model**
- Each thread processes exactly one array element
- Threads execute in parallel across GPU cores
- Thread index determines which data element to process

### **Memory Transfer Pattern**
1. **Allocate** GPU memory buffer
2. **Copy** data from CPU to GPU (Host → Device)
3. **Execute** kernel on GPU
4. **Copy** results from GPU to CPU (Device → Host)
5. **Dispose** GPU resources

### **Accelerator Selection Logic**
1. **Try CUDA** - NVIDIA GPU acceleration (best performance)
2. **Try OpenCL** - Cross-platform GPU acceleration
3. **Fallback to CPU** - Always available, good for debugging

## ⚡ **Performance Notes**

### **Memory Coalescing**
- Sequential memory access patterns are optimal
- Each thread accesses consecutive memory locations
- Maximizes memory bandwidth utilization

### **Thread Divergence**
- Conditional branches can reduce performance
- The `SafeMultiplyKernel` shows minimal divergence impact
- All threads in a warp should follow similar execution paths

### **Occupancy**
- Auto-grouped kernels optimize thread block size
- Maximizes GPU utilization automatically
- Good starting point for performance optimization

## 🛠️ **Customization Ideas**

### **Try These Modifications**
1. **Change array sizes** - Experiment with different data sizes
2. **Add more operations** - Implement complex mathematical functions
3. **Multiple kernels** - Chain multiple kernel executions
4. **Different data types** - Use `double`, `Vector2`, custom structs

### **Performance Experiments**
1. **Manual thread grouping** - Try `LoadStreamKernel` with custom group sizes
2. **Memory access patterns** - Compare sequential vs random access
3. **Computation intensity** - Add more mathematical operations per thread

## 🔗 **Next Steps**

After mastering this sample:
1. **02-MemoryManagement** - Advanced memory patterns and optimization
2. **03-MultiBackend** - Cross-platform compatibility strategies
3. **04-IndexingPatterns** - Complex indexing and thread coordination

## 📚 **Related Concepts**

- **CUDA Programming** - Similar concepts apply to native CUDA
- **OpenCL** - Cross-platform parallel computing standard
- **Parallel Programming** - General parallel computing principles
- **Memory Hierarchies** - GPU memory architecture understanding

This sample provides the foundation for all GPU programming with ILGPU. Master these concepts before proceeding to more advanced phases!