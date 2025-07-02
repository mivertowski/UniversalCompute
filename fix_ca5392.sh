#\!/bin/bash

echo "Fixing CA5392 errors - Adding DefaultDllImportSearchPaths to P/Invoke methods..."

# List of files to fix
files=(
    "Src/ILGPU/Runtime/OneAPI/Native/SYCLNative.cs"
    "Src/ILGPU/Backends/OneAPI/Native/OneAPINative.cs"
    "Src/ILGPU/Backends/Metal/Native/MetalNative.cs"
    "Src/ILGPU/Apple/NeuralEngine/Native/ANENative.cs"
    "Src/ILGPU/Intel/NPU/Native/NPUNative.cs"
    "Src/ILGPU/Intel/AMX/Native/AMXNative.cs"
    "Src/ILGPU/Runtime/AMX/Native/AMXNative.cs"
    "Src/ILGPU/Intel/IPP/Native/IPPNative.cs"
    "Src/ILGPU/Runtime/ROCm/Native/ROCmNative.cs"
    "Src/ILGPU/Runtime/Vulkan/Native/VulkanNative.cs"
)

fixed_count=0

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        
        # Check if file doesn't already have DefaultDllImportSearchPaths
        if \! grep -q "DefaultDllImportSearchPaths" "$file"; then
            # Add the attribute before each DllImport that doesn't already have it
            sed -i 's/^\(\s*\)\[DllImport\(/\1[DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]\n\1[DllImport(/g' "$file"
            echo "  ✓ Fixed $file"
            ((fixed_count++))
        else
            echo "  ⚠ $file already has DefaultDllImportSearchPaths"
        fi
    else
        echo "  ❌ File not found: $file"
    fi
done

echo "Fixed $fixed_count files for CA5392 errors"
