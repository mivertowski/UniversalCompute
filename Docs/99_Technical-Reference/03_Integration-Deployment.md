// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
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

# Enterprise Integration and Deployment Guide

## Container Deployment Patterns

### 1. Docker Containerization

```dockerfile
# Multi-stage Dockerfile for ILGPU applications
# Stage 1: Build environment
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Install CUDA toolkit for GPU support (optional)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common

# CUDA repository setup
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-0

# Copy project files
COPY ["MyApp/MyApp.csproj", "MyApp/"]
COPY ["MyApp.ILGPU/MyApp.ILGPU.csproj", "MyApp.ILGPU/"]
RUN dotnet restore "MyApp/MyApp.csproj"

COPY . .
WORKDIR "/src/MyApp"
RUN dotnet build "MyApp.csproj" -c Release -o /app/build

# Stage 2: Publish
FROM build AS publish
RUN dotnet publish "MyApp.csproj" -c Release -o /app/publish

# Stage 3: Runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libnvidia-compute-470 \
    libnvidia-gl-470 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN addgroup --system ilgpu && adduser --system --group ilgpu
USER ilgpu

WORKDIR /app
COPY --from=publish /app/publish .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV ASPNETCORE_URLS=http://+:8080
ENV ILGPU_ENABLE_ASSERTIONS=false
ENV ILGPU_CACHE_KERNELS=true

EXPOSE 8080
ENTRYPOINT ["dotnet", "MyApp.dll"]
```

### 2. Kubernetes Deployment

```yaml
# ILGPU application deployment with GPU support
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ilgpu-compute-service
  namespace: compute
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ilgpu-compute-service
  template:
    metadata:
      labels:
        app: ilgpu-compute-service
    spec:
      containers:
      - name: compute-service
        image: myregistry/ilgpu-compute:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: ASPNETCORE_ENVIRONMENT
          value: "Production"
        - name: ILGPU_LOG_LEVEL
          value: "Information"
        - name: CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: connection-string
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: cache-volume
        emptyDir: {}
      - name: config-volume
        configMap:
          name: ilgpu-config
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: ilgpu-compute-service
  namespace: compute
spec:
  selector:
    app: ilgpu-compute-service
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ilgpu-config
  namespace: compute
data:
  appsettings.json: |
    {
      "ILGPU": {
        "Context": {
          "EnableDebugMode": false,
          "EnableCaching": true,
          "OptimizationLevel": "O2"
        },
        "Memory": {
          "DefaultPoolSize": 268435456,
          "EnableMemoryPooling": true,
          "MemoryPressureThreshold": 0.8
        },
        "Performance": {
          "MaxConcurrentOperations": 10,
          "HealthCheckInterval": "00:00:30"
        }
      }
    }
```

### 3. Helm Chart for ILGPU Services

```yaml
# values.yaml
replicaCount: 3

image:
  repository: myregistry/ilgpu-compute
  tag: "latest"
  pullPolicy: IfNotPresent

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1

gpu:
  enabled: true
  type: "nvidia"
  count: 1

ilgpu:
  config:
    enableDebugMode: false
    enableCaching: true
    optimizationLevel: "O2"
    memoryPoolSize: 268435456
    maxConcurrentOperations: 10

monitoring:
  enabled: true
  prometheus:
    enabled: true
    port: 9090

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector:
  accelerator: nvidia-gpu

tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

## Cloud Platform Integration

### 1. Azure Container Instances with GPU

```yaml
# Azure Container Instance ARM template
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "ilgpu-compute-group"
    },
    "containerName": {
      "type": "string",
      "defaultValue": "ilgpu-compute"
    },
    "image": {
      "type": "string",
      "defaultValue": "myregistry/ilgpu-compute:latest"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-09-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "sku": "Standard",
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "[parameters('image')]",
              "resources": {
                "requests": {
                  "memoryInGB": 4,
                  "cpu": 2,
                  "gpu": {
                    "count": 1,
                    "sku": "V100"
                  }
                }
              },
              "ports": [
                {
                  "protocol": "TCP",
                  "port": 8080
                }
              ],
              "environmentVariables": [
                {
                  "name": "ASPNETCORE_ENVIRONMENT",
                  "value": "Production"
                },
                {
                  "name": "ILGPU_ENABLE_ASSERTIONS",
                  "value": "false"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "restartPolicy": "OnFailure",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "protocol": "TCP",
              "port": 8080
            }
          ]
        }
      }
    }
  ]
}
```

### 2. AWS ECS with GPU Support

```yaml
# ECS Task Definition
{
  "family": "ilgpu-compute-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ilgpu-compute",
      "image": "myregistry/ilgpu-compute:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ilgpu-compute",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "ASPNETCORE_ENVIRONMENT",
          "value": "Production"
        },
        {
          "name": "AWS_REGION",
          "value": "us-west-2"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8080/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 3. Google Cloud Run with GPU

```yaml
# Cloud Run service configuration
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ilgpu-compute-service
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/gpu-type: nvidia-tesla-t4
        run.googleapis.com/gpu-count: "1"
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/myproject/ilgpu-compute:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "myproject"
        - name: ILGPU_LOG_LEVEL
          value: "Information"
```

## CI/CD Integration

### 1. GitHub Actions Pipeline

```yaml
# .github/workflows/ilgpu-deploy.yml
name: ILGPU CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ilgpu-compute

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        dotnet-version: ['6.0.x', '7.0.x', '8.0.x']
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup .NET
      uses: actions/setup-dotnet@v3
      with:
        dotnet-version: ${{ matrix.dotnet-version }}
    
    - name: Restore dependencies
      run: dotnet restore
    
    - name: Build
      run: dotnet build --no-restore --configuration Release
    
    - name: Test CPU Backend
      run: dotnet test --no-build --configuration Release --filter "Category=CPU"
    
    - name: Test Cross-Platform Compatibility
      run: dotnet test --no-build --configuration Release --filter "Category=CrossPlatform"
    
    - name: Performance Regression Tests
      run: dotnet test --no-build --configuration Release --filter "Category=Performance"
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.dotnet-version }}
        path: TestResults/

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        ignore-unfixed: true
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        kubectl set image deployment/ilgpu-compute-service \
          compute-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop-${{ github.sha }} \
          --namespace=staging

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - name: Deploy to production
      run: |
        # Deploy to production environment
        kubectl set image deployment/ilgpu-compute-service \
          compute-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} \
          --namespace=production
```

### 2. Azure DevOps Pipeline

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main
    - develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  buildConfiguration: 'Release'
  imageName: 'ilgpu-compute'
  registryEndpoint: 'myacr.azurecr.io'

stages:
- stage: Test
  jobs:
  - job: UnitTests
    displayName: 'Run Unit Tests'
    steps:
    - task: UseDotNet@2
      displayName: 'Use .NET 8 SDK'
      inputs:
        packageType: 'sdk'
        version: '8.0.x'
    
    - task: DotNetCoreCLI@2
      displayName: 'Restore packages'
      inputs:
        command: 'restore'
        projects: '**/*.csproj'
    
    - task: DotNetCoreCLI@2
      displayName: 'Build solution'
      inputs:
        command: 'build'
        projects: '**/*.csproj'
        arguments: '--configuration $(buildConfiguration) --no-restore'
    
    - task: DotNetCoreCLI@2
      displayName: 'Run tests'
      inputs:
        command: 'test'
        projects: '**/*Tests.csproj'
        arguments: '--configuration $(buildConfiguration) --no-build --logger trx --collect "XPlat Code Coverage"'
    
    - task: PublishTestResults@2
      displayName: 'Publish test results'
      inputs:
        testResultsFormat: 'VSTest'
        testResultsFiles: '**/*.trx'
    
    - task: PublishCodeCoverageResults@1
      displayName: 'Publish code coverage'
      inputs:
        codeCoverageTool: 'Cobertura'
        summaryFileLocation: '$(Agent.TempDirectory)/**/coverage.cobertura.xml'

- stage: Build
  dependsOn: Test
  condition: succeeded()
  jobs:
  - job: BuildImage
    displayName: 'Build Docker Image'
    steps:
    - task: Docker@2
      displayName: 'Build and push image'
      inputs:
        containerRegistry: $(registryEndpoint)
        repository: $(imageName)
        command: 'buildAndPush'
        Dockerfile: '**/Dockerfile'
        tags: |
          $(Build.BuildId)
          latest

- stage: Deploy
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployToProduction
    displayName: 'Deploy to Production'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: 'Deploy to Kubernetes'
            inputs:
              action: 'deploy'
              kubernetesServiceConnection: 'production-k8s'
              namespace: 'production'
              manifests: |
                k8s/deployment.yaml
                k8s/service.yaml
              containers: '$(registryEndpoint)/$(imageName):$(Build.BuildId)'
```

## Monitoring and Observability

### 1. Application Metrics

```csharp
// Metrics collection for ILGPU applications
public class ILGPUMetrics
{
    private readonly IMetricsLogger metricsLogger;
    private readonly Counter kernelExecutions;
    private readonly Histogram executionDuration;
    private readonly Gauge memoryUtilization;
    private readonly Counter errorCount;
    
    public ILGPUMetrics(IMetricsLogger metricsLogger)
    {
        this.metricsLogger = metricsLogger;
        
        this.kernelExecutions = metricsLogger.CreateCounter(
            "ilgpu_kernel_executions_total",
            "Total number of kernel executions",
            new[] { "accelerator_type", "kernel_name", "status" });
        
        this.executionDuration = metricsLogger.CreateHistogram(
            "ilgpu_kernel_execution_duration_seconds",
            "Kernel execution duration",
            new[] { "accelerator_type", "kernel_name" });
        
        this.memoryUtilization = metricsLogger.CreateGauge(
            "ilgpu_memory_utilization_ratio",
            "Memory utilization ratio",
            new[] { "accelerator_type" });
        
        this.errorCount = metricsLogger.CreateCounter(
            "ilgpu_errors_total",
            "Total number of ILGPU errors",
            new[] { "error_type", "accelerator_type" });
    }
    
    public void RecordKernelExecution(
        string acceleratorType,
        string kernelName,
        TimeSpan duration,
        bool success)
    {
        kernelExecutions.WithTags(
            ("accelerator_type", acceleratorType),
            ("kernel_name", kernelName),
            ("status", success ? "success" : "failure"))
            .Increment();
        
        if (success)
        {
            executionDuration.WithTags(
                ("accelerator_type", acceleratorType),
                ("kernel_name", kernelName))
                .Record(duration.TotalSeconds);
        }
    }
    
    public void RecordMemoryUtilization(string acceleratorType, double utilization)
    {
        memoryUtilization.WithTags(("accelerator_type", acceleratorType))
            .Set(utilization);
    }
    
    public void RecordError(string errorType, string acceleratorType)
    {
        errorCount.WithTags(
            ("error_type", errorType),
            ("accelerator_type", acceleratorType))
            .Increment();
    }
}

// Integration with telemetry
public class TelemetryConfiguration
{
    public static void ConfigureServices(IServiceCollection services)
    {
        services.AddOpenTelemetry()
            .WithTracing(builder =>
            {
                builder.AddSource("ILGPU")
                       .AddHttpClientInstrumentation()
                       .AddAspNetCoreInstrumentation()
                       .AddJaegerExporter();
            })
            .WithMetrics(builder =>
            {
                builder.AddMeter("ILGPU")
                       .AddPrometheusExporter()
                       .AddConsoleExporter();
            });
    }
}
```

### 2. Health Checks

```csharp
// ILGPU-specific health checks
public class ILGPUHealthCheck : IHealthCheck
{
    private readonly ILGPUContextManager contextManager;
    private readonly ILogger<ILGPUHealthCheck> logger;
    
    public ILGPUHealthCheck(ILGPUContextManager contextManager, ILogger<ILGPUHealthCheck> logger)
    {
        this.contextManager = contextManager;
        this.logger = logger;
    }
    
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var accelerator = contextManager.GetOptimalAccelerator(new WorkloadCharacteristics
            {
                WorkloadType = WorkloadType.Debug
            });
            
            // Test basic kernel compilation and execution
            var testResult = await TestBasicKernelExecution(accelerator);
            
            if (testResult.Success)
            {
                return HealthCheckResult.Healthy("ILGPU is functioning correctly", new Dictionary<string, object>
                {
                    ["accelerator_type"] = accelerator.AcceleratorType.ToString(),
                    ["memory_size"] = accelerator.MemorySize,
                    ["test_duration"] = testResult.Duration.TotalMilliseconds
                });
            }
            else
            {
                return HealthCheckResult.Degraded($"ILGPU test failed: {testResult.Error}");
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "ILGPU health check failed");
            return HealthCheckResult.Unhealthy("ILGPU is not available", ex);
        }
    }
    
    private async Task<TestResult> TestBasicKernelExecution(IAccelerator accelerator)
    {
        const int testSize = 100;
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            using var inputBuffer = accelerator.Allocate1D<float>(testSize);
            using var outputBuffer = accelerator.Allocate1D<float>(testSize);
            
            var testData = Enumerable.Range(0, testSize).Select(i => (float)i).ToArray();
            inputBuffer.CopyFromCPU(testData);
            
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>>(TestKernel);
            
            kernel(testSize, inputBuffer.View, outputBuffer.View);
            accelerator.Synchronize();
            
            var result = outputBuffer.GetAsArray1D();
            var isValid = result.SequenceEqual(testData.Select(x => x * 2));
            
            stopwatch.Stop();
            
            return new TestResult
            {
                Success = isValid,
                Duration = stopwatch.Elapsed,
                Error = isValid ? null : "Output validation failed"
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            return new TestResult
            {
                Success = false,
                Duration = stopwatch.Elapsed,
                Error = ex.Message
            };
        }
    }
    
    static void TestKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index < input.Length)
        {
            output[index] = input[index] * 2.0f;
        }
    }
}

public class TestResult
{
    public bool Success { get; set; }
    public TimeSpan Duration { get; set; }
    public string Error { get; set; }
}

// Registration
public static class HealthCheckExtensions
{
    public static IServiceCollection AddILGPUHealthChecks(this IServiceCollection services)
    {
        services.AddHealthChecks()
            .AddCheck<ILGPUHealthCheck>("ilgpu", tags: new[] { "ilgpu", "compute" })
            .AddCheck("memory", () =>
            {
                var allocated = GC.GetTotalMemory(false);
                var threshold = 1024 * 1024 * 1024; // 1GB
                
                return allocated < threshold
                    ? HealthCheckResult.Healthy($"Memory usage: {allocated / (1024 * 1024)} MB")
                    : HealthCheckResult.Degraded($"High memory usage: {allocated / (1024 * 1024)} MB");
            });
        
        return services;
    }
}
```

### 3. Distributed Tracing

```csharp
// Distributed tracing for ILGPU operations
public class TracedKernelManager : IKernelManager
{
    private readonly IKernelManager inner;
    private readonly ActivitySource activitySource;
    
    public TracedKernelManager(IKernelManager inner)
    {
        this.inner = inner;
        this.activitySource = new ActivitySource("ILGPU");
    }
    
    public async Task<ICompiledKernel<T>> CompileKernelAsync<T>(
        string kernelName,
        Delegate kernelMethod,
        CompilationOptions options = null) where T : struct
    {
        using var activity = activitySource.StartActivity("kernel.compile");
        activity?.SetTag("kernel.name", kernelName);
        activity?.SetTag("kernel.method", kernelMethod.Method.Name);
        
        try
        {
            var result = await inner.CompileKernelAsync<T>(kernelName, kernelMethod, options);
            activity?.SetStatus(ActivityStatusCode.Ok);
            return new TracedCompiledKernel<T>(result, activitySource);
        }
        catch (Exception ex)
        {
            activity?.SetStatus(ActivityStatusCode.Error, ex.Message);
            throw;
        }
    }
    
    public async Task<ExecutionResult> ExecuteAsync<T>(
        ICompiledKernel<T> kernel,
        KernelConfig config,
        params object[] arguments) where T : struct
    {
        using var activity = activitySource.StartActivity("kernel.execute");
        activity?.SetTag("kernel.name", kernel.Name);
        activity?.SetTag("config.grid_size", config.GridSize.ToString());
        activity?.SetTag("config.group_size", config.GroupSize.ToString());
        
        try
        {
            var result = await inner.ExecuteAsync(kernel, config, arguments);
            
            activity?.SetTag("execution.success", result.Success);
            activity?.SetTag("execution.duration_ms", result.ExecutionTime.TotalMilliseconds);
            
            if (result.Success)
            {
                activity?.SetStatus(ActivityStatusCode.Ok);
            }
            else
            {
                activity?.SetStatus(ActivityStatusCode.Error, result.Error?.Message);
            }
            
            return result;
        }
        catch (Exception ex)
        {
            activity?.SetStatus(ActivityStatusCode.Error, ex.Message);
            throw;
        }
    }
}

public class TracedCompiledKernel<T> : ICompiledKernel<T> where T : struct
{
    private readonly ICompiledKernel<T> inner;
    private readonly ActivitySource activitySource;
    
    public TracedCompiledKernel(ICompiledKernel<T> inner, ActivitySource activitySource)
    {
        this.inner = inner;
        this.activitySource = activitySource;
    }
    
    public string Name => inner.Name;
    
    public async Task<ExecutionResult> ExecuteAsync(KernelConfig config, params object[] arguments)
    {
        using var activity = activitySource.StartActivity($"kernel.{Name}.execute");
        
        // Add contextual information
        activity?.SetTag("operation", "kernel_execution");
        activity?.SetTag("kernel.name", Name);
        
        return await inner.ExecuteAsync(config, arguments);
    }
}
```

## Security Considerations

### 1. Secure Configuration

```csharp
// Secure configuration management
public class SecureILGPUConfiguration
{
    private readonly IConfiguration configuration;
    private readonly IDataProtector dataProtector;
    
    public SecureILGPUConfiguration(IConfiguration configuration, IDataProtectionProvider provider)
    {
        this.configuration = configuration;
        this.dataProtector = provider.CreateProtector("ILGPU.Configuration");
    }
    
    public ILGPUConfiguration GetConfiguration()
    {
        var config = new ILGPUConfiguration();
        configuration.GetSection("ILGPU").Bind(config);
        
        // Validate configuration
        ValidateConfiguration(config);
        
        return config;
    }
    
    private void ValidateConfiguration(ILGPUConfiguration config)
    {
        // Security validations
        if (config.Context.EnableDebugMode && IsProductionEnvironment())
        {
            throw new InvalidOperationException("Debug mode cannot be enabled in production");
        }
        
        if (config.Memory.DefaultPoolSize > GetMaxAllowedMemory())
        {
            throw new InvalidOperationException("Memory pool size exceeds security limits");
        }
        
        if (config.Performance.MaxConcurrentOperations > 100)
        {
            throw new InvalidOperationException("Too many concurrent operations allowed");
        }
    }
    
    private bool IsProductionEnvironment()
    {
        return Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") == "Production";
    }
    
    private long GetMaxAllowedMemory()
    {
        // Return maximum allowed memory based on security policy
        return 2L * 1024 * 1024 * 1024; // 2GB limit
    }
}
```

### 2. Input Validation and Sanitization

```csharp
// Input validation for ILGPU operations
public class SecureILGPUService : IILGPUService
{
    private readonly IILGPUService inner;
    private readonly IInputValidator validator;
    private readonly ILogger logger;
    
    public SecureILGPUService(IILGPUService inner, IInputValidator validator, ILogger logger)
    {
        this.inner = inner;
        this.validator = validator;
        this.logger = logger;
    }
    
    public async Task<TResult[]> ProcessBatchAsync<TInput, TResult>(
        TInput[] data,
        Func<Index1D, ArrayView<TInput>, ArrayView<TResult>> kernel)
        where TInput : unmanaged
        where TResult : unmanaged
    {
        // Validate inputs
        await validator.ValidateAsync(data);
        await validator.ValidateKernelAsync(kernel);
        
        // Log security-relevant information
        logger.LogInformation($"Processing batch of {data.Length} items");
        
        return await inner.ProcessBatchAsync(data, kernel);
    }
}

public interface IInputValidator
{
    Task ValidateAsync<T>(T[] data) where T : unmanaged;
    Task ValidateKernelAsync<T>(Delegate kernel);
}

public class InputValidator : IInputValidator
{
    private readonly ILogger<InputValidator> logger;
    
    public InputValidator(ILogger<InputValidator> logger)
    {
        this.logger = logger;
    }
    
    public async Task ValidateAsync<T>(T[] data) where T : unmanaged
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        
        if (data.Length == 0)
            throw new ArgumentException("Data array cannot be empty", nameof(data));
        
        if (data.Length > 10_000_000) // 10M limit
            throw new ArgumentException("Data array too large", nameof(data));
        
        // Validate data content for suspicious patterns
        await ValidateDataContent(data);
    }
    
    public async Task ValidateKernelAsync<T>(Delegate kernel)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));
        
        // Validate kernel method signature
        var method = kernel.Method;
        
        if (!IsValidKernelSignature(method))
            throw new ArgumentException("Invalid kernel signature", nameof(kernel));
        
        // Check for potentially dangerous operations
        await ValidateKernelSafety(method);
    }
    
    private async Task ValidateDataContent<T>(T[] data) where T : unmanaged
    {
        // Check for NaN, infinity, or other suspicious values
        if (typeof(T) == typeof(float))
        {
            var floatData = data as float[];
            foreach (var value in floatData)
            {
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    logger.LogWarning("Suspicious float value detected: {Value}", value);
                }
            }
        }
        
        await Task.CompletedTask;
    }
    
    private bool IsValidKernelSignature(MethodInfo method)
    {
        var parameters = method.GetParameters();
        
        // First parameter must be Index1D, Index2D, or Index3D
        if (parameters.Length == 0)
            return false;
        
        var firstParam = parameters[0].ParameterType;
        return firstParam == typeof(Index1D) || 
               firstParam == typeof(Index2D) || 
               firstParam == typeof(Index3D);
    }
    
    private async Task ValidateKernelSafety(MethodInfo method)
    {
        // Static analysis to check for unsafe operations
        // This would involve IL analysis to detect potentially harmful patterns
        
        await Task.CompletedTask;
    }
}
```

This integration and deployment guide provides comprehensive patterns for enterprise ILGPU deployment across various platforms and environments. Would you like me to continue with the final technical reference document covering operational best practices and troubleshooting?