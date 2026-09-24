#include <cuda_runtime.h>
#include <cstdio>

__global__ void verify_kernel(int *out) { out[threadIdx.x] = threadIdx.x * 3 + 7; }

int main() {
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    int *device = nullptr;
    int host[32] = {};
    cudaError_t error = cudaMalloc(&device, sizeof(host));
    if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); return 1; }
    verify_kernel<<<1, 32>>>(device);
    error = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); return 2; }
    for (int i = 0; i < 32; ++i) if (host[i] != i * 3 + 7) return 3;
    std::printf("runtime=%d driver=%d sm_121 kernel PASS\n", runtime, driver);
    return runtime >= 13030 ? 0 : 4;
}
