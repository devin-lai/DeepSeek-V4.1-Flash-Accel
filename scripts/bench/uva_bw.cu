// How fast can one RTX 5090 pull expert weights out of pinned host memory?
//
//   A. kernel reads through UVA with 16-byte vector loads (what Marlin does today)
//   B. kernel gather-copy host -> device staging with cp.async.bulk (TMA) 16 KiB chunks
//   C. cudaMemcpyAsync host -> device (copy engine / DMA), whole buffer
//   D. cudaMemcpyAsync per "expert" slice (17.7 MB each), many small DMA requests
//
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_120f,code=sm_120f uva_bw.cu -o uva_bw
// Run:   numactl --membind=<node of the GPU> ./uva_bw [MiB] [iters]
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <functional>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

// A: plain vectorised loads; each thread streams 16 B per iteration; result reduced so loads are not dead.
__global__ void read_ldg(const uint4* __restrict__ src, size_t n16, unsigned long long* sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  unsigned long long acc = 0;
#pragma unroll 4
  for (; i < n16; i += stride) {
    uint4 v = __ldg(src + i);
    acc += v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x1234567887654321ull) atomicAdd(sink, acc);
}

// A2: plain loads + store to a device staging buffer (a copy kernel)
__global__ void copy_ldg(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
#pragma unroll 4
  for (; i < n16; i += stride) dst[i] = __ldg(src + i);
}

// B: TMA bulk copy host->smem->device.  Each CTA owns chunks of CHUNK bytes; NSTAGE
// chunks in flight to cover PCIe latency.
constexpr int CHUNK = 16384;
constexpr int NSTAGE = 4;

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      " @p bra DONE;\n bra LAB_WAIT;\n DONE:\n}" ::"r"((unsigned)__cvta_generic_to_shared(bar)), "r"(parity));
}
__device__ __forceinline__ void bulk_g2s(void* smem_dst, const void* gsrc, unsigned bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"((unsigned)__cvta_generic_to_shared(smem_dst)), "l"(gsrc), "r"(bytes),
               "r"((unsigned)__cvta_generic_to_shared(bar)) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* gdst, const void* smem_src, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(gdst), "r"((unsigned)__cvta_generic_to_shared(smem_src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__global__ void __launch_bounds__(32) copy_tma(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t nbytes) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bars[NSTAGE];
  const size_t nchunks = nbytes / CHUNK;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSTAGE; s++) mbar_init(&bars[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncwarp();
  if (threadIdx.x != 0) return;  // single elected thread drives the pipeline
  size_t my = blockIdx.x;  // chunk index owned by this CTA
  const size_t step = gridDim.x;
  // prologue: issue NSTAGE loads
  unsigned parity[NSTAGE] = {0, 0, 0, 0};
  size_t issued[NSTAGE];
  int inflight = 0;
  for (int s = 0; s < NSTAGE && my < nchunks; s++, my += step) {
    mbar_expect_tx(&bars[s], CHUNK);
    bulk_g2s(smem + s * CHUNK, src + my * CHUNK, CHUNK, &bars[s]);
    issued[s] = my; inflight++;
  }
  int s = 0;
  while (inflight > 0) {
    mbar_wait(&bars[s], parity[s]); parity[s] ^= 1;
    // write the chunk out to device memory via bulk store, then refill the stage
    bulk_s2g(dst + issued[s] * CHUNK, smem + s * CHUNK, CHUNK);
    bulk_commit();
    bulk_wait_read<0>();  // smem stage is free again once the store has read it
    inflight--;
    if (my < nchunks) {
      mbar_expect_tx(&bars[s], CHUNK);
      bulk_g2s(smem + s * CHUNK, src + my * CHUNK, CHUNK, &bars[s]);
      issued[s] = my; my += step; inflight++;
    }
    s = (s + 1) % NSTAGE;
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

static float time_it(cudaStream_t st, int iters, const std::function<void()>& f) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  f(); CK(cudaStreamSynchronize(st));  // warm
  std::vector<float> ms;
  for (int i = 0; i < iters; i++) { CK(cudaEventRecord(a, st)); f(); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? atoll(argv[1]) : 1024;
  int iters = argc > 2 ? atoi(argv[2]) : 5;
  size_t nbytes = mib << 20;
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  printf("device %d %s, buffer %zu MiB, iters %d\n", dev, p.name, mib, iters);

  // pinned host buffer, page-locked and mapped (UVA)
  uint8_t* h = nullptr;
  CK(cudaHostAlloc((void**)&h, nbytes, cudaHostAllocMapped | cudaHostAllocPortable));
  for (size_t i = 0; i < nbytes; i += 4096) h[i] = (uint8_t)i;  // touch pages
  uint8_t* hdev = nullptr; CK(cudaHostGetDevicePointer((void**)&hdev, h, 0));
  uint8_t* d = nullptr; CK(cudaMalloc((void**)&d, nbytes));
  unsigned long long* sink; CK(cudaMalloc((void**)&sink, 8)); CK(cudaMemset(sink, 0, 8));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  const double gib = (double)nbytes / 1e9;
  int sms = p.multiProcessorCount;

  for (int blocks_per_sm : {2, 4, 8}) {
    int grid = sms * blocks_per_sm;
    float ms = time_it(st, iters, [&] { read_ldg<<<grid, 256, 0, st>>>((const uint4*)hdev, nbytes / 16, sink); });
    printf("A  UVA read  LDG.128   grid=%5d x256   %8.2f ms  %6.1f GB/s\n", grid, ms, gib / (ms / 1e3));
  }
  for (int blocks_per_sm : {4, 8}) {
    int grid = sms * blocks_per_sm;
    float ms = time_it(st, iters, [&] { copy_ldg<<<grid, 256, 0, st>>>((const uint4*)hdev, (uint4*)d, nbytes / 16); });
    printf("A2 UVA copy  LDG/STG   grid=%5d x256   %8.2f ms  %6.1f GB/s\n", grid, ms, gib / (ms / 1e3));
  }
  CK(cudaFuncSetAttribute(copy_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, CHUNK * NSTAGE));
  for (int grid : {sms, sms * 2, sms * 4, sms * 8}) {
    float ms = time_it(st, iters, [&] { copy_tma<<<grid, 32, CHUNK * NSTAGE, st>>>(hdev, d, nbytes); });
    printf("B  UVA copy  TMA bulk  grid=%5d x32 x%dKBx%d %8.2f ms  %6.1f GB/s\n", grid, CHUNK / 1024, NSTAGE, ms, gib / (ms / 1e3));
  }
  {
    float ms = time_it(st, iters, [&] { CK(cudaMemcpyAsync(d, h, nbytes, cudaMemcpyHostToDevice, st)); });
    printf("C  memcpyAsync H2D whole buffer            %8.2f ms  %6.1f GB/s\n", ms, gib / (ms / 1e3));
  }
  {
    const size_t slice = 17700000; size_t n = nbytes / slice;
    float ms = time_it(st, iters, [&] { for (size_t i = 0; i < n; i++) CK(cudaMemcpyAsync(d + i * slice, h + i * slice, slice, cudaMemcpyHostToDevice, st)); });
    printf("D  memcpyAsync H2D %zu x 17.7MB slices       %8.2f ms  %6.1f GB/s\n", n, ms, (double)(n * slice) / 1e9 / (ms / 1e3));
  }
  // sanity: verify TMA copy correctness on a small prefix
  std::vector<uint8_t> chk(1 << 20);
  CK(cudaMemcpy(chk.data(), d, chk.size(), cudaMemcpyDeviceToHost));
  int bad = 0; for (size_t i = 0; i < chk.size(); i += 4096) bad += chk[i] != (uint8_t)i;
  printf("copy check: %d bad pages of %zu\n", bad, chk.size() / 4096);
  return 0;
}
