// Standalone micro-benchmark for FlashInfer's SM120 sparse-MLA *MG prefill* kernel
// (DSv4 layout), used to measure how a runtime page block size changes kernel
// time versus the stock compile-time PAGE_BLOCK_SIZE.
//
// Build (see build.sh):  nvcc -gencode=arch=compute_120f,code=sm_120f -O3 -std=c++17
//   --expt-relaxed-constexpr -use_fast_math -DSMLA_PBS_MODE=<0..3> [-DSMLA_IO_MAXNREG=..]
//   -I<variant include root> bench_mg.cu -o bench_mg_<variant>
//
// The include root is produced by make_variants.py (a patched copy of the
// sparse_mla_sm120 tree).  SMLA_PBS_MODE=0 compiles the stock arithmetic.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>
#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

#ifndef SMLA_PBS_MODE
#define SMLA_PBS_MODE 0
#endif
#ifndef SMLA_IO_MAXNREG
#define SMLA_IO_MAXNREG 32
#endif
#ifndef SMLA_MATH_MAXNREG
#define SMLA_MATH_MAXNREG 232
#endif

// The sparse_mla_sm120 headers define everything in the global namespace.
using KV4 = KVCacheTraits<ModelType::DSV4>;

#ifdef SMLA_PRISTINE
// Stock headers have no PageGeom / geom fields; provide inert stand-ins.
struct PageGeom { int pbs, log2_pbs; uint32_t magic, shift; };
static PageGeom make_page_geom(int pbs) { PageGeom g{}; g.pbs = pbs; return g; }
#define SET_GEOM(cold, g, gx) do { (void)(g); (void)(gx); } while (0)
#else
#define SET_GEOM(cold, g, gx) do { (cold).geom = (g); (cold).geom_extra = (gx); } while (0)
#endif

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                         \
    }                                                                                  \
  } while (0)

struct Args {
  int nh = 8, topk = 128, pbs = 64, tokens = 4096, pool = 65536, iters = 20, warmup = 3;
  std::string cm = "bf16", idx = "window";
  int dual = 0, extra_topk = 512, extra_pbs = 64, extra_pool = 65536;
  int seed = 1234;
  int check = 0;
};

static uint64_t fnv1a(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

// Build a DSv4 packed KV pool: pages of `pbs` tokens, each page = pbs*576 data bytes
// (448 B fp8 nope + 128 B bf16 rope per token) followed by pbs*8 scale-footer bytes.
static std::vector<uint8_t> make_kv_pool(int pool_tokens, int pbs, std::mt19937& rng) {
  const int pages = (pool_tokens + pbs - 1) / pbs;
  const size_t page_bytes = (size_t)pbs * KV4::KV_GMEM_STRIDE;  // 584 * pbs
  std::vector<uint8_t> kv(pages * page_bytes);
  std::uniform_int_distribution<int> mant(0, 0x2F);   // small positive/negative e4m3 magnitudes
  std::uniform_int_distribution<int> sign(0, 1);
  std::uniform_real_distribution<float> rr(-1.f, 1.f);
  std::uniform_int_distribution<int> sc(125, 128);    // UE8M0 exponents near 1.0
  for (int p = 0; p < pages; p++) {
    uint8_t* page = kv.data() + (size_t)p * page_bytes;
    for (int t = 0; t < pbs; t++) {
      uint8_t* tok = page + (size_t)t * 576;
      for (int i = 0; i < KV4::D_NOPE; i++) tok[i] = (uint8_t)(mant(rng) | (sign(rng) ? 0x80 : 0));
      __nv_bfloat16* rope = reinterpret_cast<__nv_bfloat16*>(tok + KV4::D_NOPE);
      for (int i = 0; i < 64; i++) rope[i] = __float2bfloat16(rr(rng));
      uint8_t* scale = page + (size_t)pbs * 576 + (size_t)t * 8;
      for (int i = 0; i < 7; i++) scale[i] = (uint8_t)sc(rng);
      scale[7] = 0;
    }
  }
  return kv;
}

static std::vector<int32_t> make_indices(int tokens, int topk, int pool_tokens, const std::string& mode,
                                         std::mt19937& rng) {
  std::vector<int32_t> idx((size_t)tokens * topk);
  std::uniform_int_distribution<int> u(0, pool_tokens - 1);
  for (int t = 0; t < tokens; t++) {
    int32_t* row = idx.data() + (size_t)t * topk;
    if (mode == "window") {
      // sliding window: the `topk` most recent slots, -1 masked before the start
      for (int k = 0; k < topk; k++) {
        int slot = t - (topk - 1) + k;
        row[k] = slot >= 0 ? slot % pool_tokens : -1;
      }
      // compact: valid entries first (kernel contract for masked rows)
      std::stable_partition(row, row + topk, [](int32_t v) { return v >= 0; });
    } else {
      for (int k = 0; k < topk; k++) row[k] = u(rng);
    }
  }
  return idx;
}

// ---------------------------------------------------------------------------------
// Kernel launchers (mirror csrc/sparse_mla_sm120_prefill.cu)
// ---------------------------------------------------------------------------------
template <ComputeMode CM, int NH, int TOPK, int PBS, int NHG>
static void launch_single(const bf16* Q, const uint8_t* KV, const int32_t* idx, bf16* out, float* lse,
                          float sm_scale, int tokens, size_t stride_kv_block, const int* topk_len,
                          PageGeom geom, cudaStream_t st) {
  constexpr size_t smem = SmemLayoutMG<ModelType::DSV4, CM>::TOTAL;
  constexpr int HEADS_PER_CTA = NHG * HPB;
  constexpr int REPLICATE_H = (NH + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
  auto kernel = sparse_mla_prefill_mg_kernel<ModelType::DSV4, CM, NH, TOPK, PBS, NHG>;
  static bool configured = false;
  if (!configured) {
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured = true;
  }
  PrefillColdParams cold{};
  cold.sm_scale = sm_scale; cold.num_tokens = tokens; cold.stride_kv_block = stride_kv_block;
  cold.stride_kv_block_extra = 0; cold.topk_extra = 0; cold.attn_sink = nullptr;
  cold.topk_length = topk_len; cold.topk_length_extra = nullptr;
  SET_GEOM(cold, geom, geom);
  const float* sink = nullptr;
  dim3 grid(tokens * REPLICATE_H), block(BLOCK_THREADS);
  cudaLaunchConfig_t cfg{grid, block, smem, st, nullptr, 0};
  void* args[] = {(void*)&Q, (void*)&KV, (void*)&idx, (void*)&out, (void*)&lse, (void*)&sink, (void*)&cold};
  CK(cudaLaunchKernelExC(&cfg, (const void*)kernel, args));
}

template <ComputeMode CM, int NH, int TOPK, int PBS, int PBSX, int NHG>
static void launch_dual(const bf16* Q, const uint8_t* KV, const int32_t* idx, const uint8_t* KVx,
                        const int32_t* idxx, bf16* out, float* lse, float sm_scale, int tokens,
                        int topk_extra, size_t stride_kv_block, size_t stride_kv_block_extra,
                        const int* topk_len, const int* topk_len_extra, PageGeom geom, PageGeom geomx,
                        cudaStream_t st) {
  constexpr size_t smem = SmemLayoutMG<ModelType::DSV4, CM>::TOTAL;
  constexpr int HEADS_PER_CTA = NHG * HPB;
  constexpr int REPLICATE_H = (NH + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
  auto kernel = sparse_mla_prefill_mg_dual_kernel<ModelType::DSV4, CM, NH, TOPK, PBS, PBSX, NHG>;
  static bool configured = false;
  if (!configured) {
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured = true;
  }
  PrefillColdParams cold{};
  cold.sm_scale = sm_scale; cold.num_tokens = tokens; cold.stride_kv_block = stride_kv_block;
  cold.stride_kv_block_extra = stride_kv_block_extra; cold.topk_extra = topk_extra; cold.attn_sink = nullptr;
  cold.topk_length = topk_len; cold.topk_length_extra = topk_len_extra;
  SET_GEOM(cold, geom, geomx);
  const float* sink = nullptr;
  dim3 grid(tokens * REPLICATE_H), block(BLOCK_THREADS);
  cudaLaunchConfig_t cfg{grid, block, smem, st, nullptr, 0};
  void* args[] = {(void*)&Q, (void*)&KV, (void*)&idx, (void*)&KVx, (void*)&idxx, (void*)&out, (void*)&lse,
                  (void*)&sink, (void*)&cold};
  CK(cudaLaunchKernelExC(&cfg, (const void*)kernel, args));
}

struct Bufs {
  bf16* Q; uint8_t* KV; int32_t* idx; bf16* out; float* lse; int* len;
  uint8_t* KVx; int32_t* idxx; int* lenx;
  size_t stride, stridex;
};

typedef void (*single_fn)(const Bufs&, const Args&, PageGeom, cudaStream_t);
typedef void (*dual_fn)(const Bufs&, const Args&, PageGeom, PageGeom, cudaStream_t);

template <ComputeMode CM, int NH, int TOPK, int PBS, int NHG>
static void single_entry(const Bufs& b, const Args& a, PageGeom g, cudaStream_t st) {
  launch_single<CM, NH, TOPK, PBS, NHG>(b.Q, b.KV, b.idx, b.out, b.lse, 1.0f / sqrtf(512.f), a.tokens, b.stride,
                                        b.len, g, st);
}
template <ComputeMode CM, int NH, int TOPK, int PBS, int PBSX, int NHG>
static void dual_entry(const Bufs& b, const Args& a, PageGeom g, PageGeom gx, cudaStream_t st) {
  launch_dual<CM, NH, TOPK, PBS, PBSX, NHG>(b.Q, b.KV, b.idx, b.KVx, b.idxx, b.out, b.lse, 1.0f / sqrtf(512.f),
                                            a.tokens, a.extra_topk, b.stride, b.stridex, b.len, b.lenx, g, gx, st);
}

struct SingleShape { const char* cm; int nh, topk, pbs; single_fn fn; };
struct DualShape { const char* cm; int nh, topk, pbs, pbsx; dual_fn fn; };

// Instantiation grid.  Compile-time PBS matters only for SMLA_PBS_MODE=0; the
// runtime variants ignore it, so they are built at PBS=64 only.
#if SMLA_PBS_MODE == 0
#define PBS_LIST(X, ...) X(32, __VA_ARGS__) X(64, __VA_ARGS__)
#else
#define PBS_LIST(X, ...) X(64, __VA_ARGS__)
#endif

#define SINGLE_ROW(PBS, CMSTR, CM, NH, TK, NHG) {CMSTR, NH, TK, PBS, &single_entry<ComputeMode::CM, NH, TK, PBS, NHG>},
#define DUAL_ROW(PBS, CMSTR, CM, NH, TK, PBSX, NHG) {CMSTR, NH, TK, PBS, PBSX, &dual_entry<ComputeMode::CM, NH, TK, PBS, PBSX, NHG>},

static const SingleShape kSingle[] = {
#ifndef SMLA_SHAPES_DUAL_ONLY
    PBS_LIST(SINGLE_ROW, "bf16", BF16, 8, 128, 1)
    PBS_LIST(SINGLE_ROW, "bf16", BF16, 16, 128, 1)
    PBS_LIST(SINGLE_ROW, "fp8", FP8, 32, 512, 2)
    PBS_LIST(SINGLE_ROW, "fp8", FP8, 64, 2048, 2)
    PBS_LIST(SINGLE_ROW, "fp8", FP8, 128, 2048, 2)
#endif
};
static const DualShape kDual[] = {
#ifndef SMLA_SHAPES_SINGLE_ONLY
    PBS_LIST(DUAL_ROW, "bf16", BF16, 8, 128, 64, 1)
    PBS_LIST(DUAL_ROW, "bf16", BF16, 32, 128, 64, 2)
#endif
};

static void usage() {
  fprintf(stderr,
          "bench_mg --nh N --topk K --cm bf16|fp8 --pbs P [--tokens T] [--pool S] [--idx window|random]\n"
          "         [--iters I] [--warmup W] [--dual 1 --extra_topk K2 --extra_pbs P2] [--check 1]\n");
  exit(64);
}

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (k == "--nh") a.nh = atoi(next().c_str());
    else if (k == "--topk") a.topk = atoi(next().c_str());
    else if (k == "--cm") a.cm = next();
    else if (k == "--pbs") a.pbs = atoi(next().c_str());
    else if (k == "--tokens") a.tokens = atoi(next().c_str());
    else if (k == "--pool") a.pool = atoi(next().c_str());
    else if (k == "--idx") a.idx = next();
    else if (k == "--iters") a.iters = atoi(next().c_str());
    else if (k == "--warmup") a.warmup = atoi(next().c_str());
    else if (k == "--dual") a.dual = atoi(next().c_str());
    else if (k == "--extra_topk") a.extra_topk = atoi(next().c_str());
    else if (k == "--extra_pbs") a.extra_pbs = atoi(next().c_str());
    else if (k == "--extra_pool") a.extra_pool = atoi(next().c_str());
    else if (k == "--seed") a.seed = atoi(next().c_str());
    else if (k == "--check") a.check = atoi(next().c_str());
    else usage();
  }
  if (a.pbs <= 1 || (a.pbs & (a.pbs - 1))) { fprintf(stderr, "pbs must be a power of two >= 2\n"); return 64; }
  a.pool += (-a.pool) % a.pbs;
  a.extra_pool += (-a.extra_pool) % a.extra_pbs;

  // --- pick the instantiation --------------------------------------------------
  const int tmpl_pbs = (SMLA_PBS_MODE == 0) ? a.pbs : 64;
  single_fn sfn = nullptr; dual_fn dfn = nullptr;
  if (!a.dual) {
    for (const auto& s : kSingle)
      if (a.cm == s.cm && a.nh == s.nh && a.topk == s.topk && tmpl_pbs == s.pbs) sfn = s.fn;
    if (!sfn) { printf("{\"error\":\"no single-cache instantiation for cm=%s nh=%d topk=%d pbs=%d\"}\n", a.cm.c_str(), a.nh, a.topk, tmpl_pbs); return 3; }
  } else {
    for (const auto& s : kDual)
      if (a.cm == s.cm && a.nh == s.nh && a.topk == s.topk && tmpl_pbs == s.pbs && a.extra_pbs == s.pbsx) dfn = s.fn;
    if (!dfn) { printf("{\"error\":\"no dual-cache instantiation for cm=%s nh=%d topk=%d pbs=%d extra_pbs=%d\"}\n", a.cm.c_str(), a.nh, a.topk, tmpl_pbs, a.extra_pbs); return 3; }
  }

  // --- data --------------------------------------------------------------------
  std::mt19937 rng(a.seed);
  std::uniform_real_distribution<float> rq(-1.f, 1.f);
  std::vector<bf16> hq((size_t)a.tokens * a.nh * 512);
  for (auto& v : hq) v = __float2bfloat16(rq(rng));
  auto hkv = make_kv_pool(a.pool, a.pbs, rng);
  auto hidx = make_indices(a.tokens, a.topk, a.pool, a.idx, rng);
  std::vector<int> hlen(a.tokens);
  for (int t = 0; t < a.tokens; t++) {
    int n = 0; for (int k = 0; k < a.topk; k++) n += hidx[(size_t)t * a.topk + k] >= 0;
    hlen[t] = n;
  }
  std::vector<uint8_t> hkvx; std::vector<int32_t> hidxx; std::vector<int> hlenx;
  if (a.dual) {
    hkvx = make_kv_pool(a.extra_pool, a.extra_pbs, rng);
    hidxx = make_indices(a.tokens, a.extra_topk, a.extra_pool, "random", rng);
    hlenx.assign(a.tokens, a.extra_topk);
    // realistic ramp: the extra (compressed) row length grows with position
    for (int t = 0; t < a.tokens; t++) hlenx[t] = std::min(a.extra_topk, std::max(0, t / 2));
  }

  Bufs b{};
  b.stride = (size_t)a.pbs * KV4::KV_GMEM_STRIDE;
  b.stridex = (size_t)a.extra_pbs * KV4::KV_GMEM_STRIDE;
  CK(cudaMalloc(&b.Q, hq.size() * sizeof(bf16)));
  CK(cudaMalloc(&b.KV, hkv.size()));
  CK(cudaMalloc(&b.idx, hidx.size() * sizeof(int32_t)));
  CK(cudaMalloc(&b.out, (size_t)a.tokens * a.nh * 512 * sizeof(bf16)));
  CK(cudaMalloc(&b.lse, (size_t)a.tokens * a.nh * sizeof(float)));
  CK(cudaMalloc(&b.len, (size_t)a.tokens * sizeof(int)));
  CK(cudaMemcpy(b.Q, hq.data(), hq.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.KV, hkv.data(), hkv.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.idx, hidx.data(), hidx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b.len, hlen.data(), hlen.size() * sizeof(int), cudaMemcpyHostToDevice));
  if (a.dual) {
    CK(cudaMalloc(&b.KVx, hkvx.size()));
    CK(cudaMalloc(&b.idxx, hidxx.size() * sizeof(int32_t)));
    CK(cudaMalloc(&b.lenx, (size_t)a.tokens * sizeof(int)));
    CK(cudaMemcpy(b.KVx, hkvx.data(), hkvx.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.idxx, hidxx.data(), hidxx.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.lenx, hlenx.data(), hlenx.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  CK(cudaMemset(b.out, 0, (size_t)a.tokens * a.nh * 512 * sizeof(bf16)));
  CK(cudaMemset(b.lse, 0, (size_t)a.tokens * a.nh * sizeof(float)));

  PageGeom g = make_page_geom(a.pbs), gx = make_page_geom(a.extra_pbs);
  cudaStream_t st; CK(cudaStreamCreate(&st));
  auto run = [&]() { if (a.dual) dfn(b, a, g, gx, st); else sfn(b, a, g, st); };

  for (int i = 0; i < a.warmup; i++) run();
  CK(cudaStreamSynchronize(st));
  CK(cudaGetLastError());

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ms(a.iters);
  for (int i = 0; i < a.iters; i++) {
    CK(cudaEventRecord(e0, st)); run(); CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms[i], e0, e1));
  }
  std::vector<float> sorted = ms; std::sort(sorted.begin(), sorted.end());
  float mean = 0; for (float v : ms) mean += v; mean /= a.iters;
  const float med = sorted[a.iters / 2], mn = sorted[0];

  std::vector<bf16> hout((size_t)a.tokens * a.nh * 512); std::vector<float> hlse((size_t)a.tokens * a.nh);
  CK(cudaMemcpy(hout.data(), b.out, hout.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hlse.data(), b.lse, hlse.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const uint64_t h_out = fnv1a(hout.data(), hout.size() * sizeof(bf16));
  const uint64_t h_lse = fnv1a(hlse.data(), hlse.size() * sizeof(float));
  int nonfinite = 0; double asum = 0;
  for (size_t i = 0; i < hout.size(); i++) { float v = __bfloat162float(hout[i]); if (!std::isfinite(v)) nonfinite++; else asum += fabs(v); }

  const double tokens_per_s = a.tokens / (med * 1e-3);
  printf("{\"mode\":%d,\"io_maxnreg\":%d,\"math_maxnreg\":%d,\"dual\":%d,\"cm\":\"%s\",\"nh\":%d,\"topk\":%d,"
         "\"pbs\":%d,\"tmpl_pbs\":%d,\"extra_topk\":%d,\"extra_pbs\":%d,\"tokens\":%d,\"pool\":%d,\"idx\":\"%s\","
         "\"iters\":%d,\"us_mean\":%.2f,\"us_median\":%.2f,\"us_min\":%.2f,\"tokens_per_s\":%.0f,"
         "\"hash_out\":\"%016llx\",\"hash_lse\":\"%016llx\",\"nonfinite\":%d,\"abs_mean\":%.6f}\n",
         SMLA_PBS_MODE, SMLA_IO_MAXNREG, SMLA_MATH_MAXNREG, a.dual, a.cm.c_str(), a.nh, a.topk, a.pbs, tmpl_pbs,
         a.dual ? a.extra_topk : 0, a.dual ? a.extra_pbs : 0, a.tokens, a.pool, a.idx.c_str(), a.iters,
         mean * 1e3, med * 1e3, mn * 1e3, tokens_per_s, (unsigned long long)h_out, (unsigned long long)h_lse,
         nonfinite, asum / hout.size());
  return 0;
}
