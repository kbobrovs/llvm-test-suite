//==------- simd_memory_access.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks functionality of the memory access APIs which are members of
// the simd class.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace sycl::INTEL::gpu;

static inline constexpr int channels_per_pixel = 4;

template <typename T, int N> struct defs {
  static inline constexpr int chunk_size = (channels_per_pixel * N);
  static inline constexpr int pixel_size = (channels_per_pixel * sizeof(T));
  static inline constexpr int N_pixels_size = (pixel_size * N);
};

// Pointer-based kernel.
template <typename T, int N> struct RGBAKernel {

  T *A; T *B;

  RGBAKernel(T *A, T *B) : A(A), B(B) {}

  void operator()(id<1> i) const SYCL_ESIMD_KERNEL {
    const uint32_t ii = static_cast<uint32_t>(i.get(0));
    simd<T, defs<T, N>::chunk_size> v;
    const auto offset = ii * (sizeof(v) / sizeof(T));
    simd<uint32_t, N> offsets(0, defs<T, N>::pixel_size);
    v.gather_rgba(A + offset, offsets);

    for (int i = 0; i < channels_per_pixel; ++i) {
      block_store<T, N>(A + i * N, v.template select<N, 1>(i*N));
    }

    simd<T, defs<T, N>::chunk_size> v1;

    for (int i = 0; i < channels_per_pixel; ++i) {
      v1.template select<N, channels_per_pixel>(i) = simd<T, N>((T)offset + i* channels_per_pixel, 1);
    }
    v1.scatter_rgba(B + offset, offsets);
  }

};

template <typename T>
int check(T *X, const char *msg, size_t size) {
  std::cout << "checking " << msg << "...\n";
  int err_cnt = 0;

  for (unsigned i = 0; i < size; ++i) {
    T gold = (T)i;
    T val = X[i];

    if (val != gold) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ": " << val
          << " != " << gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
      << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
      << (size - err_cnt) << "/" << size << ")\n";
  }
  return err_cnt;
}

// The main test routine.
template <typename T, int N> bool test(queue q, size_t size) {
  std::cout << "Testing T=" << typeid(T).name() << ", N=" << N << "...\n";

  T *A = reinterpret_cast<T *>(sycl::malloc_shared(size, q));
  T *B = reinterpret_cast<T *>(sycl::malloc_shared(size, q));

  for (unsigned N_chunk = 0; N_chunk < size / defs<T, N>::chunk_size; ++N_chunk) {
    int chunk_start = N_chunk * defs<T, N>::chunk_size;
    int val = chunk_start;

    for (int ch = 0; ch < channels_per_pixel; ++ch) {
      for (int i = 0; i < N; ++i) {
        B[chunk_start + ch + i * channels_per_pixel] = val++;
      }
    }
  }

  try {
    range<1> glob_range{ size / defs<T, N>::chunk_size };

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(glob_range, RGBAKernel<T, N>{A, B});
    });
    e.wait_and_throw();
  }
  catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    return false;
  }

  int err_cnt = 0;
  err_cnt += check<T>(A, "gather_rgba", size);
  err_cnt += check<T>(B, "scatter_rgba", size);

  sycl::free(A, q);
  sycl::free(B, q);

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(int argc, char **argv) {
  size_t size = 32 * 4 * 7;

  if (argc > 1) {
    size = atoi(argv[1]);
    size = size == 0 ? 32 * 4 * 7 : size;
  }
  if (size % (32 * 4) != 0) {
    std::cerr << "*** ERROR: size (" << size << ") must be a multiple of 128\n";
    return 2;
  }
  std::cout << "Using size=" << size << "\n";
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<int, 32>(q, size);
  //passed &= test<unsigned int, 32>(q, size);
  //passed &= test<float, 32>(q, size);
  //passed &= test<int, 16>(q, size);
  //passed &= test<unsigned int, 16>(q, size);
  //passed &= test<float, 16>(q, size);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
