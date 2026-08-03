// Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
//
// Naming conventions:
//
//   * gl_ prefix for global arrays
//   * lo_ prefix for (some) local arrays
//   * ix means "neuron index"
//   * N_ and n_ prefix means "number of"
#ifndef SHARED_CL
#define SHARED_CL

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Requires the following defines
#if !defined(IS_AOC) || !defined(N_NEU_BLOCKS) || !defined(FP_BITS)
#error "Requires IS_AOC, N_NEU_BLOCKS, and FP_BITS!"
#endif

// Good utility
#define ALIGN_TO(x, y)      (((x) + (y) - 1) / (y) * (y))

#define MAX(a, b)               ((a) > (b) ? (a) : (b))
#define MIN(a, b)               ((a) < (b) ? (a) : (b))
#define CLAMP(a, lo, hi)        (MIN(MAX(a, lo), hi))



// Not supported by NVIDIA OpenCL
#if IS_AOC == 0

#define ARR_LOAD_BC(arr, ix)   (arr[(ix)])
#define ASSERT(cond)        if (!(cond)) { printf("%s L%4d: %s fail\n", __FILE__, __LINE__, #cond); }

#else

#define ARR_LOAD_BC(arr, ix)   (__burst_coalesced_load(&arr[(ix)]))
#define ASSERT(cond)

#endif

#if FP_BITS == 16
typedef half fp_t;
#elif FP_BITS == 32
typedef float fp_t;
#else
typedef double fp_t;
#endif

typedef short sb_t;


// Simplifies indexing calculations in ND arrays.
#define IX2D(d1, d2, e1, e2) \
    ((d2) * (e1) + (e2))

#define IX3D(d1, d2, d3, e1, e2, e3) \
    ((d2) * (d3) * (e1) + (d3) * (e2) + (e3))

#define IX4D(d1, d2, d3, d4, e1, e2, e3, e4) \
    ((d4) * (d3) * (d2) * (e1) + (d4) * (d3) * (e2) + (d4) * (e3) + (e4))


#endif
