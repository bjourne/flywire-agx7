// Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
#include "shared.cl"

#define N_NEURONS       (N_NEU_BLOCKS * NEU_ALIGN)
#define N_SEND_BUF      (64 * 1024 / NEU_ALIGN)

#define MAX_N_SYNS      (32 * 1024)
#define N_SYN_BUF       (32 * 1024 / N_LANES)

#define N_BLK_SYNS  (SYN_ALIGN * SYN_GRP_ALIGN)

#if IS_AOC==1
#define ARR_LOAD(arr, ix)   (__burst_coalesced_load(&arr[(ix)]))
#else
#define ARR_LOAD(arr, ix)   (arr[(ix)])
#endif

#define PER_NEU_COUNTS    1

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
sim(
    uint n_ticks,
    fp_t alpha, fp_t beta, fp_t gamma,
    fp_t v_th,
    fp_t w_int, fp_t w_ext,

    // Initial membrane and pre-synaptic potential
    global const fp_t * restrict gl_v_init,
    global const fp_t * restrict gl_g_init,

    // External stimuli
    global const uchar * restrict gl_ext,

    // CSR data in src -> dst format
    global const uint * restrict gl_ix,
    global const int * restrict gl_syn,

    global uint * restrict ret
) {
    // Initialize state
    uint
        __attribute__((force_pow2_depth(0)))
        lo_ix[N_NEURONS];
    uchar
        __attribute__((force_pow2_depth(0)))
        lo_r[N_NEURONS];
    fp_t
        __attribute__((force_pow2_depth(0)))
        lo_g[N_NEURONS];
    fp_t
        __attribute__((force_pow2_depth(0)))
        lo_v[N_NEURONS];

#if PER_NEU_COUNTS==1
    ushort
        __attribute__((force_pow2_depth(0)))
        lo_per_neu[N_NEURONS];
#endif

    sb_t
        __attribute__((memory, force_pow2_depth(0)))
        lo_int[N_NEURONS][N_LANES][SYN_GRP_ALIGN];
    ushort
        __attribute__((numbanks(NEU_ALIGN)))
        send_buf[N_SEND_BUF][NEU_ALIGN];
    ushort n_send[NEU_ALIGN];


#pragma disable_loop_pipelining
    for (uint i = 0; i < N_NEU_BLOCKS; i++) {
#pragma unroll
        for (uint j = 0; j < NEU_ALIGN; j++) {
            uint ix = NEU_ALIGN * i + j;
            lo_v[ix] = gl_v_init[ix];
            lo_g[ix] = gl_g_init[ix];
            lo_r[ix] = 0;
            lo_ix[ix] = gl_ix[ix];
#if PER_NEU_COUNTS==1
            lo_per_neu[ix] = 0;
#endif
        }
    }
    uint tot = 0;

#pragma disable_loop_pipelining
    for (uint i = 0; i < n_ticks; i++) {
        uint syn_base[MAX_N_SYNS];
        ushort syn_blks[MAX_N_SYNS];
        uint syn_n = 0;
        for (uint j = 0; j < NEU_ALIGN; j++) {
            uint to_send = (i > 0) ? n_send[j] : 0;
            ASSERT(to_send < N_SEND_BUF);
            for (uint k = 0; k < to_send; k++) {
                uint ix = NEU_ALIGN * send_buf[k][j] + j;
                ASSERT(syn_n < MAX_N_SYNS);

                uint k0 = lo_ix[ix];
                uint k1 = lo_ix[ix + 1];
                syn_base[syn_n] = k0;
                syn_blks[syn_n] = k1 - k0;
                ASSERT(k1 - k0 < 0xffff);
                syn_n++;
            }
            n_send[j] = 0;
        }

#pragma ivdep
        for (uint j = 0; j < syn_n; j++) {
#pragma ivdep
            for (ushort k = 0; k < syn_blks[j]; k++) {
#pragma unroll
                for (uchar l = 0; l < SYN_GRP_ALIGN; l++) {
#pragma unroll
                    for (uchar m = 0; m < SYN_ALIGN; m++) {
                        uint block_ix = syn_base[j] + k;
                        uint syn_ix = N_BLK_SYNS * block_ix +
                                      SYN_ALIGN * l + m;
                        int data = ARR_LOAD(gl_syn, syn_ix);
                        uint dst = (uint)data & 0x3ffff;
                        sb_t cnt = data >> 18;
                        uchar lane_ix = j & (N_LANES - 1);
                        lo_int[SYN_ALIGN * dst + m][lane_ix][l] += cnt;
                    }
                }
            }
        }

#pragma ivdep
        for (uint j = 0; j < N_NEU_BLOCKS; j++) {
#pragma unroll
            for (uint k = 0; k < NEU_ALIGN; k++) {

                uint ix = NEU_ALIGN * j + k;

                // Receive spikes
                sb_t int_in = 0;
#pragma unroll
                for (uint l = 0; l < SYN_GRP_ALIGN; l++) {
#pragma unroll
                    for (uint m = 0; m < N_LANES; m++) {
                        int_in += (i > 0) ? lo_int[ix][m][l] : 0;
                        lo_int[ix][m][l] = 0;
                    }
                }
                fp_t int_in_fp = (fp_t)int_in;

                // Address to external input
                uint ext_row = i & (EXT_BUF_ROWS - 1);
                uint ext_ix = IX3D(
                    EXT_BUF_ROWS, N_NEU_BLOCKS, NEU_ALIGN,
                    ext_row, j, k
                );
                fp_t ext_in_fp = (fp_t)gl_ext[ext_ix];
                fp_t curr_in = int_in_fp * w_int + ext_in_fp * w_ext;

                uchar r = lo_r[ix];
                fp_t v = lo_v[ix];
                fp_t g = lo_g[ix];

                fp_t vp = alpha * v + gamma * g;
                fp_t gp = beta * g + curr_in;

                bool sub_th = (vp <= v_th) && (r == 0);
                bool spike = (vp > v_th) && (r == 0);
                r = spike * T_REF + (1 - spike) * r;

                lo_r[ix] = r - (r > 0 ? 1 : 0);
                lo_v[ix] = sub_th * vp;
                lo_g[ix] = sub_th * gp;

#if PER_NEU_COUNTS==1
                lo_per_neu[ix] += spike;
#endif

                tot += spike;
                if (r == T_REF - T_DLY + 1) {
                    uint at = n_send[k];
                    ASSERT(at < N_SEND_BUF);
                    ASSERT(j < 0xffff);
                    send_buf[at][k] = j;
                    n_send[k] = at + 1;
                }
            }
        }
    }
    ret[0] = tot;
#if PER_NEU_COUNTS==1
    for (uint i = 0; i < N_NEURONS; i++) {
        ret[1 + i] = lo_per_neu[i];
    }
#endif
}
