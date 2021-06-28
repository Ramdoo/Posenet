#include <ap_int.h>
#include <hls_stream.h>
#include "Posenet.h"


void LoadWgt1(wgt16_T* weight, wgt1_T wgt1[WGT_SIZE1][POSE_PE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums1*config[iter_block].ic_nums2*48*16/POSE_SIMD1/POSE_PE1; ++rep) {
        for (int pe = 0 ; pe < POSE_PE1; ++pe) {
            ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd = weight[parm_size[iter_block].w1 + rep*POSE_SIMD1*POSE_PE1 + pe];
            wgt1[rep][pe]   = temp_wgt_simd;
        }
    }
}


void LoadWgt2(wgt16_T* weight, wgt2_T wgt2[WGT_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48*3*3/POSE_SIMD2; ++rep) {
        ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
        for (int p = 0; p < POSE_SIMD2/16; ++p) {
            temp_wgt_simd((p+1)*16*POSE_W_BIT-1, p*16*POSE_W_BIT)
                    = weight[parm_size[iter_block].w2 + rep*POSE_SIMD2 + p];
        }
        wgt2[rep] = temp_wgt_simd;
    }
}


void LoadWgt3(wgt16_T* weight, wgt3_T wgt3[WGT_SIZE3][POSE_PE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*config[iter_block].oc_nums3*48*16/POSE_SIMD3/POSE_PE3; ++rep) {
        for (int pe = 0 ; pe < POSE_PE3; ++pe) {
            ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd = weight[parm_size[iter_block].w3 + rep*POSE_SIMD3*POSE_PE3 + pe];
            wgt3[rep][pe]   = temp_wgt_simd;
        }
    }
}


void LoadBias1(bias8_T* bias, bias1_pe_T bias1[BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE1; ++rep) {
        bias1[rep] = bias[parm_size[iter_block].b1 + rep];
    }
}


void LoadBias2(bias8_T* bias, bias2_pe_T bias2[BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE2; ++rep) {
        bias2_pe_T temp_bias;
        for (int pe = 0; pe < POSE_PE2/8; ++pe) {
            temp_bias((pe+1)*8*POSE_BIAS_BIT-1, pe*8*POSE_BIAS_BIT) =
                    bias[parm_size[iter_block].b2 + rep*POSE_PE2 + pe];
        }
        bias2[rep] = temp_bias;
    }
}


void LoadBias3(bias8_T* bias, bias3_pe_T bias3[BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*16/POSE_PE3; ++rep) {
        bias3[rep] = bias[parm_size[iter_block].b3 + rep];
    }
}


void LoadM1(m16_T * m0, m0_1pe_T m0_1[BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE1; rep=rep+2) {
        m16_T temp_m0 = m0[parm_size[iter_block].m1 + rep];
        m0_1[rep] = temp_m0(POSE_PE1*POSE_M0_BIT-1, 0);
        m0_1[rep+1] = temp_m0(2*POSE_PE1*POSE_M0_BIT-1, POSE_PE1*POSE_M0_BIT);
    }
}


void LoadM2(m16_T * m0, m0_2pe_T m0_2[BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE2; ++rep) {
        m0_2pe_T temp_m0;
        for (int pe = 0; pe < POSE_PE2/16; ++pe) {
            temp_m0((pe+1)*16*POSE_M0_BIT-1, pe*16*POSE_M0_BIT) =
                    m0[parm_size[iter_block].m2 + rep*POSE_PE2 + pe];
        }
        m0_2[rep] = temp_m0;
    }
}


void LoadM3(m16_T * m0, m0_3pe_T m0_3[BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*16/POSE_PE3; rep=rep+2) {
        m16_T temp_m0 = m0[parm_size[iter_block].m3 + rep];
        m0_3[rep] = temp_m0(POSE_PE3*POSE_M0_BIT-1, 0);
        m0_3[rep+1] = temp_m0(2*POSE_PE3*POSE_M0_BIT-1, POSE_PE3*POSE_M0_BIT);
    }
}
