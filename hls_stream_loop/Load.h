#pragma once
#include "Posenet.h"


void LoadWgt1(wgt16_T* weight, wgt1_T wgt1[WGT_SIZE1][POSE_PE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums1*config[iter_block].ic_nums2*48*16/POSE_SIMD1/POSE_PE1; ++rep) {
        for (int pe = 0 ; pe < POSE_PE1; ++pe) {
            ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd;
            memcpy(&temp_wgt_simd, weight+parm_size[iter_block].w1+rep*POSE_PE1+pe, POSE_SIMD1*sizeof(ap_int<POSE_W_BIT>));
            wgt1[rep][pe] = temp_wgt_simd;
        }
    }
}


void LoadWgt2(wgt16_T* weight, wgt2_T wgt2[WGT_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48*3*3/POSE_SIMD2; ++rep) {
        ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
        for (int p = 0; p < POSE_SIMD2/16; ++p) {
            wgt16_T data;
            memcpy(&data, weight+parm_size[iter_block].w2+rep*3+p, 16*sizeof(ap_int<POSE_W_BIT>));
            temp_wgt_simd((p+1)*16*POSE_W_BIT-1, p*16*POSE_W_BIT) = data;
        }
        wgt2[rep] = temp_wgt_simd;
    }
}


void LoadWgt3(wgt16_T* weight, wgt3_T wgt3[WGT_SIZE3][POSE_PE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*config[iter_block].oc_nums3*48*16/POSE_SIMD3/POSE_PE3; ++rep) {
        for (int pe = 0 ; pe < POSE_PE3; ++pe) {
            ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd;
            memcpy(&temp_wgt_simd, weight+parm_size[iter_block].w3+rep*POSE_PE3+pe, POSE_SIMD3*sizeof(ap_int<POSE_W_BIT>));
            wgt3[rep][pe]   = temp_wgt_simd;
        }
    }
}


void LoadBias1(bias8_T* bias, bias_T bias1[POSE_PE1][BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE1; ++rep) {
        bias8_T data;
        memcpy(&data, bias+parm_size[iter_block].b1+rep, POSE_PE1*sizeof(bias_T));
        for (int p = 0; p < POSE_PE1; ++p) {
            bias1[p][rep] = data((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT);
        }
    }
}


void LoadBias2(bias8_T* bias, bias_T bias2[POSE_PE2][BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE2; ++rep) {
        for (int iter_p = 0; iter_p < POSE_PE2/8; ++iter_p) {
            bias8_T data;
            memcpy(&data, bias+parm_size[iter_block].b2+rep*6+iter_p, 8*sizeof(bias_T));
            for (int pe = 0; pe < 8; ++pe) {
                bias2[iter_p*8+pe][rep] = data((pe+1)*POSE_BIAS_BIT-1, pe*POSE_BIAS_BIT);
            }
        }
    }
}


void LoadBias3(bias8_T* bias, bias_T bias3[POSE_PE3][BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*16/POSE_PE3; ++rep) {
        bias8_T data;
        memcpy(&data, bias+parm_size[iter_block].b3+rep, POSE_PE3* sizeof(bias_T));
        for (int pe = 0; pe < POSE_PE3; ++pe) {
            bias3[pe][rep] = data((pe+1)*POSE_BIAS_BIT-1, pe*POSE_BIAS_BIT);
        }
    }
}


void LoadM1(m8_T* m0, m0_T m0_1[POSE_PE1][BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE1; ++rep) {
        ap_uint<POSE_PE1*POSE_M0_BIT> data;
        memcpy(&data, m0+parm_size[iter_block].b1+rep, POSE_PE1* sizeof(m0_T));
        for (int p = 0; p < POSE_PE1; ++p) {
            m0_1[p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
        }
    }
}


void LoadM2(m8_T* m0, m0_T m0_2[POSE_PE2][BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*48/POSE_PE2; ++rep) {
        for(int iter_p = 0; iter_p < POSE_PE2/8; ++iter_p) {
            m8_T data;
            memcpy(&data, m0+parm_size[iter_block].b2+rep*6+iter_p, 8*sizeof(m0_T));
            for (int p = 0; p < 8; ++p) {
                m0_2[iter_p*8+p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
            }
        }
    }
}


void LoadM3(m8_T* m0, m0_T m0_3[POSE_PE3][BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*16/POSE_PE3; ++rep) {
        ap_uint<POSE_PE3*POSE_M0_BIT> data;
        memcpy(&data, m0+parm_size[iter_block].b3+rep, POSE_PE3* sizeof(m0_T));
        for (int p = 0; p < POSE_PE3; ++p) {
            m0_3[p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
        }
    }
}