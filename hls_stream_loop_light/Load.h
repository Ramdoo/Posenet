#pragma once
#include "Posenet.h"


void LoadWgt1(wgt16_T* weight, wgt1_T wgt1[WGT_SIZE1][POSE_PE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums1*config[iter_block].ic_nums2*POSE_INTER_CH*POSE_OUT_CH/POSE_SIMD1/POSE_PE1; ++rep) {
        int p_addr = 0;
        for (int pe = 0 ; pe < POSE_PE1; pe+=2) {
            wgt16_T data;
            memcpy(&data, weight + parm_size[iter_block].w1+rep*4+p_addr, 16*sizeof(ap_int<POSE_W_BIT>));
            wgt1[rep][pe]   = data(8*POSE_W_BIT-1,0);
            wgt1[rep][pe+1] = data(16*POSE_W_BIT-1, 8*POSE_W_BIT);
            p_addr++;
        }
    }
}


void LoadWgt2(wgt16_T* weight, wgt2_T wgt2[WGT_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*POSE_INTER_CH*3*3/POSE_SIMD2; ++rep) {
        wgt16_T data;
        memcpy(&data, weight+parm_size[iter_block].w2+rep, 16*sizeof(ap_int<POSE_W_BIT>));
        wgt2[rep] = data;
    }
}


void LoadWgt3(wgt16_T* weight, wgt3_T wgt3[WGT_SIZE3][POSE_PE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*config[iter_block].oc_nums3*POSE_INTER_CH*POSE_OUT_CH/POSE_SIMD3/POSE_PE3; ++rep) {
        int p_addr = 0;
        for (int pe = 0 ; pe < POSE_PE3; pe+=2) {
            wgt16_T data;
            memcpy(&data, weight + parm_size[iter_block].w3+rep*4+p_addr, 16*sizeof(ap_int<POSE_W_BIT>));
            wgt3[rep][pe]   = data(8*POSE_W_BIT-1,0);
            wgt3[rep][pe+1] = data(16*POSE_W_BIT-1, 8*POSE_W_BIT);
            p_addr++;
        }
    }
}


void LoadBias1(bias8_T* bias, bias_T bias1[POSE_PE1][BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*POSE_INTER_CH/POSE_PE1; ++rep) {
        bias8_T data;
        memcpy(&data, bias+parm_size[iter_block].b1+rep, 8*sizeof(bias_T));
        for (int p = 0; p < 8; ++p) {
            bias1[p][rep] = data((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT);
        }
    }
}


void LoadBias2(bias8_T* bias, bias_T bias2[POSE_PE2][BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*POSE_INTER_CH/POSE_PE2; ++rep) {
        for (int iter_p = 0; iter_p < POSE_PE2/POSE_PE1; ++iter_p) {
            bias8_T data;
            memcpy(&data, bias+parm_size[iter_block].b2+rep*POSE_PE2/POSE_PE1+iter_p, POSE_PE1*sizeof(bias_T));
            for (int pe = 0; pe < 8; ++pe) {
                bias2[iter_p*8+pe][rep] = data((pe+1)*POSE_BIAS_BIT-1, pe*POSE_BIAS_BIT);
            }
        }
    }
}


void LoadBias3(bias8_T* bias, bias_T bias3[POSE_PE3][BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*POSE_OUT_CH/POSE_PE3; ++rep) {
        bias8_T data;
        memcpy(&data, bias + parm_size[iter_block].b3+rep, POSE_PE1*sizeof(bias_T));
        for (int pe = 0; pe < POSE_PE3; ++pe) {
            bias3[pe][rep] = data((pe + 1) * POSE_BIAS_BIT - 1, pe * POSE_BIAS_BIT);
        }
    }
}


void LoadM1(m8_T* m0, m0_T m0_1[POSE_PE1][BIAS_M0_SIZE1], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*POSE_INTER_CH/POSE_PE1; ++rep) {
        m8_T data;
        memcpy(&data, m0+parm_size[iter_block].m1+rep, 8*sizeof(m0_T));
        for (int p = 0; p < POSE_PE1; ++p) {
            m0_1[p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
        }
    }
}


void LoadM2(m8_T* m0, m0_T m0_2[POSE_PE2][BIAS_M0_SIZE2], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].ic_nums2*POSE_INTER_CH/POSE_PE2; ++rep) {
        for(int iter_p = 0; iter_p < POSE_PE2/POSE_PE1; ++iter_p) {
            m8_T data;
            memcpy(&data, m0+parm_size[iter_block].m2+rep*POSE_PE2/POSE_PE1+iter_p, 8*sizeof(m0_T));
            for (int p = 0; p < 8; ++p) {
                m0_2[iter_p*8+p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
            }
        }
    }
}


void LoadM3(m8_T* m0, m0_T m0_3[POSE_PE3][BIAS_M0_SIZE3], unsigned iter_block, bool enable) {
    if (!enable)
        return;
    for (int rep = 0; rep < config[iter_block].oc_nums3*POSE_OUT_CH/POSE_PE3; ++rep) {
        m8_T data;
        memcpy(&data, m0+parm_size[iter_block].m3+rep, 8*sizeof(m0_T));
        for (int p = 0; p < POSE_PE3; ++p) {
            m0_3[p][rep] = data((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT);
        }
    }
}
