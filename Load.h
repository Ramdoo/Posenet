
#include <ap_int.h>
#include <hls_stream.h>
#include "Posenet.h"


void LoadPwcvWgtAlpha1(wgt_T* wgt, wgt_T wgt_buf[POSE_PE][WGT_PW_SIZE_ALPHA1]) {
    for (unsigned iter = 0; iter < WGT_PW_SIZE_ALPHA1; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            wgt_buf[p][iter] = wgt[iter*POSE_PE+p];
        }
    }
}


void LoadDwcvWgtAlpha2(wgt_T* wgt, wgt_T wgt_buf[WGT_DW_SIZE_ALPHA2]) {
    for (unsigned iter = 0; iter < WGT_DW_SIZE_ALPHA2; ++iter) {
        wgt_T data = wgt[iter];
        wgt_buf[iter] = data;
    }
}


void LoadPwcvWgtAlpha3(wgt_T* wgt, wgt_T wgt_buf[POSE_PE][WGT_PW_SIZE_ALPHA3]) {
    for (unsigned iter = 0; iter < WGT_PW_SIZE_ALPHA1; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            wgt_buf[p][iter] = wgt[iter*POSE_PE+p];
        }
    }
}


void LoadPwcvWgtBeta1(wgt_T* wgt, wgt_T wgt_buf[POSE_PE][WGT_PW_SIZE_BETA]) {
    for (unsigned iter = 0; iter < WGT_PW_SIZE_BETA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            wgt_buf[p][iter] = wgt[iter*POSE_PE+p];
        }
    }
}


void LoadDwcvWgtBeta2(wgt_T* wgt, wgt_T wgt_buf[WGT_DW_SIZE_BETA]) {
    for (unsigned iter = 0; iter < WGT_DW_SIZE_BETA; ++iter) {
            wgt_T data = wgt[iter];
            wgt_buf[iter] = data;
    }
}


void LoadPwcvWgtBeta3(wgt_T* wgt, wgt_T wgt_buf[POSE_PE][WGT_PW_SIZE_BETA]) {
    for (unsigned iter = 0; iter < WGT_PW_SIZE_BETA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            wgt_buf[p][iter] = wgt[iter*POSE_PE+p];
        }
    }
}


void LoadBiasAlpha(bias_T* bias, bias_T bias_buf[POSE_PE][BIAS_M0_SIZE_ALPHA]) {
    for (unsigned iter = 0; iter < BIAS_M0_SIZE_ALPHA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            bias_buf[p][iter] = bias[iter*POSE_PE+p];
        }
    }
}


void LoadBiasBeta(bias_T* bias, bias_T bias_buf[POSE_PE][BIAS_M0_SIZE_BETA]) {
    for (unsigned iter = 0; iter < BIAS_M0_SIZE_BETA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            bias_buf[p][iter] = bias[iter*POSE_PE+p];
        }
    }
}


void LoadM0Alpha(m0_T* m0, m0_T m0_buf[POSE_PE][BIAS_M0_SIZE_ALPHA]) {
    for (unsigned iter = 0; iter < BIAS_M0_SIZE_ALPHA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            m0_buf[p][iter] = m0[iter*POSE_PE+p];
        }
    }
}


void LoadM0Beta(m0_T* m0, m0_T m0_buf[POSE_PE][BIAS_M0_SIZE_BETA]) {
    for (unsigned iter = 0; iter < BIAS_M0_SIZE_BETA; ++iter) {
        for (unsigned p = 0; p < POSE_PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            m0_buf[p][iter] = m0[iter*POSE_PE+p];
        }
    }
}
