#pragma once
#include <ap_int.h>
#include <hls_stream.h>


#define POSE_IN_CH              16
#define POSE_OUT_CH             16
#define POSE_INTER_CH           16

#define POSE_IN_BIT             8
#define POSE_OUT_BIT            8
#define POSE_W_BIT              8
#define POSE_BIAS_BIT           16
#define POSE_M0_BIT             16
#define POSE_MUL_BIT            32

#define POSE_SIMD               16
#define POSE_PE                 16

//480 channels, 12 cols
#define WGT_PW_SIZE_ALPHA1      150
#define WGT_DW_SIZE_ALPHA2      270
#define WGT_PW_SIZE_ALPHA3      300
#define BIAS_M0_SIZE_ALPHA      30

//128 channels, 96 cols
#define WGT_PW_SIZE_BETA        64
#define WGT_DW_SIZE_BETA        72
#define BIAS_M0_SIZE_BETA       8

#define INST_WIDTH              32



// ************************************************************************* //
// do some typedef
// ************************************************************************* //
typedef ap_int<POSE_IN_CH*POSE_IN_BIT>      infm_T;
typedef ap_int<POSE_OUT_CH*POSE_OUT_BIT>    outfm_T;
typedef ap_int<POSE_PE*POSE_OUT_BIT>        addfm_T;
typedef ap_int<POSE_INTER_CH*POSE_OUT_BIT>  innerfm_T;

typedef ap_int<POSE_SIMD*POSE_W_BIT>        adj_T;
typedef ap_int<POSE_PE*POSE_W_BIT>          mvau_T;
typedef ap_int<POSE_SIMD*POSE_W_BIT>        wgt_T;
typedef ap_int<32*POSE_W_BIT>               wgt32_T;
typedef ap_int<POSE_BIAS_BIT>               bias_T;
typedef ap_int<POSE_MUL_BIT>                mul_T;
typedef ap_uint<POSE_M0_BIT>                m0_T;
