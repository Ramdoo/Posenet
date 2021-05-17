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



// ************************************************************************* //
// convtranspose 8 layers
// ************************************************************************* //
#define POSE_PWCV0_ROW        8
#define POSE_PWCV0_COL        6
#define POSE_PWCV0_INCH       160
#define POSE_PWCV0_OUTCH      128
#define POSE_PWCV0_SIMD       4
#define POSE_PWCV0_PE         4
#define WGT_PWCV0_SIZE        1280
#define BIAS_M0_PWCV0_SIZE    32

#define POSE_DECV1_ROW        8
#define POSE_DECV1_COL        6
#define POSE_DECV1_INCH       128
#define POSE_DECV1_OUTCH      128
#define POSE_DECV1_SIMD       16
#define POSE_DECV1_PE         16
#define WGT_DECV1_SIZE        288
#define BIAS_M0_DECV1_SIZE    32

#define POSE_PWCV2_ROW        16
#define POSE_PWCV2_COL        12
#define POSE_PWCV2_INCH       128
#define POSE_PWCV2_OUTCH      128
#define POSE_PWCV2_SIMD       4
#define POSE_PWCV2_PE         4
#define WGT_PWCV2_SIZE        1024
#define BIAS_M0_PWCV2_SIZE    32

#define POSE_DECV3_ROW        16
#define POSE_DECV3_COL        12
#define POSE_DECV3_INCH       128
#define POSE_DECV3_OUTCH      128
#define POSE_DECV3_SIMD       16
#define POSE_DECV3_PE         16
#define WGT_DECV3_SIZE        288
#define BIAS_M0_DECV3_SIZE    32

#define POSE_PWCV4_ROW        32
#define POSE_PWCV4_COL        24
#define POSE_PWCV4_INCH       128
#define POSE_PWCV4_OUTCH      128
#define POSE_PWCV4_SIMD       4
#define POSE_PWCV4_PE         4
#define WGT_PWCV4_SIZE        1024
#define BIAS_M0_PWCV4_SIZE    32

#define POSE_DECV5_ROW        34
#define POSE_DECV5_COL        24
#define POSE_DECV5_INCH       128
#define POSE_DECV5_OUTCH      128
#define POSE_DECV5_SIMD       16
#define POSE_DECV5_PE         16
#define WGT_DECV5_SIZE        288
#define BIAS_M0_DECV5_SIZE    32

#define POSE_PWCV6_ROW        64
#define POSE_PWCV6_COL        48
#define POSE_PWCV6_INCH       128
#define POSE_PWCV6_OUTCH      128
#define POSE_PWCV6_SIMD       4
#define POSE_PWCV6_PE         4
#define WGT_PWCV6_SIZE        1024
#define BIAS_M0_PWCV6_SIZE    32

#define POSE_PWCV7_ROW        64
#define POSE_PWCV7_COL        48
#define POSE_PWCV7_INCH       128
#define POSE_PWCV7_OUTCH      17
#define POSE_PWCV7_SIMD       4
#define POSE_PWCV7_PE         4
#define WGT_PWCV7_SIZE        136
#define BIAS_M0_PWCV7_SIZE    32



// ************************************************************************* //
// weight, bias and m0   array size
// ************************************************************************* //
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

typedef ap_int<POSE_PE*POSE_SIMD*POSE_W_BIT>  wgt_pe_T;
typedef ap_int<POSE_PE*POSE_BIAS_BIT>         bias_pe_T;
typedef ap_uint<POSE_PE*POSE_M0_BIT>          m0_pe_T;
