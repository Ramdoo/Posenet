#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>
#include <stdio.h>
#include <string.h>


#define IN_IMAGE_WIDTH  480
#define IN_IMAGE_HEIGHT 640

#define RESIZE_IMAGE_WIDTH 192
#define RESIZE_IMAGE_HEIGHT 256

#define POSE_IN_CH              16
#define POSE_OUT_CH             16
#define POSE_INTER_CH           48

#define POSE_IN_BIT             8
#define POSE_OUT_BIT            8
#define POSE_W_BIT              8
#define POSE_BIAS_BIT           32
#define POSE_M0_BIT             16
#define POSE_MUL_BIT            32

#define POSE_SIMD1              16
#define POSE_PE1                16
#define POSE_SIMD2              48
#define POSE_PE2                48
#define POSE_SIMD3              16
#define POSE_PE3                16

#define WGT_SIZE1               80*480/POSE_SIMD1/POSE_PE1
#define WGT_SIZE2               9*480/POSE_SIMD2
#define WGT_SIZE3               160*480/POSE_SIMD3/POSE_PE3
#define BIAS_M0_SIZE1           480/POSE_PE1
#define BIAS_M0_SIZE2           480/POSE_PE2
#define BIAS_M0_SIZE3           160/POSE_PE3

// ************************************************************************* //
// head 3 layers
// ************************************************************************* //
#define POSE_HCV0_ROW        256
#define POSE_HCV0_COL        192
#define POSE_HCV0_INCH       3
#define POSE_HCV0_OUTCH      16
#define POSE_HCV0_SIMD       3
#define POSE_HCV0_PE         16
#define WGT_HCV0_SIZE        POSE_HCV0_INCH*POSE_HCV0_OUTCH*9/POSE_HCV0_SIMD/POSE_HCV0_PE
#define BIAS_M0_HCV0_SIZE    POSE_HCV0_OUTCH/POSE_HCV0_PE

#define POSE_HCV1_ROW        128
#define POSE_HCV1_COL        96
#define POSE_HCV1_INCH       16
#define POSE_HCV1_OUTCH      16
#define POSE_HCV1_SIMD       8
#define POSE_HCV1_LOG2_SIMD  3
#define POSE_HCV1_PE         8
#define WGT_HCV1_SIZE        POSE_HCV1_INCH*9/POSE_HCV1_SIMD
#define BIAS_M0_HCV1_SIZE    POSE_HCV1_OUTCH/POSE_HCV1_PE

#define POSE_HCV2_ROW        128
#define POSE_HCV2_COL        96
#define POSE_HCV2_INCH       16
#define POSE_HCV2_OUTCH      8
#define POSE_HCV2_SIMD       4
#define POSE_HCV2_PE         4
#define WGT_HCV2_SIZE        POSE_HCV2_INCH*POSE_HCV2_OUTCH/POSE_HCV2_SIMD/POSE_HCV2_PE
#define BIAS_M0_HCV2_SIZE    POSE_HCV2_OUTCH/POSE_HCV2_PE


// ************************************************************************* //
// convtranspose 8 layers
// ************************************************************************* //
#define POSE_PWCV0_ROW        8
#define POSE_PWCV0_COL        6
#define POSE_PWCV0_INCH       160
#define POSE_PWCV0_OUTCH      128
#define POSE_PWCV0_SIMD       4
#define POSE_PWCV0_PE         4
#define WGT_PWCV0_SIZE        POSE_PWCV0_INCH*POSE_PWCV0_OUTCH/POSE_PWCV0_SIMD/POSE_PWCV0_PE
#define BIAS_M0_PWCV0_SIZE    POSE_PWCV0_OUTCH/POSE_PWCV0_PE

#define POSE_DECV1_ROW        8
#define POSE_DECV1_COL        6
#define POSE_DECV1_INCH       128
#define POSE_DECV1_OUTCH      128
#define POSE_DECV1_SIMD       4
#define POSE_DECV1_LOG2_SIMD  2
#define POSE_DECV1_PE         4
#define WGT_DECV1_SIZE        POSE_DECV1_INCH*9/POSE_DECV1_SIMD
#define BIAS_M0_DECV1_SIZE    POSE_DECV1_OUTCH/POSE_DECV1_PE

#define POSE_PWCV2_ROW        16
#define POSE_PWCV2_COL        12
#define POSE_PWCV2_INCH       128
#define POSE_PWCV2_OUTCH      128
#define POSE_PWCV2_SIMD       8
#define POSE_PWCV2_PE         4
#define WGT_PWCV2_SIZE        POSE_PWCV2_INCH*POSE_PWCV2_OUTCH/POSE_PWCV2_SIMD/POSE_PWCV2_PE
#define BIAS_M0_PWCV2_SIZE    POSE_PWCV2_OUTCH/POSE_PWCV2_PE

#define POSE_DECV3_ROW        16
#define POSE_DECV3_COL        12
#define POSE_DECV3_INCH       128
#define POSE_DECV3_OUTCH      128
#define POSE_DECV3_SIMD       8
#define POSE_DECV3_LOG2_SIMD  3
#define POSE_DECV3_PE         8
#define WGT_DECV3_SIZE        POSE_DECV3_INCH*9/POSE_DECV3_SIMD
#define BIAS_M0_DECV3_SIZE    POSE_DECV3_OUTCH/POSE_DECV3_PE

#define POSE_PWCV4_ROW        32
#define POSE_PWCV4_COL        24
#define POSE_PWCV4_INCH       128
#define POSE_PWCV4_OUTCH      128
#define POSE_PWCV4_SIMD       16
#define POSE_PWCV4_PE         8
#define WGT_PWCV4_SIZE        POSE_PWCV4_INCH*POSE_PWCV4_OUTCH/POSE_PWCV4_SIMD/POSE_PWCV4_PE
#define BIAS_M0_PWCV4_SIZE    POSE_PWCV4_OUTCH/POSE_PWCV4_PE

#define POSE_DECV5_ROW        32
#define POSE_DECV5_COL        24
#define POSE_DECV5_INCH       128
#define POSE_DECV5_OUTCH      128
#define POSE_DECV5_SIMD       32
#define POSE_DECV5_LOG2_SIMD  5
#define POSE_DECV5_PE         32
#define WGT_DECV5_SIZE        POSE_DECV5_INCH*9/POSE_DECV5_SIMD
#define BIAS_M0_DECV5_SIZE    POSE_DECV5_OUTCH/POSE_DECV5_PE

#define POSE_PWCV6_ROW        64
#define POSE_PWCV6_COL        48
#define POSE_PWCV6_INCH       128
#define POSE_PWCV6_OUTCH      128
#define POSE_PWCV6_SIMD       32
#define POSE_PWCV6_PE         16
#define WGT_PWCV6_SIZE        POSE_PWCV6_INCH*POSE_PWCV6_OUTCH/POSE_PWCV6_SIMD/POSE_PWCV6_PE
#define BIAS_M0_PWCV6_SIZE    POSE_PWCV6_OUTCH/POSE_PWCV6_PE

#define POSE_CV7_ROW        64
#define POSE_CV7_COL        48
#define POSE_CV7_INCH       128
#define POSE_CV7_OUTCH      17
#define POSE_CV7_SIMD       4
#define POSE_CV7_PE         17
#define WGT_CV7_SIZE        POSE_CV7_INCH*POSE_CV7_OUTCH/POSE_CV7_SIMD/POSE_CV7_PE
#define BIAS_M0_CV7_SIZE    POSE_CV7_OUTCH/POSE_CV7_PE



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
typedef ap_int<POSE_IN_CH*POSE_IN_BIT>          infm_T;
typedef ap_int<POSE_OUT_CH*POSE_OUT_BIT>        outfm_T;
typedef ap_int<POSE_PE3*POSE_OUT_BIT>           addfm_T;
typedef ap_int<POSE_INTER_CH*POSE_OUT_BIT>      innerfm_T;

typedef ap_int<POSE_SIMD1*POSE_W_BIT>           wgt1_T;
typedef ap_int<POSE_SIMD2*POSE_W_BIT>           wgt2_T;
typedef ap_int<POSE_SIMD3*POSE_W_BIT>           wgt3_T;
typedef ap_int<16*POSE_W_BIT>                   wgt16_T;
typedef ap_int<8*POSE_BIAS_BIT>                 bias8_T;
typedef ap_int<POSE_MUL_BIT>                    mul_T;
typedef ap_uint<16*POSE_M0_BIT>                 m16_T;
typedef ap_int<POSE_BIAS_BIT>                   bias_T;
typedef ap_uint<POSE_M0_BIT>                    m0_T;

typedef ap_int<POSE_PE1*POSE_SIMD1*POSE_W_BIT>  wgt1_pe_T;
typedef ap_int<POSE_PE2*POSE_SIMD2*POSE_W_BIT>  wgt2_pe_T;
typedef ap_int<POSE_PE3*POSE_SIMD3*POSE_W_BIT>  wgt3_pe_T;
typedef ap_int<POSE_PE1*POSE_BIAS_BIT>          bias1_pe_T;
typedef ap_int<POSE_PE2*POSE_BIAS_BIT>          bias2_pe_T;
typedef ap_int<POSE_PE3*POSE_BIAS_BIT>          bias3_pe_T;
typedef ap_uint<POSE_PE1*POSE_M0_BIT>           m0_1pe_T;
typedef ap_uint<POSE_PE2*POSE_M0_BIT>           m0_2pe_T;
typedef ap_uint<POSE_PE3*POSE_M0_BIT>           m0_3pe_T;



// ************************************************************************* //
// config
// ************************************************************************* //
#define BLOCK_NUMS 16
struct block
{
    char name[10];
    int ih, iw, ih3, iw3;
    int ic_nums1, ic_nums2, oc_nums3;
    int s, is_add, next_add;
};


struct parm
{
    char name[10];
    int w1, w2, w3;
    int b1, b2, b3;
    int m1, m2, m3;
};


static block config[BLOCK_NUMS] = {
        { "blk1",  128,96,64,48, 1,1,1,   2,0,1 },
        { "blk2",  64,48,64,48,  1,2,1,   1,1,0 },
        { "blk3",  64,48,32,24,  1,2,1,   2,0,1 },
        { "blk4",  32,24,32,24,  1,2,1,   1,1,1 },
        { "blk5",  32,24,32,24,  1,2,1,   1,1,0 },
        { "blk6",  32,24,16,12,  1,2,2,   2,0,1 },
        { "blk7",  16,12,16,12,  2,4,2,   1,1,1 },
        { "blk8",  16,12,16,12,  2,4,2,   1,1,1 },
        { "blk9",  16,12,16,12,  2,4,2,   1,1,0 },
        { "blk10", 16,12,16,12,  2,4,3,   1,0,1 },
        { "blk11", 16,12,16,12,  3,6,3,   1,1,1 },
        { "blk12", 16,12,16,12,  3,6,3,   1,1,0 },
        { "blk13", 16,12,8,6,    3,6,5,   2,0,1 },
        { "blk14", 8,6,8,6,      5,10,5,  1,1,1 },
        { "blk15", 8,6,8,6,      5,10,5,  1,1,0 },
        { "blk16", 8,6,8,6,      5,10,10, 1,0,0 }
};

static parm parm_size[BLOCK_NUMS] = {
        { "blk1",  0,48,75,           0,6,12,      0,3,6       },
        { "blk2",  123,219,273,       14,26,38,    7,13,19     },
        { "blk3",  369,465,519,       40,52,64,    20,26,32    },
        { "blk4",  615,711,765,       66,78,90,    33,39,45    },
        { "blk5",  861,957,1011,      92,104,116,  46,52,58    },
        { "blk6",  1107,1203,1257,    118,130,142, 59,65,71    },
        { "blk7",  1449,1833,1941,    146,170,194, 73,85,97    },
        { "blk8",  2325,2709,2817,    198,222,246, 99,111,123  },
        { "blk9",  3201,3585,3693,    250,274,298, 125,137,149 },
        { "blk10", 4077,4461,4569,    302,326,350, 151,163,175 },
        { "blk11", 5145,6009,6171,    356,392,428, 178,196,214 },
        { "blk12", 7035,7899,8061,    434,470,506, 217,235,253 },
        { "blk13", 8925,9789,9951,    512,548,584, 256,274,292 },
        { "blk14", 11391,13791,14061, 594,654,714, 297,327,357 },
        { "blk15", 16461,18861,19131, 724,784,844, 362,392,422 },
        { "blk16", 21531,23931,24201, 854,914,974, 427,457,487 }
};
