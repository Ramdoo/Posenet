#include <hls_stream.h>

#include "ConvLayer.h"
#include "Load.h"
#include "Posenet.h"


using namespace std;
using namespace hls;



//480channels, 12 cols
void PosenetBlockAlpha(
        stream<infm_T> &in, stream<outfm_T> &out, stream<addfm_T> &add_fm,
        const wgt_T wgt1[POSE_PE][WGT_PW_SIZE_ALPHA1], const wgt_T wgt2[WGT_DW_SIZE_ALPHA2], const wgt_T wgt3[POSE_PE][WGT_PW_SIZE_ALPHA3],
        const bias_T bias1[POSE_PE][BIAS_M0_SIZE_ALPHA], const bias_T bias2[POSE_PE][BIAS_M0_SIZE_ALPHA], const bias_T bias3[POSE_PE][BIAS_M0_SIZE_ALPHA],
        const m0_T m0_1[POSE_PE][BIAS_M0_SIZE_ALPHA], const m0_T m0_2[POSE_PE][BIAS_M0_SIZE_ALPHA], const m0_T m0_3[POSE_PE][BIAS_M0_SIZE_ALPHA],
        const unsigned ROW1, const unsigned ROW2, const unsigned ROW3, const unsigned COL1, const unsigned COL2, const unsigned COL3,
        const unsigned CH_NUMS1, const unsigned CH_NUMS2, const unsigned  CH_NUMS3, const unsigned STRIDE, const unsigned IS_ADD
) {
#pragma HLS DATAFLOW

    stream<innerfm_T> pw1_out("pw1_out");
#pragma HLS STREAM variable=pw1_out depth=128 dim=1

    PwConvLayer<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD,POSE_PE,0,WGT_PW_SIZE_ALPHA1,BIAS_M0_SIZE_ALPHA>
            (in, pw1_out, wgt1, bias1, m0_1, ROW1, COL1, CH_NUMS1);

    stream<innerfm_T> dw2_out("dw2_out");
#pragma HLS STREAM variable=dw2_out depth=128 dim=1

    DwConvLayerAlpha<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_SIMD,POSE_PE,0,WGT_DW_SIZE_ALPHA2,BIAS_M0_SIZE_ALPHA>
            (pw1_out, dw2_out, wgt2, bias2, m0_2, ROW2, COL2, STRIDE, CH_NUMS2);

    PwConvAddLayer<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD,POSE_PE,0,WGT_PW_SIZE_ALPHA3,BIAS_M0_SIZE_ALPHA>
            (dw2_out, out, add_fm, wgt3, bias3, m0_3, ROW3, COL3, CH_NUMS3, IS_ADD);

}



// 128 channels, 96 cols
void PosenetBlockBeta(
        stream<infm_T> &in, stream<outfm_T> &out, stream<addfm_T> &add_fm,
        const wgt_T wgt1[POSE_PE][WGT_PW_SIZE_BETA], const wgt_T wgt2[WGT_DW_SIZE_BETA], const wgt_T wgt3[POSE_PE][WGT_PW_SIZE_BETA],
        const bias_T bias1[POSE_PE][BIAS_M0_SIZE_BETA], const bias_T bias2[POSE_PE][BIAS_M0_SIZE_BETA], const bias_T bias3[POSE_PE][BIAS_M0_SIZE_BETA],
        const m0_T m0_1[POSE_PE][BIAS_M0_SIZE_BETA], const m0_T m0_2[POSE_PE][BIAS_M0_SIZE_BETA], const m0_T m0_3[POSE_PE][BIAS_M0_SIZE_BETA],
        const unsigned ROW1, const unsigned ROW2, const unsigned ROW3, const unsigned COL1, const unsigned COL2, const unsigned COL3,
        const unsigned CH_NUMS1, const unsigned CH_NUMS2, const unsigned  CH_NUMS3, const unsigned STRIDE, const unsigned IS_ADD, const unsigned IS_DECONV
) {
#pragma HLS DATAFLOW

    stream<innerfm_T> pw1_out("pw1_out");
#pragma HLS STREAM variable=pw1_out depth=128 dim=1

    PwConvLayer<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD,POSE_PE,0,WGT_PW_SIZE_BETA,BIAS_M0_SIZE_BETA>
            (in, pw1_out, wgt1, bias1, m0_1, ROW1, COL1, CH_NUMS1);

    stream<innerfm_T> dw2_out("dw2_out");
#pragma HLS STREAM variable=dw2_out depth=128 dim=1

    DwConvDeConvLayerBeta<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_SIMD,POSE_PE,0,WGT_DW_SIZE_BETA,BIAS_M0_SIZE_BETA>
            (pw1_out, dw2_out, wgt2, bias2, m0_2, ROW2, COL2, STRIDE, CH_NUMS2, IS_DECONV);

    PwConvAddLayer<POSE_IN_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD,POSE_PE,0,WGT_PW_SIZE_BETA,BIAS_M0_SIZE_BETA>
            (dw2_out, out, add_fm, wgt3, bias3, m0_3, ROW3, COL3, CH_NUMS3, IS_ADD);

}


void PosenetAlpha(
        stream<infm_T> &in, stream<outfm_T> &out, stream<addfm_T> &add_fm,
        wgt_T* wgt1, wgt_T* wgt2, wgt_T* wgt3,
        bias_T* bias1, bias_T* bias2, bias_T* bias3,
        m0_T* m0_1, m0_T* m0_2, m0_T* m0_3,
        const unsigned ROW1, const unsigned ROW2, const unsigned ROW3, const unsigned COL1, const unsigned COL2, const unsigned COL3,
        const unsigned CH_NUMS1, const unsigned CH_NUMS2, const unsigned  CH_NUMS3, const unsigned STRIDE, const unsigned IS_ADD,
        const unsigned PINGPONG
        ) {
#pragma HLS INTERFACE m_axi depth=2400  port=wgt1 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=270  port=wgt2 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=4800  port=wgt3 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=480  port=bias1 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=480  port=bias2 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=480  port=bias3 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=480  port=m0_1 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=480  port=m0_2 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=480  port=m0_3 offset=slave bundle=bm

#pragma HLS stream variable=add_fm depth=1024 dim=1

    wgt_T wgt1_buf_alpha_ping[POSE_PE][WGT_PW_SIZE_ALPHA1];
    wgt_T wgt2_buf_alpha_ping[WGT_DW_SIZE_ALPHA2];
    wgt_T wgt3_buf_alpha_ping[POSE_PE][WGT_PW_SIZE_ALPHA3];

    bias_T bias1_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];
    bias_T bias2_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];
    bias_T bias3_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];

    m0_T m0_1_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];
    m0_T m0_2_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];
    m0_T m0_3_buf_alpha_ping[POSE_PE][BIAS_M0_SIZE_ALPHA];

#pragma HLS ARRAY_PARTITION variable=wgt1_buf_alpha_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=wgt3_buf_alpha_ping complete dim=1

#pragma HLS ARRAY_PARTITION variable=bias1_buf_alpha_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_buf_alpha_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_buf_alpha_ping complete dim=1

#pragma HLS ARRAY_PARTITION variable=m0_1_buf_alpha_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_buf_alpha_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_buf_alpha_ping complete dim=1

    wgt_T wgt1_buf_alpha_pong[POSE_PE][WGT_PW_SIZE_ALPHA1];
    wgt_T wgt2_buf_alpha_pong[WGT_DW_SIZE_ALPHA2];
    wgt_T wgt3_buf_alpha_pong[POSE_PE][WGT_PW_SIZE_ALPHA3];

    bias_T bias1_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];
    bias_T bias2_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];
    bias_T bias3_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];

    m0_T m0_1_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];
    m0_T m0_2_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];
    m0_T m0_3_buf_alpha_pong[POSE_PE][BIAS_M0_SIZE_ALPHA];

#pragma HLS ARRAY_PARTITION variable=wgt1_buf_alpha_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=wgt3_buf_alpha_pong complete dim=1

#pragma HLS ARRAY_PARTITION variable=bias1_buf_alpha_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_buf_alpha_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_buf_alpha_pong complete dim=1

#pragma HLS ARRAY_PARTITION variable=m0_1_buf_alpha_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_buf_alpha_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_buf_alpha_pong complete dim=1

    //Ping - 1 , Pong - 0
    if (PINGPONG) {
        LoadPwcvWgtAlpha1(wgt1, wgt1_buf_alpha_pong);
        LoadDwcvWgtAlpha2(wgt2, wgt2_buf_alpha_pong);
        LoadPwcvWgtAlpha3(wgt3, wgt3_buf_alpha_pong);
        LoadBiasAlpha(bias1, bias1_buf_alpha_pong);
        LoadBiasAlpha(bias2, bias2_buf_alpha_pong);
        LoadBiasAlpha(bias3, bias3_buf_alpha_pong);
        LoadM0Alpha(m0_1, m0_1_buf_alpha_pong);
        LoadM0Alpha(m0_2, m0_2_buf_alpha_pong);
        LoadM0Alpha(m0_3, m0_3_buf_alpha_pong);
        PosenetBlockAlpha(in, out, add_fm, wgt1_buf_alpha_ping, wgt2_buf_alpha_ping, wgt3_buf_alpha_ping,
                          bias1_buf_alpha_ping, bias2_buf_alpha_ping, bias3_buf_alpha_ping,
                          m0_1_buf_alpha_ping, m0_2_buf_alpha_ping, m0_3_buf_alpha_ping,
                          ROW1, ROW2, ROW3, COL1, COL2, COL3, CH_NUMS1, CH_NUMS2, CH_NUMS3, STRIDE, IS_ADD);
    } else {
        LoadPwcvWgtAlpha1(wgt1, wgt1_buf_alpha_ping);
        LoadDwcvWgtAlpha2(wgt2, wgt2_buf_alpha_ping);
        LoadPwcvWgtAlpha3(wgt3, wgt3_buf_alpha_ping);
        LoadBiasAlpha(bias1, bias1_buf_alpha_ping);
        LoadBiasAlpha(bias2, bias2_buf_alpha_ping);
        LoadBiasAlpha(bias3, bias3_buf_alpha_ping);
        LoadM0Alpha(m0_1, m0_1_buf_alpha_ping);
        LoadM0Alpha(m0_2, m0_2_buf_alpha_ping);
        LoadM0Alpha(m0_3, m0_3_buf_alpha_ping);
        PosenetBlockAlpha(in, out, add_fm, wgt1_buf_alpha_pong, wgt2_buf_alpha_pong, wgt3_buf_alpha_pong,
                          bias1_buf_alpha_pong, bias2_buf_alpha_pong, bias3_buf_alpha_pong,
                          m0_1_buf_alpha_pong, m0_2_buf_alpha_pong, m0_3_buf_alpha_pong,
                          ROW1, ROW2, ROW3, COL1, COL2, COL3, CH_NUMS1, CH_NUMS2, CH_NUMS3, STRIDE, IS_ADD);
    }


}



void PosenetBeta(
        stream<infm_T> &in, stream<outfm_T> &out, stream<addfm_T> &add_fm,
        wgt_T* wgt1, wgt_T* wgt2, wgt_T* wgt3,
        bias_T* bias1, bias_T* bias2, bias_T* bias3,
        m0_T* m0_1, m0_T* m0_2, m0_T* m0_3,
        const unsigned ROW1, const unsigned ROW2, const unsigned ROW3, const unsigned COL1, const unsigned COL2, const unsigned COL3,
        const unsigned CH_NUMS1, const unsigned CH_NUMS2, const unsigned  CH_NUMS3, const unsigned STRIDE, const unsigned IS_ADD, const unsigned IS_DECONV,
        const unsigned PINGPONG
) {
#pragma HLS INTERFACE m_axi depth=1024  port=wgt1 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=72  port=wgt2 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=1024  port=wgt3 offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=128  port=bias1 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=128  port=bias2 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=128  port=bias3 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=128  port=m0_1 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=128  port=m0_2 offset=slave bundle=bm
#pragma HLS INTERFACE m_axi depth=128  port=m0_3 offset=slave bundle=bm

#pragma HLS stream variable=add_fm depth=1024 dim=1

    wgt_T wgt1_buf_beta_ping[POSE_PE][WGT_PW_SIZE_BETA];
    wgt_T wgt2_buf_beta_ping[WGT_DW_SIZE_BETA];
    wgt_T wgt3_buf_beta_ping[POSE_PE][WGT_PW_SIZE_BETA];

    bias_T bias1_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];
    bias_T bias2_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];
    bias_T bias3_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];

    m0_T m0_1_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];
    m0_T m0_2_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];
    m0_T m0_3_buf_beta_ping[POSE_PE][BIAS_M0_SIZE_BETA];

#pragma HLS ARRAY_PARTITION variable=wgt1_buf_beta_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=wgt3_buf_beta_ping complete dim=1

#pragma HLS ARRAY_PARTITION variable=bias1_buf_beta_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_buf_beta_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_buf_beta_ping complete dim=1

#pragma HLS ARRAY_PARTITION variable=m0_1_buf_beta_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_buf_beta_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_buf_beta_ping complete dim=1

    wgt_T wgt1_buf_beta_pong[POSE_PE][WGT_PW_SIZE_BETA];
    wgt_T wgt2_buf_beta_pong[WGT_DW_SIZE_BETA];
    wgt_T wgt3_buf_beta_pong[POSE_PE][WGT_PW_SIZE_BETA];

    bias_T bias1_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];
    bias_T bias2_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];
    bias_T bias3_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];

    m0_T m0_1_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];
    m0_T m0_2_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];
    m0_T m0_3_buf_beta_pong[POSE_PE][BIAS_M0_SIZE_BETA];

#pragma HLS ARRAY_PARTITION variable=wgt1_buf_beta_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=wgt3_buf_beta_pong complete dim=1

#pragma HLS ARRAY_PARTITION variable=bias1_buf_beta_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_buf_beta_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_buf_beta_pong complete dim=1

#pragma HLS ARRAY_PARTITION variable=m0_1_buf_beta_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_buf_beta_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_buf_beta_pong complete dim=1

    if (PINGPONG) {
        LoadPwcvWgtBeta1(wgt1, wgt1_buf_beta_pong);
        LoadDwcvWgtBeta2(wgt2, wgt2_buf_beta_pong);
        LoadPwcvWgtBeta3(wgt3, wgt3_buf_beta_pong);
        LoadBiasBeta(bias1, bias1_buf_beta_pong);
        LoadBiasBeta(bias2, bias2_buf_beta_pong);
        LoadBiasBeta(bias3, bias3_buf_beta_pong);
        LoadM0Beta(m0_1, m0_1_buf_beta_pong);
        LoadM0Beta(m0_2, m0_2_buf_beta_pong);
        LoadM0Beta(m0_3, m0_3_buf_beta_pong);

        PosenetBlockBeta(in, out, add_fm, wgt1_buf_beta_ping, wgt2_buf_beta_ping, wgt3_buf_beta_ping,
                         bias1_buf_beta_ping, bias2_buf_beta_ping, bias3_buf_beta_ping,
                         m0_1_buf_beta_ping, m0_2_buf_beta_ping, m0_3_buf_beta_ping,
                         ROW1, ROW2, ROW3, COL1, COL2, COL3, CH_NUMS1, CH_NUMS2, CH_NUMS3, STRIDE, IS_ADD, IS_DECONV);

    } else {
        LoadPwcvWgtBeta1(wgt1, wgt1_buf_beta_ping);
        LoadDwcvWgtBeta2(wgt2, wgt2_buf_beta_ping);
        LoadPwcvWgtBeta3(wgt3, wgt3_buf_beta_ping);
        LoadBiasBeta(bias1, bias1_buf_beta_ping);
        LoadBiasBeta(bias2, bias2_buf_beta_ping);
        LoadBiasBeta(bias3, bias3_buf_beta_ping);
        LoadM0Beta(m0_1, m0_1_buf_beta_ping);
        LoadM0Beta(m0_2, m0_2_buf_beta_ping);
        LoadM0Beta(m0_3, m0_3_buf_beta_ping);

        PosenetBlockBeta(in, out, add_fm, wgt1_buf_beta_pong, wgt2_buf_beta_pong, wgt3_buf_beta_pong,
                         bias1_buf_beta_pong, bias2_buf_beta_pong, bias3_buf_beta_pong,
                         m0_1_buf_beta_pong, m0_2_buf_beta_pong, m0_3_buf_beta_pong,
                         ROW1, ROW2, ROW3, COL1, COL2, COL3, CH_NUMS1, CH_NUMS2, CH_NUMS3, STRIDE, IS_ADD, IS_DECONV);
    }

}


void Top(
        stream<infm_T> &in, stream<outfm_T> &out, stream<addfm_T> &add_fm,
        wgt_T* wgt1_alpha, wgt_T* wgt2_alpha, wgt_T* wgt3_alpha,
        bias_T* bias1_alpha, bias_T* bias2_alpha, bias_T* bias3_alpha,
        m0_T* m0_1_alpha, m0_T* m0_2_alpha, m0_T* m0_3_alpha,
        const unsigned ROW1_ALPHA, const unsigned ROW2_ALPHA, const unsigned ROW3_ALPHA, const unsigned COL1_ALPHA, const unsigned COL2_ALPHA, const unsigned COL3_ALPHA,
        const unsigned CH_NUMS1_ALPHA, const unsigned CH_NUMS2_ALPHA, const unsigned  CH_NUMS3_ALPHA, const unsigned STRIDE_ALPHA, const unsigned IS_ADD_ALPHA, const unsigned PingPongAlpha,

        stream<infm_T> &in1, stream<outfm_T> &out1, stream<addfm_T> &add_fm1,
        wgt_T* wgt1_1, wgt_T* wgt2_1, wgt_T* wgt3_1,
        bias_T* bias1_1, bias_T* bias2_1, bias_T* bias3_1,
        m0_T* m0_1_1, m0_T* m0_2_1, m0_T* m0_3_1,
        const unsigned ROW1_BETA, const unsigned ROW2_BETA, const unsigned ROW3_BETA, const unsigned COL1_BETA, const unsigned COL2_BETA, const unsigned COL3_BETA,
        const unsigned CH_NUMS1_BETA, const unsigned CH_NUMS2_BETA, const unsigned  CH_NUMS3_BETA, const unsigned STRIDE_BETA, const unsigned IS_ADD_BETA, const unsigned IS_DECONV_BETA, const unsigned PingPongBeta
        ) {


//480 channels, 12 cols
    PosenetAlpha(in, out, add_fm, wgt1_alpha, wgt2_alpha, wgt3_alpha, bias1_alpha, bias2_alpha, bias3_alpha, m0_1_alpha, m0_2_alpha, m0_3_alpha, ROW1_ALPHA, ROW2_ALPHA, ROW3_ALPHA, COL1_ALPHA, COL2_ALPHA,
                 COL3_ALPHA, CH_NUMS1_ALPHA, CH_NUMS2_ALPHA, CH_NUMS3_ALPHA, STRIDE_ALPHA, IS_ADD_ALPHA, PingPongAlpha);

//128 channels, 96 cols
    PosenetBeta(in1, out1, add_fm1, wgt1_1, wgt2_1, wgt3_1, bias1_1, bias2_1, bias3_1, m0_1_1, m0_2_1, m0_3_1, ROW1_BETA,
                ROW2_BETA, ROW3_BETA, COL1_BETA, COL2_BETA, COL3_BETA, CH_NUMS1_BETA, CH_NUMS2_BETA, CH_NUMS3_BETA, STRIDE_BETA, IS_ADD_BETA, IS_DECONV_BETA, PingPongBeta);

}
