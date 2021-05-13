//
// Created by 90573 on 2021/4/12.
//

#include <ap_int.h>
#include <hls_stream.h>
#include "MatrixVectorActUnit.h"


template<
        unsigned IN_CH,//=16,
        unsigned IN_BIT,//=8,
        unsigned OUT_CH,//=16,
        unsigned OUT_BIT,//=8,
        unsigned W_BIT,//=8,
        unsigned MUL_BIT,//=32,
        unsigned BIAS_BIT,//=16,
        unsigned M0_BIT,//=16,
        unsigned K,//=3,
        unsigned SIMD,//=16,
        unsigned PE,//=16,
        unsigned RSHIFT,//=0,          //TODO:
        unsigned WGT_ARRAYSIZE,//=16,//TODO:
        unsigned BIAS_M0_ARRAYSIZE//=1//TODO:
        >
void DwConvLayer(
        stream<ap_uint<IN_CH*IN_BIT>> &in,
        stream<ap_uint<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S,
        const unsigned IN_CH_NUMS
        ) {
#pragma HLS DATAFLOW
    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
    Padding<IN_CH, IN_BIT, 1>(in, padding_out, IN_CH_NUMS, IN_ROW, IN_COL);

    stream<ap_uint<IN_CH*IN_BIT>> swu_out("swu_out");
    SWU<K,IN_BIT,IN_CH>(padding_out, swu_out, IN_CH_NUMS, INTER_ROW, INTER_COL, S);

    stream<ap_uint<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(swu_out, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_uint<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>  //TODO: for DEBUG test
            (adj_out, mvau_out, weights, bias, m0, IN_CH_NUMS*IN_CH*K*K, OUT_CH, IN_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}


template<
        unsigned IN_CH,//=16,
        unsigned IN_BIT,//=8,
        unsigned OUT_CH,//=16,
        unsigned OUT_BIT,//=8,
        unsigned W_BIT,//=8,
        unsigned MUL_BIT,//=32,
        unsigned BIAS_BIT,//=16,
        unsigned M0_BIT,//=16,
        unsigned K,//=1,
        unsigned SIMD,//=16,
        unsigned PE,//=16,
        unsigned RSHIFT,//=0,          //TODO:
        unsigned WGT_ARRAYSIZE,//=900,//TODO:
        unsigned BIAS_M0_ARRAYSIZE//=270//TODO:
>
void PwConvLayer(
        stream<ap_uint<IN_CH*IN_BIT>> &in,
        stream<ap_uint<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned IN_CH_NUMS
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_uint<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(in, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_uint<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE> //TODO: for DEBUG test
            (adj_out, mvau_out, weights, bias, m0, IN_CH*IN_CH_NUMS, OUT_CH*IN_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}
