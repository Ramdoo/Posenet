#include <ap_int.h>
#include <hls_stream.h>
#include "Padding.h"
#include "SlidingWindowUnit.h"
#include "StreamTools.h"
#include "MatrixVectorActUnit.h"



template<
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned OUT_CH,
        unsigned OUT_BIT,
        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,
        unsigned K,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
        >
void DwConvLayerAlpha(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S,
        const unsigned IN_CH_NUMS
        ) {
#pragma HLS DATAFLOW
    unsigned INTER_ROW = IN_ROW + 2;
    unsigned INTER_COL = IN_COL + 2;

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
    Padding<IN_CH, IN_BIT, 1>(in, padding_out, IN_CH_NUMS, IN_ROW, IN_COL);

    stream<ap_int<IN_CH*IN_BIT>> swu_out("swu_out");
    SWU<K,IN_BIT,IN_CH>(padding_out, swu_out, IN_CH_NUMS, INTER_ROW, INTER_COL, S);

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(swu_out, adj_out, K*K*INTER_ROW*INTER_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>
            (adj_out, mvau_out, weights, bias, m0, IN_CH_NUMS*IN_CH*K*K, OUT_CH, IN_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}




template<
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned OUT_CH,
        unsigned OUT_BIT,
        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,
        unsigned K,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void DwConvDeConvLayerBeta(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S,
        const unsigned IN_CH_NUMS,
        const unsigned IS_DECONV
) {
#pragma HLS DATAFLOW
    unsigned INTER_ROW = IN_ROW + 2;
    unsigned INTER_COL = IN_COL + 2;

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
    if (IS_DECONV) {
        stream<ap_int<IN_CH*IN_BIT> > deconvpad_out("deconvpad_out");
#pragma HLS stream variable=deconvpad_out depth=16 dim=1
        INTER_ROW += IN_ROW;
        INTER_COL += IN_COL;
        DilationPadding<IN_CH, IN_BIT, 1>(in, deconvpad_out, IN_CH_NUMS, IN_ROW, IN_COL);
        Padding<IN_CH, IN_BIT, 1>(deconvpad_out, padding_out, IN_CH_NUMS, 2*IN_ROW, 2*IN_COL);
    } else {
        Padding<IN_CH, IN_BIT, 1>(in, padding_out, IN_CH_NUMS, IN_ROW, IN_COL);
    }

    stream<ap_int<IN_CH*IN_BIT>> swu_out("swu_out");
    SWUBeta<K,IN_BIT,IN_CH>(padding_out, swu_out, IN_CH_NUMS, INTER_ROW, INTER_COL, S);

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(swu_out, adj_out, K*K*INTER_ROW*INTER_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>
            (adj_out, mvau_out, weights, bias, m0, IN_CH_NUMS*IN_CH*K*K, OUT_CH, IN_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}



template<
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned OUT_CH,
        unsigned OUT_BIT,
        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,
        unsigned K,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwConvLayer(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
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

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(in, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>
            (adj_out, mvau_out, weights, bias, m0, IN_CH*IN_CH_NUMS, OUT_CH*IN_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}



template<
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned OUT_CH,
        unsigned OUT_BIT,
        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,
        unsigned K,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwConvAddLayer(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        stream<ap_int<PE*OUT_BIT>> &add_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned IN_CH_NUMS,
        const ap_uint<1> IS_ADD
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(in, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvAddMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>
            (adj_out, mvau_out, add_fm, weights, bias, m0, IN_CH*IN_CH_NUMS, OUT_CH*IN_CH_NUMS, OUT_ROW*OUT_COL, IS_ADD);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS);
}
