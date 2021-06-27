#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include "Padding.h"
#include "SlidingWindowUnit.h"
#include "StreamTools.h"
#include "MatrixVectorActUnit.h"



//通用的函数， 将尺寸大小和通道数作为入参传入
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
        unsigned RSHIFT           //TODO:
        >
void DwConvActLayerAlpha(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        stream<ap_int<SIMD*W_BIT>> &weights,
        stream<ap_int<PE*BIAS_BIT>> &bias,
        stream<ap_uint<PE*M0_BIT>> &m0,
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S,
        const unsigned IN_CH_NUMS
        ) {
#pragma HLS DATAFLOW
    unsigned INTER_ROW = IN_ROW + 2;
    unsigned INTER_COL = IN_COL + 2;

    const unsigned OUT_ROW = (INTER_ROW-K)/S+1;
    const unsigned OUT_COL = (INTER_COL-K)/S+1;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_SRL
    Padding<IN_CH, IN_BIT, 1>(in, padding_out, IN_CH_NUMS, IN_ROW, IN_COL);
    //cout << dec << "padding_out size: " << padding_out.size() << endl;

    stream<ap_int<IN_CH*IN_BIT>> swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_SRL
    SWU<K,IN_BIT,IN_CH>(padding_out, swu_out, IN_CH_NUMS, INTER_ROW, INTER_COL, S);
#if 0
    cout << dec << "swu_out size: " << swu_out.size() << endl;
    const unsigned STEPS = ((INTER_COL - K) / S + 1) * ((INTER_ROW - K) / S + 1);
    ofstream fpblk1cv2swu("..\\Test\\blk1cv2swu.txt", ios::out);
    if (!fpblk1cv2swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS*IN_CH_NUMS; step++) {
        //cout << dec << "------ step: " << step << "------" << endl;
        fpblk1cv2swu << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                fpblk1cv2swu << "[";
                //cout /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                cout << hex;
                fpblk1cv2swu /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                //cout << "] ";
                fpblk1cv2swu << "] ";
            }
            //cout << endl;
            fpblk1cv2swu << endl;
        }
        //cout << "-------------------------" << endl;
        fpblk1cv2swu << "-------------------------" << endl;
    }
    fpblk1cv2swu.close();
#endif

    //stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    //StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(swu_out, adj_out, K*K*INTER_ROW*INTER_COL*IN_CH_NUMS, IN_CH_NUMS);

    //stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");

    DwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT>
            (swu_out, out, weights, bias, m0, IN_CH_NUMS*IN_CH*K*K, OUT_CH, IN_CH_NUMS, OUT_ROW*OUT_COL);

    //StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*IN_CH_NUMS, IN_CH_NUMS*IN_CH/OUT_CH);
}



//通用的函数， 将尺寸大小和通道数作为入参传入
//在通用的block中，作为前面的pwcv， 不处理shortcut
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
        unsigned RSHIFT           //TODO:
>
void PwConvActLayer(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        stream<ap_int<PE*SIMD*W_BIT>> &weights,
        stream<ap_int<PE*BIAS_BIT>> &bias,
        stream<ap_uint<PE*M0_BIT>> &m0,
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned IN_CH_NUMS,
        const unsigned OUT_CH_NUMS
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(in, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT>
            (adj_out, mvau_out, weights, bias, m0, IN_CH*IN_CH_NUMS, OUT_CH*OUT_CH_NUMS, OUT_ROW*OUT_COL);

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*OUT_CH_NUMS, IN_CH_NUMS);
}



//通用的函数， 将尺寸大小和通道数作为入参传入
//在通用的block中，作为后面的pwcv， 需处理是否有shortcut， 有则add
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
        unsigned RSHIFT           //TODO:
>
void PwConvAddLayer(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        stream<ap_int<PE*OUT_BIT>> &add_in,
#ifdef DEBUG
        stream<ap_int<PE*OUT_BIT>> &add_out,
#endif
        stream<ap_int<PE*SIMD*W_BIT>> &weights,
        stream<ap_int<PE*BIAS_BIT>> &bias,
        stream<ap_uint<PE*M0_BIT>> &m0,
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned IN_CH_NUMS,
        const unsigned OUT_CH_NUMS,
        const ap_uint<1> IS_ADD
#ifdef DEBUG
        ,
        const ap_uint<1> NEXT_ADD
#endif
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT>(in, adj_out, K*K*IN_ROW*IN_COL*IN_CH_NUMS, IN_CH_NUMS);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvAddMatrixVectorUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT>
            (adj_out, mvau_out, add_in,
#ifdef DEBUG
             add_out,
#endif
             weights, bias, m0, IN_CH*IN_CH_NUMS, OUT_CH*OUT_CH_NUMS, OUT_ROW*OUT_COL, IS_ADD
#ifdef DEBUG
             , NEXT_ADD
#endif
             );

    StreamingDataWidthConverter_Batch<PE*IN_BIT, OUT_CH*IN_BIT>(mvau_out, out, OUT_ROW*OUT_COL*OUT_CH_NUMS, OUT_CH_NUMS);

}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参数
template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned IN_CH_NUMS,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned K,
        unsigned SIMD,
        unsigned LOG2_SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void DwConvLayerT(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_SRL
    PaddingT<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out);
    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;

    //stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    //StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, INTER_ROW*INTER_COL>(padding_out, adj_out);

    stream<ap_int<SIMD*IN_BIT>> swu_out("swu_out");
    SWUT<K,INTER_ROW, INTER_COL,IN_BIT,IN_CH,IN_CH/SIMD,SIMD,LOG2_SIMD,1>(padding_out, swu_out);
    //cout << dec << "swu_out size: " << swu_out.size() << endl;
#if 0
    const unsigned STEPS = ((INTER_COL - K) / 1 + 1) * ((INTER_ROW - K) / 1 + 1);
    ofstream fpconv1swu("..\\Test\\hconv1swu.txt", ios::out);
    if (!fpconv1swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS; step++) {
        //cout << dec << "------ step: " << step << "------" << endl;
        fpconv1swu << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                fpconv1swu << "[";
                //cout /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                cout << hex;
                fpconv1swu /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                //cout << "] ";
                fpconv1swu << "] ";
            }
            //cout << endl;
            fpconv1swu << endl;
        }
        //cout << "-------------------------" << endl;
        fpconv1swu << "-------------------------" << endl;
    }
    fpconv1swu.close();
#endif

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnitT<IN_CH_NUMS*SIMD*K*K, OUT_CH, IN_BIT, IN_CH_NUMS, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
            RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
            (swu_out, mvau_out, weights, bias, m0);

    StreamingDataWidthConverter_BatchT<PE*IN_BIT, OUT_CH*IN_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(mvau_out, out);
}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参数
template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned IN_CH_NUMS,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned K,
        unsigned SIMD,
        unsigned LOG2_SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void DeConvLayerT(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW+IN_ROW;
    const unsigned OUT_COL = IN_COL+IN_COL;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_SRL
    stream<ap_int<IN_CH*IN_BIT> > deconvpad_out("deconvpad_out");
#pragma HLS RESOURCE variable=deconvpad_out core=FIFO_SRL
    DilationPaddingT<IN_ROW, IN_COL, IN_CH, IN_BIT>(in, deconvpad_out);
#if 0
    cout << dec << "depadding_out size: " << deconvpad_out.size() << endl;
    ofstream fpdecvde1_depad("..\\Test\\decv_de1_depad.txt", ios::out);
    if (!fpdecvde1_depad)
        cout << "no such file" << endl;
    for (int h = 0; h < 2*IN_ROW; ++h) {
        for (int w = 0; w < 2*IN_COL; ++w) {
            ap_int<IN_CH * POSE_IN_BIT> temp = deconvpad_out.read();
            for (int ch = 0; ch < IN_CH; ++ch) {
                cout << dec;
                fpdecvde1_depad << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvde1_depad << endl;
        }
    }
    fpdecvde1_depad.close();
#endif
    PaddingT<IN_ROW+IN_ROW, IN_COL+IN_COL, IN_CH, IN_BIT, 1>(deconvpad_out, padding_out);
    const unsigned INTER_ROW = IN_ROW+IN_ROW + 2;
    const unsigned INTER_COL = IN_COL+IN_COL + 2;

#if 0
    cout << dec << "padding_out size: " << padding_out.size() << endl;
    ofstream fpdecvde1_pad("..\\Test\\decv_de1_pad.txt", ios::out);
    if (!fpdecvde1_pad)
        cout << "no such file" << endl;
    for (int h = 0; h < INTER_ROW; ++h) {
        for (int w = 0; w < INTER_COL; ++w) {
            ap_int<IN_CH * POSE_IN_BIT> temp = padding_out.read();
            for (int ch = 0; ch < IN_CH; ++ch) {
                cout << dec;
                fpdecvde1_pad << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvde1_pad << endl;
        }
    }
    fpdecvde1_pad.close();
#endif

    //stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    //StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, INTER_ROW*INTER_COL>(padding_out, adj_out);

    stream<ap_int<SIMD*IN_BIT>> swu_out("swu_out");
    SWUT<K,INTER_ROW, INTER_COL,IN_BIT,IN_CH,IN_CH/SIMD,SIMD,LOG2_SIMD,1>(padding_out, swu_out);
#if 0
    const unsigned STEPS = ((INTER_COL - K) / 1 + 1) * ((INTER_ROW - K) / 1 + 1);
    ofstream fpde1swu("..\\Test\\de1swu.txt", ios::out);
    if (!fpde1swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS*IN_CH/SIMD; step++) {
        //cout << dec << "------ step: " << step << "------" << endl;
        fpde1swu << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                fpde1swu << "[";
                //cout /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                cout << hex;
                fpde1swu /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                //cout << "] ";
                fpde1swu << "] ";
            }
            //cout << endl;
            fpde1swu << endl;
        }
        //cout << "-------------------------" << endl;
        fpde1swu << "-------------------------" << endl;
    }
    fpde1swu.close();
#endif

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnitT<IN_CH_NUMS*SIMD*K*K, OUT_CH, IN_BIT, IN_CH_NUMS, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
                                RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
                            (swu_out, mvau_out, weights, bias, m0);

    StreamingDataWidthConverter_BatchT<PE*IN_BIT, OUT_CH*IN_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(mvau_out, out);
}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参数
template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwConvLayer3(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, IN_ROW*IN_COL>(in, adj_out);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvMatrixVectorUnitT<IN_CH, OUT_CH, IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
                                RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
                            (adj_out, mvau_out, weights, bias, m0);

    StreamingDataWidthConverter_BatchT<PE*IN_BIT, OUT_CH*IN_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(mvau_out, out);
}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参数
template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwConvActLayerT(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, IN_ROW*IN_COL>(in, adj_out);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    PwcvMatrixVectorActUnitT<IN_CH, OUT_CH, IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
            RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
            (adj_out, mvau_out, weights, bias, m0);

    StreamingDataWidthConverter_BatchT<PE*IN_BIT, OUT_CH*IN_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(mvau_out, out);
}

template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned K,
        unsigned S,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void ConvLayerT(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW


    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_SRL
    PaddingT<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out);
    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;
    const unsigned OUT_ROW = (INTER_ROW - K) / S + 1;
    const unsigned OUT_COL = (INTER_COL - K) / S + 1;

    stream<ap_int<IN_CH*IN_BIT>> swu_out("swu_out");
    SWUCvT<K,INTER_ROW, INTER_COL,IN_BIT,IN_CH,S>(padding_out, swu_out);
    //cout << dec << "swu_out size: " << swu_out.size() << endl;
#if 0
    const unsigned STEPS = ((INTER_COL - K) / S + 1) * ((INTER_ROW - K) / S + 1);
    ofstream fpconv0swu("..\\Test\\hconv0swu.txt", ios::out);
    if (!fpconv0swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS; step++) {
        //cout << dec << "------ step: " << step << "------" << endl;
        fpconv0swu << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                fpconv0swu << "[";
                //cout /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                cout << hex;
                fpconv0swu /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                //cout << "] ";
                fpconv0swu << "] ";
            }
            //cout << endl;
            fpconv0swu << endl;
        }
        //cout << "-------------------------" << endl;
        fpconv0swu << "-------------------------" << endl;
    }
    fpconv0swu.close();
#endif

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, K*K*OUT_ROW*OUT_COL>(swu_out, adj_out);

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    MatrixVectorActUnitT<IN_CH*K*K, OUT_CH, IN_BIT,  OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
            RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
            (adj_out, mvau_out, weights, bias, m0);

    StreamingDataWidthConverter_BatchT<PE*IN_BIT, OUT_CH*IN_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(mvau_out, out);
}

template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,

        unsigned OUT_CH,
        unsigned OUT_BIT,

        unsigned W_BIT,
        unsigned MUL_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned K,
        unsigned S,
        unsigned SIMD,
        unsigned PE,
        unsigned RSHIFT,           //TODO:
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void LastConvLayerT(
        stream<ap_int<IN_CH*IN_BIT>> &in,
        stream<ap_int<OUT_CH*OUT_BIT>> &out,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    stream<ap_int<SIMD*IN_BIT>> adj_out("adj_out");
    StreamingDataWidthConverter_BatchT<IN_CH*IN_BIT, SIMD*IN_BIT, K*K*OUT_ROW*OUT_COL>(in, adj_out);

    LastPwcvMatrixVectorUnitT<IN_CH*K*K, OUT_CH, IN_BIT,  OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
            RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
            (adj_out, out, weights, bias, m0);

}
