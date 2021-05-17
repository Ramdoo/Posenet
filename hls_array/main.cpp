#include <iostream>

#include <hls_stream.h>
#include <ap_int.h>
#include <fstream>
#include "Padding.h"
#include "SlidingWindowUnit.h"
#include "StreamTools.h"
#include "MatrixVectorActUnit.h"
#include "ConvLayer.h"
#include "GenParam.h"//for DEBUG test
#include "DebugParams.h"//for DEBUG test

#include "Posenet.h"


#define DwCv 1
#define PwCv 0
#define Dilation 0

using namespace std;
using namespace hls;

int main() {
    //GenParamW<1,4,4,8,WGT_PWCV7_SIZE>("..\\Test\\testw.txt");
    //GenParamW<16,1,8,WGT_DECV1_SIZE>("..\\Test\\testw.txt");
    //GenParamB<16,16,BIAS_M0_DECV1_SIZE>("..\\Test\\testb.txt");


    const unsigned CONV0_IN_ROW = 3;
    const unsigned CONV0_IN_COL = 5;

    const unsigned CONV0_P = 1;
    const unsigned CONV0_S = 1;
#if PwCv
    const unsigned K = 1;
#endif
#if DwCv
    const unsigned K = 3;
#endif
#if PwCv
    const unsigned CONV0_SIMD = 16;
#endif
#if DwCv
    const unsigned CONV0_SIMD = 16;
#endif
    const unsigned CONV0_PE = 16;

    const unsigned in_ch_nums = 2;

    hls::stream<infm_T> testin("input stream");


    ap_int<POSE_IN_CH*POSE_IN_BIT> count = 1;
    for (int h = 0; h < CONV0_IN_ROW; ++h) {
        for (int w = 0; w < CONV0_IN_COL ; ++w) {
            cout << "[ ";
            for (int nums = 0; nums < in_ch_nums; ++nums) {
                ap_int<POSE_IN_CH*POSE_IN_BIT> tmp_in;
                for (int c = 0; c < POSE_IN_CH; ++c) {
                    tmp_in = tmp_in << POSE_IN_BIT;
                    tmp_in |= count;
                    count += 1;
                }
                cout << hex << "tmp_in:" << tmp_in << " " ;
                testin.write(tmp_in);
            }
            cout << " ]";
            cout << endl;
        }
    }


    unsigned INTER_ROW = CONV0_IN_ROW;
    unsigned INTER_COL = CONV0_IN_COL;

#if 1//DwCv
    INTER_ROW += 2;
    INTER_COL += 2;

    hls::stream<innerfm_T> padding_out("pad_put stream");
    Padding<POSE_IN_CH, POSE_IN_BIT,1>(testin, padding_out, in_ch_nums, CONV0_IN_ROW, CONV0_IN_COL);
#endif

#if Dilation
    INTER_ROW *= 2;
    INTER_COL *= 2;
    hls::stream<innerfm_T> padding_out("pad_put stream");
    stream<ap_int<POSE_IN_CH*POSE_IN_BIT> > deconvpad_out("deconvpad_out");
#pragma HLS stream variable=deconvpad_out depth=16 dim=1
    DilationPadding<POSE_IN_CH, POSE_IN_BIT, 1>(testin, deconvpad_out, in_ch_nums, CONV0_IN_ROW, CONV0_IN_COL);
    INTER_ROW += 2;
    INTER_COL += 2;
    Padding<POSE_IN_CH, POSE_IN_BIT, 1>(deconvpad_out, padding_out, in_ch_nums, 2*CONV0_IN_ROW, 2*CONV0_IN_COL);
#endif

#if 0
    ofstream fpconv0pad("..\\Test\\conv0pad.txt", ios::out);
    if (!fpconv0pad)
        cout << "no such file" << endl;
    for (int h = 0; h < INTER_ROW; ++h) {
        for (int w = 0; w < INTER_COL ; ++w) {
            fpconv0pad << "[";
            for (int c = 0; c < in_ch_nums; ++c) {
                fpconv0pad << hex << " " << padding_out.read() << " ";
            }
            fpconv0pad << "] ";
        }
        fpconv0pad << endl;
    }
    fpconv0pad.close();
#endif

#if DwCv
    stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> swu_out("swu_out");
    cout << dec << " INTER_ROW: "<< INTER_ROW << endl;
    SWU<3, POSE_IN_BIT, POSE_IN_CH>(padding_out,swu_out, in_ch_nums, INTER_ROW, INTER_COL, CONV0_S);


    cout << dec << "swu_out size: " << swu_out.size() << endl;
#endif
#if 0
    const unsigned STEPS = ((INTER_COL - K) / CONV0_S + 1) * ((INTER_ROW - K) / CONV0_S + 1);
    ofstream fpconv0swu("..\\Test\\conv0swu.txt", ios::out);
    if (!fpconv0swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS; step++) {
        //cout << dec << "------ step: " << step << "------" << endl;
        fpconv0swu << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                //cout << "[";
                fpconv0swu << "[";
                for (int nums = 0; nums < in_ch_nums; ++nums) {
                    //cout /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                    cout << hex;
                    fpconv0swu /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << swu_out.read() << " ";
                }
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

    stream<ap_int<CONV0_SIMD*POSE_IN_BIT>> adj_out("adj_out");

#if PwCv
    const unsigned IsDwCv = 0;
    StreamingDataWidthConverter_Batch<POSE_IN_CH*POSE_IN_BIT, CONV0_SIMD*POSE_IN_BIT>
            (testin, adj_out, K*K*CONV0_IN_ROW*CONV0_IN_COL*in_ch_nums, in_ch_nums);
#endif
#if DwCv
    const unsigned IsDwCv = 1;
    const unsigned OUT_ROW = (INTER_ROW - 3) / 1 + 1;
    const unsigned OUT_COL = (INTER_COL - 3) / 1 + 1;
    StreamingDataWidthConverter_Batch<POSE_IN_CH*POSE_IN_BIT, POSE_SIMD*POSE_IN_BIT>
            (swu_out, adj_out, 3*3*OUT_ROW*OUT_COL*in_ch_nums, in_ch_nums);
#endif
    cout << dec << "adj_out size: " << adj_out.size() << endl;

#if 0
    ofstream fpconv0adj("..\\Test\\conv0adj.txt", ios::out);
    if (!fpconv0adj)
        cout << "no such file" << endl;
    const unsigned STEPS = ((CONV0_INTER_COL - K) / CONV0_S + 1) * ((CONV0_INTER_ROW - K) / CONV0_S + 1);
    for (int step = 0; step < STEPS; step++) {
        fpconv0adj << dec << "------ step: " << step << "------" << endl;
        for (int h = 0; h < K ; ++h) {
            for (int w = 0; w < K; ++w) {
                fpconv0adj << "[";
                for (int nums = 0; nums < in_ch_nums; ++nums) {
                    cout << hex;
                    fpconv0adj /*<< setw(2+POSE_IN_CH*POSE_IN_BIT/4) */<< hex << adj_out.read() << " ";
                }
                //cout << "] ";
                fpconv0adj << "] ";
            }
            //cout << endl;
            fpconv0adj << endl;
        }
        //cout << "-------------------------" << endl;
        fpconv0adj << "-------------------------" << endl;
    }
    fpconv0adj.close();
#endif
#if 0
    ofstream fpconv0adj("..\\Test\\conv0adj.txt", ios::out);
    if (!fpconv0adj)
        cout << "no such file" << endl;
    for (int h = 0; h < CONV0_IN_ROW; ++h) {
        for (int w = 0; w < CONV0_IN_COL; ++w) {
            fpconv0adj << "[";
            for (int nums = 0; nums < in_ch_nums; ++nums) {
                cout << hex;
                fpconv0adj << hex << adj_out.read() << " ";
            }
            fpconv0adj << "]";
        }
        fpconv0adj << endl;
    }
    fpconv0adj.close();
#endif

    //GenParamW<1>("..\\Test\\testw.txt");
    //GenParamB("..\\Test\\testb.txt");
    //GenParamM("..\\Test\\testm.txt");
    stream<ap_int<CONV0_PE*POSE_IN_BIT>> mvau_out("mvau_out");
#if PwCv
    const unsigned VECT_NUMS = CONV0_IN_ROW*CONV0_IN_COL;
    PwcvMatrixVectorActUnit<8,8,32,8,16,16,16,16,0,4,1>
            (adj_out, mvau_out, conv0_w, conv0_bias, conv0_m0, in_ch_nums*POSE_IN_CH, 32/*OUT_CH*/, VECT_NUMS);
#endif
#if DwCv
    const unsigned VECT_NUMS = CONV0_IN_ROW*CONV0_IN_COL;
    DwcvMatrixVectorActUnit<8,8,32,8,16,16,16,16,0,18,2>
            (adj_out, mvau_out, conv0_w_dw, conv0_bias_dw, conv0_m0_dw, in_ch_nums*POSE_IN_CH*3*3, CONV0_PE, 2/*in_ch_nums*/,VECT_NUMS);
#endif
    cout << dec << "mvau_out size: " << mvau_out.size() << endl;


#if 0
    ofstream fpconv0mvau("..\\Test\\conv0mvau.txt", ios::out);
    if (!fpconv0mvau)
        cout << "no such file" << endl;
    for (int i = 0; i < VECT_NUMS*in_ch_nums; ++i) {
        cout << hex;
        fpconv0mvau << hex << mvau_out.read() << endl;
    }
    fpconv0mvau.close();
#endif

    return 0;
}
