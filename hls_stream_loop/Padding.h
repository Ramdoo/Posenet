#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#define ZERO 0

using namespace std;
using namespace hls;

//普通padding，在上下左右padding，padding的数量为P
template<
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned P
                >
void Padding(
        stream<ap_int<IN_CH*IN_BIT>> & in_fm,
        stream<ap_int<IN_CH*IN_BIT>> &out_fm,
        const unsigned IN_CH_NUMS,
        const unsigned IN_ROW,
        const unsigned IN_COL)
{
    unsigned OUT_COL = IN_COL + 2*P;
    ap_int<IN_CH*IN_BIT> tmp_out = 0;

    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=98 max=98
#pragma HLS PIPELINE II=1
            for (int nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
                out_fm.write(ZERO);
            }
        }
    }
    for (int h = 0; h < IN_ROW; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=98 max=98
#pragma HLS PIPELINE II=1
            for (int nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
                if ((w < P) || (w >= OUT_COL-P)) {
                    tmp_out = ZERO;
                } else {
                    tmp_out = in_fm.read();
                }
                out_fm.write(tmp_out);
            }
        }
    }
    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=98 max=98
#pragma HLS PIPELINE II=1
            for (int nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
                out_fm.write(ZERO);
            }
        }
    }
}


//Padding and preprocess
template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned P,
        unsigned M0_BIT,
        unsigned BIAS_BIT
>
void FirstLayerPaddingT(
        stream<ap_uint<IN_CH*IN_BIT>> & in_fm,
        stream<ap_int<IN_CH*IN_BIT>> &out_fm,
        const ap_uint<M0_BIT> m0[IN_CH],
        const ap_uint<BIAS_BIT> const0_16[IN_CH])
        {

    unsigned OUT_COL = IN_COL + 2*P;
    ap_int<IN_CH*IN_BIT> tmp_out = 0;

    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
            out_fm.write(ZERO);
        }
    }
    for (int h = 0; h < IN_ROW; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=128
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
            if ((w < P) || (w >= OUT_COL-P)) {
                tmp_out = ZERO;
            } else {
                tmp_out = in_fm.read();
                for (ap_uint<8> ic = 0; ic < IN_CH; ++ic) {
#pragma HLS UNROLL
                    ap_int<32> temp = ap_uint<IN_BIT>(tmp_out((ic<<3)+IN_BIT-1, (ic<<3)));
                    temp = (temp * m0[ic] - const0_16[ic]) >> 16;
                    tmp_out((ic<<3)+IN_BIT-1, (ic<<3)) = temp;
                }
            }
            out_fm.write(tmp_out);
        }
    }
    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
            out_fm.write(ZERO);
        }
    }
}


template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT,
        unsigned P
>
void PaddingT(stream<ap_int<IN_CH*IN_BIT>> & in_fm,
              stream<ap_int<IN_CH*IN_BIT>> &out_fm) {
    unsigned OUT_COL = IN_COL + 2*P;
    ap_int<IN_CH*IN_BIT> tmp_out = 0;

    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
            out_fm.write(ZERO);
        }
    }
    for (int h = 0; h < IN_ROW; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=128
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
            if ((w < P) || (w >= OUT_COL-P)) {
                tmp_out = ZERO;
            } else {
                tmp_out = in_fm.read();
            }
            out_fm.write(tmp_out);
        }
    }
    for (int h = 0; h < P; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=96
#pragma HLS PIPELINE II=1
            out_fm.write(ZERO);
        }
    }
}



template<
        unsigned IN_ROW,
        unsigned IN_COL,
        unsigned IN_CH,
        unsigned IN_BIT
        >
void DilationPaddingT(
        stream<ap_int<IN_CH*IN_BIT>> &in_fm,
        stream<ap_int<IN_CH*IN_BIT>> &out_fm
        ) {

    const unsigned OUT_ROW = IN_ROW + IN_ROW;
    const unsigned OUT_COL = IN_COL + IN_COL;
    ap_int<IN_CH*IN_BIT> tmp_out = 0;

    for (int h = 0; h < OUT_ROW; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=32
        for (int w = 0; w < OUT_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=24
#pragma HLS PIPELINE II=1
            if (h % 2 == 0) {
                if (w % 2 == 0) {
                    tmp_out = in_fm.read();
                } else {
                    tmp_out = ZERO;
                }
            } else {
                tmp_out = ZERO;
            }
#if 0
            cout << dec << "h: " << h << ", w: " << w;
                cout << hex << ", tmp_out: " << tmp_out << endl;
#endif
            out_fm.write(tmp_out);
        }
    }

}
