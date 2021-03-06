#pragma once

#define MVAU_DEBUG 0
#define RELU_DEBUG 0

using namespace hls;
using namespace std;

#define LOG_W_BIT 3
#define LOG_IN_BIT 3
#define LOG_OUT_BIT 3


template<
        unsigned W_BIT,
        unsigned IN_BIT,
        unsigned MUL_BIT,
        unsigned SIMD
                >
ap_int<MUL_BIT> SimdMul(
        ap_int<SIMD*W_BIT> weights,
        ap_int<SIMD*IN_BIT> in)
{
    ap_int<MUL_BIT> res = 0;

    for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
        ap_int<W_BIT> temp_wgt = weights( p<<LOG_W_BIT+(W_BIT-1), p<<LOG_W_BIT);
        ap_int<IN_BIT> temp_in = in( p<<LOG_IN_BIT+(IN_BIT-1), p<<LOG_IN_BIT);
        ap_int<W_BIT+IN_BIT> mul;
        mul = temp_wgt * temp_in;
        res += mul;
    }
    return res;
}



template<
        unsigned W_BIT,
        unsigned IN_BIT,
        unsigned MUL_BIT,
        unsigned SIMD
                >
ap_int<MUL_BIT> SimdMulLut(
        ap_int<SIMD*W_BIT> weights,
        ap_int<SIMD*IN_BIT> in)
{
    ap_int<MUL_BIT> res = 0;

    for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
        ap_int<W_BIT> temp_wgt = weights( p<<LOG_W_BIT+(W_BIT-1), p<<LOG_W_BIT);
        ap_int<IN_BIT> temp_in = in( p<<LOG_IN_BIT+(IN_BIT-1), p<<LOG_IN_BIT);
        ap_int<W_BIT+IN_BIT> mul;
#pragma HLS RESOURCE variable=mul core=Mul_LUT
        mul = temp_wgt * temp_in;
        res += mul;
    }
    return res;
}



template<
        unsigned IN_BIT,
        unsigned OUT_BIT,
        unsigned M0_BIT,
        unsigned BIAS_BIT,
        unsigned RSHIFT//=0 // TODO
>
ap_int<OUT_BIT> ReLU(
        ap_int<IN_BIT> in,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0
)
{
    ap_int<32> temp;
    temp = in + bias;
    ap_int<OUT_BIT> res;

#if RELU_DEBUG
    cout << dec << "ReLU:  " << "temp:" << temp << ", in: " << in << ", bias: " << bias << ", m0:" << m0;
#endif
    if (temp > 0) {
        temp = (temp * m0) >> RSHIFT;
#if RELU_DEBUG
        cout << ", debug-> temp: " << temp;
#endif
        if (temp > 127) {
            res = 127;
        } else {
            res = temp;
        }
    } else {
        res = 0;
    }
#if RELU_DEBUG
    cout << ", res:" << res << endl;
#endif
    return res;
}



template<
        unsigned IN_BIT,
        unsigned OUT_BIT,
        unsigned M0_BIT,
        unsigned BIAS_BIT,
        unsigned RSHIFT//=0 // TODO
>
ap_int<OUT_BIT> ReLUAdd(
        ap_int<IN_BIT> in,
        ap_int<IN_BIT> add_fm,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0,
        const ap_uint<1> IS_ADD
)
{
    ap_int<32> temp;
    ap_int<OUT_BIT> res;

    temp = in + bias;

    if (IS_ADD) {     //Add
        temp = (temp * m0) >> RSHIFT;
        temp += add_fm;
    }
    else {            //ReLU
        if (temp > 0) {
            temp = (temp * m0) >> RSHIFT;
            if (temp > 127) {
                res = 127;
            } else {
                res = temp;
            }
        } else {
            res = 0;
        }
    }

    return res;
}


//?????????????????? ?????????????????????????????????????????????
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT,           //TODO
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwcvMatrixVectorActUnit(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned MAT_ROW,
        const unsigned MAT_COL,
        const unsigned VECT_NUMS
) {
    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    cout << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << "total_reps : " << total_reps << endl;
#endif
    ap_int<SIMD*IN_BIT> row_store[30]; //480/16
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned tile         = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=4800 max=43200
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }
#if MVAU_DEBUG
        cout << dec << "{\nrep: " << rep << ", "<< "out_fold_cnt: " << out_fold_cnt << ", " << "row_store: " << endl;
        for (unsigned i = 0; i < INPUT_FOLD; ++i) {
            cout << hex << i << ": " << row_store[i] << " ; ";
        }
        cout << "\n}" << endl;
#endif

        for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt = weights[p][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
            acc[p] += SimdMul<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt, temp_in);
#if MVAU_DEBUG
            cout << hex << ", temp_wgt:" << temp_wgt  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                out_buf( p<<LOG_OUT_BIT+(OUT_BIT-1), p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                << ", acc["<< p << "]: " << acc[p] << ", bias: "
                << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
            }
            cout << endl;
#endif
            out_fm.write(out_buf);
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                tile = 0;
            }
        }
    }
}



//?????????????????? ?????????????????????????????????????????????
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT,            //TODO
        unsigned WGT_ARRAYSIZE,     //TODO:
        unsigned BIAS_M0_ARRAYSIZE  //TODO:
>
void DwcvMatrixVectorActUnit(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned MAT_ROW, // K*K*CH
        const unsigned MAT_COL, // 1
        const unsigned IN_CH_NUMS,
        const unsigned VECT_NUMS
) {

    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    //const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned OUTPUT_FOLE = 1;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    const unsigned CH = SIMD*IN_CH_NUMS;
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "total_reps : " << total_reps << endl;
    cout << dec << "CH : " << CH << endl;
#endif
    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned tile         = 0;
    unsigned ch_nums      = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[480/*SIMD*IN_CH_NUMS*/];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=4320 max=12960
#pragma HLS PIPELINE II=1
        temp_in = in_fm.read();
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < 480/*SIMD*IN_CH_NUMS*/; ++p) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        ap_int<SIMD*W_BIT> temp_wgt_simd = weights[tile];
#if MVAU_DEBUG
        cout << hex << "temp_wgt_simd: " << temp_wgt_simd << endl;
        cout << hex << "temp_in_simd: " << temp_in << endl;
#endif
        for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            unsigned index = ch_nums*SIMD + p;
#if MVAU_DEBUG
            cout << hex << "before acc[" << index << "]: " << acc[p];
#endif
            ap_int<W_BIT> temp_wgt = temp_wgt_simd( p<<LOG_W_BIT+(W_BIT-1), p<<LOG_W_BIT);
            ap_int<IN_BIT> temp = temp_in( p<<LOG_IN_BIT+(IN_BIT-1), p<<LOG_IN_BIT);
            ap_int<W_BIT+IN_BIT> mul;
            mul = temp_wgt * temp;
            acc[index] += mul;
#if MVAU_DEBUG
            cout << hex << ", temp_wgt: " << temp_wgt  << ", temp: " << temp;
            cout << hex << ", acc[" << index << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        ++ch_nums;
        if (ch_nums == IN_CH_NUMS) {
            ch_nums = 0;
        }
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=30
                for (ap_int<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                    out_buf(p<<LOG_OUT_BIT+OUT_BIT-1, p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(
                            acc[nums*PE+p], bias[p][nums], m0[p][nums]);
                }
#if MVAU_DEBUG
                cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
                for (unsigned p = 0; p < PE; ++p) {
                    cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                         << ", acc["<< nums*PE+p << "]: " << acc[nums*PE+p] << ", bias: "
                         << bias[p][nums] << ", m0: " << m0[p][nums] << endl;
                }
                cout << endl;
#endif
                out_fm.write(out_buf);
#if MVAU_DEBUG
                cout << hex << "out_but: " << out_buf << endl;
#endif
            }
            tile = 0;
        }
    }
}


#if 0
//?????????????????? ?????????????????????????????????????????????
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT,            //TODO
        unsigned WGT_ARRAYSIZE,     //TODO:
        unsigned BIAS_M0_ARRAYSIZE  //TODO:
>
void DwcvMatrixVectorActUnit1(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
        const unsigned MAT_ROW, // K*K*CH
        const unsigned MAT_COL, // 1
        const unsigned IN_CH_NUMS,
        const unsigned VECT_NUMS
) {

    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    //const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned OUTPUT_FOLE = 1;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    const unsigned CH = SIMD*IN_CH_NUMS;
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "total_reps : " << total_reps << endl;
    cout << dec << "CH : " << CH << endl;
#endif
    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned tile         = 0;
    unsigned ch_nums      = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[128/*SIMD*IN_CH_NUMS*/];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=6912 max=331776
#pragma HLS PIPELINE II=1
        temp_in = in_fm.read();
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < 128/*SIMD*IN_CH_NUMS*/; ++p) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        ap_int<SIMD*W_BIT> temp_wgt_simd = weights[tile];
#if MVAU_DEBUG
        cout << hex << "temp_wgt_simd: " << temp_wgt_simd << endl;
#endif
        for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            unsigned index = ch_nums*SIMD + p;
#if MVAU_DEBUG
            cout << hex << "before acc[" << index << "]: " << acc[p];
#endif
            ap_int<W_BIT> temp_wgt = temp_wgt_simd( p<<LOG_W_BIT+(W_BIT-1), p<<LOG_W_BIT);
            ap_int<IN_BIT> temp = temp_in( p<<LOG_IN_BIT+(IN_BIT-1), p<<LOG_IN_BIT);
            ap_int<W_BIT+IN_BIT> mul;
            mul = temp_wgt * temp;
            acc[index] += mul;
#if MVAU_DEBUG
            cout << hex << ", temp_wgt: " << temp_wgt  << ", temp: " << temp;
            cout << hex << ", acc[" << index << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        ++ch_nums;
        if (ch_nums == IN_CH_NUMS) {
            ch_nums = 0;
        }
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
                for (ap_int<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                    out_buf(p<<LOG_OUT_BIT+OUT_BIT-1, p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(
                            acc[nums*PE+p], bias[p][nums], m0[p][nums]);
                }
#if MVAU_DEBUG
                cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                     << ", acc["<< nums*PE+p << "]: " << acc[nums*PE+p] << ", bias: "
                     << bias[p][nums] << ", m0: " << m0[p][nums] << endl;
            }
            cout << endl;
#endif
                out_fm.write(out_buf);
#if MVAU_DEBUG
                cout << hex << "out_but: " << out_buf << endl;
#endif
            }
            tile = 0;
        }
    }
}
#endif


//?????????????????? ?????????????????????????????????????????????
template<
unsigned IN_BIT,
unsigned OUT_BIT,

unsigned MUL_BIT,
unsigned W_BIT,
unsigned BIAS_BIT,
unsigned M0_BIT,

unsigned SIMD,
unsigned PE,

unsigned RSHIFT,           //TODO
unsigned WGT_ARRAYSIZE,    //TODO:
unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void PwcvAddMatrixVectorActUnit(
    stream<ap_int<SIMD*IN_BIT>> &in_fm,
    stream<ap_int<PE*OUT_BIT>> &out_fm,
    stream<ap_int<PE*OUT_BIT>> &add_fm,
    const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
    const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
    const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
    const unsigned MAT_ROW,
    const unsigned MAT_COL,
    const unsigned VECT_NUMS,
    const ap_uint<1> IS_ADD
) {
    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    cout << "INPUT_FOLD : " << INPUT_FOLD << endl;
cout << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
cout << "total_reps : " << total_reps << endl;
#endif
    ap_int<SIMD*IN_BIT> row_store[30]; //480/16
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned tile         = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=43200
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }
#if MVAU_DEBUG
        cout << dec << "{\nrep: " << rep << ", "<< "out_fold_cnt: " << out_fold_cnt << ", " << "row_store: " << endl;
    for (unsigned i = 0; i < INPUT_FOLD; ++i) {
        cout << hex << i << ": " << row_store[i] << " ; ";
    }
    cout << "\n}" << endl;
#endif

        for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt = weights[p][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
            acc[p] += SimdMul<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt, temp_in);
#if MVAU_DEBUG
            cout << hex << ", temp_wgt:" << temp_wgt  << ", temp_in:" << temp_in;
        cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            ap_int<PE*OUT_BIT> addfm_buf;
            if (IS_ADD)
                addfm_buf = add_fm.read();
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                ap_int<IN_BIT> temp_addfm = addfm_buf(p<<LOG_OUT_BIT+(OUT_BIT-1), p<<LOG_OUT_BIT);
                out_buf( p<<LOG_OUT_BIT+(OUT_BIT-1), p<<LOG_OUT_BIT) = ReLUAdd<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(acc[p], temp_addfm, bias[p][out_fold_cnt], m0[p][out_fold_cnt], IS_ADD);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
        for (unsigned p = 0; p < PE; ++p) {
            cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
            << ", acc["<< p << "]: " << acc[p] << ", bias: "
            << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
        }
        cout << endl;
#endif
            out_fm.write(out_buf);
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                tile = 0;
            }
        }
    }
}


#if 0
template<
unsigned IN_BIT,
unsigned OUT_BIT,

unsigned MUL_BIT,
unsigned W_BIT,
unsigned BIAS_BIT,
unsigned M0_BIT,

unsigned SIMD,
unsigned PE,

unsigned RSHIFT,           //TODO
unsigned WGT_ARRAYSIZE,    //TODO:
unsigned BIAS_M0_ARRAYSIZE //TODO:
>
void DecvMatrixVectorActUnit(
    stream<ap_int<SIMD*IN_BIT>> &in_fm,
    stream<ap_int<PE*OUT_BIT>> &out_fm,
    const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
    const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
    const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE],
    const unsigned MAT_ROW,
    const unsigned MAT_COL,
    const unsigned VECT_NUMS
) {
const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
const unsigned OUTPUT_FOLE = MAT_COL/PE;
const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
cout << "INPUT_FOLD : " << INPUT_FOLD << endl;
cout << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
cout << "total_reps : " << total_reps << endl;
#endif
ap_int<SIMD*IN_BIT> row_store[128];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

unsigned in_fold_cnt  = 0;
unsigned out_fold_cnt = 0;
unsigned tile         = 0;

ap_int<SIMD*IN_BIT> temp_in;
ap_int<MUL_BIT> acc[PE];

for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=12288 max=196608
#pragma HLS PIPELINE II=1
    if (out_fold_cnt == 0) {
        temp_in = in_fm.read();
        row_store[in_fold_cnt] = temp_in;
    } else {
        temp_in = row_store[in_fold_cnt];
    }

    if (in_fold_cnt == 0) {
        for (unsigned p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            acc[p] = 0;
        }
    }
#if MVAU_DEBUG
    cout << dec << "{\nrep: " << rep << ", "<< "out_fold_cnt: " << out_fold_cnt << ", " << "row_store: " << endl;
    for (unsigned i = 0; i < INPUT_FOLD; ++i) {
        cout << hex << i << ": " << row_store[i] << " ; ";
    }
    cout << "\n}" << endl;
#endif

    for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
        ap_int<SIMD*W_BIT> temp_wgt = weights[p][tile];
#if MVAU_DEBUG
        cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
        acc[p] += SimdMul<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt, temp_in);
#if MVAU_DEBUG
        cout << hex << ", temp_wgt:" << temp_wgt  << ", temp_in:" << temp_in;
        cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
    }

    ++tile;
    ++in_fold_cnt;
    if (in_fold_cnt == INPUT_FOLD) {
        in_fold_cnt = 0;
        ap_int<PE*OUT_BIT> out_buf;
        for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                out_buf( p<<LOG_OUT_BIT+(OUT_BIT-1), p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                << ", acc["<< p << "]: " << acc[p] << ", bias: "
                << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
            }
            cout << endl;
#endif
            out_fm.write(out_buf);
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                tile = 0;
            }
        }
    }
}
#endif


//??????????????????T, ???????????????????????????Template?????? ???????????????
template<
        unsigned MAT_ROW,
        unsigned MAT_COL,

        unsigned IN_BIT,
        unsigned IN_CH_NUMS,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT,            //TODO
        unsigned WGT_ARRAYSIZE,     //TODO:
        unsigned BIAS_M0_ARRAYSIZE,  //TODO:
        unsigned VECT_NUMS
>
void DwcvMatrixVectorActUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {

    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    //const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned OUTPUT_FOLE = 1;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    const unsigned CH = SIMD*IN_CH_NUMS;
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "total_reps : " << total_reps << endl;
    cout << dec << "CH : " << CH << endl;
#endif
    unsigned in_fold_cnt  = 0;
    unsigned ch_nums      = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[128];
#pragma HLS ARRAY_PARTITION variable=acc block factor=8 dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=221184 max=221184
#pragma HLS PIPELINE II=1
        temp_in = in_fm.read();
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < 128; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        ap_int<SIMD*W_BIT> temp_wgt_simd = weights[in_fold_cnt];
#if MVAU_DEBUG
        cout << hex << "temp_wgt_simd: " << temp_wgt_simd << endl;
        cout << hex << "temp_in_simd: " << temp_in << endl;
#endif
        for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
            unsigned index = ch_nums*SIMD + p;
#if MVAU_DEBUG
            cout << hex << "before acc[" << index << "]: " << acc[p];
#endif
            ap_int<W_BIT> temp_wgt = temp_wgt_simd( p<<LOG_W_BIT+(W_BIT-1), p<<LOG_W_BIT);
            ap_int<IN_BIT> temp = temp_in( p<<LOG_IN_BIT+(IN_BIT-1), p<<LOG_IN_BIT);
            ap_int<W_BIT+IN_BIT> mul;
            mul = temp_wgt * temp;
            acc[index] += mul;
#if MVAU_DEBUG
            cout << hex << ", temp_wgt: " << temp_wgt  << ", temp: " << temp;
            cout << hex << ", acc[" << index << "]: " << acc[p] << endl;
#endif
        }

        ++in_fold_cnt;
        ++ch_nums;
        if (ch_nums == IN_CH_NUMS) {
            ch_nums = 0;
        }
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
                for (ap_int<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
                    out_buf(p<<LOG_OUT_BIT+OUT_BIT-1, p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(
                            acc[nums*PE+p], bias[p][nums], m0[p][nums]);
                }
#if MVAU_DEBUG
                cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
                for (unsigned p = 0; p < PE; ++p) {
                    cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                         << ", acc["<< nums*PE+p << "]: " << acc[nums*PE+p] << ", bias: "
                         << bias[p][nums] << ", m0: " << m0[p][nums] << endl;
                }
                cout << endl;
#endif
                out_fm.write(out_buf);
#if MVAU_DEBUG
                cout << hex << "out_but: " << out_buf << endl;
#endif
            }
        }
    }
}


//??????????????????T, ???????????????????????????Template?????? ???????????????
template<
        unsigned MAT_ROW,
        unsigned MAT_COL,

        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT,           //TODO
        unsigned WGT_ARRAYSIZE,    //TODO:
        unsigned BIAS_M0_ARRAYSIZE, //TODO:
        unsigned VECT_NUMS
>
void PwcvMatrixVectorActUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    cout << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << "total_reps : " << total_reps << endl;
#endif
    ap_int<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned tile         = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=4800 max=43200
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }
#if MVAU_DEBUG
        cout << dec << "{\nrep: " << rep << ", "<< "out_fold_cnt: " << out_fold_cnt << ", " << "row_store: " << endl;
        for (unsigned i = 0; i < INPUT_FOLD; ++i) {
            cout << hex << i << ": " << row_store[i] << " ; ";
        }
        cout << "\n}" << endl;
#endif

        for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt = weights[p][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
            acc[p] += SimdMul<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt, temp_in);
#if MVAU_DEBUG
            cout << hex << ", temp_wgt:" << temp_wgt  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
#pragma HLS UNROLL
                out_buf( p<<LOG_OUT_BIT+(OUT_BIT-1), p<<LOG_OUT_BIT) = ReLU<MUL_BIT, OUT_BIT, W_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                << ", acc["<< p << "]: " << acc[p] << ", bias: "
                << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
            }
            cout << endl;
#endif
            out_fm.write(out_buf);
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                tile = 0;
            }
        }
    }
}
