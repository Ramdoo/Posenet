#pragma once

#define MVAU_DEBUG 0
#define RELU_DEBUG 0
#include "Function.h"
#include "Posenet.h"
using namespace hls;
using namespace std;

#if 0
#define LOG_W_BIT 3
#define LOG_BIAS_BIT 4
#define LOG_M0_BIT 4
#define LOG_IN_BIT 3
#define LOG_OUT_BIT 3
#define LOG_SIMD 4 // PE = 16
#define LOG_PE 4 // PE = 16


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
        ap_int<W_BIT> temp_wgt = weights( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT));
        ap_int<IN_BIT> temp_in = in( (p<<LOG_IN_BIT)+(IN_BIT-1), (p<<LOG_IN_BIT));
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
ap_int<45> SimdMulReuse(
        ap_int<SIMD*W_BIT> weights_a,
        ap_int<SIMD*W_BIT> weights_b,
        ap_int<SIMD*IN_BIT> in)
{
    ap_int<MUL_BIT> res_a = 0;
    ap_int<MUL_BIT> res_b = 0;
    ap_int<45> res = 0;
    for (ap_uint<8> p = 0; p < SIMD; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL
        ap_int<27> temp_wgt_a = ap_int<8>(weights_a( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT)));
        ap_int<27> temp_wgt_b = ap_int<8>(weights_b( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT)));
        ap_int<18> temp_in    = ap_int<8>(in((p<<LOG_IN_BIT)+(IN_BIT-1), (p<<LOG_IN_BIT)));
        temp_wgt_a = temp_wgt_a << 18;
        ap_int<45> mul;
        mul = (temp_wgt_a + temp_wgt_b) * temp_in;
#if 0
        cout << hex << "hex temp_in: " << temp_in << dec << ", dec" << temp_in << endl;
        cout << hex << "hex temp_wgt_a: " << ap_int<27>(temp_wgt_a) << ", hex temp_wgt_b: " << temp_wgt_b << ", hex temp: " << (temp_wgt_a + temp_wgt_b) << endl;
        cout << dec << "dec temp_wgt_a: " << ap_int<8>(temp_wgt_a) << ", dec temp_wgt_b: " << ap_int<8>(temp_wgt_b) << ", dec temp: " << (temp_wgt_a + temp_wgt_b) << endl;
        cout << hex << "mul: " << mul << endl;
#endif
        res_a += ap_int<16>(mul(33,18)) + mul(15,15);
        res_b += ap_int<16>(mul(15,0));
#if 0
        cout << dec << "    simd temp_wgt_a: " << ap_int<27>(temp_wgt_a) << ", temp_in:" << ap_int<8>(temp_in) << ", mul_a: " << ap_int<16>(mul(33,18)) + mul(15,15)  << hex << ", hex mul_a : " << ap_int<16>(mul(33,18)) + mul(15,15) << endl;
        cout << dec << "    simd temp_wgt_b: " << ap_int<27>(temp_wgt_b) << ", temp_in:" << ap_int<8>(temp_in) << ", mul_b: " << ap_int<16>(mul(15,0)) << hex << ", hex mul_b: " << ap_int<16>(mul(15,0)) << endl;
#endif
    }
    res(20,0) = res_b(20,0);
    res(41,21) = res_a(20,0);
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
        ap_int<W_BIT> temp_wgt = weights( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT));
        ap_int<IN_BIT> temp_in = in( (p<<LOG_IN_BIT)+(IN_BIT-1), (p<<LOG_IN_BIT));
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
    ap_int<26> temp;
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
ap_int<OUT_BIT> ShortcutAddbias(
        ap_int<IN_BIT> in,
        ap_int<IN_BIT> add_fm,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0,
        const ap_uint<1> IS_ADD
)
{
    ap_int<26> temp;
    ap_int<OUT_BIT> res;

    temp = in + bias;

    if (IS_ADD) {     //Add
        temp += add_fm;
    }
    res = (temp * m0) >> RSHIFT;

    return res;
}


template<
        unsigned IN_BIT,
        unsigned OUT_BIT,
        unsigned M0_BIT,
        unsigned BIAS_BIT,
        unsigned RSHIFT//=0 // TODO
>
ap_int<OUT_BIT> AddBias(
        ap_int<IN_BIT> in,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0
)
{
    ap_int<26> temp;
    temp = in + bias;
    ap_int<OUT_BIT> res;
    res = (temp * m0) >> RSHIFT;
    return res;
}
#endif

//通用的函数， 将尺寸大小和通道数作为入参传�??
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT           //TODO
>
void PwcvMatrixVectorActUnit(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        ap_int<SIMD*W_BIT> weights[WGT_SIZE1][PE],
        ap_int<BIAS_BIT> bias[PE][BIAS_M0_SIZE1],
        ap_uint<M0_BIT> m0[PE][BIAS_M0_SIZE1],
        const ap_uint<16> MAT_ROW,
        const ap_uint<16> MAT_COL,
        const ap_uint<16> VECT_NUMS
) {
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0 complete dim=1

    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned total_fold = INPUT_FOLD * OUTPUT_FOLE;
    const unsigned total_reps = total_fold * VECT_NUMS;

#if MVAU_DEBUG
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "VECT_NUMS: "   << VECT_NUMS << endl;
    cout << dec << "total_reps : " << total_reps << endl;
#endif

    ap_int<SIMD*IN_BIT> row_store[5]; //80/16
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned vect_cnt     = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=36864 max=36864
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
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

        {
            ap_int<SIMD*W_BIT> temp_wgt_a = weights[rep % total_fold][0];
            ap_int<SIMD*W_BIT> temp_wgt_b = weights[rep % total_fold][1];
            ap_int<45> res = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
            acc[0] += ap_int<21>(res(41,21));  // a
            acc[1] += ap_int<21>(res(20,0));   // b

            ap_int<SIMD*W_BIT> temp_wgt_c = weights[rep % total_fold][2];
            ap_int<SIMD*W_BIT> temp_wgt_d = weights[rep % total_fold][3];
            ap_int<45> res1 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_c, temp_wgt_d, temp_in);
            acc[2] += ap_int<21>(res1(41,21));  // c
            acc[3] += ap_int<21>(res1(20,0));   // d

            ap_int<SIMD*W_BIT> temp_wgt_e = weights[rep % total_fold][4];
            ap_int<SIMD*W_BIT> temp_wgt_f = weights[rep % total_fold][5];
            ap_int<45> res2 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_e, temp_wgt_f, temp_in);
            acc[4] += ap_int<21>(res2(41,21));  // c
            acc[5] += ap_int<21>(res2(20,0));   // d

            ap_int<SIMD*W_BIT> temp_wgt_g = weights[rep % total_fold][6];
            ap_int<SIMD*W_BIT> temp_wgt_h = weights[rep % total_fold][7];
            ap_int<45> res3 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_g, temp_wgt_h, temp_in);
            acc[6] += ap_int<21>(res3(41,21));  // c
            acc[7] += ap_int<21>(res3(20,0));   // d
        }

        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT>  out_buf;
            {
                out_buf(7, 0) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[0], bias[0][out_fold_cnt], m0[0][out_fold_cnt]);

                out_buf(15, 8) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[1], bias[1][out_fold_cnt], m0[1][out_fold_cnt]);

                out_buf(23, 16) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[2], bias[2][out_fold_cnt], m0[2][out_fold_cnt]);

                out_buf(31, 24) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[3], bias[3][out_fold_cnt], m0[3][out_fold_cnt]);

                out_buf(39, 32) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[4], bias[4][out_fold_cnt], m0[4][out_fold_cnt]);

                out_buf(47, 40) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[5], bias[5][out_fold_cnt], m0[5][out_fold_cnt]);

                out_buf(55, 48) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[6], bias[6][out_fold_cnt], m0[6][out_fold_cnt]);

                out_buf(63, 56) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                        (acc[7], bias[7][out_fold_cnt], m0[7][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << dec << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                << ", acc["<< p << "]: " << acc[p]
                << ", bias: " << bias[p][out_fold_cnt]
                << ", m0: " << m0[p][out_fold_cnt] << endl;
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
                ++vect_cnt;
            }
        }
    }
}



//通用的函数， 将尺寸大小和通道数作为入参传�??
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT            //TODO
>
void DwcvMatrixVectorActUnit(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        ap_int<SIMD*W_BIT> weights[WGT_SIZE2],
        ap_int<BIAS_BIT> bias[PE][BIAS_M0_SIZE2],
        ap_uint<M0_BIT> m0[PE][BIAS_M0_SIZE2],
        const unsigned MAT_ROW, // K*K*CH
        const unsigned MAT_COL, // 1
        const unsigned IN_CH_NUMS,
        const unsigned VECT_NUMS
) {

    //const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned INPUT_FOLD  = 9;
    //const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned OUTPUT_FOLE = IN_CH_NUMS;
    //const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
    const unsigned total_fold = IN_CH_NUMS * INPUT_FOLD;
    const unsigned total_reps = total_fold * VECT_NUMS;
#if MVAU_DEBUG
    const unsigned CH = SIMD*IN_CH_NUMS;
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "total_reps : " << total_reps << endl;
    cout << dec << "CH : " << CH << endl;
#endif
    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt  = 0;
    unsigned vect_cnt     = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[SIMD];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=27648 max=27648
#pragma HLS PIPELINE II=1
        temp_in = in_fm.read();
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < SIMD; ++p) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        ap_int<SIMD*W_BIT> wgt_buf = weights[rep % total_fold];
#if MVAU_DEBUG
        cout << hex << "wgt_buf: " << wgt_buf << endl;
        cout << hex << "temp_in_simd: " << temp_in << endl;
#endif
        {
            acc[0] += ap_int<W_BIT>(wgt_buf(7,0)) * ap_int<IN_BIT>(temp_in(7,0));
            acc[1] += ap_int<W_BIT>(wgt_buf(15,8)) * ap_int<IN_BIT>(temp_in(15,8));
            acc[2] += ap_int<W_BIT>(wgt_buf(23,16)) * ap_int<IN_BIT>(temp_in(23,16));
            acc[3] += ap_int<W_BIT>(wgt_buf(31,24)) * ap_int<IN_BIT>(temp_in(31,24));
            acc[4] += ap_int<W_BIT>(wgt_buf(39,32)) * ap_int<IN_BIT>(temp_in(39,32));
            acc[5] += ap_int<W_BIT>(wgt_buf(47,40)) * ap_int<IN_BIT>(temp_in(47,40));
            acc[6] += ap_int<W_BIT>(wgt_buf(55,48)) * ap_int<IN_BIT>(temp_in(55,48));
            acc[7] += ap_int<W_BIT>(wgt_buf(63,56)) * ap_int<IN_BIT>(temp_in(63,56));
            acc[8] += ap_int<W_BIT>(wgt_buf(71,64)) * ap_int<IN_BIT>(temp_in(71,64));
            acc[9] += ap_int<W_BIT>(wgt_buf(79,72)) * ap_int<IN_BIT>(temp_in(79,72));
            acc[10] += ap_int<W_BIT>(wgt_buf(87,80)) * ap_int<IN_BIT>(temp_in(87,80));
            acc[11] += ap_int<W_BIT>(wgt_buf(95,88)) * ap_int<IN_BIT>(temp_in(95,88));
            acc[12] += ap_int<W_BIT>(wgt_buf(103,96)) * ap_int<IN_BIT>(temp_in(103,96));
            acc[13] += ap_int<W_BIT>(wgt_buf(111,104)) * ap_int<IN_BIT>(temp_in(111,104));
            acc[14] += ap_int<W_BIT>(wgt_buf(119,112)) * ap_int<IN_BIT>(temp_in(119,112));
            acc[15] += ap_int<W_BIT>(wgt_buf(127,120)) * ap_int<IN_BIT>(temp_in(127,120));

            acc[16] += ap_int<W_BIT>(wgt_buf(135,128)) * ap_int<IN_BIT>(temp_in(135,128));
            acc[17] += ap_int<W_BIT>(wgt_buf(143,136)) * ap_int<IN_BIT>(temp_in(143,136));
            acc[18] += ap_int<W_BIT>(wgt_buf(151,144)) * ap_int<IN_BIT>(temp_in(151,144));
            acc[19] += ap_int<W_BIT>(wgt_buf(159,152)) * ap_int<IN_BIT>(temp_in(159,152));
            acc[20] += ap_int<W_BIT>(wgt_buf(167,160)) * ap_int<IN_BIT>(temp_in(167,160));
            acc[21] += ap_int<W_BIT>(wgt_buf(175,168)) * ap_int<IN_BIT>(temp_in(175,168));
            acc[22] += ap_int<W_BIT>(wgt_buf(183,176)) * ap_int<IN_BIT>(temp_in(183,176));
            acc[23] += ap_int<W_BIT>(wgt_buf(191,184)) * ap_int<IN_BIT>(temp_in(191,184));
            acc[24] += ap_int<W_BIT>(wgt_buf(199,192)) * ap_int<IN_BIT>(temp_in(199,192));
            acc[25] += ap_int<W_BIT>(wgt_buf(207,200)) * ap_int<IN_BIT>(temp_in(207,200));
            acc[26] += ap_int<W_BIT>(wgt_buf(215,208)) * ap_int<IN_BIT>(temp_in(215,208));
            acc[27] += ap_int<W_BIT>(wgt_buf(223,216)) * ap_int<IN_BIT>(temp_in(223,216));
            acc[28] += ap_int<W_BIT>(wgt_buf(231,224)) * ap_int<IN_BIT>(temp_in(231,224));
            acc[29] += ap_int<W_BIT>(wgt_buf(239,232)) * ap_int<IN_BIT>(temp_in(239,232));
            acc[30] += ap_int<W_BIT>(wgt_buf(247,240)) * ap_int<IN_BIT>(temp_in(247,240));
            acc[31] += ap_int<W_BIT>(wgt_buf(255,248)) * ap_int<IN_BIT>(temp_in(255,248));

            acc[32] += ap_int<W_BIT>(wgt_buf(263,256)) * ap_int<IN_BIT>(temp_in(263,256));
            acc[33] += ap_int<W_BIT>(wgt_buf(271,264)) * ap_int<IN_BIT>(temp_in(271,264));
            acc[34] += ap_int<W_BIT>(wgt_buf(279,272)) * ap_int<IN_BIT>(temp_in(279,272));
            acc[35] += ap_int<W_BIT>(wgt_buf(287,280)) * ap_int<IN_BIT>(temp_in(287,280));
            acc[36] += ap_int<W_BIT>(wgt_buf(295,288)) * ap_int<IN_BIT>(temp_in(295,288));
            acc[37] += ap_int<W_BIT>(wgt_buf(303,296)) * ap_int<IN_BIT>(temp_in(303,296));
            acc[38] += ap_int<W_BIT>(wgt_buf(311,304)) * ap_int<IN_BIT>(temp_in(311,304));
            acc[39] += ap_int<W_BIT>(wgt_buf(319,312)) * ap_int<IN_BIT>(temp_in(319,312));
            acc[40] += ap_int<W_BIT>(wgt_buf(327,320)) * ap_int<IN_BIT>(temp_in(327,320));
            acc[41] += ap_int<W_BIT>(wgt_buf(335,328)) * ap_int<IN_BIT>(temp_in(335,328));
            acc[42] += ap_int<W_BIT>(wgt_buf(343,336)) * ap_int<IN_BIT>(temp_in(343,336));
            acc[43] += ap_int<W_BIT>(wgt_buf(351,344)) * ap_int<IN_BIT>(temp_in(351,344));
            acc[44] += ap_int<W_BIT>(wgt_buf(359,352)) * ap_int<IN_BIT>(temp_in(359,352));
            acc[45] += ap_int<W_BIT>(wgt_buf(367,360)) * ap_int<IN_BIT>(temp_in(367,360));
            acc[46] += ap_int<W_BIT>(wgt_buf(375,368)) * ap_int<IN_BIT>(temp_in(375,368));
            acc[47] += ap_int<W_BIT>(wgt_buf(383,376)) * ap_int<IN_BIT>(temp_in(383,376));
        }

        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT>  out_buf;
            {
                out_buf(7,0) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[0], bias[0][out_fold_cnt], m0[0][out_fold_cnt]);
                out_buf(15,8) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[1], bias[1][out_fold_cnt], m0[1][out_fold_cnt]);
                out_buf(23,16) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[2], bias[2][out_fold_cnt], m0[2][out_fold_cnt]);
                out_buf(31,24) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[3], bias[3][out_fold_cnt], m0[3][out_fold_cnt]);
                out_buf(39,32) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[4], bias[4][out_fold_cnt], m0[4][out_fold_cnt]);
                out_buf(47,40) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[5], bias[5][out_fold_cnt], m0[5][out_fold_cnt]);
                out_buf(55,48) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[6], bias[6][out_fold_cnt], m0[6][out_fold_cnt]);
                out_buf(63,56) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[7], bias[7][out_fold_cnt], m0[7][out_fold_cnt]);
                out_buf(71,64) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[8], bias[8][out_fold_cnt], m0[8][out_fold_cnt]);
                out_buf(79,72) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[9], bias[9][out_fold_cnt], m0[9][out_fold_cnt]);
                out_buf(87,80) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[10], bias[10][out_fold_cnt], m0[10][out_fold_cnt]);
                out_buf(95,88) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[11], bias[11][out_fold_cnt], m0[11][out_fold_cnt]);
                out_buf(103,96) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[12], bias[12][out_fold_cnt], m0[12][out_fold_cnt]);
                out_buf(111,104) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[13], bias[13][out_fold_cnt], m0[13][out_fold_cnt]);
                out_buf(119,112) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[14], bias[14][out_fold_cnt], m0[14][out_fold_cnt]);
                out_buf(127,120) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[15], bias[15][out_fold_cnt], m0[15][out_fold_cnt]);

                out_buf(135,128) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[16], bias[16][out_fold_cnt], m0[16][out_fold_cnt]);
                out_buf(143,136) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[17], bias[17][out_fold_cnt], m0[17][out_fold_cnt]);
                out_buf(151,144) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[18], bias[18][out_fold_cnt], m0[18][out_fold_cnt]);
                out_buf(159,152) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[19], bias[19][out_fold_cnt], m0[19][out_fold_cnt]);
                out_buf(167,160) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[20], bias[20][out_fold_cnt], m0[20][out_fold_cnt]);
                out_buf(175,168) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[21], bias[21][out_fold_cnt], m0[21][out_fold_cnt]);
                out_buf(183,176) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[22], bias[22][out_fold_cnt], m0[22][out_fold_cnt]);
                out_buf(191,184) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[23], bias[23][out_fold_cnt], m0[23][out_fold_cnt]);
                out_buf(199,192) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[24], bias[24][out_fold_cnt], m0[24][out_fold_cnt]);
                out_buf(207,200) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[25], bias[25][out_fold_cnt], m0[25][out_fold_cnt]);
                out_buf(215,208) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[26], bias[26][out_fold_cnt], m0[26][out_fold_cnt]);
                out_buf(223,216) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[27], bias[27][out_fold_cnt], m0[27][out_fold_cnt]);
                out_buf(231,224) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[28], bias[28][out_fold_cnt], m0[28][out_fold_cnt]);
                out_buf(239,232) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[29], bias[29][out_fold_cnt], m0[29][out_fold_cnt]);
                out_buf(247,240) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[30], bias[30][out_fold_cnt], m0[30][out_fold_cnt]);
                out_buf(255,248) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[31], bias[31][out_fold_cnt], m0[31][out_fold_cnt]);

                out_buf(263,256) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[32], bias[32][out_fold_cnt], m0[32][out_fold_cnt]);
                out_buf(271,264) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[33], bias[33][out_fold_cnt], m0[33][out_fold_cnt]);
                out_buf(279,272) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[34], bias[34][out_fold_cnt], m0[34][out_fold_cnt]);
                out_buf(287,280) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[35], bias[35][out_fold_cnt], m0[35][out_fold_cnt]);
                out_buf(295,288) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[36], bias[36][out_fold_cnt], m0[36][out_fold_cnt]);
                out_buf(303,296) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[37], bias[37][out_fold_cnt], m0[37][out_fold_cnt]);
                out_buf(311,304) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[38], bias[38][out_fold_cnt], m0[38][out_fold_cnt]);
                out_buf(319,312) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[39], bias[39][out_fold_cnt], m0[39][out_fold_cnt]);
                out_buf(327,320) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[40], bias[40][out_fold_cnt], m0[40][out_fold_cnt]);
                out_buf(335,328) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[41], bias[41][out_fold_cnt], m0[41][out_fold_cnt]);
                out_buf(343,336) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[42], bias[42][out_fold_cnt], m0[42][out_fold_cnt]);
                out_buf(351,344) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[43], bias[43][out_fold_cnt], m0[43][out_fold_cnt]);
                out_buf(359,352) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[44], bias[44][out_fold_cnt], m0[44][out_fold_cnt]);
                out_buf(367,360) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[45], bias[45][out_fold_cnt], m0[45][out_fold_cnt]);
                out_buf(375,368) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[46], bias[46][out_fold_cnt], m0[46][out_fold_cnt]);
                out_buf(383,376) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[47], bias[47][out_fold_cnt], m0[47][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$n" << endl;
            cout << dec << "BIAS_BIT:" << BIAS_BIT << endl;
//            cout << hex << "bias_buf: " << bias[p][] << endl;
//            cout << hex << "m0_buf: " << m0_buf << endl;
            cout << "out_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << dec << p << ": " << ap_int<OUT_BIT>(out_buf((p+1)*OUT_BIT-1, p*OUT_BIT))
                     << ", acc["<< p << "]: " << acc[p]
                     << ", bias: " << ap_int<BIAS_BIT>(bias[p][out_fold_cnt])
                     << ", m0: "   << ap_uint<M0_BIT>(m0[p][out_fold_cnt]) << endl;
            }
            cout << endl;
#endif
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            out_fm.write(out_buf);
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                ++vect_cnt;
            }
        }
    }
}



//通用的函数， 将尺寸大小和通道数作为入参传�??
template<
        unsigned IN_BIT,
        unsigned OUT_BIT,

        unsigned MUL_BIT,
        unsigned W_BIT,
        unsigned BIAS_BIT,
        unsigned M0_BIT,

        unsigned SIMD,
        unsigned PE,

        unsigned RSHIFT           //TODO
>
void PwcvAddMatrixVectorUnit(
    stream<ap_int<SIMD*IN_BIT>> &in_fm,
    stream<ap_int<PE*OUT_BIT>> &out_fm,
    stream<ap_int<PE*OUT_BIT>> &add_in,
#ifdef DEBUG
    stream<ap_int<PE*OUT_BIT>> &add_out,
#endif
    ap_int<SIMD*W_BIT> weights[WGT_SIZE3][PE],
    ap_int<BIAS_BIT> bias[PE][BIAS_M0_SIZE3],
    ap_uint<M0_BIT> m0[PE][BIAS_M0_SIZE3],
    const ap_uint<16> MAT_ROW,
    const ap_uint<16> MAT_COL,
    const ap_uint<16> VECT_NUMS,
    const ap_uint<1> IS_ADD
#ifdef DEBUG
    ,
    const ap_uint<1> NEXT_ADD
#endif
) {

    const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned OUTPUT_FOLE = MAT_COL/PE;
    const unsigned total_fold = INPUT_FOLD * OUTPUT_FOLE;
    const unsigned total_reps = total_fold * VECT_NUMS;

#if MVAU_DEBUG
    cout << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << "total_reps : " << total_reps << endl;
#endif

    ap_int<SIMD*IN_BIT> row_store[30]; //480/16
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt = 0;
    unsigned vect_cnt     = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS LOOP_TRIPCOUNT min=9216 max=9216
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; ++p) {
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

        {
            ap_int<SIMD*W_BIT> temp_wgt_a = weights[rep % total_fold][0];
            ap_int<SIMD*W_BIT> temp_wgt_b = weights[rep % total_fold][1];
            ap_int<45> res = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
            acc[0] += ap_int<21>(res(41,21));  // a
            acc[1] += ap_int<21>(res(20,0));   // b

            ap_int<SIMD*W_BIT> temp_wgt_c = weights[rep % total_fold][2];
            ap_int<SIMD*W_BIT> temp_wgt_d = weights[rep % total_fold][3];
            ap_int<45> res1 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_c, temp_wgt_d, temp_in);
            acc[2] += ap_int<21>(res1(41,21));  // c
            acc[3] += ap_int<21>(res1(20,0));   // d

            ap_int<SIMD*W_BIT> temp_wgt_e = weights[rep % total_fold][4];
            ap_int<SIMD*W_BIT> temp_wgt_f = weights[rep % total_fold][5];
            ap_int<45> res2 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_e, temp_wgt_f, temp_in);
            acc[4] += ap_int<21>(res2(41,21));  // c
            acc[5] += ap_int<21>(res2(20,0));   // d

            ap_int<SIMD*W_BIT> temp_wgt_g = weights[rep % total_fold][6];
            ap_int<SIMD*W_BIT> temp_wgt_h = weights[rep % total_fold][7];
            ap_int<45> res3 = Simd16MulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_g, temp_wgt_h, temp_in);
            acc[6] += ap_int<21>(res3(41,21));  // c
            acc[7] += ap_int<21>(res3(20,0));   // d
        }

        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            ap_int<PE*OUT_BIT> addfm_buf = 0;
            if (IS_ADD)
                addfm_buf = add_in.read();
            {
                ap_int<IN_BIT>   temp_addfm0 = ap_int<OUT_BIT>(addfm_buf(7, 0));
                out_buf(7, 0) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                    (acc[0], temp_addfm0, bias[0][out_fold_cnt], m0[0][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm1 = ap_int<OUT_BIT>(addfm_buf(15,8));
                out_buf(15, 8) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                    (acc[1], temp_addfm1, bias[1][out_fold_cnt], m0[1][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm2 = ap_int<OUT_BIT>(addfm_buf(23,16));
                out_buf(23, 16) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[2], temp_addfm2, bias[2][out_fold_cnt], m0[2][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm3 = ap_int<OUT_BIT>(addfm_buf(31,24));
                out_buf(31, 24) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[3], temp_addfm3, bias[3][out_fold_cnt], m0[3][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm4 = ap_int<OUT_BIT>(addfm_buf(39,32));
                out_buf(39, 32) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[4], temp_addfm4, bias[4][out_fold_cnt], m0[4][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm5 = ap_int<OUT_BIT>(addfm_buf(47,40));
                out_buf(47, 40) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[5], temp_addfm5, bias[5][out_fold_cnt], m0[5][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm6 = ap_int<OUT_BIT>(addfm_buf(55,48));
                out_buf(55, 48) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[6], temp_addfm6, bias[6][out_fold_cnt], m0[6][out_fold_cnt], IS_ADD);

                ap_int<IN_BIT>   temp_addfm7 = ap_int<OUT_BIT>(addfm_buf(63,56));
                out_buf(63, 56) = ShortcutAddbias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>
                                     (acc[7], temp_addfm7, bias[7][out_fold_cnt], m0[7][out_fold_cnt], IS_ADD);
            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
        for (unsigned p = 0; p < PE; ++p) {
            cout << dec << p << ": " << ap_int<OUT_BIT>(out_buf((p+1)*OUT_BIT-1, p*OUT_BIT))
            << ", acc[" << p << "]: " << acc[p]
            << ", bias: " << ap_int<BIAS_BIT>(bias_buf((p<<LOG_BIAS_BIT)+(BIAS_BIT-1), (p<<LOG_BIAS_BIT)))
            << ", m0: "   << ap_uint<M0_BIT>(m0_buf(  (p<<LOG_M0_BIT)  +(M0_BIT-1),   (p<<LOG_M0_BIT)))
            << ", add_in: " << ap_int<OUT_BIT>(addfm_buf((p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)))
                    << endl;
        }
        cout << endl;
#endif
            out_fm.write(out_buf);
#ifdef DEBUG
            if (NEXT_ADD)
                add_out.write(out_buf);
#endif
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                ++vect_cnt;
            }
        }
    }
}



//函数名后面加T, 表示参数都放在模板Template中， 固定的参�??
template<
        unsigned MAT_ROW,
        unsigned MAT_COL,

        unsigned IN_BIT,
        unsigned SIMD_NUMS,
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
        unsigned short VECT_NUMS
>
void DwcvMatrixVectorActUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
    static_assert(SIMD == PE, "SIMD is not PE");

    //const unsigned INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned short INPUT_FOLD  = 9;
    const unsigned short OUTPUT_FOLE = MAT_COL/PE;
    const ap_int<25> total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
#if MVAU_DEBUG
    const unsigned CH = SIMD*SIMD_NUMS;
    cout << dec << "INPUT_FOLD : " << INPUT_FOLD << endl;
    cout << dec << "OUTPUT_FOLD: " << OUTPUT_FOLE << endl;
    cout << dec << "VECT_NUMS: " << VECT_NUMS << endl;
    cout << dec << "total_reps : " << total_reps << endl;
    cout << dec << "CH : " << CH << endl;
#endif
    unsigned in_fold_cnt  = 0;
    unsigned out_fold_cnt  = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS PIPELINE II=1
        temp_in = in_fm.read();
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        ap_int<SIMD*W_BIT> temp_wgt_simd = weights[out_fold_cnt*INPUT_FOLD+in_fold_cnt];
#if MVAU_DEBUG
        cout << hex << "temp_wgt_simd: " << ap_uint<SIMD*W_BIT>(temp_wgt_simd) << endl;
        cout << hex << "temp_in_simd: " << temp_in << endl;
#endif
        for (ap_uint<16> p = 0; p < SIMD; ++p) {
#pragma HLS UNROLL
            //unsigned index = ch_nums*SIMD + p;
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]: " << acc[p];
#endif
            ap_int<W_BIT> temp_wgt = ap_int<W_BIT>(temp_wgt_simd( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT)));
            ap_int<IN_BIT> temp = ap_int<IN_BIT>(temp_in( (p<<LOG_IN_BIT)+(IN_BIT-1), (p<<LOG_IN_BIT)));
            ap_int<W_BIT+IN_BIT> mul;
            mul = temp_wgt * temp;
            acc[p] += mul;
#if MVAU_DEBUG
            cout << hex << ", hex temp_wgt: " << temp_wgt  << ", temp: " << temp;
            cout << dec << ", dec temp_wgt: " << temp_wgt  << ", temp: " << temp;
            cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_int<16> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                out_buf((p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(
                        acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
            }
#if MVAU_DEBUG
            cout << dec << "Rshift: " << RSHIFT << endl;
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                     << ", acc["<< p << "]: " << acc[p] << ", bias: "
                     << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
            }
            cout << hex << "out_buf: " << out_buf << endl;
            cout << endl;
#endif
            out_fm.write(out_buf);
            if (++out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
            }
        }
    }
}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参�??
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
        unsigned short VECT_NUMS
>
void PwcvMatrixVectorUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
    const unsigned short INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned short OUTPUT_FOLE = MAT_COL/PE;
    const ap_uint<25> total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
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
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
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

        static_assert(PE % 2 == 0, "PE mod 2 is not 0!");
        for (ap_uint<8> p = 0; p < PE; p+=2) {
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt_a = weights[p][tile];
            ap_int<SIMD*W_BIT> temp_wgt_b = weights[p+1][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
            ap_int<45> res = SimdMulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
            acc[p] += ap_int<21>(res(41,21));
            acc[p+1] += ap_int<21>(res(20,0));
#if MVAU_DEBUG
            cout << hex << ", temp_wgt_a:" << temp_wgt_a  << ", temp_in:" << temp_in;
            cout << hex << ", temp_wgt_b:" << temp_wgt_b  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                out_buf( (p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)) = AddBias<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
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


//函数名后面加T, 表示参数都放在模板Template中， 固定的参�??
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
        unsigned short VECT_NUMS
>
void PwcvMatrixVectorActUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
    const unsigned short INPUT_FOLD  = MAT_ROW/SIMD;
    const unsigned short OUTPUT_FOLE = MAT_COL/PE;
    const ap_uint<25> total_reps = INPUT_FOLD * OUTPUT_FOLE * VECT_NUMS;
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
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
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

        static_assert(PE % 2 == 0, "PE mod 2 is not 0!");
        for (ap_uint<8> p = 0; p < PE; p+=2) {
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt_a = weights[p][tile];
            ap_int<SIMD*W_BIT> temp_wgt_b = weights[p+1][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p];
#endif
            ap_int<45> res = SimdMulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
            acc[p] += ap_int<21>(res(41,21));
            acc[p+1] += ap_int<21>(res(20,0));
#if MVAU_DEBUG
            cout << hex << ", temp_wgt_a:" << temp_wgt_a  << ", temp_in:" << temp_in;
            cout << hex << ", temp_wgt_b:" << temp_wgt_b  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                out_buf( (p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
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
void MatrixVectorActUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
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
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
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

        for (ap_uint<8> p = 0; p < PE; p+=2) {
#pragma HLS UNROLL
            ap_int<SIMD*W_BIT> temp_wgt_a = weights[p][tile];
            ap_int<SIMD*W_BIT> temp_wgt_b = weights[p+1][tile];
#if MVAU_DEBUG
            cout << hex << "before acc[" << p << "]:" << acc[p] << ", before acc[" << p+1 << "]:" << acc[p+1] << endl;
#endif
            ap_int<45> res = SimdMulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
            acc[p] += ap_int<21>(res(41,21));
            acc[p+1] += ap_int<21>(res(20,0));
#if MVAU_DEBUG
            cout << hex << "  temp_wgt_a:" << temp_wgt_a  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << dec << ", dec acc[" << p << "]: " << acc[p] << endl;
            cout << hex << "  temp_wgt_b:" << temp_wgt_b  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p+1 << "]: " << acc[p+1] << dec << ", dec acc[" << p+1 << "]: " << acc[p+1] << endl;
#endif
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_int<PE*OUT_BIT> out_buf;
            for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                out_buf( (p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)) = ReLU<MUL_BIT, OUT_BIT, M0_BIT, BIAS_BIT, RSHIFT>(acc[p], bias[p][out_fold_cnt], m0[p][out_fold_cnt]);
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
void LastPwcvMatrixVectorUnitT(
        stream<ap_int<SIMD*IN_BIT>> &in_fm,
        stream<ap_int<PE*OUT_BIT>> &out_fm,
        const ap_int<SIMD*W_BIT> weights[PE][WGT_ARRAYSIZE],
        const ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE],
        const ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE]
) {
#pragma HLS DATAFLOW
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
    ap_int<OUT_BIT> vect_num     = 0;

    ap_int<SIMD*IN_BIT> temp_in;
    ap_int<MUL_BIT> acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
    ap_int<MUL_BIT> max_acc[PE];// = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=max_acc complete dim=0
    ap_int<OUT_BIT> max_pos[PE];// = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=max_pos complete dim=0

    for (unsigned rep = 0; rep < total_reps; ++rep) {
#pragma HLS PIPELINE II=1
        if (out_fold_cnt == 0) {
            temp_in = in_fm.read();
            row_store[in_fold_cnt] = temp_in;
        } else {
            temp_in = row_store[in_fold_cnt];
        }

        if (in_fold_cnt == 0) {
            for (ap_uint<8> p = 0; p < PE; ++p) {
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
        {
            for (ap_uint<8> p = 0; p < 16; p+=2) {
#pragma HLS UNROLL
                ap_int<SIMD*W_BIT> temp_wgt_a = weights[p][tile];
                ap_int<SIMD*W_BIT> temp_wgt_b = weights[p+1][tile];
#if MVAU_DEBUG
                cout << hex << "before acc[" << p << "]:" << acc[p] << ", before acc[" << p+1 << "]:" << acc[p+1] << endl;
#endif
                ap_int<45> res = SimdMulReuse<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt_a, temp_wgt_b, temp_in);
                acc[p] += ap_int<21>(res(41,21));
                acc[p+1] += ap_int<21>(res(20,0));
#if MVAU_DEBUG
                cout << hex << "  temp_wgt_a:" << temp_wgt_a  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p << "]: " << acc[p] << dec << ", dec acc[" << p << "]: " << acc[p] << endl;
            cout << hex << "  temp_wgt_b:" << temp_wgt_b  << ", temp_in:" << temp_in;
            cout << hex << ", acc[" << p+1 << "]: " << acc[p+1] << dec << ", dec acc[" << p+1 << "]: " << acc[p+1] << endl;
#endif
            }
            ap_int<SIMD*W_BIT> temp_wgt = weights[16][tile];
            acc[16] += SimdMul<W_BIT, IN_BIT, MUL_BIT, SIMD>(temp_wgt, temp_in);
        }

        ++tile;
        ++in_fold_cnt;
        if (in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
//            ap_int<PE*OUT_BIT> out_buf;
//            for (ap_uint<8> p = 0; p < PE; ++p) {
//#pragma HLS UNROLL
//                out_buf( (p<<LOG_OUT_BIT)+(OUT_BIT-1), (p<<LOG_OUT_BIT)) = acc[p];
//            }
#if MVAU_DEBUG
            cout << "$$$$$$$$$$$$$$$\nout_buf: " << endl;
            for (unsigned p = 0; p < PE; ++p) {
                cout << hex << p << ": " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT)
                << ", acc["<< p << "]: " << acc[p] << ", bias: "
                << bias[p][out_fold_cnt] << ", m0: " << m0[p][out_fold_cnt] << endl;
            }
            cout << endl;
#endif
            //out_fm.write(out_buf);
#if MVAU_DEBUG
            cout << hex << "out_but: " << out_buf << endl;
#endif
            ++out_fold_cnt;
            if (out_fold_cnt == OUTPUT_FOLE) {
                out_fold_cnt = 0;
                tile = 0;
                for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                    if (max_acc[p] < acc[p]) {
                        max_acc[p] = acc[p];
                        max_pos[p] = vect_num;
                        //cout << "Compare p:" << p << ", vect_num: " << vect_num << ", " << max_acc[p] << " vs " << acc[p] << endl;
                    }
                }
                ++vect_num;
                if (vect_num == VECT_NUMS) {
                    ap_int<PE*OUT_BIT> out_buf;
                    for (ap_uint<8> p = 0; p < PE; ++p) {
#pragma HLS UNROLL
                        //cout << dec << "max_acc[" << p << "]: " << max_acc[p] << ", max_pos:" << max_pos[p] << endl;
                        if (max_acc[p] > m0[p][0]) {
                            out_buf( (p+1)*OUT_BIT-1, p*OUT_BIT) = max_pos[p];
                            //cout << hex << out_buf( (p+1)*OUT_BIT-1, p*OUT_BIT) << endl;
                        } else {
                            out_buf = 0;
                        }
                    }
                    out_fm.write(out_buf);
                }
            }
        }
    }
}
