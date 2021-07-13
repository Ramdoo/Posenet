using namespace hls;
using namespace std;

#define LOG_W_BIT 3
#define LOG_BIAS_BIT 5
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
#pragma HLS UNROLL
        ap_int<25> temp_wgt_a = ap_int<8>(weights_a( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT)));
        ap_int<25> temp_wgt_b = ap_int<8>(weights_b( (p<<LOG_W_BIT)+(W_BIT-1), (p<<LOG_W_BIT)));
        ap_int<18> temp_in    = ap_int<8>(in((p<<LOG_IN_BIT)+(IN_BIT-1), (p<<LOG_IN_BIT)));
        temp_wgt_a = temp_wgt_a << 16;
        ap_int<45> mul;
        mul = (temp_wgt_a + temp_wgt_b) * temp_in;
#if 0
        cout << hex << "hex temp_in: " << temp_in << dec << ", dec" << temp_in << endl;
        cout << hex << "hex temp_wgt_a: " << ap_int<27>(temp_wgt_a) << ", hex temp_wgt_b: " << temp_wgt_b << ", hex temp: " << (temp_wgt_a + temp_wgt_b) << endl;
        cout << dec << "dec temp_wgt_a: " << ap_int<8>(temp_wgt_a) << ", dec temp_wgt_b: " << ap_int<8>(temp_wgt_b) << ", dec temp: " << (temp_wgt_a + temp_wgt_b) << endl;
        cout << hex << "mul: " << mul << endl;
#endif
        res_a += ap_int<16>(mul(31,16)) + mul(15,15);
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
ap_int<45> Simd16MulReuse(
        ap_int<SIMD*W_BIT> weights_a,
        ap_int<SIMD*W_BIT> weights_b,
        ap_int<SIMD*IN_BIT> in)
{
    ap_int<MUL_BIT> res_a[16];
    ap_int<MUL_BIT> res_b[8];

    ap_int<MUL_BIT> res_i[8];
    ap_int<MUL_BIT> res_j[4];
    ap_int<MUL_BIT> res_k[2];
    ap_int<MUL_BIT> res_l = 0;

    ap_int<MUL_BIT> res_m[4];
    ap_int<MUL_BIT> res_n[2];
    ap_int<MUL_BIT> res_o = 0;

    ap_int<45> res = 0;
    {
        ap_int<25> temp_wgt_a = ap_int<8>(weights_a(7,0));
        ap_int<25> temp_wgt_b = ap_int<8>(weights_b(7,0));
        ap_int<18> temp_in0    = ap_int<8>(in(7,0));
        temp_wgt_a = temp_wgt_a << 16;
        ap_int<45> mul0 = (temp_wgt_a + temp_wgt_b) * temp_in0;
        res_a[0] += ap_int<16>(mul0(31,16)) + mul0(15,15);

        ap_int<25> temp_wgt_c = ap_int<8>(weights_a(15,8));
        ap_int<25> temp_wgt_d = ap_int<8>(weights_b(15,8));
        ap_int<18> temp_in1    = ap_int<8>(in(15,8));
        temp_wgt_c = temp_wgt_c << 16;
        ap_int<45> mul1 = (temp_wgt_c + temp_wgt_d) * temp_in1;
        res_a[1] += ap_int<16>(mul1(31,16)) + mul1(15,15);
        res_b[0] = ap_int<16>(mul1(15,0)) + ap_int<16>(mul0(15,0));

        ap_int<25> temp_wgt_e = ap_int<8>(weights_a(23,16));
        ap_int<25> temp_wgt_f = ap_int<8>(weights_b(23,16));
        ap_int<18> temp_in2    = ap_int<8>(in(23,16));
        temp_wgt_e = temp_wgt_e << 16;
        ap_int<45> mul2 = (temp_wgt_e + temp_wgt_f) * temp_in2;
        res_a[2] += ap_int<16>(mul2(31,16)) + mul2(15,15);

        ap_int<25> temp_wgt_g = ap_int<8>(weights_a(31,24));
        ap_int<25> temp_wgt_h = ap_int<8>(weights_b(31,24));
        ap_int<18> temp_in3    = ap_int<8>(in(31,24));
        temp_wgt_g = temp_wgt_g << 16;
        ap_int<45> mul3 = (temp_wgt_g + temp_wgt_h) * temp_in3;
        res_a[3] += ap_int<16>(mul3(31,16)) + mul3(15,15);
        res_b[1] = ap_int<16>(mul3(15,0)) + ap_int<16>(mul2(15,0));

        ap_int<25> temp_wgt_i = ap_int<8>(weights_a(39,32));
        ap_int<25> temp_wgt_j = ap_int<8>(weights_b(39,32));
        ap_int<18> temp_in4    = ap_int<8>(in(39,32));
        temp_wgt_i = temp_wgt_i << 16;
        ap_int<45> mul4 = (temp_wgt_i + temp_wgt_j) * temp_in4;
        res_a[4] += ap_int<16>(mul4(31,16)) + mul4(15,15);

        ap_int<25> temp_wgt_k = ap_int<8>(weights_a(47,40));
        ap_int<25> temp_wgt_l = ap_int<8>(weights_b(47,40));
        ap_int<18> temp_in5    = ap_int<8>(in(47,40));
        temp_wgt_k = temp_wgt_k << 16;
        ap_int<45> mul5 = (temp_wgt_k + temp_wgt_l) * temp_in5;
        res_a[5] += ap_int<16>(mul5(31,16)) + mul5(15,15);
        res_b[2] = ap_int<16>(mul5(15,0)) + ap_int<16>(mul4(15,0));

        ap_int<25> temp_wgt_m = ap_int<8>(weights_a(55,48));
        ap_int<25> temp_wgt_n = ap_int<8>(weights_b(55,48));
        ap_int<18> temp_in6    = ap_int<8>(in(55,48));
        temp_wgt_m = temp_wgt_m << 16;
        ap_int<45> mul6 = (temp_wgt_m + temp_wgt_n) * temp_in6;
        res_a[6] += ap_int<16>(mul6(31,16)) + mul6(15,15);

        ap_int<25> temp_wgt_o = ap_int<8>(weights_a(63,56));
        ap_int<25> temp_wgt_p = ap_int<8>(weights_b(63,56));
        ap_int<18> temp_in7    = ap_int<8>(in(63,56));
        temp_wgt_o = temp_wgt_o << 16;
        ap_int<45> mul7 = (temp_wgt_o + temp_wgt_p) * temp_in7;
        res_a[7] += ap_int<16>(mul7(31,16)) + mul7(15,15);
        res_b[3] = ap_int<16>(mul7(15,0)) + ap_int<16>(mul6(15,0));

        ap_int<25> temp_wgt_q = ap_int<8>(weights_a(71,64));
        ap_int<25> temp_wgt_r = ap_int<8>(weights_b(71,64));
        ap_int<18> temp_in8    = ap_int<8>(in(71,64));
        temp_wgt_q = temp_wgt_q << 16;
        ap_int<45> mul8 = (temp_wgt_q + temp_wgt_r) * temp_in8;
        res_a[8] += ap_int<16>(mul8(31,16)) + mul8(15,15);

        ap_int<25> temp_wgt_s = ap_int<8>(weights_a(79,72));
        ap_int<25> temp_wgt_t = ap_int<8>(weights_b(79,72));
        ap_int<18> temp_in9   = ap_int<8>(in(79,72));
        temp_wgt_s = temp_wgt_s << 16;
        ap_int<45> mul9 = (temp_wgt_s + temp_wgt_t) * temp_in9;
        res_a[9] += ap_int<16>(mul9(31,16)) + mul9(15,15);
        res_b[4] = ap_int<16>(mul9(15,0)) + ap_int<16>(mul8(15,0));

        ap_int<25> temp_wgt_u = ap_int<8>(weights_a(87,80));
        ap_int<25> temp_wgt_v = ap_int<8>(weights_b(87,80));
        ap_int<18> temp_in10  = ap_int<8>(in(87,80));
        temp_wgt_u = temp_wgt_u << 16;
        ap_int<45> mul10 = (temp_wgt_u + temp_wgt_v) * temp_in10;
        res_a[10] += ap_int<16>(mul10(31,16)) + mul10(15,15);

        ap_int<25> temp_wgt_w = ap_int<8>(weights_a(95,88));
        ap_int<25> temp_wgt_x = ap_int<8>(weights_b(95,88));
        ap_int<18> temp_in11  = ap_int<8>(in(95,88));
        temp_wgt_w = temp_wgt_w << 16;
        ap_int<45> mul11 = (temp_wgt_w + temp_wgt_x) * temp_in11;
        res_a[11] += ap_int<16>(mul11(31,16)) + mul11(15,15);
        res_b[5] = ap_int<16>(mul11(15,0)) + ap_int<16>(mul10(15,0));

        ap_int<25> temp_wgt_y = ap_int<8>(weights_a(103,96));
        ap_int<25> temp_wgt_z = ap_int<8>(weights_b(103,96));
        ap_int<18> temp_in12  = ap_int<8>(in(103,96));
        temp_wgt_y = temp_wgt_y << 16;
        ap_int<45> mul12 = (temp_wgt_y + temp_wgt_z) * temp_in12;
        res_a[12] += ap_int<16>(mul12(31,16)) + mul12(15,15);

        ap_int<25> temp_wgt_A = ap_int<8>(weights_a(111,104));
        ap_int<25> temp_wgt_B = ap_int<8>(weights_b(111,104));
        ap_int<18> temp_in13  = ap_int<8>(in(111,104));
        temp_wgt_A = temp_wgt_A << 16;
        ap_int<45> mul13 = (temp_wgt_A + temp_wgt_B) * temp_in13;
        res_a[13] += ap_int<16>(mul13(31,16)) + mul13(15,15);
        res_b[6] = ap_int<16>(mul13(15,0)) + ap_int<16>(mul12(15,0));

        ap_int<25> temp_wgt_C = ap_int<8>(weights_a(119,112));
        ap_int<25> temp_wgt_D = ap_int<8>(weights_b(119,112));
        ap_int<18> temp_in14  = ap_int<8>(in(119,112));
        temp_wgt_C = temp_wgt_C << 16;
        ap_int<45> mul14 = (temp_wgt_C + temp_wgt_D) * temp_in14;
        res_a[14] += ap_int<16>(mul14(31,16)) + mul14(15,15);

        ap_int<25> temp_wgt_E = ap_int<8>(weights_a(127,120));
        ap_int<25> temp_wgt_F = ap_int<8>(weights_b(127,120));
        ap_int<18> temp_in15  = ap_int<8>(in(127,120));
        temp_wgt_E = temp_wgt_E << 16;
        ap_int<45> mul15 = (temp_wgt_E + temp_wgt_F) * temp_in15;
        res_a[15] += ap_int<16>(mul15(31,16)) + mul15(15,15);
        res_b[7] = ap_int<16>(mul15(15,0)) + ap_int<16>(mul14(15,0));

        res_i[0] = res_a[0] + res_a[1];
        res_i[1] = res_a[2] + res_a[3];
        res_i[2] = res_a[4] + res_a[5];
        res_i[3] = res_a[6] + res_a[7];
        res_i[4] = res_a[8] + res_a[9];
        res_i[5] = res_a[10] + res_a[11];
        res_i[6] = res_a[12] + res_a[13];
        res_i[7] = res_a[14] + res_a[15];

        res_j[0] = res_i[0] + res_i[1];
        res_j[1] = res_i[2] + res_i[3];
        res_j[2] = res_i[4] + res_i[5];
        res_j[3] = res_i[6] + res_i[7];
        res_k[0] = res_j[0] + res_j[1];
        res_k[1] = res_j[2] + res_j[3];
        res_l = res_k[0] + res_k[1];

        res_m[0] = res_b[0] + res_b[1];
        res_m[1] = res_b[2] + res_b[3];
        res_m[2] = res_b[4] + res_b[5];
        res_m[3] = res_b[6] + res_b[7];
        res_n[0] = res_m[0] + res_m[1];
        res_n[1] = res_m[2] + res_m[3];
        res_o = res_n[0] + res_n[1];


    }
    res(20,0) = res_o(20,0);
    res(41,21) = res_l(20,0);
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
        temp = (temp * m0 + 32768) >> RSHIFT;
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
ap_int<OUT_BIT> ReLU1(
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
        temp = (temp * m0 + 32768) >> RSHIFT;
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
        ap_int<OUT_BIT> add_fm,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0,
        const ap_uint<1> IS_ADD
)
{
    ap_int<26> temp;
    ap_int<OUT_BIT> res;

    temp = in + bias;

    res = (temp * m0 + 32768) >> RSHIFT;

    if (IS_ADD) {     //Add
        res += add_fm;
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
ap_int<OUT_BIT> AddBias(
        ap_int<IN_BIT> in,
        ap_int<BIAS_BIT> bias,
        ap_uint<M0_BIT> m0
)
{
    ap_int<26> temp;
    temp = in + bias;
    ap_int<OUT_BIT> res;
    res = (temp * m0 + 32768) >> RSHIFT;
    return res;
}
