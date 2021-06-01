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

    if (IS_ADD) {     //Add
        temp += add_fm;
    }
    res = (temp * m0 + 32768) >> RSHIFT;

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
