#define AP_INT_MAX_W 7680
#include <hls_stream.h>

#include "ConvLayer.h"
#include "Posenet.h"
#include "Params.h"
#include <assert.h>


using namespace std;
using namespace hls;



void PosenetBlockAlpha(
        stream<infm_T> &in,        stream<outfm_T> &out,      stream<addfm_T> &add_in,   stream<addfm_T> &add_out,
        stream<wgt1_pe_T> &wgt1,   stream<wgt2_T> &wgt2,      stream<wgt3_pe_T> &wgt3,
        stream<bias1_pe_T> &bias1, stream<bias2_pe_T> &bias2, stream<bias3_pe_T> &bias3,
        stream<m0_1pe_T> &m0_1,    stream<m0_2pe_T> &m0_2,    stream<m0_3pe_T> &m0_3,
        ap_uint<8> ROW1,       ap_uint<8> ROW2,        ap_uint<8> ROW3,
        ap_uint<8> COL1,       ap_uint<8> COL2,        ap_uint<8> COL3,
        ap_uint<4> INCH_NUMS1, ap_uint<4> OUTCH_NUMS1, ap_uint<4> CH_NUMS2,
        ap_uint<4> INCH_NUMS3, ap_uint<4> OUTCH_NUMS3, ap_uint<2> STRIDE,
        ap_uint<1> IS_ADD,     ap_uint<1> NEXT_ADD
) {
#pragma HLS DATAFLOW

    stream<innerfm_T> pw1_out("pw1_out");
#pragma HLS STREAM variable=pw1_out depth=128 dim=1

    PwConvActLayer<POSE_IN_CH,POSE_IN_BIT,POSE_INTER_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD1,POSE_PE1,16>
            (in, pw1_out, wgt1, bias1, m0_1, ROW1, COL1, INCH_NUMS1, OUTCH_NUMS1);
#if 0
    cout << dec << "pw1_out size: " << pw1_out.size() << endl;
    ofstream fpblk1cv1("..\\Test\\blk1cv1.txt", ios::out);
    if (!fpblk1cv1)
        cout << "no such file" << endl;
    for (int h = 0; h < ROW1; ++h) {
        for (int w = 0; w < COL1 ; ++w) {
            for (int nums = 0; nums < OUTCH_NUMS1; nums++) {
                ap_int<POSE_INTER_CH * POSE_IN_BIT> temp = pw1_out.read();
                for (int ch = 0; ch < POSE_INTER_CH; ++ch) {
                    cout << dec;
                    fpblk1cv1 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk1cv1 << endl;
        }
    }
    fpblk1cv1.close();
#endif

    stream<innerfm_T> dw2_out("dw2_out");
#pragma HLS STREAM variable=dw2_out depth=128 dim=1

    DwConvActLayerAlpha<POSE_INTER_CH,POSE_IN_BIT,POSE_INTER_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_SIMD2,POSE_PE2,16>
            (pw1_out, dw2_out, wgt2, bias2, m0_2, ROW2, COL2, STRIDE, CH_NUMS2);
#if 0
    cout << dec << "dw2_out size: " << dw2_out.size() << endl;
    ofstream fpblk1cv2("..\\Test\\blk1cv2.txt", ios::out);
    if (!fpblk1cv2)
        cout << "no such file" << endl;
    for (int h = 0; h < ROW3; ++h) {
        for (int w = 0; w < COL3 ; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> temp = dw2_out.read();
                for (int ch = 0; ch < POSE_OUT_CH; ++ch) {
                    cout << dec;
                    fpblk1cv2 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk1cv2 << endl;
        }
    }
    fpblk1cv2.close();
#endif

    PwConvAddLayer<POSE_INTER_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD3,POSE_PE3,16>
            (dw2_out, out, add_in, add_out, wgt3, bias3, m0_3, ROW3, COL3, INCH_NUMS3, OUTCH_NUMS3, IS_ADD, NEXT_ADD);
#if 0
    cout << dec << "out size: " << out.size() << endl;
    ofstream fpblk1cv3("..\\Test\\blk1cv3.txt", ios::out);
    if (!fpblk1cv3)
        cout << "no such file" << endl;
    for (int h = 0; h < ROW3; ++h) {
        for (int w = 0; w < COL3 ; ++w) {
            for (int nums = 0; nums < OUTCH_NUMS3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> temp = out.read();
                for (int ch = 0; ch < POSE_OUT_CH; ++ch) {
                    cout << dec;
                    fpblk1cv3 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk1cv3 << endl;
        }
    }
    fpblk1cv3.close();
#endif
#if 0
    cout << dec << "add_out size: " << add_out.size() << endl;
    ofstream fpblk1cv3addout("..\\Test\\blk1cv3addout.txt", ios::out);
    if (!fpblk1cv3addout)
        cout << "no such file" << endl;
    for (int h = 0; h < ROW3; ++h) {
        for (int w = 0; w < COL3 ; ++w) {
            for (int nums = 0; nums < OUTCH_NUMS3*POSE_OUT_CH/POSE_PE3; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> temp = add_out.read();
                for (int ch = 0; ch < POSE_PE3; ++ch) {
                    cout << dec;
                    fpblk1cv3addout << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk1cv3addout << endl;
        }
    }
    fpblk1cv3addout.close();
#endif

}


void PosenetAlpha(
        stream<infm_T> &in,        stream<outfm_T> &out,      stream<addfm_T> &add_in,   stream<addfm_T> &add_out,
        stream<wgt1_pe_T> &wgt1,   stream<wgt2_T> &wgt2,      stream<wgt3_pe_T> &wgt3,
        stream<bias1_pe_T> &bias1, stream<bias2_pe_T> &bias2, stream<bias3_pe_T> &bias3,
        stream<m0_1pe_T> &m0_1,    stream<m0_2pe_T> &m0_2,    stream<m0_3pe_T> &m0_3,
        const unsigned ROW1, const unsigned ROW2, const unsigned ROW3, const unsigned COL1, const unsigned COL2, const unsigned COL3,
        const unsigned INCH_NUMS1, const unsigned OUTCH_NUMS1, const unsigned CH_NUMS2,
        const unsigned INCH_NUMS3, const unsigned OUTCH_NUMS3, const unsigned STRIDE,
        const unsigned IS_ADD, const unsigned NEXT_ADD
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE axis port=add_in
#pragma HLS INTERFACE axis port=add_out
#pragma HLS INTERFACE axis port=wgt1
#pragma HLS INTERFACE axis port=wgt2
#pragma HLS INTERFACE axis port=wgt3
#pragma HLS INTERFACE axis port=bias1
#pragma HLS INTERFACE axis port=bias2
#pragma HLS INTERFACE axis port=bias3
#pragma HLS INTERFACE axis port=m0_1
#pragma HLS INTERFACE axis port=m0_2
#pragma HLS INTERFACE axis port=m0_3
#pragma HLS INTERFACE axis port=insts

	//assert(
	//           (ROW1==8 && ROW2==8 && ROW3==8 && COL1==6 && COL2==6 && COL3==6 && INCH_NUMS1==5 && OUTCH_NUMS1==10 && CH_NUMS2==10 && INCH_NUMS3==10 && OUTCH_NUMS3==10 && STRIDE==1 && IS_ADD==0 && NEXT_ADD==1)
	//        || (ROW1==128 && ROW2==128 && ROW3==64 && COL1==96 && COL2==96 && COL3==48 && INCH_NUMS1==1 && OUTCH_NUMS1==1 && CH_NUMS2==1 && INCH_NUMS3==1 && OUTCH_NUMS3==1 && STRIDE==2 && IS_ADD==1 && NEXT_ADD==1)
    //        || (ROW1==64 && ROW2==64 && ROW3==32 && COL1==48 && COL2==48 && COL3==24 && INCH_NUMS1==1 && OUTCH_NUMS1==2 && CH_NUMS2==2 && INCH_NUMS3==2 && OUTCH_NUMS3==1 && STRIDE==2 && IS_ADD==0 && NEXT_ADD==1)
	//        );


    PosenetBlockAlpha(in, out, add_in, add_out,
                      wgt1, wgt2, wgt3,
                      bias1, bias2, bias3,
                      m0_1, m0_2, m0_3,
                      ROW1, ROW2, ROW3, COL1, COL2, COL3,
                      INCH_NUMS1, CH_NUMS2*3, CH_NUMS2, CH_NUMS2*3, OUTCH_NUMS3,
                      STRIDE, IS_ADD, NEXT_ADD);

}


#if 0
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

#if 0
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
#endif
        LoadPwcvWgtBeta1(wgt1, wgt1_buf_beta_ping);
        LoadDwcvWgtBeta2(wgt2, wgt2_buf_beta_ping);
        LoadPwcvWgtBeta3(wgt3, wgt3_buf_beta_ping);
        LoadBiasBeta(bias1, bias1_buf_beta_ping);
        LoadBiasBeta(bias2, bias2_buf_beta_ping);
        LoadBiasBeta(bias3, bias3_buf_beta_ping);

        PosenetBlockBeta(in, out, add_fm, wgt1_buf_beta_ping, wgt2_buf_beta_ping, wgt3_buf_beta_ping,
                         bias1_buf_beta_ping, bias2_buf_beta_ping, bias3_buf_beta_ping,
                         m0_1_buf_beta_ping, m0_2_buf_beta_ping, m0_3_buf_beta_ping,
                         ROW1, ROW2, ROW3, COL1, COL2, COL3, CH_NUMS1, CH_NUMS2, CH_NUMS3, STRIDE, IS_ADD, IS_DECONV);
}
#endif


void PosenetDecv(
        stream<ap_int<POSE_PWCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_CV7_OUTCH * 12>> &out
) {
#pragma HLS DATAFLOW

#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out

#pragma HLS ARRAY_PARTITION variable=pwcv0_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv0_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv0_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=decv1_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=decv1_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=pwcv2_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv2_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv2_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=decv3_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=decv3_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=pwcv4_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv4_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv4_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=decv5_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=decv5_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=pwcv6_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv6_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv6_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=pwcv7_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv7_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=pwcv7_m0   complete dim=1

    stream<ap_int<POSE_PWCV0_OUTCH*POSE_OUT_BIT>> pw0_out("pw0_out");
#pragma HLS RESOURCE variable=pw0_out core=FIFO_SRL

    PwConvActLayerT<POSE_PWCV0_ROW,POSE_PWCV0_COL,POSE_PWCV0_INCH,POSE_IN_BIT,POSE_PWCV0_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_PWCV0_SIMD,POSE_PWCV0_PE,16,WGT_PWCV0_SIZE,BIAS_M0_PWCV0_SIZE>
            (in, pw0_out, pwcv0_w, pwcv0_bias, pwcv0_m0);
#if 0
    cout << dec << "pw0_out size: " << pw0_out.size() << endl;
    ofstream fpdecvpw0("..\\Test\\decv_pw0.txt", ios::out);
    if (!fpdecvpw0)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_PWCV0_ROW; ++h) {
        for (int w = 0; w < POSE_PWCV0_COL ; ++w) {
            ap_int<POSE_PWCV0_OUTCH * POSE_IN_BIT> temp = pw0_out.read();
            for (int ch = 0; ch < POSE_PWCV0_OUTCH; ++ch) {
                cout << dec;
                fpdecvpw0 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvpw0 << endl;
        }
    }
    fpdecvpw0.close();
#endif
    stream<ap_int<POSE_DECV1_OUTCH*POSE_OUT_BIT>> de1_out("de1_out");
#pragma HLS RESOURCE variable=de1_out core=FIFO_SRL

    DeConvLayerT<POSE_DECV1_ROW,POSE_DECV1_COL,POSE_DECV1_INCH,POSE_IN_BIT,POSE_DECV1_INCH/POSE_DECV1_SIMD,POSE_DECV1_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_DECV1_SIMD,POSE_DECV1_LOG2_SIMD,POSE_DECV1_PE,16,WGT_DECV1_SIZE,BIAS_M0_DECV1_SIZE>
            (pw0_out, de1_out, decv1_w, decv1_bias, decv1_m0);

#if 0
    cout << dec << "de1_out size: " << de1_out.size() << endl;
    ofstream fpdecvde1("..\\Test\\decv_de1.txt", ios::out);
    if (!fpdecvde1)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_PWCV2_ROW; ++h) {
        for (int w = 0; w < POSE_PWCV2_COL ; ++w) {
            ap_int<POSE_DECV1_OUTCH * POSE_IN_BIT> temp = de1_out.read();
            for (int ch = 0; ch < POSE_DECV1_OUTCH; ++ch) {
                cout << dec;
                fpdecvde1 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvde1 << endl;
        }
    }
    fpdecvde1.close();
#endif

    stream<ap_int<POSE_PWCV2_OUTCH*POSE_OUT_BIT>> pw2_out("pw2_out");
#pragma HLS RESOURCE variable=pw2_out core=FIFO_SRL

    PwConvActLayerT<POSE_PWCV2_ROW,POSE_PWCV2_COL,POSE_PWCV2_INCH,POSE_IN_BIT,POSE_PWCV2_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_PWCV2_SIMD,POSE_PWCV2_PE,16,WGT_PWCV2_SIZE,BIAS_M0_PWCV2_SIZE>
            (de1_out, pw2_out, pwcv2_w, pwcv2_bias, pwcv2_m0);

#if 0
    cout << dec << "pw2_out size: " << pw2_out.size() << endl;
    ofstream fpdecvpw2("..\\Test\\decv_pw2.txt", ios::out);
    if (!fpdecvpw2)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_PWCV2_ROW; ++h) {
        for (int w = 0; w < POSE_PWCV2_COL ; ++w) {
            ap_int<POSE_PWCV2_OUTCH * POSE_IN_BIT> temp = pw2_out.read();
            for (int ch = 0; ch < POSE_PWCV2_OUTCH; ++ch) {
                cout << dec;
                fpdecvpw2 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvpw2 << endl;
        }
    }
    fpdecvpw2.close();
#endif

    stream<ap_int<POSE_DECV3_OUTCH*POSE_OUT_BIT>> de3_out("de3_out");
#pragma HLS RESOURCE variable=de3_out core=FIFO_SRL

    DeConvLayerT<POSE_DECV3_ROW,POSE_DECV3_COL,POSE_DECV3_INCH,POSE_IN_BIT,POSE_DECV3_INCH/POSE_DECV3_SIMD,POSE_DECV3_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_DECV3_SIMD,POSE_DECV3_LOG2_SIMD,POSE_DECV3_PE,16,WGT_DECV3_SIZE,BIAS_M0_DECV3_SIZE>
            (pw2_out, de3_out, decv3_w, decv3_bias, decv3_m0);

#if 0
    cout << dec << "de3_out size: " << de3_out.size() << endl;
    ofstream fpdecvde3("..\\Test\\decv_de3.txt", ios::out);
    if (!fpdecvde3)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_PWCV4_ROW; ++h) {
        for (int w = 0; w < POSE_PWCV4_COL ; ++w) {
            ap_int<POSE_DECV3_OUTCH * POSE_IN_BIT> temp = de3_out.read();
            for (int ch = 0; ch < POSE_DECV3_OUTCH; ++ch) {
                cout << dec;
                fpdecvde3 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvde3 << endl;
        }
    }
    fpdecvde3.close();
#endif
    stream<ap_int<POSE_PWCV4_OUTCH*POSE_OUT_BIT>> pw4_out("pw4_out");
#pragma HLS RESOURCE variable=pw4_out core=FIFO_SRL

    PwConvActLayerT<POSE_PWCV4_ROW,POSE_PWCV4_COL,POSE_PWCV4_INCH,POSE_IN_BIT,POSE_PWCV4_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_PWCV4_SIMD,POSE_PWCV4_PE,16,WGT_PWCV4_SIZE,BIAS_M0_PWCV4_SIZE>
            (de3_out, pw4_out, pwcv4_w, pwcv4_bias, pwcv4_m0);

    stream<ap_int<POSE_DECV5_OUTCH*POSE_OUT_BIT>> de5_out("de5_out");
#pragma HLS RESOURCE variable=de5_out core=FIFO_SRL

    DeConvLayerT<POSE_DECV5_ROW,POSE_DECV5_COL,POSE_DECV5_INCH,POSE_IN_BIT,POSE_DECV5_INCH/POSE_DECV5_SIMD,POSE_DECV5_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_DECV5_SIMD,POSE_DECV5_LOG2_SIMD,POSE_DECV5_PE,16,WGT_DECV5_SIZE,BIAS_M0_DECV5_SIZE>
            (pw4_out, de5_out, decv5_w, decv5_bias, decv5_m0);


    stream<ap_int<POSE_PWCV6_OUTCH*POSE_OUT_BIT>> pw6_out("pw6_out");
#pragma HLS RESOURCE variable=pw6_out core=FIFO_SRL

    PwConvActLayerT<POSE_PWCV6_ROW,POSE_PWCV6_COL,POSE_PWCV6_INCH,POSE_IN_BIT,POSE_PWCV6_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_PWCV6_SIMD,POSE_PWCV6_PE,16,WGT_PWCV6_SIZE,BIAS_M0_PWCV6_SIZE>
            (de5_out, pw6_out, pwcv6_w, pwcv6_bias, pwcv6_m0);

#if 0
    cout << dec << "pw6_out size: " << pw6_out.size() << endl;
    ofstream fpdecvpw6("..\\Test\\decv_pw6.txt", ios::out);
    if (!fpdecvpw6)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_PWCV6_ROW; ++h) {
        for (int w = 0; w < POSE_PWCV6_COL ; ++w) {
            ap_int<POSE_PWCV6_OUTCH * POSE_IN_BIT> temp = pw6_out.read();
            for (int ch = 0; ch < POSE_PWCV6_OUTCH; ++ch) {
                cout << dec;
                fpdecvpw6 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
            }
            fpdecvpw6 << endl;
        }
    }
    fpdecvpw6.close();
#endif
    //TODO:
    LastConvLayerT<POSE_CV7_ROW,POSE_CV7_COL,POSE_CV7_INCH,POSE_IN_BIT, POSE_CV7_OUTCH,12,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,1,POSE_CV7_SIMD,POSE_CV7_PE,0, WGT_CV7_SIZE, BIAS_M0_CV7_SIZE>
            (pw6_out, out, pwcv7_w, pwcv7_bias, pwcv7_m0);
}


void PosenetHead(
        stream<ap_int<POSE_HCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_IN_CH * POSE_OUT_BIT>> &out
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out

#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=hcv0_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=hcv0_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=hcv0_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=hcv1_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=hcv1_m0   complete dim=1

#pragma HLS ARRAY_PARTITION variable=hcv2_w    complete dim=1
#pragma HLS ARRAY_PARTITION variable=hcv2_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=hcv2_m0   complete dim=1

    stream<ap_int<POSE_HCV0_OUTCH*POSE_OUT_BIT>> cv0_out("cv0_out");
#pragma HLS RESOURCE variable=cv0_out core=FIFO_SRL

    //TODO:
    ConvLayerT<POSE_HCV0_ROW,POSE_HCV0_COL,POSE_HCV0_INCH,POSE_IN_BIT, POSE_HCV0_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,2,POSE_HCV0_SIMD,POSE_HCV0_PE,16, WGT_HCV0_SIZE, BIAS_M0_HCV0_SIZE>
            (in, cv0_out, hcv0_w, hcv0_bias, hcv0_m0);

    //cout << dec << "cv0_out size:" << cv0_out.size() << endl;
#if 0
        ofstream fphconv0("..\\Test\\hconv0.txt", ios::out);
    if (!fphconv0)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV1_ROW; ++h) {
        for (int w = 0; w < POSE_HCV1_COL ; ++w) {
            ap_int<POSE_HCV0_OUTCH*POSE_IN_BIT> temp =  cv0_out.read();
            for (int ch = 0; ch < POSE_HCV0_OUTCH; ++ch) {
                cout << dec;
                fphconv0 << dec << temp((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT) << "  ";
            }
            fphconv0 << endl;
        }
    }
    fphconv0.close();
#endif

    stream<ap_int<POSE_HCV1_OUTCH*POSE_OUT_BIT>> cv1_out("cv1_out");
#pragma HLS RESOURCE variable=cv1_out core=FIFO_SRL

    DwConvLayerT<POSE_HCV1_ROW,POSE_HCV1_COL,POSE_HCV1_INCH,POSE_IN_BIT,POSE_HCV1_INCH/POSE_HCV1_SIMD,POSE_HCV1_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,POSE_HCV1_SIMD,POSE_HCV1_LOG2_SIMD,POSE_HCV1_PE,16,WGT_HCV1_SIZE,BIAS_M0_HCV1_SIZE>
            (cv0_out, cv1_out, hcv1_w, hcv1_bias, hcv1_m0);

#if 0
    ofstream fphconv1("..\\Test\\hconv1.txt", ios::out);
    if (!fphconv1)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV1_ROW; ++h) {
        for (int w = 0; w < POSE_HCV1_COL ; ++w) {
            ap_int<POSE_HCV0_OUTCH*POSE_IN_BIT> temp =  cv1_out.read();
            for (int ch = 0; ch < POSE_HCV0_OUTCH; ++ch) {
                cout << dec;
                fphconv1 << dec << temp((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT) << "  ";
            }
            fphconv1 << endl;
        }
    }
    fphconv1.close();
#endif

    stream<ap_int<POSE_HCV2_OUTCH*POSE_OUT_BIT>> cv2_out("cv2_out");
#pragma HLS RESOURCE variable=cv1_out core=FIFO_SRL
    PwConvLayer3<POSE_HCV2_ROW,POSE_HCV2_COL,POSE_HCV2_INCH,POSE_IN_BIT,POSE_HCV2_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_HCV2_SIMD,POSE_HCV2_PE,16,WGT_HCV2_SIZE,BIAS_M0_HCV2_SIZE>
           (cv1_out, cv2_out, hcv2_w, hcv2_bias, hcv2_m0);

    for (int h = 0; h < POSE_HCV2_ROW; ++h) {
        for (int w = 0; w < POSE_HCV2_COL; ++w) {
            ap_int<POSE_IN_CH*POSE_IN_BIT> temp = cv2_out.read();
            out.write(temp);
        }
    }

#if 0
    ofstream fphconv2("..\\Test\\hconv2.txt", ios::out);
    if (!fphconv2)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV2_ROW; ++h) {
        for (int w = 0; w < POSE_HCV2_COL ; ++w) {
            ap_int<POSE_IN_CH*POSE_IN_BIT> temp =  out.read();
            for (int ch = 0; ch < POSE_IN_CH; ++ch) {
                cout << dec;
                fphconv2 << dec << ap_int<8>(temp((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT)) << "  ";
            }
            fphconv2 << endl;
        }
    }
    fphconv2.close();
#endif
}

#if 0
void Top(
        stream<infm_T>     &in,          stream<outfm_T> &out,            stream<addfm_T> &add_in, stream<addfm_T> &add_out,
        stream<wgt1_pe_T>  &wgt1_alpha,  stream<wgt2_T> &wgt2_alpha,      stream<wgt3_pe_T> &wgt3_alpha,
        stream<bias1_pe_T> &bias1_alpha, stream<bias2_pe_T> &bias2_alpha, stream<bias3_pe_T> &bias3_alpha,
        stream<m0_1pe_T>   &m0_1_alpha,  stream<m0_2pe_T> &m0_2_alpha,    stream<m0_3pe_T> &m0_3_alpha,
        const unsigned ROW1_ALPHA, const unsigned ROW2_ALPHA, const unsigned ROW3_ALPHA, const unsigned COL1_ALPHA, const unsigned COL2_ALPHA, const unsigned COL3_ALPHA,
        const unsigned INCH_NUMS1_ALPHA, const unsigned OUTCH_NUMS1_ALPHA, const unsigned CH_NUMS2_ALPHA, const unsigned  INCH_NUMS3_ALPHA, const unsigned OUTCH_NUMS3_ALPHA, const unsigned STRIDE_ALPHA, const unsigned IS_ADD_ALPHA, const unsigned PingPongAlpha,
#if 0
        stream<infm_T> &in1, stream<outfm_T> &out1, stream<addfm_T> &add_fm1,
        wgt_T* wgt1_1, wgt_T* wgt2_1, wgt_T* wgt3_1,
        bias_T* bias1_1, bias_T* bias2_1, bias_T* bias3_1,
        m0_T* m0_1_1, m0_T* m0_2_1, m0_T* m0_3_1,
        const unsigned ROW1_BETA, const unsigned ROW2_BETA, const unsigned ROW3_BETA, const unsigned COL1_BETA, const unsigned COL2_BETA, const unsigned COL3_BETA,
        const unsigned CH_NUMS1_BETA, const unsigned CH_NUMS2_BETA, const unsigned  CH_NUMS3_BETA, const unsigned STRIDE_BETA, const unsigned IS_ADD_BETA, const unsigned IS_DECONV_BETA, const unsigned PingPongBeta,
#endif
        stream<ap_int<POSE_PWCV0_INCH*POSE_IN_BIT>> &dein, stream<ap_int<POSE_CV7_OUTCH * POSE_OUT_BIT>> &deout
        ) {


//480 channels, 12 cols
    PosenetAlpha(in, out, add_in, add_out, wgt1_alpha, wgt2_alpha, wgt3_alpha, bias1_alpha, bias2_alpha, bias3_alpha, m0_1_alpha, m0_2_alpha, m0_3_alpha, ROW1_ALPHA, ROW2_ALPHA, ROW3_ALPHA, COL1_ALPHA, COL2_ALPHA,
                 COL3_ALPHA, INCH_NUMS1_ALPHA, OUTCH_NUMS1_ALPHA, CH_NUMS2_ALPHA, INCH_NUMS3_ALPHA, OUTCH_NUMS3_ALPHA, STRIDE_ALPHA, IS_ADD_ALPHA, PingPongAlpha);
#if 0
//128 channels, 96 cols
    PosenetBeta(in1, out1, add_fm1, wgt1_1, wgt2_1, wgt3_1, bias1_1, bias2_1, bias3_1, m0_1_1, m0_2_1, m0_3_1, ROW1_BETA,
                ROW2_BETA, ROW3_BETA, COL1_BETA, COL2_BETA, COL3_BETA, CH_NUMS1_BETA, CH_NUMS2_BETA, CH_NUMS3_BETA, STRIDE_BETA, IS_ADD_BETA, IS_DECONV_BETA, PingPongBeta);

#endif
    PosenetDecv(dein, deout);
}
#endif