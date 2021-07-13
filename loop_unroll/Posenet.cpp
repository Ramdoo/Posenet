#define AP_INT_MAX_W 7680
#include <hls_stream.h>

#include "ConvLayer.h"
#include "Posenet.h"
#include "Params.h"
#include <assert.h>
#include "Load.h"


using namespace std;
using namespace hls;

void stream_to_mat (hls::stream<ap_uint<24>>&in,
                    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> & raw_img) {
#pragma HLS dataflow
    for (int i=0; i<IN_IMAGE_HEIGHT; i++) {
        for (int j=0; j<IN_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix;
            ap_uint<24> in_data = in.read();
            for (unsigned int p=0; p < 3; p ++) {
                pix.val[p] = in_data((p<<3)+7, (p<<3));
            }
            raw_img << pix;
        }
    }

}


void mat_to_stream (hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> & resize_img,
                    hls::stream<ap_uint<24>> & out ) {
#pragma HLS dataflow
    for (int i=0; i<RESIZE_IMAGE_HEIGHT; i++) {
        for (int j=0; j<RESIZE_IMAGE_WIDTH; j++) {
#pragma HLS pipeline II = 1
            hls::Scalar<3, ap_uint<8>> pix;
            resize_img >> pix;
            ap_uint<24> out_data;
            for (unsigned int p=0; p < 3; p ++) {
                out_data((p<<3)+7, (p<<3)) = pix.val[p];
            }
            out.write(out_data);
        }
    }

}


void resize(hls::stream<ap_uint<24>> &in, hls::stream<ap_uint<24>> & out) {
#pragma HLS dataflow
    hls::Mat<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, HLS_8UC3> raw_img;
#pragma HLS STREAM variable=raw_img depth=128 dim=1
    hls::Mat<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, HLS_8UC3> resize_img;
#pragma HLS STREAM variable=resize_img depth=128 dim=1
    stream_to_mat(in, raw_img);
    // hls::Resize(raw_img, resize_img, HLS_INTER_LINEAR);
    hls::Resize_opr_linear(raw_img, resize_img);
    mat_to_stream(resize_img, out);
}


void PosenetHeadResize(
        stream<ap_uint<POSE_HCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_IN_CH * POSE_OUT_BIT>> &out
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

    stream<ap_uint<POSE_HCV0_INCH*POSE_OUT_BIT>> resize_stream("resize_stream");
#pragma HLS RESOURCE variable=resize_stream core=FIFO_SRL
    resize(in, resize_stream);

    stream<ap_int<POSE_HCV0_OUTCH*POSE_OUT_BIT>> cv0_out("cv0_out");
#pragma HLS RESOURCE variable=cv0_out core=FIFO_SRL

    FirstLayerT<POSE_HCV0_ROW,POSE_HCV0_COL,POSE_HCV0_INCH,POSE_IN_BIT, POSE_HCV0_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,2,POSE_HCV0_SIMD,POSE_HCV0_PE,16, WGT_HCV0_SIZE, BIAS_M0_HCV0_SIZE>
            (resize_stream, cv0_out, hcv0_w, hcv0_bias, hcv0_m0, preprocess_m0, preprocess_const0_16);

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


void PosenetHead(
        stream<ap_uint<POSE_HCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_IN_CH * POSE_OUT_BIT>> &out
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

    FirstLayerT<POSE_HCV0_ROW,POSE_HCV0_COL,POSE_HCV0_INCH,POSE_IN_BIT, POSE_HCV0_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,3,2,POSE_HCV0_SIMD,POSE_HCV0_PE,16, WGT_HCV0_SIZE, BIAS_M0_HCV0_SIZE>
            (in, cv0_out, hcv0_w, hcv0_bias, hcv0_m0, preprocess_m0, preprocess_const0_16);

#if 0
    //cout << dec << "cv0_out size:" << cv0_out.size() << endl;
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


void PosenetBlockAlpha(
        stream<infm_T> &in,        stream<outfm_T> &out,
        stream<addfm_T> &add_in,
        wgt1_T wgt1[WGT_SIZE1][POSE_PE1],        wgt2_T wgt2[WGT_SIZE2],                 wgt3_T wgt3[WGT_SIZE3][POSE_PE3],
        bias_T bias1[POSE_PE1][BIAS_M0_SIZE1],   bias_T bias2[POSE_PE2][BIAS_M0_SIZE2],  bias_T bias3[POSE_PE3][BIAS_M0_SIZE3],
        m0_T m0_1[POSE_PE1][BIAS_M0_SIZE1],      m0_T m0_2[POSE_PE2][BIAS_M0_SIZE2],     m0_T m0_3[POSE_PE3][BIAS_M0_SIZE3],
        ap_uint<8> ROW1,       ap_uint<8> ROW2,        ap_uint<8> ROW3,
        ap_uint<8> COL1,       ap_uint<8> COL2,        ap_uint<8> COL3,
        ap_uint<4> INCH_NUMS1, ap_uint<4> OUTCH_NUMS1, ap_uint<4> CH_NUMS2,
        ap_uint<4> INCH_NUMS3, ap_uint<4> OUTCH_NUMS3, ap_uint<2> STRIDE,
        ap_uint<1> IS_ADD
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
    ofstream fpblk1cv2("..\\Test\\blk2cv2.txt", ios::out);
    if (!fpblk1cv2)
        cout << "no such file" << endl;
    for (int h = 0; h < ROW3; ++h) {
        for (int w = 0; w < COL3 ; ++w) {
            for (int nums = 0; nums < CH_NUMS2; ++nums) {
                ap_int<POSE_INTER_CH * POSE_IN_BIT> data = dw2_out.read();
                for (int ch = 0; ch < POSE_INTER_CH; ++ch) {
                    cout << dec;
                    fpblk1cv2 << dec << ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk1cv2 << endl;
        }
    }
    fpblk1cv2.close();
#endif

    PwConvAddLayer<POSE_INTER_CH,POSE_IN_BIT,POSE_OUT_CH,POSE_OUT_BIT,POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,POSE_SIMD3,POSE_PE3,16>
            (dw2_out, out, add_in, wgt3, bias3, m0_3, ROW3, COL3, INCH_NUMS3, OUTCH_NUMS3, IS_ADD);
#if 0
    cout << dec << "out size: " << out.size() << endl;
    ofstream fpblk1cv3("..\\Test\\blk2cv3.txt", ios::out);
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
        stream<infm_T> &in,       stream<outfm_T> &out,
        stream<addfm_T> &add_in,  stream<ap_uint<8>> &add_flag,
        wgt16_T* weight, bias8_T* bias, m8_T* m0
) {

#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=add_in
#pragma HLS INTERFACE axis register both port=add_flag
#pragma HLS INTERFACE m_axi depth=14500 port=weight offset=direct bundle=wt
#pragma HLS INTERFACE m_axi depth=994   port=bias   offset=direct bundle=bias
#pragma HLS INTERFACE m_axi depth=497   port=m0     offset=direct bundle=m0

    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt1_ping[WGT_SIZE1][POSE_PE1];
    ap_int<POSE_SIMD2*POSE_W_BIT>  wgt2_ping[WGT_SIZE2];
    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt3_ping[WGT_SIZE3][POSE_PE1];
    ap_int<POSE_BIAS_BIT> bias1_ping[POSE_PE1][BIAS_M0_SIZE1];
    ap_int<POSE_BIAS_BIT> bias2_ping[POSE_PE2][BIAS_M0_SIZE2];
    ap_int<POSE_BIAS_BIT> bias3_ping[POSE_PE3][BIAS_M0_SIZE3];
    ap_uint<POSE_M0_BIT>  m0_1_ping[POSE_PE1][BIAS_M0_SIZE1];
    ap_uint<POSE_M0_BIT>  m0_2_ping[POSE_PE2][BIAS_M0_SIZE2];
    ap_uint<POSE_M0_BIT>  m0_3_ping[POSE_PE3][BIAS_M0_SIZE3];
#pragma HLS ARRAY_PARTITION variable=wgt1_ping complete dim=2
#pragma HLS ARRAY_PARTITION variable=wgt3_ping complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias1_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_1_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_ping complete dim=1

    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt1_pong[WGT_SIZE1][POSE_PE1];
    ap_int<POSE_SIMD2*POSE_W_BIT>  wgt2_pong[WGT_SIZE2];
    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt3_pong[WGT_SIZE3][POSE_PE1];
    ap_int<POSE_BIAS_BIT> bias1_pong[POSE_PE1][BIAS_M0_SIZE1];
    ap_int<POSE_BIAS_BIT> bias2_pong[POSE_PE2][BIAS_M0_SIZE2];
    ap_int<POSE_BIAS_BIT> bias3_pong[POSE_PE3][BIAS_M0_SIZE3];
    ap_uint<POSE_M0_BIT>  m0_1_pong[POSE_PE1][BIAS_M0_SIZE1];
    ap_uint<POSE_M0_BIT>  m0_2_pong[POSE_PE2][BIAS_M0_SIZE2];
    ap_uint<POSE_M0_BIT>  m0_3_pong[POSE_PE3][BIAS_M0_SIZE3];
#pragma HLS ARRAY_PARTITION variable=wgt1_pong complete dim=2
#pragma HLS ARRAY_PARTITION variable=wgt3_pong complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias1_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias3_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_1_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_2_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=m0_3_pong complete dim=1

#pragma HLS ALLOCATION instances=PosenetBlockAlpha	 limit=1 function
#pragma HLS ALLOCATION instances=LoadWgt1    limit=1 function
#pragma HLS ALLOCATION instances=LoadWgt2    limit=1 function
#pragma HLS ALLOCATION instances=LoadWgt3    limit=1 function
#pragma HLS ALLOCATION instances=LoadBias1   limit=1 function
#pragma HLS ALLOCATION instances=LoadBias2   limit=1 function
#pragma HLS ALLOCATION instances=LoadBias3   limit=1 function
#pragma HLS ALLOCATION instances=LoadM1      limit=1 function
#pragma HLS ALLOCATION instances=LoadM2      limit=1 function
#pragma HLS ALLOCATION instances=LoadM3      limit=1 function

    //TODO: Load ping
    LoadWgt1(weight, wgt1_ping, 0, true);
    LoadBias1(bias, bias1_ping, 0, true);
    LoadM1(m0, m0_1_ping, 0, true);
    LoadWgt2(weight, wgt2_ping, 0, true);
    LoadBias2(bias, bias2_ping, 0, true);
    LoadM2(m0, m0_2_ping, 0, true);
    LoadWgt3(weight, wgt3_ping, 0, true);
    LoadBias3(bias, bias3_ping, 0, true);
    LoadM3(m0, m0_3_ping, 0, true);
    for (ap_uint<8> iter_block = 0; iter_block < BLOCK_NUMS; ++iter_block) {
        ap_uint<8> ROW1 = config[iter_block].ih;
        ap_uint<8> ROW2 = config[iter_block].ih;
        ap_uint<8> ROW3 = config[iter_block].ih3;
        ap_uint<8> COL1 = config[iter_block].iw;
        ap_uint<8> COL2 = config[iter_block].iw;
        ap_uint<8> COL3 = config[iter_block].iw3;
        ap_uint<4> INCH_NUMS1 = config[iter_block].ic_nums1;
        ap_uint<4> OUTCH_NUMS1 = config[iter_block].ic_nums2;
        ap_uint<4> CH_NUMS2 = config[iter_block].ic_nums2;
        ap_uint<4> INCH_NUMS3 = config[iter_block].ic_nums2;
        ap_uint<4> OUTCH_NUMS3 = config[iter_block].oc_nums3;
        ap_uint<2> STRIDE = config[iter_block].s;
        ap_uint<1> IS_ADD = config[iter_block].is_add;
        ap_uint<1> NEXT_ADD = config[iter_block].next_add;

        ap_uint<8> raw_add_flag;
        raw_add_flag(0,0) = IS_ADD;
        raw_add_flag(5,1) = iter_block;
        add_flag.write(raw_add_flag);

        if (~iter_block[0]) {
            PosenetBlockAlpha(in, out, add_in,
            wgt1_ping, wgt2_ping, wgt3_ping,
            bias1_ping, bias2_ping, bias3_ping,
            m0_1_ping, m0_2_ping, m0_3_ping,
            ROW1, ROW2, ROW3, COL1, COL2, COL3,
            INCH_NUMS1, CH_NUMS2, CH_NUMS2, CH_NUMS2, OUTCH_NUMS3,
            STRIDE, IS_ADD
            );
            //TODO: Load pong
            LoadWgt1(weight,  wgt1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias1(bias,  bias1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM1(      m0,  m0_1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadWgt2(weight,  wgt2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias2(bias,  bias2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM2(      m0,  m0_2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadWgt3(weight,  wgt3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias3(bias,  bias3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM3(      m0,  m0_3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
        } else {
            PosenetBlockAlpha(in, out, add_in,
                wgt1_pong, wgt2_pong, wgt3_pong,
                bias1_pong, bias2_pong, bias3_pong,
                m0_1_pong, m0_2_pong, m0_3_pong,
                ROW1, ROW2, ROW3, COL1, COL2, COL3,
                INCH_NUMS1, CH_NUMS2, CH_NUMS2, CH_NUMS2, OUTCH_NUMS3,
                STRIDE, IS_ADD
            );
            //TODO: Load ping
            LoadWgt1(weight,  wgt1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias1(bias,  bias1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM1(      m0,  m0_1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadWgt2(weight,  wgt2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias2(bias,  bias2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM2(      m0,  m0_2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadWgt3(weight,  wgt3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadBias3(bias,  bias3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
            LoadM3(      m0,  m0_3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
        }
    }

}




//void PosenetAlphaTest(
//        stream<infm_T> &in,       stream<outfm_T> &out,
//        stream<addfm_T> &add_in,  stream<ap_uint<8>> &add_flag,
//        wgt16_T* weight, bias8_T* bias, m8_T* m0
//) {
//
//#pragma HLS INTERFACE axis register both port=in
//#pragma HLS INTERFACE axis register both port=out
//#pragma HLS INTERFACE axis register both port=add_in
//#pragma HLS INTERFACE axis register both port=add_flag
//#pragma HLS INTERFACE m_axi depth=14500 port=weight offset=direct bundle=wt
//#pragma HLS INTERFACE m_axi depth=994   port=bias   offset=direct bundle=bias
//#pragma HLS INTERFACE m_axi depth=497   port=m0     offset=direct bundle=m0
//
//    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt1_ping[WGT_SIZE1][POSE_PE1];
//    ap_int<POSE_SIMD2*POSE_W_BIT>  wgt2_ping[WGT_SIZE2];
//    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt3_ping[WGT_SIZE3][POSE_PE1];
//    ap_int<POSE_BIAS_BIT> bias1_ping[POSE_PE1][BIAS_M0_SIZE1];
//    ap_int<POSE_BIAS_BIT> bias2_ping[POSE_PE2][BIAS_M0_SIZE2];
//    ap_int<POSE_BIAS_BIT> bias3_ping[POSE_PE3][BIAS_M0_SIZE3];
//    ap_uint<POSE_M0_BIT>  m0_1_ping[POSE_PE1][BIAS_M0_SIZE1];
//    ap_uint<POSE_M0_BIT>  m0_2_ping[POSE_PE2][BIAS_M0_SIZE2];
//    ap_uint<POSE_M0_BIT>  m0_3_ping[POSE_PE3][BIAS_M0_SIZE3];
//#pragma HLS ARRAY_PARTITION variable=wgt1_ping complete dim=2
//#pragma HLS ARRAY_PARTITION variable=wgt3_ping complete dim=2
//#pragma HLS ARRAY_PARTITION variable=bias1_ping complete dim=1
//#pragma HLS ARRAY_PARTITION variable=bias2_ping complete dim=1
//#pragma HLS ARRAY_PARTITION variable=bias3_ping complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_1_ping complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_2_ping complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_3_ping complete dim=1
//
//    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt1_pong[WGT_SIZE1][POSE_PE1];
//    ap_int<POSE_SIMD2*POSE_W_BIT>  wgt2_pong[WGT_SIZE2];
//    ap_int<POSE_SIMD1*POSE_W_BIT>  wgt3_pong[WGT_SIZE3][POSE_PE1];
//    ap_int<POSE_BIAS_BIT> bias1_pong[POSE_PE1][BIAS_M0_SIZE1];
//    ap_int<POSE_BIAS_BIT> bias2_pong[POSE_PE2][BIAS_M0_SIZE2];
//    ap_int<POSE_BIAS_BIT> bias3_pong[POSE_PE3][BIAS_M0_SIZE3];
//    ap_uint<POSE_M0_BIT>  m0_1_pong[POSE_PE1][BIAS_M0_SIZE1];
//    ap_uint<POSE_M0_BIT>  m0_2_pong[POSE_PE2][BIAS_M0_SIZE2];
//    ap_uint<POSE_M0_BIT>  m0_3_pong[POSE_PE3][BIAS_M0_SIZE3];
//#pragma HLS ARRAY_PARTITION variable=wgt1_pong complete dim=2
//#pragma HLS ARRAY_PARTITION variable=wgt3_pong complete dim=2
//#pragma HLS ARRAY_PARTITION variable=bias1_pong complete dim=1
//#pragma HLS ARRAY_PARTITION variable=bias2_pong complete dim=1
//#pragma HLS ARRAY_PARTITION variable=bias3_pong complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_1_pong complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_2_pong complete dim=1
//#pragma HLS ARRAY_PARTITION variable=m0_3_pong complete dim=1
//
//#pragma HLS ALLOCATION instances=PosenetBlockAlpha	 limit=1 function
//#pragma HLS ALLOCATION instances=LoadWgt1    limit=1 function
//#pragma HLS ALLOCATION instances=LoadWgt2    limit=1 function
//#pragma HLS ALLOCATION instances=LoadWgt3    limit=1 function
//#pragma HLS ALLOCATION instances=LoadBias1   limit=1 function
//#pragma HLS ALLOCATION instances=LoadBias2   limit=1 function
//#pragma HLS ALLOCATION instances=LoadBias3   limit=1 function
//#pragma HLS ALLOCATION instances=LoadM1      limit=1 function
//#pragma HLS ALLOCATION instances=LoadM2      limit=1 function
//#pragma HLS ALLOCATION instances=LoadM3      limit=1 function
//
//    //TODO: Load ping
//    LoadWgt1(weight, wgt1_ping, 1, true);
//    LoadBias1(bias, bias1_ping, 1, true);
//    LoadM1(      m0, m0_1_ping, 1, true);
//    LoadWgt2(weight, wgt2_ping, 1, true);
//    LoadBias2(bias, bias2_ping, 1, true);
//    LoadM2(      m0, m0_2_ping, 1, true);
//    LoadWgt3(weight, wgt3_ping, 1, true);
//    LoadBias3(bias, bias3_ping, 1, true);
//    LoadM3(      m0, m0_3_ping, 1, true);
//    for (ap_uint<8> iter_block = 1; iter_block < 2; ++iter_block) {
//        ap_uint<8> ROW1 = config[iter_block].ih;
//        ap_uint<8> ROW2 = config[iter_block].ih;
//        ap_uint<8> ROW3 = config[iter_block].ih3;
//        ap_uint<8> COL1 = config[iter_block].iw;
//        ap_uint<8> COL2 = config[iter_block].iw;
//        ap_uint<8> COL3 = config[iter_block].iw3;
//        ap_uint<4> INCH_NUMS1 = config[iter_block].ic_nums1;
//        ap_uint<4> OUTCH_NUMS1 = config[iter_block].ic_nums2;
//        ap_uint<4> CH_NUMS2 = config[iter_block].ic_nums2;
//        ap_uint<4> INCH_NUMS3 = config[iter_block].ic_nums2;
//        ap_uint<4> OUTCH_NUMS3 = config[iter_block].oc_nums3;
//        ap_uint<2> STRIDE = config[iter_block].s;
//        ap_uint<1> IS_ADD = config[iter_block].is_add;
//        ap_uint<1> NEXT_ADD = config[iter_block].next_add;
//
//        ap_uint<8> raw_add_flag;
//        raw_add_flag(0,0) = IS_ADD;
//        raw_add_flag(5,1) = iter_block;
//        add_flag.write(raw_add_flag);
//
//        if (1) {
//            PosenetBlockAlpha(in, out, add_in,
//                              wgt1_ping, wgt2_ping, wgt3_ping,
//                              bias1_ping, bias2_ping, bias3_ping,
//                              m0_1_ping, m0_2_ping, m0_3_ping,
//                              ROW1, ROW2, ROW3, COL1, COL2, COL3,
//                              INCH_NUMS1, CH_NUMS2, CH_NUMS2, CH_NUMS2, OUTCH_NUMS3,
//                              STRIDE, IS_ADD
//            );
//            //TODO: Load pong
//            LoadWgt1(weight,  wgt1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias1(bias,  bias1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM1(      m0,  m0_1_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadWgt2(weight,  wgt2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias2(bias,  bias2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM2(      m0,  m0_2_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadWgt3(weight,  wgt3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias3(bias,  bias3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM3(      m0,  m0_3_pong, iter_block+1, iter_block != (BLOCK_NUMS-1));
//        } else {
//            PosenetBlockAlpha(in, out, add_in,
//                              wgt1_pong, wgt2_pong, wgt3_pong,
//                              bias1_pong, bias2_pong, bias3_pong,
//                              m0_1_pong, m0_2_pong, m0_3_pong,
//                              ROW1, ROW2, ROW3, COL1, COL2, COL3,
//                              INCH_NUMS1, CH_NUMS2, CH_NUMS2, CH_NUMS2, OUTCH_NUMS3,
//                              STRIDE, IS_ADD
//            );
//            //TODO: Load ping
//            LoadWgt1(weight,  wgt1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias1(bias,  bias1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM1(      m0,  m0_1_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadWgt2(weight,  wgt2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias2(bias,  bias2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM2(      m0,  m0_2_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadWgt3(weight,  wgt3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadBias3(bias,  bias3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//            LoadM3(      m0,  m0_3_ping, iter_block+1, iter_block != (BLOCK_NUMS-1));
//        }
//    }
//
//}



void PosenetDecv(
        stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> &in, stream<ap_int<POSE_CV7_OUTCH * 16>> &out
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

    stream<ap_int<POSE_PWCV0_INCH*POSE_OUT_BIT>> pw0_in("pw0_in");
#pragma HLS RESOURCE variable=pw0_in core=FIFO_SRL
    StreamingDataWidthConverter_BatchT<POSE_IN_CH*POSE_IN_BIT, POSE_PWCV0_INCH*POSE_IN_BIT, 8*6*10>(in, pw0_in);

    stream<ap_int<POSE_PWCV0_OUTCH*POSE_OUT_BIT>> pw0_out("pw0_out");
#pragma HLS RESOURCE variable=pw0_out core=FIFO_SRL

    PwConvActLayerT<POSE_PWCV0_ROW,POSE_PWCV0_COL,POSE_PWCV0_INCH,POSE_IN_BIT,POSE_PWCV0_OUTCH,POSE_OUT_BIT,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,POSE_PWCV0_SIMD,POSE_PWCV0_PE,16,WGT_PWCV0_SIZE,BIAS_M0_PWCV0_SIZE>
            (pw0_in, pw0_out, pwcv0_w, pwcv0_bias, pwcv0_m0);
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
    LastConvLayerT<POSE_CV7_ROW,POSE_CV7_COL,POSE_CV7_INCH,POSE_IN_BIT, POSE_CV7_OUTCH,16,
            POSE_W_BIT,POSE_MUL_BIT,POSE_BIAS_BIT,POSE_M0_BIT,1,1,POSE_CV7_SIMD,POSE_CV7_PE,0, WGT_CV7_SIZE, BIAS_M0_CV7_SIZE>
            (pw6_out, out, pwcv7_w, pwcv7_bias, pwcv7_m0);
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
