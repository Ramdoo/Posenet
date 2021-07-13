#define AP_INT_MAX_W 7680
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
#include "weight.h"
#include "bias.h"
#include "M0.h"

extern
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
);

extern void PosenetHead(
        stream<ap_uint<POSE_HCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_IN_CH * POSE_OUT_BIT>> &out
);

extern void PosenetDecv(
        stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> &in, stream<ap_int<POSE_CV7_OUTCH * 16>> &out
);

extern void PosenetAlpha(
        stream<infm_T> &in,       stream<outfm_T> &out,
        stream<addfm_T> &add_in,  stream<ap_uint<8>> &add_flag,
        wgt16_T* weight, bias8_T* bias, m8_T* m0
);



extern void PosenetAlphaTest(
        stream<infm_T> &in,       stream<outfm_T> &out,
        stream<addfm_T> &add_in,  stream<ap_uint<8>> &add_flag,
        wgt16_T* weight, bias8_T* bias, m8_T* m0
);



void ReadData8(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
    }
    fread(img, sizeof(int8_t), size, fp);
    fclose(fp);
    printf("ReadData8 success!\n");
}


void ReadData16(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
    }
    fread(img, sizeof(uint16_t), size, fp);
    fclose(fp);
    printf("ReadData16 success!\n");
}

void ReadData32(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
    }
    fread(img, sizeof(int32_t), size, fp);
    fclose(fp);
    printf("ReadData32 success!\n");
}


int main() {

    stream<ap_uint<POSE_HCV0_INCH * POSE_IN_BIT>> in("testin");
    int8_t *img = (int8_t *) malloc(256 * 192 * 3 * sizeof(int8_t));
    ReadData8("..\\data\\input_256x192_0_255.bin", (char *) img, 256 * 192 * 3);

    ofstream fpconv0in("..\\Test\\hconv0in.txt", ios::out);
    if (!fpconv0in)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV0_ROW; ++h) {
        for (int w = 0; w < POSE_HCV0_COL; ++w) {
            ap_uint<POSE_HCV0_INCH * POSE_IN_BIT> temp_in;
            for (int ch = 0; ch < POSE_HCV0_INCH; ++ch) {
                temp_in((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = img[h * POSE_HCV0_COL * POSE_HCV0_INCH +
                                                                            w * POSE_HCV0_INCH + ch];
                cout << hex;
                //fpconv0in <<  ap_uint<8>(temp_in((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT)) << " ";
            }
            fpconv0in << "\"" << temp_in << "\", ";
            fpconv0in << endl;
            in.write(temp_in);
        }
    }
    fpconv0in.close();

    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk_in("blk_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk_out("blk_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add_in("add_in");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add_out("add_out");
    stream<ap_uint<8>> add_flag("add_flag");

    PosenetHead(in, blk_in);

    //ofstream fphcv2("..\\Test\\hcv2.txt", ios::out);
    //if (!fphcv2)
    //    cout << "no suck file!" << endl;
    //for (int h = 0; h < 128; ++h) {
    //    for (int w = 0; w < 96; ++w) {
    //        ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk_in.read();
    //        for (int ch = 0; ch < 8; ++ch) {
    //            cout << dec;
    //            fphcv2 << dec << ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
    //        }
    //        fphcv2 << endl;
    //    }
    //}
    //fphcv2.close();

    int8_t *WEIGHT = (int8_t *)   malloc(464016* sizeof(int8_t));
    int32_t *BIAS  = (int32_t *)  malloc(7952*   sizeof(int32_t));
    uint16_t *M0   = (uint16_t *) malloc(7952*   sizeof(uint16_t));

    WEIGHT = reinterpret_cast<int8_t *>(weights);
    BIAS   = reinterpret_cast<int32_t *>(bias);
    M0     = reinterpret_cast<uint16_t*>(m0);

#if BLK1
    PosenetAlphaTest(blk_in, blk_out, add_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk1("..\\Test\\bin\\blk1out.dat", ios::out | ios::binary);
    if (!fpblk1)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk_out.read();
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                //fpblk1 << dec << ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                fpblk1.write((char *)&binout, sizeof(ap_int<8>));
            }
            //fpblk1 << endl;
        }
    }
    fpblk1.close();
#endif

#if 1//BLK2
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk2_in("blk2_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk2_out("blk2_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add2_in("add2_in");

    ifstream fpblk2in("..\\Test\\bin\\blk1out.dat", ios::in | ios::binary);
    if (!fpblk2in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binin;
                fpblk2in.read((char *)&binin, sizeof(ap_int<8>));
                data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
            }
            blk2_in.write(data);
        }
    }
    fpblk2in.close();
    ifstream fpblk2addin("..\\Test\\bin\\blk1out.dat", ios::in | ios::binary);
    if (!fpblk2addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk2addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add2_in.write(data);
            }
        }
    }
    fpblk2addin.close();

    PosenetAlphaTest(blk2_in, blk2_out, add2_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

//    ofstream fpblk2("..\\Test\\bin\\blk2out.dat", ios::out | ios::binary);
//    if (!fpblk2)
//        cout << "no suck file!" << endl;
//    for (int h = 0; h < 64; ++h) {
//        for (int w = 0; w < 48; ++w) {
//            ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk2_out.read();
//            for (int ch = 0; ch < 16; ++ch) {
//                cout << dec;
//                ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
//                fpblk2.write((char *)&binout, sizeof(ap_int<8>));
//            }
//        }
//    }
//    fpblk2.close();
#endif

#if BLK3
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk3_in("blk3_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk3_out("blk3_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add3_in("add3_in");
    ifstream fpblk3in("..\\Test\\bin\\blk2out.dat", ios::in | ios::binary);
    if (!fpblk3in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binin;
                fpblk3in.read((char *)&binin, sizeof(ap_int<8>));
                data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
            }
            blk3_in.write(data);
        }
    }
    fpblk3in.close();
    PosenetAlphaTest(blk3_in, blk3_out, add3_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk3("..\\Test\\bin\\blk3out.dat", ios::out | ios::binary);
    if (!fpblk3)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk3_out.read();
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                fpblk3.write((char *)&binout, sizeof(ap_int<8>));
            }
        }
    }
    fpblk3.close();
#endif

#if BLK4
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk4_in("blk4_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk4_out("blk4_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add4_in("add4_in");

    ifstream fpblk4in("..\\Test\\bin\\blk3out.dat", ios::in | ios::binary);
    if (!fpblk4in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binin;
                fpblk4in.read((char *)&binin, sizeof(ap_int<8>));
                data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
            }
            blk4_in.write(data);
        }
    }
    fpblk4in.close();
    ifstream fpblk4addin("..\\Test\\bin\\blk3out.dat", ios::in | ios::binary);
    if (!fpblk4addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk4addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add4_in.write(data);
            }
        }
    }
    fpblk4addin.close();

    PosenetAlphaTest(blk4_in, blk4_out, add4_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk4("..\\Test\\bin\\blk4out.dat", ios::out | ios::binary);
    if (!fpblk4)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk4_out.read();
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                fpblk4.write((char *)&binout, sizeof(ap_int<8>));
            }
        }
    }
    fpblk4.close();
#endif

#if BLK5
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk5_in("blk5_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk5_out("blk5_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add5_in("add5_in");

    ifstream fpblk5in("..\\Test\\bin\\blk4out.dat", ios::in | ios::binary);
    if (!fpblk5in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binin;
                fpblk5in.read((char *)&binin, sizeof(ap_int<8>));
                data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
            }
            blk5_in.write(data);
        }
    }
    fpblk5in.close();
    ifstream fpblk5addin("..\\Test\\bin\\blk4out.dat", ios::in | ios::binary);
    if (!fpblk5addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk5addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add5_in.write(data);
            }
        }
    }
    fpblk5addin.close();

    PosenetAlphaTest(blk5_in, blk5_out, add5_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk5("..\\Test\\bin\\blk5out.dat", ios::out | ios::binary);
    if (!fpblk5)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk5_out.read();
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                fpblk5.write((char *)&binout, sizeof(ap_int<8>));
            }
        }
    }
    fpblk5.close();
#endif

#if BLK6
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk6_in("blk6_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk6_out("blk6_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add6_in("add6_in");

    ifstream fpblk6in("..\\Test\\bin\\blk5out.dat", ios::in | ios::binary);
    if (!fpblk6in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 24; ++w) {
            ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
            for (int ch = 0; ch < 16; ++ch) {
                cout << dec;
                ap_int<8> binin;
                fpblk6in.read((char *)&binin, sizeof(ap_int<8>));
                data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
            }
            blk6_in.write(data);
        }
    }
    fpblk6in.close();

    PosenetAlphaTest(blk6_in, blk6_out, add6_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk6("..\\Test\\bin\\blk6out.dat", ios::out | ios::binary);
    if (!fpblk6)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk6_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk6.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk6.close();
#endif

#if BLK7
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk7_in("blk7_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk7_out("blk7_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add7_in("add7_in");

    ifstream fpblk7in("..\\Test\\bin\\blk6out.dat", ios::in | ios::binary);
    if (!fpblk7in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk7in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk7_in.write(data);
            }
        }
    }
    fpblk7in.close();
    ifstream fpblk7addin("..\\Test\\bin\\blk6out.dat", ios::in | ios::binary);
    if (!fpblk7addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 4; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk7addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add7_in.write(data);
            }
        }
    }
    fpblk7addin.close();

    PosenetAlphaTest(blk7_in, blk7_out, add7_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk7("..\\Test\\bin\\blk7out.dat", ios::out | ios::binary);
    if (!fpblk7)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk7_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk7.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk7.close();
#endif

#if BLK8
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk8_in("blk8_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk8_out("blk8_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add8_in("add8_in");

    ifstream fpblk8in("..\\Test\\bin\\blk7out.dat", ios::in | ios::binary);
    if (!fpblk8in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk8in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk8_in.write(data);
            }
        }
    }
    fpblk8in.close();
    ifstream fpblk8addin("..\\Test\\bin\\blk7out.dat", ios::in | ios::binary);
    if (!fpblk8addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 4; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk8addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add8_in.write(data);
            }
        }
    }
    fpblk8addin.close();

    PosenetAlphaTest(blk8_in, blk8_out, add8_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk8("..\\Test\\bin\\blk8out.dat", ios::out | ios::binary);
    if (!fpblk8)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk8_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk8.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk8.close();
#endif

#if BLK9
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk9_in("blk9_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk9_out("blk9_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add9_in("add9_in");

    ifstream fpblk9in("..\\Test\\bin\\blk8out.dat", ios::in | ios::binary);
    if (!fpblk9in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk9in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk9_in.write(data);
            }
        }
    }
    fpblk9in.close();
    ifstream fpblk9addin("..\\Test\\bin\\blk8out.dat", ios::in | ios::binary);
    if (!fpblk9addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 4; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk9addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add9_in.write(data);
            }
        }
    }
    fpblk9addin.close();

    PosenetAlphaTest(blk9_in, blk9_out, add9_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk9("..\\Test\\bin\\blk9out.dat", ios::out | ios::binary);
    if (!fpblk9)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk9_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk9.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk9.close();
#endif

#if BLK10
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk10_in("blk10_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk10_out("blk10_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add10_in("add10_in");

    ifstream fpblk10in("..\\Test\\bin\\blk9out.dat", ios::in | ios::binary);
    if (!fpblk10in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 2; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk10in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk10_in.write(data);
            }
        }
    }
    fpblk10in.close();

    PosenetAlphaTest(blk10_in, blk10_out, add10_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk10("..\\Test\\bin\\blk10out.dat", ios::out | ios::binary);
    if (!fpblk10)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk10_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk10.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk10.close();
#endif

#if BLK11
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk11_in("blk11_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk11_out("blk11_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add11_in("add11_in");

    ifstream fpblk11in("..\\Test\\bin\\blk10out.dat", ios::in | ios::binary);
    if (!fpblk11in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk11in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk11_in.write(data);
            }
        }
    }
    fpblk11in.close();
    ifstream fpblk11addin("..\\Test\\bin\\blk10out.dat", ios::in | ios::binary);
    if (!fpblk11addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 6; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk11addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add11_in.write(data);
            }
        }
    }
    fpblk11addin.close();

    PosenetAlphaTest(blk11_in, blk11_out, add11_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk11("..\\Test\\bin\\blk11out.dat", ios::out | ios::binary);
    if (!fpblk11)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk11_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk11.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk11.close();
#endif

#if BLK12
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk12_in("blk12_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk12_out("blk12_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add12_in("add12_in");

    ifstream fpblk12in("..\\Test\\bin\\blk11out.dat", ios::in | ios::binary);
    if (!fpblk12in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk12in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk12_in.write(data);
            }
        }
    }
    fpblk12in.close();
    ifstream fpblk12addin("..\\Test\\bin\\blk11out.dat", ios::in | ios::binary);
    if (!fpblk12addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 6; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk12addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add12_in.write(data);
            }
        }
    }
    fpblk12addin.close();

    PosenetAlphaTest(blk12_in, blk12_out, add12_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk12("..\\Test\\bin\\blk12out.dat", ios::out | ios::binary);
    if (!fpblk12)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk12_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk12.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk12.close();
#endif

#if BLK13
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk13_in("blk13_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk13_out("blk13_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add13_in("add13_in");

    ifstream fpblk13in("..\\Test\\bin\\blk12out.dat", ios::in | ios::binary);
    if (!fpblk13in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int nums = 0; nums < 3; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk13in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk13_in.write(data);
            }
        }
    }
    fpblk13in.close();

    PosenetAlphaTest(blk13_in, blk13_out, add13_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk13("..\\Test\\bin\\blk13out.dat", ios::out | ios::binary);
    if (!fpblk13)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk13_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk13.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk13.close();
#endif

#if BLK14
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk14_in("blk14_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk14_out("blk14_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add14_in("add14_in");

    ifstream fpblk14in("..\\Test\\bin\\blk13out.dat", ios::in | ios::binary);
    if (!fpblk14in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk14in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk14_in.write(data);
            }
        }
    }
    fpblk14in.close();
    ifstream fpblk14addin("..\\Test\\bin\\blk13out.dat", ios::in | ios::binary);
    if (!fpblk14addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 10; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk14addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add14_in.write(data);
            }
        }
    }
    fpblk14addin.close();

    PosenetAlphaTest(blk14_in, blk14_out, add14_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk14("..\\Test\\bin\\blk14out.dat", ios::out | ios::binary);
    if (!fpblk14)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk14_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk14.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk14.close();
#endif

#if BLK15
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk15_in("blk15_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk15_out("blk15_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add15_in("add15_in");

    ifstream fpblk15in("..\\Test\\bin\\blk14out.dat", ios::in | ios::binary);
    if (!fpblk15in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk15in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk15_in.write(data);
            }
        }
    }
    fpblk15in.close();
    ifstream fpblk15addin("..\\Test\\bin\\blk14out.dat", ios::in | ios::binary);
    if (!fpblk15addin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 10; nums++) {
                ap_int<POSE_PE3 * POSE_IN_BIT> data;
                for (int p = 0; p < POSE_PE3; ++p) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk15addin.read((char *)&binin, sizeof(ap_int<8>));
                    data((p + 1) * POSE_IN_BIT - 1, p * POSE_IN_BIT) = binin;
                }
                add15_in.write(data);
            }
        }
    }
    fpblk15addin.close();

    PosenetAlphaTest(blk15_in, blk15_out, add15_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk15("..\\Test\\bin\\blk15out.dat", ios::out | ios::binary);
    if (!fpblk15)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk15_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk15.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk15.close();
#endif

#if BLK16
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> blk16_in("blk16_in");
    stream<ap_int<POSE_OUT_CH * POSE_IN_BIT>> blk16_out("blk16_out");
    stream<ap_int<POSE_PE3 * POSE_IN_BIT>> add16_in("add16_in");

    ifstream fpblk16in("..\\Test\\bin\\blk15out.dat", ios::in | ios::binary);
    if (!fpblk16in)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 5; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpblk16in.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                blk16_in.write(data);
            }
        }
    }
    fpblk16in.close();

    PosenetAlphaTest(blk16_in, blk16_out, add16_in, add_flag, (wgt16_T *)WEIGHT, (bias8_T *)BIAS, (m8_T *)M0);

    ofstream fpblk16("..\\Test\\bin\\blk16out.dat", ios::out | ios::binary);
    if (!fpblk16)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 10; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data = blk16_out.read();
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binout = ap_int<8>(data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT));
                    fpblk16.write((char *) &binout, sizeof(ap_int<8>));
                }
            }
        }
    }
    fpblk16.close();
#endif

#if DECV
    stream<ap_int<POSE_IN_CH * POSE_IN_BIT>> decv_in("decv_in");
    stream<ap_int<17 * 16>> decv_out("decv_out");

    ifstream fpdecvin("..\\Test\\bin\\blk16out.dat", ios::in | ios::binary);
    if (!fpdecvin)
        cout << "no suck file!" << endl;
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 6; ++w) {
            for (int nums = 0; nums < 10; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> data;
                for (int ch = 0; ch < 16; ++ch) {
                    cout << dec;
                    ap_int<8> binin;
                    fpdecvin.read((char *) &binin, sizeof(ap_int<8>));
                    data((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT) = binin;
                }
                decv_in.write(data);
            }
        }
    }
    fpdecvin.close();

    PosenetDecv(decv_in, decv_out);
#endif
}