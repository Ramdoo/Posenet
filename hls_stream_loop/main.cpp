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
extern
void PosenetBlockAlpha(
        stream<infm_T> &in,        stream<outfm_T> &out,
        stream<addfm_T> &add_in,   stream<addfm_T> &add_out,
        wgt1_T wgt1[WGT_SIZE1][POSE_PE1],  wgt2_T wgt2[WGT_SIZE2],           wgt3_T wgt3[WGT_SIZE3][POSE_PE3],
        bias1_pe_T bias1[BIAS_M0_SIZE1],   bias2_pe_T bias2[BIAS_M0_SIZE2],  bias3_pe_T bias3[BIAS_M0_SIZE3],
        m0_1pe_T m0_1[BIAS_M0_SIZE1],       m0_2pe_T m0_2[BIAS_M0_SIZE2],    m0_3pe_T m0_3[BIAS_M0_SIZE3],
        ap_uint<8> ROW1,       ap_uint<8> ROW2,        ap_uint<8> ROW3,
        ap_uint<8> COL1,       ap_uint<8> COL2,        ap_uint<8> COL3,
        ap_uint<4> INCH_NUMS1, ap_uint<4> OUTCH_NUMS1, ap_uint<4> CH_NUMS2,
        ap_uint<4> INCH_NUMS3, ap_uint<4> OUTCH_NUMS3, ap_uint<2> STRIDE,
        ap_uint<1> IS_ADD,     ap_uint<1> NEXT_ADD
);

extern void PosenetHead(
        stream<ap_int<POSE_HCV0_INCH*POSE_IN_BIT>> &in, stream<ap_int<POSE_IN_CH * POSE_OUT_BIT>> &out
);

extern void PosenetDecv(
        stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> &in, stream<ap_int<POSE_CV7_OUTCH * 12>> &out
);

//static block config[16] = {
//        { "blk1",  128,96, 1,1,1,   2,0,1 },
//        { "blk2",  64,48,  1,2,1,   1,1,0 },
//        { "blk3",  64,48,  1,2,1,   2,0,1 },
//        { "blk4",  32,24,  1,2,1,   1,1,1 },
//        { "blk5",  32,24,  1,2,1,   1,1,0 },
//        { "blk6",  32,24,  1,2,2,   2,0,1 },
//        { "blk7",  16,12,  2,4,2,   1,1,1 },
//        { "blk8",  16,12,  2,4,2,   1,1,1 },
//        { "blk9",  16,12,  2,4,2,   1,1,0 },
//        { "blk10", 16,12,  2,4,3,   1,0,1 },
//        { "blk11", 16,12,  3,6,3,   1,1,1 },
//        { "blk12", 16,12,  3,6,3,   1,1,0 },
//        { "blk13", 16,12,  3,6,5,   2,0,1 },
//        { "blk14", 8,6,    5,10,5,  1,1,1 },
//        { "blk15", 8,6,    5,10,5,  1,1,0 },
//        { "blk16", 8,6,    5,10,10, 1,0,0 }
//};

static parm parm_size_debug[16] = {
        { "blk1",  0,768,1200,           0,48,96 },
        { "blk2",  1968,3504,4368,       112,208,304 },
        { "blk3",  5904,7440,8304,       320,416,512 },
        { "blk4",  9840,11376,12240,     528,624,720 },
        { "blk5",  13776,15312,16176,    736,832,928 },
        { "blk6",  17712,19248,20112,    944,1040,1136 },
        { "blk7",  23184,29328,31056,    1168,1360,1552 },
        { "blk8",  37200,43344,45072,    1584,1776,1968 },
        { "blk9",  51216,57360,59088,    2000,2192,2384 },
        { "blk10", 65232,71376,73104,    2416,2608,2800 },
        { "blk11", 82320,96144,98736,    2848,3136,3424 },
        { "blk12", 112560,126384,128976, 3472,3760,4048 },
        { "blk13", 142800,156624,159216, 4096,4384,4672 },
        { "blk14", 182256,220656,224976, 4752,5232,5712 },
        { "blk15", 263376,301776,306096, 5792,6272,6752 },
        { "blk16", 344496,382896,387216, 6832,7312,7792 }
};

#define DwCv 1
#define PwCv 0
#define Dilation 0

using namespace std;
using namespace hls;
#if 0
#define IN_BIT  8
#define OUT_BIT  8
#define W_BIT  8
#define BIAS_BIT  16
#define M0_BIT  16
#define MUL_BIT  32

#define IN_ROW  3
#define IN_COL  5
#define IN_CH   16
#define OUT_CH  16
#define K       3
#define SIMD    16
#define PE      16
#define S       2
#define RSHIFT  0
#define WGT_ARRAYSIZE K*K*IN_CH/SIMD
#define BIAS_M0_ARRAYSIZE OUT_CH/PE
#define INCH_NUMS 2
#endif

#if 0
ap_int<SIMD*W_BIT> weights[WGT_ARRAYSIZE]= {
         "0x01010000010000000000010101010101","0x01000100010000010000010000010100","0x01000100010101000101000101000101",
         "0x01000100000101010101010000010000","0x010001000000010100","0x01010000000000000100000100010100",
         "0x0101010101000000010001000101","0x01010101000000010001010100","0x01000000010000000101010101010101",

         "0x01010100010000000000010000010001","0x010001000101010000010000000001","0x010000010001010000000001010001",
         "0x010101000101000100010100010100","0x010000000101000101010101010001","0x010100010100000000",
         "0x010001000101000001000000010000","0x0101010000000100000101010100","0x010100000001010100010101010001"

};

ap_int<BIAS_BIT> bias[PE][BIAS_M0_ARRAYSIZE] = {
        {"0x2","0x1"},
        {"0x1","0x1"},
        {"0x0","0x0"},
        {"0x1","0x2"},
        {"0x2","0x0"},
        {"0x2","0x2"},
        {"0x2","0x0"},
        {"0x1","0x0"},
        {"0x2","0x1"},
        {"0x2","0x1"},
        {"0x1","0x2"},
        {"0x1","0x0"},
        {"0x1","0x0"},
        {"0x2","0x0"},
        {"0x1","0x1"},
        {"0x1","0x1"}

};

ap_uint<M0_BIT> m0[PE][BIAS_M0_ARRAYSIZE] = {
        {"0x1","0x1"},
        {"0x1","0x1"},
        {"0x1","0x1"},
        {"0x1","0x1"},
        {"0x1","0x0"},
        {"0x1","0x1"},
        {"0x1","0x0"},
        {"0x1","0x0"},
        {"0x1","0x1"},
        {"0x1","0x1"},
        {"0x1","0x1"},
        {"0x1","0x0"},
        {"0x1","0x0"},
        {"0x1","0x0"},
        {"0x1","0x1"},
        {"0x1","0x1"}

};
#endif

void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}


void ReadData8(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
        return;
    }
    fread(img, sizeof(int8_t), size, fp);
    fclose(fp);
    printf("ReadData8 success!\n");
}


void ReadData16(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
        return;
    }
    fread(img, sizeof(uint16_t), size, fp);
    fclose(fp);
    printf("ReadData16 success!\n");
}

void ReadData32(const char* path, char* img, unsigned int size) {
    FILE* fp = fopen(path ,"rb");
    if (fp == NULL) {
        printf("Can't not open file ReadImg!\n");
        return;
    }
    fread(img, sizeof(int32_t), size, fp);
    fclose(fp);
    printf("ReadData32 success!\n");
}


int main() {
    stream<ap_int<POSE_HCV0_INCH*POSE_IN_BIT>> in("testin");
    int8_t * img = (int8_t *) malloc(256*192*3* sizeof(int8_t));
    ReadData8("..\\data\\input_256x192.bin", (char*)img, 256*192*3);

    ofstream fpconv0in("..\\Test\\hconv0in.txt", ios::out);
    if (!fpconv0in)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV0_ROW; ++h) {
        for (int w = 0; w < POSE_HCV0_COL; ++w) {
            ap_int<POSE_HCV0_INCH*POSE_IN_BIT> temp_in;
            for (int ch = 0; ch < POSE_HCV0_INCH; ++ch) {
                temp_in((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT) = img[h*POSE_HCV0_COL*POSE_HCV0_INCH + w*POSE_HCV0_INCH + ch];
                cout << dec;
                fpconv0in << ap_int<8>(temp_in((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT)) << "  ";
            }
            fpconv0in << endl;
            in.write(temp_in);
        }
    }
    fpconv0in.close();

    stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> blk_in("blk_in");
    stream<ap_int<POSE_OUT_CH*POSE_IN_BIT>> blk_out("blk_out");
    stream<ap_int<POSE_PE3*POSE_IN_BIT>> add_in("add_in");
    stream<ap_int<POSE_PE3*POSE_IN_BIT>> add_out("add_out");

    PosenetHead(in, blk_in);

    int8_t *WEIGHT = (int8_t *)   malloc(464016* sizeof(int8_t));
    int32_t *BIAS  = (int32_t *)  malloc(7952*   sizeof(int32_t));
    uint16_t *M0   = (uint16_t *) malloc(7952*   sizeof(uint16_t));

    ReadData8("..\\data\\weight.bin", (char *) WEIGHT, 464016);
    ReadData32("..\\data\\bias.bin", (char *) BIAS, 7952);
    ReadData16("..\\data\\M0.bin", (char *)M0, 7952);

    for (int i = 0; i < 16; i=i+2) {

        wgt1_T wgt1[WGT_SIZE1][POSE_PE1];
        ofstream fpblk1wgt1("..\\Test\\blk1wgt1.txt", ios::out);
        if (!fpblk1wgt1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums1*config[i].ic_nums2*48*16/POSE_SIMD1/POSE_PE1; ++rep) {
            wgt1_pe_T temp_wgt;
            for (int pe = 0 ; pe < POSE_PE1; ++pe) {
                ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd;
                for (int p = 0; p < POSE_SIMD1; ++p) {
                    temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                        = WEIGHT[parm_size_debug[i].w1 + rep * POSE_SIMD1 * POSE_PE1 + pe * POSE_SIMD1 + p];
                }
                cout << hex;
                wgt1[rep][pe] = temp_wgt_simd;
                temp_wgt((pe+1)*POSE_SIMD1*POSE_W_BIT-1, pe*POSE_SIMD1*POSE_W_BIT) = temp_wgt_simd;
            }
            fpblk1wgt1 << temp_wgt << "  " << endl;
        }
        fpblk1wgt1.close();

        wgt2_T wgt2[WGT_SIZE2];
        ofstream fpblk1wgt2("..\\Test\\blk1wgt2.txt", ios::out);
        if (!fpblk1wgt2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*48*3*3/POSE_SIMD2; ++rep) {
            ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD2; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                    = WEIGHT[parm_size_debug[i].w2 + rep * POSE_SIMD2 + p];
            }
            cout << hex;
            fpblk1wgt2 << temp_wgt_simd << "  " << endl;
            wgt2[rep] = temp_wgt_simd;
        }
        fpblk1wgt2.close();

        //stream<wgt3_pe_T> wgt3("wgt3");
        wgt3_T wgt3[WGT_SIZE3][POSE_PE3];
        ofstream fpblk1wgt3("..\\Test\\blk1wgt3.txt", ios::out);
        if (!fpblk1wgt3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*config[i].oc_nums3*48*16/POSE_SIMD3/POSE_PE3; ++rep) {
            wgt3_pe_T temp_wgt;
            for (int pe = 0 ; pe < POSE_PE3; ++pe) {
                ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd;
                for (int p = 0; p < POSE_SIMD3; ++p) {
                    temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                        = WEIGHT[parm_size_debug[i].w3 + rep * POSE_SIMD3 * POSE_PE3 + pe * POSE_SIMD3 + p];
                }
                cout << hex;
                fpblk1wgt3 << temp_wgt_simd << "  " << endl;
                wgt3[rep][pe] = temp_wgt_simd;
            }
        }
        fpblk1wgt3.close();


        bias1_pe_T bias1[BIAS_M0_SIZE1];
        ofstream fpblk1bias1("..\\Test\\blk1bias1.txt", ios::out);
        if (!fpblk1bias1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*48/POSE_PE1; ++rep) {
            bias1_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE1; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i].b1 + rep * POSE_PE1 + p];
                cout << hex;
                //fpblk1bias1 << temp_bias << "  " << endl;
            }
            fpblk1bias1 << temp_bias << "  " << endl;
            bias1[rep] = temp_bias;
        }
        fpblk1bias1.close();

        bias2_pe_T bias2[BIAS_M0_SIZE2];
        ofstream fpblk1bias2("..\\Test\\blk1bias2.txt", ios::out);
        if (!fpblk1bias2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*48/POSE_PE2; ++rep) {
            bias2_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE2; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i].b2 + rep * POSE_PE2 + p];
                cout << hex;
                //fpblk1bias2 << temp_bias << "  " << endl;
            }
            fpblk1bias2 << temp_bias << "  " << endl;
            bias2[rep] = temp_bias;
        }
        fpblk1bias2.close();

        bias3_pe_T bias3[BIAS_M0_SIZE3];
        ofstream fpblk1bias3("..\\Test\\blk1bias3.txt", ios::out);
        if (!fpblk1bias3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].oc_nums3*16/POSE_PE3; ++rep) {
            bias3_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE3; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i].b3 + rep * POSE_PE3 + p];
                cout << hex;
                //fpblk1bias3 << temp_bias << "  " << endl;
            }
            fpblk1bias3 << temp_bias << "  " << endl;
            bias3[rep] = temp_bias;
        }
        fpblk1bias3.close();

        m0_1pe_T m0_1[BIAS_M0_SIZE1];
        ofstream fpblk1m1("..\\Test\\blk1m1.txt", ios::out);
        if (!fpblk1m1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*48/POSE_PE1; ++rep) {
            m0_1pe_T temp_m;
            for (int p = 0 ; p < POSE_PE1; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i].b1 + rep * POSE_PE1 + p];
                cout << hex;
            }
            fpblk1m1 << temp_m << "  " << endl;
            m0_1[rep] = temp_m;
        }
        fpblk1m1.close();

        m0_2pe_T m0_2[BIAS_M0_SIZE2];
        ofstream fpblk1m2("..\\Test\\blk1m2.txt", ios::out);
        if (!fpblk1m2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].ic_nums2*48/POSE_PE2; ++rep) {
            m0_2pe_T temp_m;
            for (int p = 0 ; p < POSE_PE2; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i].b2 + rep * POSE_PE2 + p];
                cout << hex;
            }
            fpblk1m2 << temp_m << "  " << endl;
            m0_2[rep] = temp_m;
        }
        fpblk1m2.close();

        m0_3pe_T m0_3[BIAS_M0_SIZE3];
        ofstream fpblk1m3("..\\Test\\blk1m3.txt", ios::out);
        if (!fpblk1m3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i].oc_nums3*16/POSE_PE3; ++rep) {
            m0_3pe_T temp_m;
            for (int p = 0 ; p < POSE_PE3; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i].b3 + rep * POSE_PE3 + p];
                cout << hex;
            }
            fpblk1m3 << temp_m << "  " << endl;
            m0_3[rep] = temp_m;
        }
        fpblk1m3.close();


        ap_uint<8> ROW1 = config[i].ih;
        ap_uint<8> ROW2 = config[i].ih;
        ap_uint<8> ROW3 = config[i].ih/config[i].s;
        ap_uint<8> COL1 = config[i].iw;
        ap_uint<8> COL2 = config[i].iw;
        ap_uint<8> COL3 = config[i].iw/config[i].s;
        ap_uint<4> INCH_NUMS1 = config[i].ic_nums1;
        ap_uint<4> OUTCH_NUMS1 = config[i].ic_nums2;
        ap_uint<4> CH_NUMS2 = config[i].ic_nums2;
        ap_uint<4> INCH_NUMS3 = config[i].ic_nums2;
        ap_uint<4> OUTCH_NUMS3 = config[i].oc_nums3;
        ap_uint<2> STRIDE = config[i].s;
        ap_uint<1> IS_ADD = config[i].is_add;
        ap_uint<1> NEXT_ADD = config[i].next_add;

        PosenetBlockAlpha(blk_in, blk_out, add_in, add_out,
                          wgt1, wgt2, wgt3, bias1, bias2, bias3, m0_1, m0_2, m0_3,
                          ROW1, ROW2, ROW3, COL1, COL2, COL3,
                          INCH_NUMS1, OUTCH_NUMS1, CH_NUMS2, INCH_NUMS3, OUTCH_NUMS3,
                          STRIDE, IS_ADD, NEXT_ADD);


        wgt1_T wgt4[WGT_SIZE1][POSE_PE1];
        ofstream fpblk2wgt1("..\\Test\\blk2wgt1.txt", ios::out);
        if (!fpblk2wgt1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums1*config[i+1].ic_nums2*48*16/POSE_SIMD1/POSE_PE1; ++rep) {
            wgt1_pe_T temp_wgt;
            for (int pe = 0 ; pe < POSE_PE1; ++pe) {
                ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd;
                for (int p = 0; p < POSE_SIMD1; ++p) {
                    temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                        = WEIGHT[parm_size_debug[i + 1].w1 + rep * POSE_SIMD1 * POSE_PE1 + pe * POSE_SIMD1 + p];
                }
                wgt4[rep][pe] = temp_wgt_simd;
                cout << hex;
                //fpblk1wgt1 << temp_wgt_simd << "  " << endl;
                temp_wgt((pe+1)*POSE_SIMD1*POSE_W_BIT-1, pe*POSE_SIMD1*POSE_W_BIT) = temp_wgt_simd;
            }
            fpblk2wgt1 << temp_wgt << "  " << endl;
        }
        fpblk2wgt1.close();

        wgt2_T wgt5[WGT_SIZE2];
        ofstream fpblk2wgt2("..\\Test\\blk2wgt2.txt", ios::out);
        if (!fpblk2wgt2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*48*3*3/POSE_SIMD2; ++rep) {
            ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD2; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                    = WEIGHT[parm_size_debug[i + 1].w2 + rep * POSE_SIMD2 + p];
            }
            cout << hex;
            fpblk2wgt1 << temp_wgt_simd << "  " << endl;
            wgt5[rep] = temp_wgt_simd;
        }
        fpblk2wgt2.close();

        wgt3_T wgt6[WGT_SIZE3][POSE_PE3];
        ofstream fpblk2wgt3("..\\Test\\blk2wgt3.txt", ios::out);
        if (!fpblk2wgt3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*config[i+1].oc_nums3*48*16/POSE_SIMD3/POSE_PE3; ++rep) {
            wgt3_pe_T temp_wgt;
            for (int pe = 0 ; pe < POSE_PE3; ++pe) {
                ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd;
                for (int p = 0; p < POSE_SIMD3; ++p) {
                    temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT)
                        = WEIGHT[parm_size_debug[i + 1].w3 + rep * POSE_SIMD3 * POSE_PE3 + pe * POSE_SIMD3 + p];
                }
                wgt6[rep][pe] = temp_wgt_simd;
                cout << hex;
                fpblk2wgt3 << temp_wgt_simd << "  " << endl;
                temp_wgt((pe+1)*POSE_SIMD3*POSE_W_BIT-1, pe*POSE_SIMD3*POSE_W_BIT) = temp_wgt_simd;
            }
        }
        fpblk2wgt3.close();

        bias1_pe_T bias4[BIAS_M0_SIZE1];
        ofstream fpblk2bias1("..\\Test\\blk2bias1.txt", ios::out);
        if (!fpblk2bias1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*48/POSE_PE1; ++rep) {
            bias1_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE1; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i + 1].b1 + rep * POSE_PE1 + p];
                cout << hex;
                //fpblk1bias1 << temp_bias << "  " << endl;
            }
            fpblk2bias1 << temp_bias << "  " << endl;
            bias4[rep] = temp_bias;
        }
        fpblk2bias1.close();

        bias2_pe_T bias5[BIAS_M0_SIZE2];
        ofstream fpblk2bias2("..\\Test\\blk2bias2.txt", ios::out);
        if (!fpblk2bias2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*48/POSE_PE2; ++rep) {
            bias2_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE2; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i + 1].b2 + rep * POSE_PE2 + p];
                cout << hex;
                //fpblk1bias2 << temp_bias << "  " << endl;
            }
            fpblk2bias2 << temp_bias << "  " << endl;
            bias5[rep] = temp_bias;
        }
        fpblk2bias2.close();

        bias3_pe_T bias6[BIAS_M0_SIZE3];
        ofstream fpblk2bias3("..\\Test\\blk2bias3.txt", ios::out);
        if (!fpblk2bias3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].oc_nums3*16/POSE_PE3; ++rep) {
            bias3_pe_T temp_bias;
            for (int p = 0 ; p < POSE_PE3; ++p) {
                temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT)
                    = BIAS[parm_size_debug[i + 1].b3 + rep * POSE_PE3 + p];
                cout << hex;
                //fpblk1bias3 << temp_bias << "  " << endl;
            }
            fpblk2bias3 << temp_bias << "  " << endl;
            bias6[rep] = temp_bias;
        }
        fpblk2bias3.close();

        m0_1pe_T m0_4[BIAS_M0_SIZE1];
        ofstream fpblk2m1("..\\Test\\blk2m1.txt", ios::out);
        if (!fpblk2m1)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*48/POSE_PE1; ++rep) {
            m0_1pe_T temp_m;
            for (int p = 0 ; p < POSE_PE1; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i + 1].b1 + rep * POSE_PE1 + p];
                cout << hex;
            }
            fpblk2m1 << temp_m << "  " << endl;
            m0_4[rep] = temp_m;
        }
        fpblk2m1.close();

        m0_2pe_T m0_5[BIAS_M0_SIZE2];
        ofstream fpblk2m2("..\\Test\\blk2m2.txt", ios::out);
        if (!fpblk2m2)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].ic_nums2*48/POSE_PE2; ++rep) {
            m0_2pe_T temp_m;
            for (int p = 0 ; p < POSE_PE2; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i + 1].b2 + rep * POSE_PE2 + p];
                cout << hex;
            }
            fpblk2m2 << temp_m << "  " << endl;
            m0_5[rep] = temp_m;
        }
        fpblk2m2.close();

        m0_3pe_T m0_6[BIAS_M0_SIZE3];
        ofstream fpblk2m3("..\\Test\\blk2m3.txt", ios::out);
        if (!fpblk2m3)
            cout << "no such file" << endl;
        for (int rep = 0; rep < config[i+1].oc_nums3*16/POSE_PE3; ++rep) {
            m0_3pe_T temp_m;
            for (int p = 0 ; p < POSE_PE3; ++p) {
                temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT)
                    = M0[parm_size_debug[i + 1].b3 + rep * POSE_PE3 + p];
                cout << hex;
            }
            fpblk2m3 << temp_m << "  " << endl;
            m0_6[rep] = temp_m;
        }
        fpblk2m3.close();

        ap_uint<8> ROW1_ = config[i+1].ih;
        ap_uint<8> ROW2_ = config[i+1].ih;
        ap_uint<8> ROW3_ = config[i+1].ih/config[i+1].s;
        ap_uint<8> COL1_ = config[i+1].iw;
        ap_uint<8> COL2_ = config[i+1].iw;
        ap_uint<8> COL3_ = config[i+1].iw/config[i+1].s;
        ap_uint<4> INCH_NUMS1_ = config[i+1].ic_nums1;
        ap_uint<4> OUTCH_NUMS1_ = config[i+1].ic_nums2;
        ap_uint<4> CH_NUMS2_ = config[i+1].ic_nums2;
        ap_uint<4> INCH_NUMS3_ = config[i+1].ic_nums2;
        ap_uint<4> OUTCH_NUMS3_ = config[i+1].oc_nums3;
        ap_uint<2> STRIDE_ = config[i+1].s;
        ap_uint<1> IS_ADD_ = config[i+1].is_add;
        ap_uint<1> NEXT_ADD_ = config[i+1].next_add;

        PosenetBlockAlpha(blk_out, blk_in, add_out, add_in,
                          wgt4, wgt5, wgt6, bias4, bias5, bias6, m0_4, m0_5, m0_6,
                          ROW1_, ROW2_, ROW3_,
                          COL1_, COL2_, COL3_,
                          INCH_NUMS1_, OUTCH_NUMS1_, CH_NUMS2_, INCH_NUMS3_, OUTCH_NUMS3_,
                          STRIDE_, IS_ADD_, NEXT_ADD_);


//        if (i == 14) {
//
//            cout << dec << "out size: " << blk_in.size() << endl;
//            cout << dec << "add_out size: " << add_in.size() << endl;
//            ofstream fpblk2cv3("..\\Test\\blk16cv3.txt", ios::out);
//            if (!fpblk2cv3)
//                cout << "no such file" << endl;
//            for (int h = 0; h < ROW3_; ++h) {
//                for (int w = 0; w < COL3_ ; ++w) {
//                    for (int nums = 0; nums < OUTCH_NUMS3_; nums++) {
//                        ap_int<POSE_OUT_CH * POSE_IN_BIT> temp = blk_in.read();
//                        for (int ch = 0; ch < POSE_OUT_CH; ++ch) {
//                            cout << dec;
//                            fpblk2cv3 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
//                        }
//                    }
//                    fpblk2cv3 << endl;
//                }
//            }
//            fpblk2cv3.close();
//
//        }
    }
    free(WEIGHT);
    free(BIAS);
    free(M0);

    stream<ap_int<POSE_CV7_OUTCH*12>> out("out");
    PosenetDecv(blk_in, out);

}

#if 0
int main() {
    stream<ap_int<POSE_HCV0_INCH*POSE_IN_BIT>> in("testin");
    int8_t * img = (int8_t *) malloc(256*192*3* sizeof(int8_t));
    ReadData8("..\\data\\input_256x192.bin", (char*)img, 256*192*3);

    ofstream fpconv0in("..\\Test\\hconv0in.txt", ios::out);
    if (!fpconv0in)
        cout << "no such file" << endl;
    for (int h = 0; h < POSE_HCV0_ROW; ++h) {
        for (int w = 0; w < POSE_HCV0_COL; ++w) {
            ap_int<POSE_HCV0_INCH*POSE_IN_BIT> temp_in;
            for (int ch = 0; ch < POSE_HCV0_INCH; ++ch) {
                temp_in((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT) = img[h*POSE_HCV0_COL*POSE_HCV0_INCH + w*POSE_HCV0_INCH + ch];
                cout << dec;
                fpconv0in << ap_int<8>(temp_in((ch+1)*POSE_IN_BIT-1, ch*POSE_IN_BIT)) << "  ";
            }
            fpconv0in << endl;
            in.write(temp_in);
        }
    }
    fpconv0in.close();

    stream<ap_int<POSE_HCV2_OUTCH*POSE_IN_BIT>> head_out("head_out");

    PosenetHead(in, head_out);

    stream<ap_int<POSE_PE3*POSE_IN_BIT>> add_in("add_in");
    stream<ap_int<POSE_PE3*POSE_IN_BIT>> add_out("add_out");
    stream<ap_int<POSE_OUT_CH*POSE_IN_BIT>> blk1_out("blk1_out");

    int8_t *weight1 = (int8_t *) malloc(16*48* sizeof(int8_t));
    ReadData8("..\\data\\bin\\3_weight.bin", (char *) weight1, 16 * 48);
    stream<wgt1_pe_T> wgt1("wgt1");
    ofstream fpblk1wgt1("..\\Test\\blk1wgt1_sp.txt", ios::out);
    if (!fpblk1wgt1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48*16/POSE_SIMD1/POSE_PE1; ++rep) {
        wgt1_pe_T temp_wgt;
        for (int pe = 0 ; pe < POSE_PE1; ++pe) {
            ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD1; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight1[rep*POSE_SIMD1*POSE_PE1 + pe*POSE_SIMD1 + p];
            }
            cout << hex;
            //fpblk1wgt1 << temp_wgt_simd << "  " << endl;
            temp_wgt((pe+1)*POSE_SIMD1*POSE_W_BIT-1, pe*POSE_SIMD1*POSE_W_BIT) = temp_wgt_simd;
        }
        fpblk1wgt1 << temp_wgt << "  " << endl;
        wgt1.write(temp_wgt);
    }
    fpblk1wgt1.close();

    int8_t *weight2 = (int8_t *) malloc(48*3*3* sizeof(int8_t));
    ReadData8("..\\data\\bin\\4_weight.bin", (char *) weight2, 48 * 3 * 3);
    stream<wgt2_T> wgt2("wgt2");
    ofstream fpblk1wgt2("..\\Test\\blk1wgt2.txt", ios::out);
    if (!fpblk1wgt2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48*3*3/POSE_SIMD2; ++rep) {
        ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
        for (int p = 0; p < POSE_SIMD2; ++p) {
            temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight2[rep*POSE_SIMD2 + p];
        }
        cout << hex;
        fpblk1wgt2 << temp_wgt_simd << "  " << endl;
        wgt2.write(temp_wgt_simd);
    }
    fpblk1wgt2.close();

    int8_t *weight3 = (int8_t *) malloc(48*16* sizeof(int8_t));
    ReadData8("..\\data\\bin\\5_weight.bin", (char *) weight3, 48 * 16);
    stream<wgt3_pe_T> wgt3("wgt3");
    ofstream fpblk1wgt3("..\\Test\\blk1wgt3.txt", ios::out);
    if (!fpblk1wgt3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48*16/POSE_SIMD3/POSE_PE3; ++rep) {
        wgt3_pe_T temp_wgt;
        for (int pe = 0 ; pe < POSE_PE3; ++pe) {
            ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD3; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight3[rep*POSE_SIMD3*POSE_PE3 + pe*POSE_SIMD3 + p];
            }
            cout << hex;
            fpblk1wgt3 << temp_wgt_simd << "  " << endl;
            temp_wgt((pe+1)*POSE_SIMD3*POSE_W_BIT-1, pe*POSE_SIMD3*POSE_W_BIT) = temp_wgt_simd;
        }
        //fpblk1wgt3 << temp_wgt << "  " << endl;
        wgt3.write(temp_wgt);
    }
    fpblk1wgt3.close();

    int32_t *b1 = (int32_t *) malloc(48* sizeof(int32_t));
    ReadData32("..\\data\\bin\\3_bias.bin", (char *) b1, 48);
    stream<bias1_pe_T> bias1("bias1");
    ofstream fpblk1bias1("..\\Test\\blk1bias1.txt", ios::out);
    if (!fpblk1bias1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48/POSE_PE1; ++rep) {
        bias1_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE1; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b1[rep*POSE_PE1 + p];
            cout << hex;
            //fpblk1bias1 << temp_bias << "  " << endl;
        }
        fpblk1bias1 << temp_bias << "  " << endl;
        bias1.write(temp_bias);
    }
    fpblk1bias1.close();

    int32_t *b2 = (int32_t *) malloc(48* sizeof(int32_t));
    ReadData32("..\\data\\bin\\4_bias.bin", (char *)b2, 48);
    stream<bias2_pe_T> bias2("bias2");
    ofstream fpblk1bias2("..\\Test\\blk1bias2.txt", ios::out);
    if (!fpblk1bias2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48/POSE_PE2; ++rep) {
        bias2_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE2; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b2[rep*POSE_PE2 + p];
            cout << hex;
            //fpblk1bias2 << temp_bias << "  " << endl;
        }
        fpblk1bias2 << temp_bias << "  " << endl;
        bias2.write(temp_bias);
    }
    fpblk1bias2.close();

    int32_t *b3 = (int32_t *) malloc(48* sizeof(int32_t));
    ReadData32("..\\data\\bin\\5_bias.bin", (char *)b3, 48);
    stream<bias3_pe_T> bias3("bias3");
    ofstream fpblk1bias3("..\\Test\\blk1bias3.txt", ios::out);
    if (!fpblk1bias3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 16/POSE_PE3; ++rep) {
        bias3_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE3; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b3[rep*POSE_PE3 + p];
            cout << hex;
            //fpblk1bias3 << temp_bias << "  " << endl;
        }
        fpblk1bias3 << temp_bias << "  " << endl;
        bias3.write(temp_bias);
    }
    fpblk1bias3.close();

    uint16_t *M0_1 = (uint16_t *) malloc(48*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\3_M0.bin", (char *)M0_1, 48);
    stream<m0_1pe_T> m0_1("m0_1");
    ofstream fpblk1m1("..\\Test\\blk1m1.txt", ios::out);
    if (!fpblk1m1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48/POSE_PE1; ++rep) {
        m0_1pe_T temp_m;
        for (int p = 0 ; p < POSE_PE1; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_1[rep*POSE_PE1 + p];
            cout << hex;
        }
        fpblk1m1 << temp_m << "  " << endl;
        m0_1.write(temp_m);
    }
    fpblk1m1.close();

    uint16_t *M0_2 = (uint16_t *) malloc(48*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\4_M0.bin", (char *)M0_2, 48);
    stream<m0_2pe_T> m0_2("m0_2");
    ofstream fpblk1m2("..\\Test\\blk1m2.txt", ios::out);
    if (!fpblk1m2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 48/POSE_PE2; ++rep) {
        m0_2pe_T temp_m;
        for (int p = 0 ; p < POSE_PE2; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_2[rep*POSE_PE2 + p];
            cout << hex;
        }
        fpblk1m2 << temp_m << "  " << endl;
        m0_2.write(temp_m);
    }
    fpblk1m2.close();

    uint16_t *M0_3 = (uint16_t *) malloc(16*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\5_M0.bin", (char *)M0_3, 16);
    stream<m0_3pe_T> m0_3("m0_3");
    ofstream fpblk1m3("..\\Test\\blk1m3.txt", ios::out);
    if (!fpblk1m3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 16/POSE_PE3; ++rep) {
        m0_3pe_T temp_m;
        for (int p = 0 ; p < POSE_PE3; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_3[rep*POSE_PE3 + p];
            cout << hex;
        }
        fpblk1m3 << temp_m << "  " << endl;
        m0_3.write(temp_m);
    }
    fpblk1m3.close();

    stream<ap_int<POSE_IN_CH*POSE_IN_BIT>> blk1_in("blk1_in");
    for (int h = 0; h < POSE_HCV2_ROW; ++h) {
        for (int w = 0; w < POSE_HCV2_COL; ++w) {
            ap_int<POSE_IN_CH*POSE_IN_BIT> temp = head_out.read();
            blk1_in.write(temp);
        }
    }
    PosenetBlockAlpha(blk1_in, blk1_out, add_in, add_out,
                 wgt1, wgt2, wgt3, bias1, bias2, bias3, m0_1, m0_2, m0_3,
                 128, 128, 64, 96, 96, 48, 1, 1, 1, 1, 1, 2, 0, 1);

#if 0
    cout << dec << "blk1_out size: " << blk1_out.size() << endl;
    ofstream fpblk1cv3("..\\Test\\blk1cv3.txt", ios::out);
    if (!fpblk1cv3)
        cout << "no such file" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48 ; ++w) {
            for (int nums = 0; nums < 1; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> temp = blk1_out.read();
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

    int8_t *weight4 = (int8_t *) malloc(16*96* sizeof(int8_t));
    ReadData8("..\\data\\bin\\6_weight.bin", (char *) weight4, 16 * 96);
    stream<wgt1_pe_T> wgt4("wgt4");
    ofstream fpblk2wgt1("..\\Test\\blk2wgt1.txt", ios::out);
    if (!fpblk2wgt1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96*16/POSE_SIMD1/POSE_PE1; ++rep) {
        wgt1_pe_T temp_wgt;
        for (int pe = 0 ; pe < POSE_PE1; ++pe) {
            ap_int<POSE_SIMD1*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD1; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight4[rep*POSE_SIMD1*POSE_PE1 + pe*POSE_SIMD1 + p];
            }
            cout << hex;
            //fpblk1wgt1 << temp_wgt_simd << "  " << endl;
            temp_wgt((pe+1)*POSE_SIMD1*POSE_W_BIT-1, pe*POSE_SIMD1*POSE_W_BIT) = temp_wgt_simd;
        }
        fpblk2wgt1 << temp_wgt << "  " << endl;
        wgt4.write(temp_wgt);
    }
    fpblk2wgt1.close();

    int8_t *weight5 = (int8_t *) malloc(96*3*3* sizeof(int8_t));
    ReadData8("..\\data\\bin\\7_weight.bin", (char *) weight5, 96 * 3 * 3);
    stream<wgt2_T> wgt5("wgt5");
    ofstream fpblk2wgt2("..\\Test\\blk2wgt2.txt", ios::out);
    if (!fpblk2wgt2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96*3*3/POSE_SIMD2; ++rep) {
        ap_int<POSE_SIMD2*POSE_W_BIT> temp_wgt_simd;
        for (int p = 0; p < POSE_SIMD2; ++p) {
            temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight5[rep*POSE_SIMD2 + p];
        }
        cout << hex;
        fpblk2wgt1 << temp_wgt_simd << "  " << endl;
        wgt5.write(temp_wgt_simd);
    }
    fpblk2wgt2.close();

    int8_t *weight6 = (int8_t *) malloc(96*16*sizeof(int8_t));
    ReadData8("..\\data\\bin\\8_weight.bin", (char *) weight6, 96*16);
    stream<wgt3_pe_T> wgt6("wgt6");
    ofstream fpblk2wgt3("..\\Test\\blk2wgt3.txt", ios::out);
    if (!fpblk2wgt3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96*16/POSE_SIMD3/POSE_PE3; ++rep) {
        wgt3_pe_T temp_wgt;
        for (int pe = 0 ; pe < POSE_PE3; ++pe) {
            ap_int<POSE_SIMD3*POSE_W_BIT> temp_wgt_simd;
            for (int p = 0; p < POSE_SIMD3; ++p) {
                temp_wgt_simd((p+1)*POSE_W_BIT-1, p*POSE_W_BIT) = weight6[rep*POSE_SIMD3*POSE_PE3 + pe*POSE_SIMD3 + p];
            }
            cout << hex;
            fpblk2wgt3 << temp_wgt_simd << "  " << endl;
            temp_wgt((pe+1)*POSE_SIMD3*POSE_W_BIT-1, pe*POSE_SIMD3*POSE_W_BIT) = temp_wgt_simd;
        }
        //fpblk1wgt3 << temp_wgt << "  " << endl;
        wgt6.write(temp_wgt);
    }
    fpblk2wgt3.close();

    int32_t *b4 = (int32_t *) malloc(96*sizeof(int32_t));
    ReadData32("..\\data\\bin\\6_bias.bin", (char *) b4, 96);
    stream<bias1_pe_T> bias4("bias4");
    ofstream fpblk2bias1("..\\Test\\blk2bias1.txt", ios::out);
    if (!fpblk2bias1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96/POSE_PE1; ++rep) {
        bias1_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE1; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b4[rep*POSE_PE1 + p];
            cout << hex;
            //fpblk1bias1 << temp_bias << "  " << endl;
        }
        fpblk2bias1 << temp_bias << "  " << endl;
        bias4.write(temp_bias);
    }
    fpblk2bias1.close();

    int32_t *b5 = (int32_t *) malloc(96*sizeof(int32_t));
    ReadData32("..\\data\\bin\\7_bias.bin", (char *)b5, 96);
    stream<bias2_pe_T> bias5("bias5");
    ofstream fpblk2bias2("..\\Test\\blk2bias2.txt", ios::out);
    if (!fpblk2bias2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96/POSE_PE2; ++rep) {
        bias2_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE2; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b5[rep*POSE_PE2 + p];
            cout << hex;
            //fpblk1bias2 << temp_bias << "  " << endl;
        }
        fpblk2bias2 << temp_bias << "  " << endl;
        bias5.write(temp_bias);
    }
    fpblk2bias2.close();

    int32_t *b6 = (int32_t *) malloc(16* sizeof(int32_t));
    ReadData32("..\\data\\bin\\8_bias.bin", (char *)b6, 16);
    stream<bias3_pe_T> bias6("bias6");
    ofstream fpblk2bias3("..\\Test\\blk2bias3.txt", ios::out);
    if (!fpblk2bias3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 16/POSE_PE3; ++rep) {
        bias3_pe_T temp_bias;
        for (int p = 0 ; p < POSE_PE3; ++p) {
            temp_bias((p+1)*POSE_BIAS_BIT-1, p*POSE_BIAS_BIT) = b6[rep*POSE_PE3 + p];
            cout << hex;
            //fpblk1bias3 << temp_bias << "  " << endl;
        }
        fpblk2bias3 << temp_bias << "  " << endl;
        bias6.write(temp_bias);
    }
    fpblk2bias3.close();

    uint16_t *M0_4 = (uint16_t *) malloc(96*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\6_M0.bin", (char *)M0_4, 96);
    stream<m0_1pe_T> m0_4("m0_4");
    ofstream fpblk2m1("..\\Test\\blk2m1.txt", ios::out);
    if (!fpblk2m1)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96/POSE_PE1; ++rep) {
        m0_1pe_T temp_m;
        for (int p = 0 ; p < POSE_PE1; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_4[rep*POSE_PE1 + p];
            cout << hex;
        }
        fpblk2m1 << temp_m << "  " << endl;
        m0_4.write(temp_m);
    }
    fpblk2m1.close();

    uint16_t *M0_5 = (uint16_t *) malloc(96*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\7_M0.bin", (char *)M0_5, 96);
    stream<m0_2pe_T> m0_5("m0_5");
    ofstream fpblk2m2("..\\Test\\blk2m2.txt", ios::out);
    if (!fpblk2m2)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 96/POSE_PE2; ++rep) {
        m0_2pe_T temp_m;
        for (int p = 0 ; p < POSE_PE2; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_5[rep*POSE_PE2 + p];
            cout << hex;
        }
        fpblk2m2 << temp_m << "  " << endl;
        m0_5.write(temp_m);
    }
    fpblk2m2.close();

    uint16_t *M0_6 = (uint16_t *) malloc(16*sizeof(uint16_t));
    ReadData16("..\\data\\bin\\8_M0.bin", (char *)M0_6, 16);
    stream<m0_3pe_T> m0_6("m0_6");
    ofstream fpblk2m3("..\\Test\\blk2m3.txt", ios::out);
    if (!fpblk2m3)
        cout << "no such file" << endl;
    for (int rep = 0; rep < 16/POSE_PE3; ++rep) {
        m0_3pe_T temp_m;
        for (int p = 0 ; p < POSE_PE3; ++p) {
            temp_m((p+1)*POSE_M0_BIT-1, p*POSE_M0_BIT) = M0_6[rep*POSE_PE3 + p];
            cout << hex;
        }
        fpblk2m3 << temp_m << "  " << endl;
        m0_6.write(temp_m);
    }
    fpblk2m3.close();

    stream<ap_int<POSE_PE3*POSE_IN_BIT>> add_blk2_out("add_out");
    stream<ap_int<POSE_OUT_CH*POSE_IN_BIT>> blk2_out("blk2_out");
    PosenetBlockAlpha(blk1_out, blk2_out, add_out, add_blk2_out,
                 wgt4, wgt5, wgt6, bias4, bias5, bias6, m0_4, m0_5, m0_6,
                 64, 64, 64, 48, 48, 48, 1, 2, 2, 2, 1, 1, 1, 0);
#if 0
    cout << dec << "blk2_out size: " << blk2_out.size() << endl;
    ofstream fpblk2cv3("..\\Test\\blk2cv3.txt", ios::out);
    if (!fpblk2cv3)
        cout << "no such file" << endl;
    for (int h = 0; h < 64; ++h) {
        for (int w = 0; w < 48 ; ++w) {
            for (int nums = 0; nums < 1; nums++) {
                ap_int<POSE_OUT_CH * POSE_IN_BIT> temp = blk2_out.read();
                for (int ch = 0; ch < POSE_OUT_CH; ++ch) {
                    cout << dec;
                    fpblk2cv3 << dec << ap_int<8>(temp((ch + 1) * POSE_IN_BIT - 1, ch * POSE_IN_BIT)) << "  ";
                }
            }
            fpblk2cv3 << endl;
        }
    }
    fpblk2cv3.close();
#endif

}
#endif

#if 0
int main() {
    GenParamW<3,16,8,WGT_HCV0_SIZE>("..\\Test\\testw.txt");
    GenParamB<16,16,BIAS_M0_HCV0_SIZE>("..\\Test\\testb.txt");
    GenParamW<8,1,8,WGT_HCV1_SIZE>("..\\Test\\testw1.txt");
    GenParamB<8,16,BIAS_M0_HCV1_SIZE>("..\\Test\\testb1.txt");
    GenParamW<4,4,8,WGT_HCV2_SIZE>("..\\Test\\testw2.txt");
    GenParamB<4,16,BIAS_M0_HCV2_SIZE>("..\\Test\\testb2.txt");


    hls::stream<ap_int<16*8>> testin("input stream");
    ap_int<SIMD*IN_BIT> count = 1;
    for (int h = 0; h < IN_ROW; ++h) {
        for (int w = 0; w < IN_COL ; ++w) {
            cout << "[ ";
            for (int nums = 0; nums < INCH_NUMS; ++nums) {
                ap_int<SIMD*IN_BIT> tmp_in;
                for (int c = 0; c < SIMD; ++c) {
                    tmp_in = tmp_in << IN_BIT;
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

    cout << dec << "test in size: " << testin.size() << endl;

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
    Padding<IN_CH, IN_BIT, 1>(testin, padding_out, INCH_NUMS, IN_ROW, IN_COL);

    cout << dec << "padding_out size: " << padding_out.size() << endl;

    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;

#if 0
    ofstream fpconv0pad("..\\Test\\conv0pad.txt", ios::out);
    if (!fpconv0pad)
        cout << "no such file" << endl;
    for (int h = 0; h < INTER_ROW; ++h) {
        for (int w = 0; w < INTER_COL ; ++w) {
            fpconv0pad << "[";
            for (int c = 0; c < INCH_NUMS; ++c) {
                cout << hex;
                fpconv0pad << hex << " " << padding_out.read() << " ";
            }
            fpconv0pad << "] ";
        }
        fpconv0pad << endl;
    }
    fpconv0pad.close();
#endif

    stream<ap_int<IN_CH*IN_BIT>> swu_out("swu_out");
    SWU<K,IN_BIT,IN_CH>(padding_out, swu_out, INCH_NUMS, INTER_ROW, INTER_COL, S);

    cout << dec << "swu_out size: " << swu_out.size() << endl;
#if 0
    const unsigned STEPS = ((INTER_COL - K) / S + 1) * ((INTER_ROW - K) / S + 1);
    ofstream fpconv0swu("..\\Test\\conv0swu.txt", ios::out);
    if (!fpconv0swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS*INCH_NUMS; step++) {
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

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;
    stream<ap_int<PE*W_BIT>> weights("weights");
    stream<ap_int<PE*BIAS_BIT>> bias("bias");
    stream<ap_uint<PE*M0_BIT>> m0("m0");

    cout << "wgt: " << endl;
    for (int i = 0; i < OUT_ROW*OUT_COL; ++i) {
        count = 1;
        for (int nums = 0; nums < INCH_NUMS; ++nums) {
            for (int h = 0; h < K; ++h) {
                cout << "[";
                for (int w = 0; w < K; ++w) {
                    ap_int<SIMD * W_BIT> tmp_wgt;
                    for (int c = 0; c < SIMD; ++c) {
                        tmp_wgt = tmp_wgt << W_BIT;
                        tmp_wgt |= count;
                        count += 1;
                    }
                    cout << hex << tmp_wgt << " ";
                    weights.write(tmp_wgt);
                }
                cout << "]" << endl;
            }
            cout << endl;
        }
    }

    cout << "bias: " << endl;
    for (int i = 0; i < OUT_ROW* OUT_COL; ++i) {
        ap_int<PE * BIAS_BIT> count_bias = 1;
        for (int nums = 0; nums < INCH_NUMS; ++nums) {
            ap_int<PE * BIAS_BIT> tmp_bias;
            for (int c = 0; c < SIMD; ++c) {
                tmp_bias = tmp_bias << BIAS_BIT;
                tmp_bias |= count_bias;
                count_bias += 1;
                count_bias = count_bias % 3;
            }
            cout << hex << tmp_bias << " ";
            bias.write(tmp_bias);
        }
        cout << endl;
    }

    cout << "m0: " << endl;
    for (int i = 0; i < OUT_ROW*OUT_COL; ++i) {
        ap_uint<PE * M0_BIT> count_m0 = 1;
        for (int nums = 0; nums < INCH_NUMS; ++nums) {
            ap_uint<PE * M0_BIT> tmp_m0;
            for (int c = 0; c < SIMD; ++c) {
                tmp_m0 = tmp_m0 << M0_BIT;
                tmp_m0 |= count_m0;
                count_m0 += 1;
                count_m0 = count_m0 % 2;
            }
            cout << hex << tmp_m0 << " ";
            m0.write(tmp_m0);
        }
        cout << endl;
    }

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnit<IN_BIT, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE, RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE>
            (swu_out, mvau_out, weights, bias, m0, INCH_NUMS*IN_CH*K*K, OUT_CH, INCH_NUMS, OUT_ROW*OUT_COL);
    cout << dec << "mvau_out size: " << mvau_out.size() << endl;
}
#endif

#if 0
int main() {
    GenParamW<8,16,8,WGT_PWCV6_SIZE>("..\\Test\\testw.txt");
    GenParamB<16,16,BIAS_M0_PWCV6_SIZE>("..\\Test\\testb.txt");

    hls::stream<ap_int<IN_CH*POSE_IN_BIT>> testin("input stream");
    ap_int<IN_CH*POSE_IN_BIT> count = 1;
    for (int h = 0; h < IN_ROW; ++h) {
        for (int w = 0; w < IN_COL ; ++w) {
            cout << "[ ";
            ap_int<IN_CH*POSE_IN_BIT> tmp_in;
            for (int c = 0; c < IN_CH; ++c) {
                tmp_in = tmp_in << POSE_IN_BIT;
                tmp_in |= count;
                count += 1;
            }
            cout << hex << "tmp_in:" << tmp_in << " " ;
            testin.write(tmp_in);
            cout << " ]";
            cout << endl;
        }
    }

    unsigned INTER_ROW = IN_ROW;
    unsigned INTER_COL = IN_COL;

    INTER_ROW *= 2;
    INTER_COL *= 2;
    stream<ap_int<IN_CH*IN_BIT> > deconvpad_out("deconvpad_out");
    DilationPaddingT<IN_ROW, IN_COL, IN_CH, IN_BIT>(testin, deconvpad_out);

    stream<ap_int<IN_CH*IN_BIT> > padding_out("samepad_out");
    PaddingT<IN_ROW+IN_ROW, IN_COL+IN_COL, IN_CH, IN_BIT, 1>(deconvpad_out, padding_out);

    INTER_ROW += 2;
    INTER_COL += 2;
#if 0
    ofstream fpconv0pad("..\\Test\\conv0pad.txt", ios::out);
    if (!fpconv0pad)
        cout << "no such file" << endl;
    for (int h = 0; h < INTER_ROW; ++h) {
        for (int w = 0; w < INTER_COL ; ++w) {
            fpconv0pad << "[";
            fpconv0pad << hex << " " << padding_out.read() << " ";
            fpconv0pad << "] ";
        }
        fpconv0pad << endl;
    }
    fpconv0pad.close();
#endif

    stream<ap_int<SIMD*IN_BIT>> swu_out("swu_out");
    SWUT<K,IN_ROW*2+2,IN_COL*2+2,IN_BIT,IN_CH,IN_CH/SIMD,SIMD,1>(padding_out, swu_out);

    cout << dec << "swu_out size: " << swu_out.size() << endl;
#if 0
    const unsigned STEPS = ((INTER_COL - K) / S + 1) * ((INTER_ROW - K) / S + 1);
    ofstream fpconv0swu("..\\Test\\conv0swu.txt", ios::out);
    if (!fpconv0swu)
        cout << "no such file" << endl;
    for (int step = 0; step < STEPS*IN_CH/SIMD; step++) {
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

    const unsigned OUT_ROW = IN_ROW+IN_ROW;
    const unsigned OUT_COL = IN_COL+IN_COL;

    stream<ap_int<PE*IN_BIT>> mvau_out("mvau_out");
    DwcvMatrixVectorActUnitT<IN_CH*K*K, OUT_CH, IN_BIT, IN_CH_NUMS, OUT_BIT, MUL_BIT, W_BIT, BIAS_BIT, M0_BIT, SIMD, PE,
            RSHIFT, WGT_ARRAYSIZE, BIAS_M0_ARRAYSIZE, OUT_ROW*OUT_COL>
            (swu_out, mvau_out, weights, bias, m0);

    cout << dec << "mvau_out size: " << mvau_out.size() << endl;
#if 1
    ofstream fpconv0mvau("..\\Test\\conv0mvau.txt", ios::out);
    if (!fpconv0mvau)
        cout << "no such file" << endl;
    for (int h = 0; h < OUT_ROW; ++h) {
        for (int w = 0; w < OUT_COL; ++w) {
            fpconv0mvau << "[";
            for (int nums = 0; nums < OUT_CH/PE; ++nums) {
                cout << hex;
                fpconv0mvau << hex << mvau_out.read() << ", ";
            }
            fpconv0mvau << "] ";
        }
        fpconv0mvau << endl;
    }
    fpconv0mvau.close();
#endif
    return 0;
}
#endif
#if 0
int main() {
    //GenParamW<16,16,8,WGT_PWCV2_SIZE>("..\\Test\\testw.txt");
    //GenParamW<16,1,8,WGT_DECV1_SIZE>("..\\Test\\testw.txt");
    //GenParamB<16,16,BIAS_M0_PWCV2_SIZE>("..\\Test\\testb.txt");


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
#endif
