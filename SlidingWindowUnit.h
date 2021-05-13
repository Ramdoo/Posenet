//
// Created by 90573 on 2021/4/12.
//

#pragma once

#include <ap_int.h>
#include <hls_stream.h>

#define SWU_DEBUG 0
using namespace hls;
using namespace std;


template<
        unsigned K,
        unsigned IN_BIT,
        unsigned IN_CH
                >
void SWU(
        stream<ap_int<IN_CH*IN_BIT>> &in_fm,
        stream<ap_int<IN_CH*IN_BIT>> &out_fm,
        const unsigned IN_CH_NUMS,
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S
        )
{
#if SWU_DEBUG
    cout << "######################## SWU DEBUG BEGIN #######################" << endl;
#endif
    const unsigned steps = (IN_COL - K) / S + 1;                // 需要滑动几次
    const unsigned line_buffer_size = K * IN_COL * IN_CH_NUMS;  // 滑动窗口的size
    ap_int<IN_CH*IN_BIT> line_buffer[1080] = {0};  //TODO:一个固定的最大值 3*12*30
    ap_int<IN_CH*IN_BIT> tmp_in;

    ap_uint<1> initial_fill = 0;
    unsigned stride = 0;
    unsigned pointer = 0;
    unsigned h = 0;
    for (unsigned rep = 0; rep < IN_ROW; rep++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
        if (h == IN_ROW) {
            initial_fill = 0;
            stride = 0;
            pointer = 0;
            h = 0;
        }
        h += 1;
#if SWU_DEBUG
        cout << dec << "width pointer: " << pointer << endl;
#endif
        for (unsigned w = 0; w < IN_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=6
#pragma HLS PIPELINE II=1
            unsigned line_buffer_pointer = pointer + w*IN_CH_NUMS;
            if (line_buffer_pointer >= line_buffer_size) {
                line_buffer_pointer = line_buffer_pointer - line_buffer_size;
            }
#if SWU_DEBUG
            cout << dec << "line_buffer_pointer: " << line_buffer_pointer << endl;
            cout << "tmp_in: ";
#endif
            for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
                tmp_in = in_fm.read();
                line_buffer[line_buffer_pointer+nums] = tmp_in;
#if SWU_DEBUG
                cout << hex << tmp_in << " ";
#endif
            }
#if SWU_DEBUG
            cout << endl;
#endif
        }

        stride +=1;
        pointer += IN_COL*IN_CH_NUMS;
        if (pointer >= line_buffer_size) {
            pointer = pointer - line_buffer_size;
            initial_fill = 1;
#if SWU_DEBUG
            cout << "initial_file set to 1!" << endl;
#endif
        }
#if SWU_DEBUG
        cout << dec << "stride: " << stride << endl;
        cout << dec << "row pointer: " << pointer << endl;
        cout << dec << "line_buffer for out:" << endl;
        for (int k = 0; k < K; ++k) {
            for (int col = 0; col < IN_COL; ++col) {
                cout << "[";
                for (int nums = 0; nums < IN_CH_NUMS; ++nums) {
                    cout << /*setw(2+IN_CH*IN_BIT/4) <<*/ hex
                         << line_buffer[k*IN_COL*IN_CH_NUMS + col*IN_CH_NUMS + nums] << " ";
                }
                cout << "]";
            }
            cout << endl;
        }
        cout << "---------" << endl;
#endif

        if (initial_fill == 1 && stride >= S) {
            stride = 0;
            unsigned s = 0;
            unsigned kx = 0;
            unsigned ky = 0;

            for (unsigned i = 0; i < steps*(K*K); ++i) {
#pragma HLS LOOP_TRIPCOUNT min=54 max=54
#pragma HLS PIPELINE II=1
                unsigned read_address = (pointer+ s*S*IN_CH_NUMS) + ky * IN_COL*IN_CH_NUMS + kx*IN_CH_NUMS;
                if (read_address >= line_buffer_size) {
                    read_address = read_address - line_buffer_size;
                }
#if SWU_DEBUG
                cout << dec << "s: " << s << endl;
                cout << dec << "kx: " << kx << endl;
                cout << dec << "ky: " << ky << endl;
                cout << dec << "read_address before: " << ((pointer+ s*S*IN_CH_NUMS) + ky * IN_COL*IN_CH_NUMS + kx*IN_CH_NUMS) << endl;
                cout << dec << "read_address after: " << read_address << endl;
                cout << dec << "------tmp out--------" << endl;
#endif
                for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=30 max=30
                    ap_int<IN_CH*IN_BIT> tmp_out = line_buffer[read_address+nums];
#if SWU_DEBUG
                    cout << setw(2+IN_BIT*IN_CH/4) << hex << tmp_out << " ";
#endif
                    out_fm.write(tmp_out);
                }
#if SWU_DEBUG
                cout << endl;
                cout << "-----------------" << endl;
#endif

                if (kx == K - 1) {
                    kx = 0;
                    if (ky == K - 1) {
                        ky = 0;
                        if (s == steps-1) {
                            s = 0;
                        } else {
                            s++;
                        }
                    } else {
                        ky++;
                    }
                } else {
                    kx++;
                }
            }
        }
    }
#if SWU_DEBUG
    cout << "######################## SWU DEBUG END #######################" << endl;
#endif
}


template<
        unsigned K,
        unsigned IN_BIT,
        unsigned IN_CH
>
void SWUBeta(
        stream<ap_int<IN_CH*IN_BIT>> &in_fm,
        stream<ap_int<IN_CH*IN_BIT>> &out_fm,
        const unsigned IN_CH_NUMS,
        const unsigned IN_ROW,
        const unsigned IN_COL,
        const unsigned S
)
{
#if SWU_DEBUG
    cout << "######################## SWU DEBUG BEGIN #######################" << endl;
#endif
    const unsigned steps = (IN_COL - K) / S + 1;                // 需要滑动几次
    const unsigned line_buffer_size = K * IN_COL * IN_CH_NUMS;  // 滑动窗口的size
    ap_int<IN_CH*IN_BIT> line_buffer[2304] = {0};  //TODO:一个固定的最大值 3*96*8
    ap_int<IN_CH*IN_BIT> tmp_in;

    ap_uint<1> initial_fill = 0;
    unsigned stride = 0;
    unsigned pointer = 0;
    unsigned h = 0;
    for (unsigned rep = 0; rep < IN_ROW; rep++) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128
        if (h == IN_ROW) {
            initial_fill = 0;
            stride = 0;
            pointer = 0;
            h = 0;
        }
        h += 1;
#if SWU_DEBUG
        cout << dec << "width pointer: " << pointer << endl;
#endif
        for (unsigned w = 0; w < IN_COL; ++w) {
#pragma HLS LOOP_TRIPCOUNT min=96 max=96
#pragma HLS PIPELINE II=1
            unsigned line_buffer_pointer = pointer + w*IN_CH_NUMS;
            if (line_buffer_pointer >= line_buffer_size) {
                line_buffer_pointer = line_buffer_pointer - line_buffer_size;
            }
#if SWU_DEBUG
            cout << dec << "line_buffer_pointer: " << line_buffer_pointer << endl;
            cout << "tmp_in: ";
#endif
            for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                tmp_in = in_fm.read();
                line_buffer[line_buffer_pointer+nums] = tmp_in;
#if SWU_DEBUG
                cout << hex << tmp_in << " ";
#endif
            }
#if SWU_DEBUG
            cout << endl;
#endif
        }

        stride +=1;
        pointer += IN_COL*IN_CH_NUMS;
        if (pointer >= line_buffer_size) {
            pointer = pointer - line_buffer_size;
            initial_fill = 1;
#if SWU_DEBUG
            cout << "initial_file set to 1!" << endl;
#endif
        }
#if SWU_DEBUG
        cout << dec << "stride: " << stride << endl;
        cout << dec << "row pointer: " << pointer << endl;
        cout << dec << "line_buffer for out:" << endl;
        for (int k = 0; k < K; ++k) {
            for (int col = 0; col < IN_COL; ++col) {
                cout << "[";
                for (int nums = 0; nums < IN_CH_NUMS; ++nums) {
                    cout << /*setw(2+IN_CH*IN_BIT/4) <<*/ hex
                         << line_buffer[k*IN_COL*IN_CH_NUMS + col*IN_CH_NUMS + nums] << " ";
                }
                cout << "]";
            }
            cout << endl;
        }
        cout << "---------" << endl;
#endif

        if (initial_fill == 1 && stride >= S) {
            stride = 0;
            unsigned s = 0;
            unsigned kx = 0;
            unsigned ky = 0;

            for (unsigned i = 0; i < steps*(K*K); ++i) {
#pragma HLS LOOP_TRIPCOUNT min=864 max=864
#pragma HLS PIPELINE II=1
                //unsigned read_address = (pointer+ s*S*IN_CH_NUMS) + ky * IN_COL*IN_CH_NUMS + kx*IN_CH_NUMS;
                unsigned read_address = pointer+ (s*S + ky*IN_COL + kx) * IN_CH_NUMS;
                if (read_address >= line_buffer_size) {
                    read_address = read_address - line_buffer_size;
                }
#if SWU_DEBUG
                cout << dec << "s: " << s << endl;
                cout << dec << "kx: " << kx << endl;
                cout << dec << "ky: " << ky << endl;
                cout << dec << "read_address before: " << ((pointer+ s*S*IN_CH_NUMS) + ky * IN_COL*IN_CH_NUMS + kx*IN_CH_NUMS) << endl;
                cout << dec << "read_address after: " << read_address << endl;
                cout << dec << "------tmp out--------" << endl;
#endif
                for (unsigned nums = 0; nums < IN_CH_NUMS; ++nums) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                    ap_int<IN_CH*IN_BIT> tmp_out = line_buffer[read_address+nums];
#if SWU_DEBUG
                    cout << setw(2+IN_BIT*IN_CH/4) << hex << tmp_out << " ";
#endif
                    out_fm.write(tmp_out);
                }
#if SWU_DEBUG
                cout << endl;
                cout << "-----------------" << endl;
#endif

                if (kx == K - 1) {
                    kx = 0;
                    if (ky == K - 1) {
                        ky = 0;
                        if (s == steps-1) {
                            s = 0;
                        } else {
                            s++;
                        }
                    } else {
                        ky++;
                    }
                } else {
                    kx++;
                }
            }
        }
    }
#if SWU_DEBUG
    cout << "######################## SWU DEBUG END #######################" << endl;
#endif
}
