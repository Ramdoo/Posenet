#pragma once

#include "ap_int.h"
#include "hls_stream.h"

using namespace std;
using namespace hls;

template <unsigned InWidth,   // width of input stream 固定值
          unsigned OutWidth  // width of output stream 固定值
        //unsigned int NumInWords // number of input words to process
>
void StreamingDataWidthConverter_Batch(stream<ap_int<InWidth>> &in,
                                       stream<ap_int<OutWidth>> &out,
                                       const unsigned NumInWords,
                                       const unsigned InChNums
                                       /*const unsigned numReps = 1*/) {
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        // CASSERT_DATAFLOW(InWidth % OutWidth == 0);
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = NumInWords * outPerIn/* * numReps*/;
        unsigned int o = 0;
        ap_int<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT min=3840 max=331776
#pragma HLS PIPELINE II = 1
            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word from the rightmost position
            ap_int<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wraparound indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    } else if (InWidth == OutWidth) {
        // straight-through copy
        for (unsigned int i = 0; i < NumInWords/* * numReps*/; i++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II = 1
            ap_int<InWidth> e = in.read();
            out.write(e);
        }
    } else { // InWidth < OutWidth
        // read multiple input words per output word emitted
        // CASSERT_DATAFLOW(OutWidth % InWidth == 0);
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords/* * numReps*/;
        unsigned int i = 0;
        ap_int<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II = 1
            // read input and shift into output buffer
            ap_int<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wraparound logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);

            }
        }
    }
}


//函数名后面加T, 表示参数都放在模板Template中， 固定的参数
template <
        unsigned InWidth,   // width of input stream 固定值
        unsigned OutWidth,  // width of output stream 固定值
        unsigned NumInWords // number of input words to process
>
void StreamingDataWidthConverter_BatchT(stream<ap_int<InWidth>> &in,
                                       stream<ap_int<OutWidth>> &out
        /*const unsigned numReps = 1*/) {
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        // CASSERT_DATAFLOW(InWidth % OutWidth == 0);
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = NumInWords * outPerIn/* * numReps*/;
        unsigned int o = 0;
        ap_int<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT min=3840 max=331776
#pragma HLS PIPELINE II = 1
            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word from the rightmost position
            ap_int<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wraparound indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    } else if (InWidth == OutWidth) {
        // straight-through copy
        for (unsigned int i = 0; i < NumInWords/* * numReps*/; i++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II = 1
            ap_int<InWidth> e = in.read();
            out.write(e);
        }
    } else { // InWidth < OutWidth
        // read multiple input words per output word emitted
        // CASSERT_DATAFLOW(OutWidth % InWidth == 0);
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords/* * numReps*/;
        unsigned int i = 0;
        ap_int<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS LOOP_TRIPCOUNT min=768 max=768
#pragma HLS PIPELINE II = 1
            // read input and shift into output buffer
            ap_int<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wraparound logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);

            }
        }
    }
}
