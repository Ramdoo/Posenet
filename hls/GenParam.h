// NOTE: for DEBUG test
#include <iostream>
#include <fstream>
#include <ap_int.h>

using namespace std;

template<
        unsigned SIMD,
        unsigned PE,
        unsigned W_BIT,
        unsigned SIZE
        >
void GenParamW(const char *path
               //const unsigned IN_CH, //TODO:
               //const unsigned OUT_CH //TODO:
               ) {
    ofstream f(path, ios::out);
    for (unsigned p = 0; p < PE; ++p) {
        f << "{";
        for (unsigned k = 0; k < SIZE; ++k) {
            ap_int<SIMD*W_BIT> temp_wgt = 0;
            for (unsigned s = 0; s < SIMD; ++s) {
                temp_wgt = temp_wgt << W_BIT;
                temp_wgt |= ap_int<SIMD*W_BIT>(ap_int<W_BIT>(rand() % 2)) ;
            }
            cout << hex ;
            f << f.fill(',') <<  "\"" << temp_wgt << "\"";
        }
        f << "}," << endl;
    }
    f.close();
}



template<
        unsigned PE,
        unsigned BIAS_BIT,
        unsigned SIZE
>
void GenParamB(const char *path) {
    ofstream f(path, ios::out);
    for (unsigned p = 0; p < PE; ++p) {
        f << "{";
        for (unsigned i = 0; i < SIZE; ++i) {
            ap_int<BIAS_BIT> temp_wgt = ap_int<BIAS_BIT>(rand() % 3);
            cout << hex ;
            f << f.fill(',') <<  "\"" << temp_wgt << "\"";
        }
        f << "}," << endl;
    }
    f.close();
}


template<
        unsigned PE=16,
        unsigned SIMD=16,
        unsigned M0_BIT=16
>
void GenParamM(const char *path) {
    ofstream f(path, ios::out);
    for (unsigned p = 0; p < PE; ++p) {
        f << "{";
        ap_int<SIMD*M0_BIT> temp_m0 = 0;
        for (unsigned s = 0; s < SIMD; ++s) {
            temp_m0 = temp_m0 << M0_BIT;
            temp_m0 |= ap_int<SIMD*M0_BIT>(ap_int<M0_BIT>(rand() % 10)) ;
        }
        cout << hex ;
        f << f.fill(',') <<  "\"" << temp_m0 << "\"";
        f << "}" << endl;
    }
    f.close();
}

