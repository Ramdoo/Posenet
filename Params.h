//
// Created by 90573 on 2021/4/29.
//
// for DEBUG test

#pragma once


//16 is SIMD
const ap_int<16*8> conv0_w[16][4] = {
        {"0x01010000010000000000010101010101","0x01010000010000000000010101010101","0x01010000010000000000010101010101","0x01010000010000000000010101010101"},
        {"0x01000100010000010000010000010100","0x01000100010000010000010000010100","0x01000100010000010000010000010100","0x01000100010000010000010000010100"},
        {"0x01000100010101000101000101000101","0x01000100010101000101000101000101","0x01000100010101000101000101000101","0x01000100010101000101000101000101"},
        {"0x01000100000101010101010000010000","0x01000100000101010101010000010000","0x01000100000101010101010000010000","0x01000100000101010101010000010000"},
        {"0x00000000000000010001000000010100","0x00000000000000010001000000010100","0x00000000000000010001000000010100","0x00000000000000010001000000010100"},
        {"0x01010000000000000100000100010100","0x01010000000000000100000100010100","0x01010000000000000100000100010100","0x01010000000000000100000100010100"},
        {"0x00000101010101000000010001000101","0x00000101010101000000010001000101","0x00000101010101000000010001000101","0x00000101010101000000010001000101"},
        {"0x00000001010101000000010001010100","0x00000001010101000000010001010100","0x00000001010101000000010001010100","0x00000001010101000000010001010100"},
        {"0x01000000010000000101010101010101","0x01000000010000000101010101010101","0x01000000010000000101010101010101","0x01000000010000000101010101010101"},
        {"0x01010100010000000000010000010001","0x01010100010000000000010000010001","0x01010100010000000000010000010001","0x01010100010000000000010000010001"},
        {"0x00010001000101010000010000000001","0x00010001000101010000010000000001","0x00010001000101010000010000000001","0x00010001000101010000010000000001"},
        {"0x00010000010001010000000001010001","0x00010000010001010000000001010001","0x00010000010001010000000001010001","0x00010000010001010000000001010001"},
        {"0x00010101000101000100010100010100","0x00010101000101000100010100010100","0x00010101000101000100010100010100","0x00010101000101000100010100010100"},
        {"0x00010000000101000101010101010001","0x00010000000101000101010101010001","0x00010000000101000101010101010001","0x00010000000101000101010101010001"},
        {"0x00000000000000010100010100000000","0x00000000000000010100010100000000","0x00000000000000010100010100000000","0x00000000000000010100010100000000"},
        {"0x00010001000101000001000000010000","0x00010001000101000001000000010000","0x00010001000101000001000000010000","0x00010001000101000001000000010000"}

};

const ap_uint<16*8> conv0_w_dw[18] = {
        {"0x01010000010000000000010101010101"},
        {"0x01000100010000010000010000010100"},
        {"0x01000100010101000101000101000101"},
        {"0x01000100000101010101010000010000"},
        {"0x00000000000000010001000000010100"},
        {"0x01010000000000000100000100010100"},
        {"0x00000101010101000000010001000101"},
        {"0x00000001010101000000010001010100"},
        {"0x01000000010000000101010101010101"},
        {"0x01000000010000000101010101010101"},
        {"0x01010100010000000000010000010001"},
        {"0x00010001000101010000010000000001"},
        {"0x00010000010001010000000001010001"},
        {"0x00010101000101000100010100010100"},
        {"0x00010000000101000101010101010001"},
        {"0x00000000000000010100010100000000"},
        {"0x00010001000101000001000000010000"}
};

const ap_int<16> conv0_bias[16][1] = {
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"},
        {"0x0"}
};

const ap_int<16> conv0_bias_dw[16][2] = {
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"},
        {"0x0","0x0"}
};

const ap_uint<16> conv0_m0[16][1] = {
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"},
        {"0x01"}
};


const ap_uint<16> conv0_m0_dw[16][2] = {
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"},
        {"0x01","0x01"}
};

