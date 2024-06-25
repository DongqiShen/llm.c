#include <stdio.h>
#include <stdlib.h>
#include <float.h>


// cpu code reference

void attention_forward_cpu(float* out, float *preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q, K, V
    // preattn, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = 3 * C;
    int hs = C / NH;
    float scale = 1.0 / sqrt(hs);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) { // 表示用于当作query的token
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    // (query_t) dot (key_t2) 
                    float val = 0.0f;
                    for (int i = 0; i < hs; ++i) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t + 1; t2 < T; ++t2) {
                    preatt_bth[t2] = -INFINITY;
                }
                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; ++t2) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 <= T; ++t2) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0;
                    }
                }

                // pass 4: accumulate weighted value into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; ++i) {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; ++i) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}