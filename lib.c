#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// before using
// gcc -fPIC -shared lib.c -o lib.so

float* group_by(float* arr, int N, int len,
                float bin_size, float max_kpc) {
    /*
    arr is [f] + [x] + [y], each of them with length N.
    return [bin_inds] + [bin_stat].
    */
    int res_len = len * 2;
    float* res = (float*) malloc(sizeof(float) * res_len);
    memset(res, 0, sizeof(float) * res_len);
    float d;
    for (int p1 = 0; p1 < N; p1++) {
        for (int p2 = 0; p2 < N; p2++) {
            d = sqrt((arr[p1+1*N] - arr[p2+1*N]) * (arr[p1+1*N] - arr[p2+1*N]) +
                     (arr[p1+2*N] - arr[p2+2*N]) * (arr[p1+2*N] - arr[p2+2*N]));
            int idx;
            if (d < max_kpc) {  // For acceleration.
                if (d <= bin_size && d > 1e-6) {idx = 1;}
                else {idx = (int)(d / bin_size);}
                res[idx+len] = ((arr[p1] * arr[p2] + res[idx+len] * res[idx]) /
                                (res[idx] + 1));
                res[idx] += 1;
            }
        }
    }
    
    return res;
}



float inner(float x, float y, float* coeff);
float* iter(float line_ratio, float S23, float* coeff);

float inner(float x, float y, float* coeff){
    float xy[10];
    xy[0] = 1; xy[1] = x; xy[2] = y; xy[3] = x*y; xy[4] = x*x;
    xy[5] = y*y; xy[6] = x*y*y; xy[7] = x*x*y; xy[8] = x*x*x; xy[9] = y*y*y;
    float res = 0;
    for (int i = 0; i < 10; i++) {res += xy[i] * coeff[i];}
    return res;
}

float* iter(float line_ratio, float S23, float* coeff){
    /*
    coeff is [coeff_logOH] + [coeff_logU].
    The length of coeff_logOH & coeff_logOH is 10.
    return metallicity and ionization parameter.
    */
    float* coeff_logOH = (float*) malloc(sizeof(float) * 10);
    memset(coeff_logOH, 0, sizeof(float) * 10);
    float* coeff_logU = (float*) malloc(sizeof(float) * 10);
    memset(coeff_logU, 0, sizeof(float) * 10);
    float* res = (float*) malloc(sizeof(float) * 2);
    memset(res, 0, sizeof(float) * 2);

    for (int i = 0; i < 10; i += 1) {
        coeff_logOH[i] = coeff[i];
        coeff_logU[i] = coeff[i+10];
    }

    float init_met = 8.7, cur_met = 8.3, logU;
    int count = 0;
    while (fabs(init_met - cur_met) > 0.005){
        init_met = cur_met;
        logU = inner(S23, init_met, coeff_logU);
        cur_met = inner(line_ratio, logU, coeff_logOH);
        count += 1;
        if (count > 10) {return res;}
    }
    res[0] = cur_met; res[1] = logU;
    return res;
}

float* iteration(float* arr, int len) {
    /*
    arr is [r_arr] + [S23_arr] + [coeff_logOH] + [coeff_logU].
    The length of r_arr is len.
    return [logOH_arr] + [logU_arr].
    */
    float* coeff = (float*) malloc(sizeof(float) * 20);
    memset(coeff, 0, sizeof(float) * 20);
    for (int i = 0; i < 20; i += 1) {coeff[i] = arr[2*len+i];}

    float* res = (float*) malloc(sizeof(float) * 2*len);
    memset(res, 0, sizeof(float) * 2*len);

    for (int p = 0; p < len; p++){
        float line_ratio = arr[p];
        float S23 = arr[p+len];
        float* pointer = iter(line_ratio, S23, coeff);
        res[p] = pointer[0];
        res[p+len] = pointer[1];
    }
    return res;
}


