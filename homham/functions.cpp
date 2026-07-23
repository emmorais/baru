#include "functions.h"

#define NUM 100
#define N 1024
#define Q 157
#define T 1024
#define W 2
#define E 0
#define LWQ 158

ZZ_pE Functions::RotLeft(ZZ_pE a, int k, ltv *scheme){
    ZZ_pE res;
    ZZ_pX pol;
    int i;
    for(i=0;i<scheme->b->n;i++){
        if (i == k)
            SetCoeff(pol, i, 1);
        else 
            SetCoeff(pol, i, 0);
    }
    res = conv<ZZ_pE>(pol);
    res = inv(res);
    res = scheme->Encrypt(res);
    res = scheme->Mul(res,a,scheme->b->q);
    return res;
}

ZZ_pE Functions::RotRight(ZZ_pE a, int k, ltv *scheme){
    ZZ_pE res;
    ZZ_pX pol;
    int i;
    for(i=0;i<scheme->b->n;i++){
        if (i == k)
            SetCoeff(pol, i, 1);
        else 
            SetCoeff(pol, i, 0);
    }
    res = conv<ZZ_pE>(pol);
    res = scheme->Encrypt(res);
    res = scheme->Mul(res,a,scheme->b->q);
    return res;
}

