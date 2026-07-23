#include "encoding.h"

ZZ_pE Encoding::EncodeRZZ(RR a, ltv *scheme){
  int i, size;
  double intpart;
  RR rint = a;
  rint *= pow(2, scheme->b->e);
  if (rint < 0)
      rint += pow(2, scheme->b->e+32);
  ZZ zint = conv<ZZ>(rint);
  ZZ_pE res;
  ZZ_pX pol;
  ZZ aux;
  for(i=0;i<scheme->b->e+32;i++){
    aux = zint % conv<ZZ>(2);
    SetCoeff(pol, i, conv<ZZ_p>(aux));
    zint /= conv<ZZ>(2); 
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

RR Encoding::DecodeRZZ(ZZ_pE a, ltv *scheme){
  int i;
  //long resint = Decode(a, scheme);
  unsigned long ua=0;
  long res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  ZZ aux;
  for(i=0;i<scheme->b->e+480;i++){
    aux = conv<ZZ>(coeff(pol, i));
    if (aux % conv<ZZ>(scheme->b->t) > scheme->b->t/2)
      aux -= scheme->b->t;
    aux *= power(conv<ZZ>(2), i);
    ua += conv<unsigned long>(aux);
  }
  res = (long)ua;

  RR resreal = conv<RR>(res)/conv<RR>(pow(2, scheme->b->e));
  return resreal;
}

ZZ_pE Encoding::EncodeR(double a, ltv *scheme){
  int i, size;
  double intpart;
  intpart = a*pow(2, scheme->b->e);
  intpart = round(intpart);
  unsigned long ua = (unsigned long)intpart;
  if (a > 0){
    size = log(a)+1;
  }
  else {
    size = log(-a)+1;
  }

  ZZ_pE res;
  ZZ_pX pol;
  ZZ aux;
  for(i=0;i<scheme->b->n;i++){
    aux = ua % 2;
    SetCoeff(pol, i, conv<ZZ_p>(aux));
    ua /= 2; 
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

double Encoding::DecodeR(ZZ_pE a, ltv *scheme){
  int i;
  unsigned long ua=0;
  double res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  ZZ aux;
  for(i=0;i<scheme->b->e+32;i++){
    aux = conv<ZZ>(coeff(pol, i));
    if (aux % conv<ZZ>(scheme->b->t) > scheme->b->t/2)
      aux -= scheme->b->t;
    aux *= power(conv<ZZ>(2), i);
    ua += conv<unsigned long>(aux);
  }
  res = (long)ua;
  res = ((double) res/pow(2, scheme->b->e));
  return res;
}

ZZ_pE Encoding::Encode(long a, ltv *scheme){
  int i;
  unsigned long ua = (unsigned long)a;
  ZZ_pE res;
  ZZ_pX pol;
  ZZ aux;
  for(i=0;i<scheme->b->e;i++){
    SetCoeff(pol, i, conv<ZZ_p>(0));
  }
  for(i=scheme->b->e;i<scheme->b->n;i++){
    aux = ua % 2;
    SetCoeff(pol, i, conv<ZZ_p>(aux));
    ua /= 2; 
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

long Encoding::Decode(ZZ_pE a, ltv *scheme){
  int i;
  unsigned long ua=0;
  long res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  ZZ aux;
  for(i=0;i<scheme->b->e+32;i++){
    aux = conv<ZZ>(coeff(pol, i));
    if (aux % conv<ZZ>(scheme->b->t) > scheme->b->t/2)
      aux -= scheme->b->t;
    aux *= power(conv<ZZ>(2), i);
    ua += conv<unsigned long>(aux);
  }
  res = (long)ua;
  return res;
}

ZZ_pE Encoding::EncodeZZ(ZZ a, ltv *scheme){
  int i;
  int size = log(a)+1;
  ZZ_pE res;
  ZZ_pX pol;
  ZZ aux;
  for(i=0;i<scheme->b->e;i++){
    SetCoeff(pol, i, conv<ZZ_p>(0));
  }
  for(i=scheme->b->e;i<scheme->b->n;i++){
    aux = a % conv<ZZ>(2);
    SetCoeff(pol, i, conv<ZZ_p>(aux));
    a /= conv<ZZ>(2); 
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ Encoding::DecodeZZ(ZZ_pE a, ltv *scheme){
  int i;
  ZZ res = conv<ZZ>(0);
  ZZ_pX pol = conv<ZZ_pX>(a);
  ZZ aux;
  for(i=0;i<scheme->b->e+480;i++){
    aux = conv<ZZ>(coeff(pol, i));
    if (aux % conv<ZZ>(scheme->b->t) > scheme->b->t/2)
      aux -= scheme->b->t;
    aux *= power(conv<ZZ>(2), i);
    res += conv<ZZ>(aux);
  }
  return res;
}
