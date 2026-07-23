#include <functions.h>
#include <iostream>
#include <colors.h>
#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

#define NUM 100
#define N 2048
#define Q 101
#define T 2048
#define W 2
#define E 0
#define LWQ 102

void lheInit(ltv *ctx){
  long lwq = Q+1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q));
  ZZ_pX P, pol;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
}

bool rheTest(){
  int count=0, ok=0, nok=0;
  ltv_base *scheme;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q)); 
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  scheme = new ltv_base();
  scheme->ParamsGen(t, w, e, n, q, delta);

  while(count < NUM){
    scheme->KeyGen();
    ZZ_pE m = scheme->SampleMessageN(t);
    ZZ_pE c1 = scheme->Encrypt(m);
    ZZ_pE mm = scheme->Decrypt(c1);
    if(m == mm)
      ok++;
    else
      nok++;
    count++;
  }
  int result = (ok == count); 
  return result; 
}

bool rheAddTest(){
  int count=0, ok=0, nok=0;
  ltv_base *scheme;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q)); 
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  scheme = new ltv_base();
  scheme->ParamsGen(t, w, e, n, q, delta);

  while(count < NUM){
    scheme->KeyGen();
    ZZ_pE m1 = scheme->SampleMessageN(t);
    ZZ_pE m2 = scheme->SampleMessageN(t);
    ZZ_pE c1 = scheme->Encrypt(m1);
    ZZ_pE c2 = scheme->Encrypt(m2);
    ZZ_pE mm = scheme->Decrypt(c1+c2);
    if(scheme->ModN(m1+m2, t) == mm){
      ok++;
    }
    else{
      nok++;
    }
    count++;
  }
  int result = (ok == count); 
  return result; 
}



bool lheTest(){
  int count=0, ok=0, nok=0;
  long lwq = Q+1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q));
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m = ctx->b->SampleMessageN(t);
    ZZ_pE c1 = ctx->Encrypt(m);
    ZZ_pE mm = ctx->Decrypt(c1);
    if (m == mm)
      ok++;
    else
      nok++;
    count++;
  }
  int result = (ok == count); 
  return result; 
}

bool lheAddTest(){
  int count=0, ok=0, nok=0;
  long lwq = Q + 1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q ;
  GenPrime(q, Q);
  ZZ_p::init(q);
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m1 = ctx->b->SampleMessageN(t);
    ZZ_pE m2 = ctx->b->SampleMessageN(t);
    ZZ_pE c1 = ctx->Encrypt(m1);
    ZZ_pE c2 = ctx->Encrypt(m2);
    ZZ_pE mm = ctx->Decrypt(c1+c2);
    if (ctx->ModN(m1+m2, t) == mm){
      ok++;
    }
    else{
      nok++;
    }
    count++;
  }
  int result = (ok == count); 
  return result; 
}

bool lheSubTest(){
  int count=0, ok=0, nok=0;
  long lwq = Q + 1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q ;
  GenPrime(q, Q);
  ZZ_p::init(q);
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m1 = Encoding::Encode(47, ctx);
    ZZ_pE m2 = Encoding::Encode(4, ctx);
    ZZ_pE c1 = ctx->Encrypt(m1);
    ZZ_pE c2 = ctx->Encrypt(m2);
    ZZ_pE mm = ctx->Decrypt(c1-c2);
    if (ctx->ModN(m1-m2, t) == mm){
      ok++;
    }
    else{
      nok++;
    }
    count++;
  }
  int result = (ok == count); 
  return result; 
}

bool lheMulTest(){
  int count=0, ok=0, nok=0;
  long lwq = Q+1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q));
  ZZ_pX P, pol;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m1 = ctx->b->SampleMessageN(t);
    ZZ_pE m2 = ctx->b->SampleMessageN(t);
    ZZ_pE c1 = ctx->Encrypt(m1);
    ZZ_pE c2 = ctx->Encrypt(m2);
    ZZ_pE cmul = ctx->Mult(c1,c1,q);
    ZZ_pE kcmul= ctx->KeySwitch(cmul);
    ZZ_pE mm = ctx->Decrypt(kcmul);
    if (ctx->ModN(m1*m1, t) == mm){
      ok++; 
    }
    else {
      nok++; 
    }
    count++;
  }
  int result = (ok == count); 
  return result; 
}

bool lheRealTest(){
  int count=0, ok=0, nok=0;
  long lwq = Q+1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q;
  GenPrime(q, Q);
  ZZ_p::init(conv<ZZ>(q));
  ZZ_pX P, pol;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m1 = Encoding::EncodeRZZ(conv<RR>(-51.25), ctx);
    ZZ_pE m2 = Encoding::EncodeRZZ(conv<RR>(23.25), ctx);
    ZZ_pE c1 = ctx->Encrypt(m1);
    ZZ_pE c2 = ctx->Encrypt(m2);
    ZZ_pE c12mul = ctx->Mul(c1,c2,q);
    ZZ_pE mm = ctx->Decrypt(c12mul);
    cout.precision(15);
    if (ctx->ModN(m1*m1, t) == mm){
      ok++; 
    }
    else {
      nok++; 
    }
    count++;
  }
  int result = (ok == count); 
  return result; 
}

void lheTiming(){
  clock_t start, end;
  double elapsed;
  int count=0, ok=0, nok=0;
  long lwq = ((Q+1)/2)+1;
  double delta = 1;
  long n = 4096;
  long t = 2;
  long w = 32;
  long e = E;
  int i, REP=100;
  ltv *ctx;
  ZZ q;
  GenPrime(q, Q);
  cout << "q: " << q << "\n";
  ZZ_p::init(conv<ZZ>(q));
  ZZ_pX P, pol;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P); 
  cout << "Leveled homomorphic encryption test (timing):\n";
  cout << "[";
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < 1){
    start = clock();
    for(i=0;i<REP;i++){
      ctx->KeyGen();
    }
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time keygen: " << elapsed/REP << " secs\n";
    
    ZZ_pE m1 = ctx->b->SampleMessageN(t);
    ZZ_pE m2 = ctx->b->SampleMessageN(t);
    //cout << "m1: " << m1 << ", m2:" << m2 << "\n";
    ZZ_pE c1;
    start = clock();
    for(i=0;i<REP;i++){
      c1 = ctx->Encrypt(m1);
    }
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time encrypt: " << elapsed/REP << " secs\n";

    ZZ_pE c2 = ctx->Encrypt(m2);

    ZZ_pE cadd;
    ZZ_pE kcadd;
    start = clock();
    for(i=0;i<REP;i++){
      cadd = c1+c1;
    }
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time add: " << elapsed/REP << " secs\n";

    ZZ_pE cmul;
    ZZ_pE kcmul;
    start = clock();
    for(i=0;i<REP;i++){
      cmul = ctx->Mult(c1,c1,q);
      kcmul= ctx->KeySwitch(cmul);
    }
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time mult: " << elapsed/REP << " secs\n";

    ZZ_pE mm;
    start = clock();
    for(i=0;i<REP;i++){
      mm = ctx->Decrypt(kcmul);
    }
    end = clock();
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time decrypt: " << elapsed/REP << " secs\n";

    cout << "mm: " << mm << "\n";
    if (ctx->ModN(m1*m1, t) == mm){
      ok++; 
      cout << "\033[1;36m.\033[0m";
    }
    else {
      nok++; 
      cout << "\033[1;31m x \033[0m,";
    }
    count++;
  }
  cout << "]\n";
  //cout << "ok,nok: " << ok << "," << nok << "\n";

}

ZZ_p HammingDistance(ZZ_pE m1, ZZ_pE m2, int size) {
  int i;
  ZZ_p res;
  ZZ_pX pol1 = conv<ZZ_pX>(m1);
  ZZ_pX pol2 = conv<ZZ_pX>(m2);
  for (i=0; i<size; i++) {
    ZZ_p c1 = coeff(pol1, i);
    ZZ_p c2 = coeff(pol2, i);
    res += (c1!=c2);
  }
  return res;
}

bool lheIrisTest(){
  clock_t start, end;
  double elapsed;
  int count=0, ok=0, nok=0;
  long lwq = Q + 1;
  double delta = 1;
  long n = N;
  long t = T;
  long w = W;
  long e = E;
  ltv *ctx;
  ZZ q;
  long SIZE  = 2000;
  GenPrime(q, Q);
  ZZ_p::init(q);
  ZZ_pX P;
  SetCoeff(P, 0, 1);
  SetCoeff(P, n, 1);
  ZZ_pE::init(P);
  ctx = new ltv();
  ctx->ParamsGen(t, w, e, n, q, delta, lwq);
  while(count < NUM){
    ctx->KeyGen();
    ZZ_pE m1, m1_inv, m2, m2_inv;
    ctx->b->SampleFeature(SIZE, &m1, &m1_inv);
    ctx->b->SampleFeature(SIZE, &m2, &m2_inv);
    cout << "m1" << m1 << "\n";
    cout << "m1_inv" << m1_inv << "\n";
    cout << "m2" << m2 << "\n";
    cout << "m2_inv" << m2_inv << "\n";
    ZZ_pE c1 = ctx->Encrypt(m1);
    ZZ_pE c1_inv = ctx->Encrypt(m1_inv);
    ZZ_pE c2 = ctx->Encrypt(m2);
    ZZ_pE c2_inv = ctx->Encrypt(m2_inv);

    start = clock();
    ZZ_pE cmul = ctx->Mult(c1-c2,c1_inv-c2_inv,q);
    //ZZ_pE kcmul= ctx->KeySwitch(cmul);
    end = clock();
    ZZ_pE mm = ctx->DecryptMul(cmul);
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "time: " << elapsed << " secs\n";
    cout << "mm" << mm << "\n";

    ZZ_pX pol = conv<ZZ_pX>(mm);
    cout << "Hamming distance: " << coeff(pol, SIZE-1) << "\n";
    ZZ_p hd = coeff(pol, SIZE-1);
    if (HammingDistance(m1,m2,SIZE) == hd){
      ok++;
    }
    else{
      nok++;
    }
    count++;
  }
  int result = (ok == count);
  return result;
}



int main(){
  srand (time(NULL));
  /*bool t1 = rheTest();
  cout << FGRN("Base LTV Encryption/Decryption test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t2 = rheAddTest();
  cout << FGRN("Base LTV Addition Homomorphism test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t3 = lheTest();
  cout << FGRN("LTV Encryption/Decryption test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t4 = lheAddTest();
  cout << FGRN("LTV Addition Homomorphism test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t5 = lheSubTest();
  cout << FGRN("LTV Subtraction Homomorphism test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t6 = lheMulTest();
  cout << FGRN("LTV Multiplication Homomorphism test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t7 = lheRealTest();
  cout << FGRN("LTV Homomorphism over reals test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  bool t8 = lheRotTest();
  cout << FGRN("LTV Rotation test result: ") << (t1 ? BOLD(FGRN("True")) : BOLD(FRED("False"))) << "\n";
  lheTiming();*/
  bool t_iris = lheIrisTest();
  cout << FGRN("Homomorphic Iris Matching: ") << (t_iris ? BOLD(FGRN("Passed")) : BOLD(FRED("Failed"))) << "\n";
  return 0;
}
