#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ.h>
#include <time.h> 

using namespace NTL;

class ltv_base{
    double delta;
  public:
    ZZ_pE f, g, h;	
    ZZ q;
    long n, t, w, e;
    ltv_base();
    ZZ CenterLift(ZZ a, ZZ q);
    ZZ DesLift(ZZ a, long q);
    ZZ_pE CenterLift(ZZ_pE a);
    ZZ_pE Mod2(ZZ_pE a);
    ZZ_pE Mod256(ZZ_pE a);
    ZZ_pE ModN(ZZ_pE a, long n);
    ZZ_pE SampleMessage();
    void SampleFeature(long size, ZZ_pE *r1, ZZ_pE *r2);
    void SampleFeatures(long num, long size, ZZ_pE *r1, ZZ_pE *r2);
    ZZ_pE SampleMessage256();
    ZZ_pE SampleMessageN(long n);
    ZZ_pE SampleErr();
    ZZ_pE SampleKey();
    void ParamsGen(long t, long w, long e, long n, ZZ q, double delta);
    ZZ_pE PrivateKey();
    ZZ_pE PublicKey();
    void KeyGen();
    ZZ_pE Encrypt(ZZ_pE m);
    ZZ_pE Decrypt(ZZ_pE c);
    ZZ_pE DecryptMul(ZZ_pE c);
    ZZ_pE DecryptLTV(ZZ_pE c);
};

