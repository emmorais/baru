#include <ltv_base.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ.h>
#include <time.h> 
#include <NTL/vector.h>

using namespace NTL;

class ltv{
    long lwq; 
    Vec<ZZ_pE> gamma;
  public:
    ltv_base *b;
    ltv();
    Vec<ZZ_pE> BitDecomp(ZZ_pE x);
    Vec<ZZ_pE> PowerOf(ZZ_pE x);
    ZZ_pE SampleMessage();
    ZZ_pE SampleMessage256();
    ZZ_pE SampleMessageN(long n);
    void ParamsGen(long t, long w, long e, long n, ZZ q, double delta, long lwq);
    void KeyGen();
    ZZ_pE Encrypt(ZZ_pE m);
    ZZ_pE Decrypt(ZZ_pE c);
    ZZ_pE DecryptMul(ZZ_pE c);
    ZZ_pE DecryptLTV(ZZ_pE c);
    Vec<ZZ_pE> KeySwitchGen();
    Vec<ZZ_pE> GetKeySwitch();
    ZZ_pE KeySwitch(ZZ_pE c);
    ZZ_pE InnerProduct(Vec<ZZ_pE> a, Vec<ZZ_pE> b);
    ZZ_pE Mod2(ZZ_pE a);
    ZZ_pE Mod256(ZZ_pE a);
    ZZ_pE ModN(ZZ_pE a, long n);
    ZZ_pE Mult(ZZ_pE a, ZZ_pE b, ZZ q);
    ZZ_pE Mul(ZZ_pE a, ZZ_pE b, ZZ q);
};

