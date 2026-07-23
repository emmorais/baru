#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ.h>
#include <NTL/vector.h>
#include <iostream>
#include <NTL/RR.h>
#include <encoding.h>

using namespace std;
using namespace NTL;

class Functions{
	public:
        static ZZ_pE RotLeft(ZZ_pE a, int k, ltv *scheme);
        static ZZ_pE RotRight(ZZ_pE a, int k, ltv *scheme);
        static ZZ_pE Equal(ZZ_pE a, ZZ_pE b, ltv *scheme);
	static ZZ_pE GeqZero(ZZ_pE a, ltv *scheme);
	static ZZ_pE LeqZero(ZZ_pE a, ltv *scheme);
	static ZZ_pE GZero(ZZ_pE a, ltv *scheme);
	static ZZ_pE LZero(ZZ_pE a, ltv *scheme);
        static ZZ_pE Max(ZZ_pE a, ZZ_pE b, ltv *scheme);
        static ZZ_pE Max(ZZ_pE a, ZZ_pE ma, ZZ_pE b, ZZ_pE mb, ltv *scheme);
};

