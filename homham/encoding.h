#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/ZZ.h>
#include <time.h> 
#include <NTL/vector.h>
#include <iostream>
#include <ltv.h>
#include <NTL/RR.h>

using namespace std;
using namespace NTL;

class Encoding{
	public:
		static ZZ_pE EncodeR(double a, ltv *scheme);
		static double DecodeR(ZZ_pE a, ltv *scheme);
		static ZZ_pE EncodeRZZ(RR a, ltv *scheme);
		static RR DecodeRZZ(ZZ_pE a, ltv *scheme);
        static ZZ_pE Encode(long a, ltv *scheme);
		static long Decode(ZZ_pE a, ltv *scheme);
		static ZZ_pE EncodeZZ(ZZ a, ltv *scheme);
		static ZZ DecodeZZ(ZZ_pE a, ltv *scheme);
};

