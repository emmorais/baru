#include <NTL/ZZ_p.h>
#include <NTL/ZZ.h>
#include <NTL/matrix.h>
#include <time.h> 


using namespace NTL;

class sample {

  public:
    static ZZ_p SampleMessage();	    
    static Mat<ZZ_p> SampleRandom(long int m, long int n);
    static Vec<ZZ_p> SampleRandomArray(int len);
    static Mat<ZZ_p> SampleRMatrix(long int m, long int N);
    static Vec<ZZ_p> SampleR(long int N);
    static Vec<ZZ_p> SampleNoiseArray(long n);
    static ZZ_p SampleNoise();
    static ZZ_p SampleKey();
};
