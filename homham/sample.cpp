#include <sample.h>
#include <math.h>

ZZ_p sample::SampleMessage(){
  int i, mm;
  ZZ_p res;
  res = rand()%2;
  return res;
}
    
Mat<ZZ_p> sample::SampleRandom(long int m, long int n){
  int i,j;
  Mat<ZZ_p> res;
  res.SetDims(m, n);
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      res[i][j] = random_ZZ_p();
    }
  }
  return res;
}

Vec<ZZ_p> sample::SampleRandomArray(int len){
  int i;
  Vec<ZZ_p> res;
  res.SetLength(len);
  for(i=0;i<len;i++){
    res[i] = random_ZZ_p();
  }
  return res;
}

/*
 * Simple implementation of Gaussian noise
 */ 
ZZ_p sample::SampleNoise(){
  ZZ_p res;
  double randNormal, randStdNormal;
  double u1 = ((double) rand()) / (RAND_MAX); 
  double u2 = ((double) rand()) / (RAND_MAX); 
  u2 *= 6.2831;
  u1 = sqrt (-2 * log(u1));
  randNormal = u1 * sin(u2); 
  randNormal = 8 * randNormal; 
  randNormal = u1 * cos(u2); 
  res = randNormal;
  return res;
}

/*
 * Returns a vector of Gaussian errors.
 */
Vec<ZZ_p> sample::SampleNoiseArray(long n){
  Vec<ZZ_p> res;
  long i;
  res.SetLength(n);
  for(i=0;i<n;i++){
    res[i] = SampleNoise();
  }
  return res;
}

/*
 * Returns a binary matrix.
 */ 
Mat<ZZ_p> sample::SampleRMatrix(long int m, long N){
  Mat<ZZ_p> res;
  long i, j;
  res.SetDims(m, N);
  for(i=0;i<m;i++){
    for(j=0;j<N;j++){
      res[i][j] = conv<ZZ_p>(SampleMessage());
    }
  }
  return res;
}

/*
 * Returns a binary vector.
 */ 
Vec<ZZ_p> sample::SampleR(long int N){
  Vec<ZZ_p> res;
  long i;
  res.SetLength(N);
  for(i=0;i<N;i++){
    res[i] = conv<ZZ_p>(SampleMessage());
  }
  return res;
}
