#include <ltv_base.h>
#include <math.h>

using namespace NTL;
using namespace std;

ltv_base::ltv_base(){ }

ZZ_pE ltv_base::SampleMessage(){
  int i, mm;
  ZZ_pE res;
  ZZ_pX pol;
  for(i=0;i<this->n;i++){
    mm = rand()%2;
    SetCoeff(pol, i, conv<ZZ_p>(mm));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

void ltv_base::SampleFeature(long size, ZZ_pE *r1, ZZ_pE *r2){
  int i, mm;
  //ZZ_pE res;
  ZZ_pX pol1;
  ZZ_pX pol2;
  for(i=0;i<size;i++){
    mm = rand()%2;
    SetCoeff(pol1, i, conv<ZZ_p>(mm));
    SetCoeff(pol2, size-i-1, conv<ZZ_p>(mm));
  }
  *r1 = conv<ZZ_pE>(pol1);
  *r2 = conv<ZZ_pE>(pol2);
  return;
}

void ltv_base::SampleFeatures(long num, long size, ZZ_pE *r1, ZZ_pE *r2){
  int i, j, mm;
  //ZZ_pE res;
  ZZ_pX pol1;
  ZZ_pX pol2;
  for (j=0; j<num; j++) {
    for(i=0;i<size;i++){
      mm = rand()%2;
      SetCoeff(pol1, j*size + i, conv<ZZ_p>(mm));
      SetCoeff(pol2, j*size + size-i-1, conv<ZZ_p>(mm));
    }
  }
  *r1 = conv<ZZ_pE>(pol1);
  *r2 = conv<ZZ_pE>(pol2);
  return;
}

ZZ_pE ltv_base::SampleMessage256(){
  int i, mm;
  ZZ_pE res;
  ZZ_pX pol;
  for(i=0;i<this->n;i++){
    mm = rand()%256;
    SetCoeff(pol, i, conv<ZZ_p>(mm));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::SampleMessageN(long n){
  int i, mm;
  ZZ_pE res;
  ZZ_pX pol;
  for(i=0;i<this->n;i++){
    mm = rand()%n;
    SetCoeff(pol, i, conv<ZZ_p>(mm));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}


ZZ_pE ltv_base::SampleErr(){
  ZZ_pE res;
  ZZ_pX pol;
  long i;
  double randNormal, randStdNormal;
  for(i = 0; i <= this->n; i++){
    double u1 = ((double) rand()) / (RAND_MAX); 
    double u2 = ((double) rand()) / (RAND_MAX); 
    u2 *= 6.2831;
    u1 = sqrt (-2 * log(u1));
    randNormal = u1 * sin(u2); 
    randNormal = this->delta*8 * randNormal; 
    SetCoeff(pol, i, conv<ZZ_p>((long)randNormal));
    i++;
    if (i <= n){
      randNormal = u1 * cos(u2); 
      randNormal = this->delta * randNormal; 
      SetCoeff(pol, i, conv<ZZ_p>((long)randNormal));
    }
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::SampleKey(){
  ZZ_pE res;
  ZZ_pX pol;
  long i;
  double randNormal, randStdNormal;
  for(i = 0; i < this->n; i++){
    double u1 = ((double) rand()) / (RAND_MAX); 
    double u2 = ((double) rand()) / (RAND_MAX); 
    u2 *= 6.2831;
    u1 = sqrt (-2 * log(u1));
    randNormal = u1 * sin(u2); 
    randNormal = this->delta/2 * randNormal; 
    SetCoeff(pol, i, conv<ZZ_p>((long)randNormal));
    i++;
    if (i <= n){
      randNormal = u1 * cos(u2); 
      randNormal = this->delta/2 * randNormal; 
      SetCoeff(pol, i, conv<ZZ_p>((long)randNormal));
    }
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

void ltv_base::ParamsGen(long t, long w, long e, long n, ZZ q, double delta){	
  this->t = t;
  this->w = w;
  this->n = n;
  this->q = q;
  this->e = e;
  this->delta = delta;
}

ZZ_pE ltv_base::PrivateKey(){
  return this->f;
}

ZZ_pE ltv_base::PublicKey(){
  return this->h;
}

void ltv_base::KeyGen(){	
  ZZ_pE finv, res;
  this->g = SampleKey();
  while(!IsOne(res)){ 
    this->f = SampleKey(); 
    this->f = this->f*this->t+1;
    finv = inv(this->f);
    res = finv * this->f;
  }
  this->h = this->t * this->g * finv;
}
    
ZZ_pE ltv_base::Encrypt(ZZ_pE m){
  ZZ_pE res;
  res = SampleErr();
  res += this->h*SampleErr();
  res += m*conv<ZZ_pE>(this->q/this->t);
  return res;
}

ZZ_pE ltv_base::CenterLift(ZZ_pE a){
  int i;
  ZZ aux;
  ZZ_pX pol = conv<ZZ_pX>(a);
  for(i=0;i<this->n;i++){
    aux = conv<ZZ>(coeff(pol, i));
    aux = this->CenterLift(aux, this->q);
    SetCoeff(pol, i, conv<ZZ_p>(aux));
  }
  return conv<ZZ_pE>(pol);
}

ZZ ltv_base::CenterLift(ZZ a, ZZ q){
  if (a > q/2)
    return a-q;
  else 
    return a;
}

ZZ ltv_base::DesLift(ZZ a, long q){
  if (a < 0)
    return a+q;
  else 
    return a;
}

ZZ_pE ltv_base::ModN(ZZ_pE a, long n){
  int i;
  ZZ_pE res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  for(i=0;i<this->n;i++){
    ZZ aux = conv<ZZ>(coeff(pol, i));
    aux = CenterLift(aux, this->q);
    SetCoeff(pol, i, conv<ZZ_p>(aux%conv<ZZ>(n)));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::Mod2(ZZ_pE a){
  int i;
  ZZ_pE res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  for(i=0;i<this->n;i++){
    ZZ aux = conv<ZZ>(coeff(pol, i));
    aux = CenterLift(aux, this->q);
    SetCoeff(pol, i, conv<ZZ_p>(aux%conv<ZZ>(2)));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::Mod256(ZZ_pE a){
  int i;
  ZZ_pE res;
  ZZ_pX pol = conv<ZZ_pX>(a);
  for(i=0;i<this->n;i++){
    ZZ aux = conv<ZZ>(coeff(pol, i));
    aux = CenterLift(aux, this->q);
    SetCoeff(pol, i, conv<ZZ_p>(aux%conv<ZZ>(256)));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::DecryptLTV(ZZ_pE c){
  int i;
  ZZ_pE res;
  ZZ_pX pol;
  ZZ_pX aux = conv<ZZ_pX>(c*this->f);
  for(i=0;i<this->n;i++){
    ZZ auxc = conv<ZZ>(coeff(aux,i));
    auxc %= q;
    if (auxc > q/2)
      auxc = (auxc - q);
    auxc %= conv<ZZ>(t);
    SetCoeff(pol, i, conv<ZZ_p>(auxc));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::Decrypt(ZZ_pE c){
  int i;
  ZZ_pE res;
  ZZ_pX pol;
  ZZ_pX aux = conv<ZZ_pX>(c*this->f);
  for(i=0;i<this->n;i++){
    ZZ auxc = conv<ZZ>(coeff(aux,i));

    auxc *= this->t;
    if (auxc >= 0)
      auxc = (auxc + q/2)/q;
    else 
      auxc = (auxc - q/2)/q;
    auxc %= conv<ZZ>(t);
    SetCoeff(pol, i, conv<ZZ_p>(auxc));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}

ZZ_pE ltv_base::DecryptMul(ZZ_pE c){
  int i;
  ZZ_pE res;
  ZZ_pX pol;
  ZZ_pX aux = conv<ZZ_pX>(c*this->f*this->f);
  for(i=0;i<this->n;i++){
    ZZ auxc = conv<ZZ>(coeff(aux,i));

    auxc *= this->t;
    if (auxc >= 0)
      auxc = (auxc + q/2)/q;
    else
      auxc = (auxc - q/2)/q;
    auxc %= conv<ZZ>(t);
    SetCoeff(pol, i, conv<ZZ_p>(auxc));
  }
  res = conv<ZZ_pE>(pol);
  return res;
}
