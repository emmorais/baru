#include <ltv.h>
#include <math.h>
#include <stdio.h>
#include <NTL/ZZX.h>

using namespace std;

ltv::ltv(){
  this->b = new ltv_base();
}

ZZ_pE ltv::SampleMessage256(){
  return this->b->SampleMessage256();
}

ZZ_pE ltv::SampleMessageN(long n){
  return this->b->SampleMessageN(n);
}

Vec<ZZ_pE> ltv::GetKeySwitch(){
  return this->gamma;
}

Vec<ZZ_pE> ltv::KeySwitchGen(){
  int i;
  Vec<ZZ_pE> e, s;
  ZZ_pE h = this->b->PublicKey();
  s.SetLength(this->lwq);
  e = this->PowerOf(this->b->PrivateKey());
  for(i=0;i<this->lwq;i++){
    e[i] += this->b->SampleErr();
    s[i] = this->b->SampleErr();
    s[i] *= h;
  }
  return e+s;
}

void ltv::KeyGen(){
  this->b->KeyGen();
  this->gamma = this->KeySwitchGen();
}

void ltv::ParamsGen(long t, long w, long e, long n, ZZ q, double delta, long lwq){	
  this->lwq = lwq;
  this->b->ParamsGen(t, w, e, n, q, delta);
}

Vec<ZZ_pE> ltv::BitDecomp(ZZ_pE x){
  int i, j;
  Vec<ZZ_pE> res;
  res.SetLength(this->lwq);
  ZZ w = power(conv<ZZ>(2), this->b->w);
  ZZ_pX pol = conv<ZZ_pX>(x);
  ZZ_pX aux;
  for(i=0; i<this->lwq; i++){
    for (j=0; j<this->b->n;j++){
      ZZ c = conv<ZZ>(coeff(pol, j));
      SetCoeff(aux, j, conv<ZZ_p>(c%w));
      SetCoeff(pol, j, conv<ZZ_p>(c/w));
    }
    res[i] = conv<ZZ_pE>(aux);
  }
  return res;
}

Vec<ZZ_pE> ltv::PowerOf(ZZ_pE x){
  int i, j;
  Vec<ZZ_pE> res;
  res.SetLength(this->lwq);
  ZZ w = power(conv<ZZ>(2), this->b->w);
  ZZ k = conv<ZZ>(1);
  ZZ_pX pol = conv<ZZ_pX>(x);
  ZZ_pX aux;
  for(i=0; i<this->lwq; i++){
    for (j=0; j<this->b->n;j++){
      ZZ c = conv<ZZ>(coeff(pol, j));
      SetCoeff(aux, j, conv<ZZ_p>(c));
      SetCoeff(pol, j, conv<ZZ_p>(c*w));
    }
    res[i] = conv<ZZ_pE>(aux);
  }
  return res;
}

ZZ_pE ltv::Encrypt(ZZ_pE m){
  return this->b->Encrypt(m);
}

ZZ_pE ltv::DecryptLTV(ZZ_pE c){
  return this->b->DecryptLTV(c);
}

ZZ_pE ltv::Decrypt(ZZ_pE c){
  return this->b->Decrypt(c);
}

ZZ_pE ltv::DecryptMul(ZZ_pE c){
  return this->b->DecryptMul(c);
}
  
ZZ_pE ltv::InnerProduct(Vec<ZZ_pE> a, Vec<ZZ_pE> b){
  int i;
  ZZ_pE res = conv<ZZ_pE>(0);
  for(i=0;i<a.length();i++){
    res += a[i]*b[i];
  }
  return res;
}

ZZ_pE ltv::KeySwitch(ZZ_pE c){
  ZZ_pE res;
  Vec<ZZ_pE> dqw = this->BitDecomp(c);
  res = InnerProduct(dqw, this->gamma);
  return res;
}

ZZ_pE ltv::Mod2(ZZ_pE a){
  return this->b->Mod2(a);
}

ZZ_pE ltv::Mod256(ZZ_pE a){
  return this->b->Mod256(a);
}

ZZ_pE ltv::ModN(ZZ_pE a, long n){
  return this->b->ModN(a, n);
}
    
ZZ_pE ltv::Mult(ZZ_pE a, ZZ_pE b, ZZ q){
  int i;
  ZZ_pEBak bak;
  bak.save();  
  ZZ qq = this->b->n*q*q+1;
  ZZ_p::init(conv<ZZ>(qq));
  ZZ_pX P, pol;
  SetCoeff(P, 0, 1);
  SetCoeff(P, this->b->n, 1);
  ZZ_pE::init(P);

  ZZ_pE c = a*b;
  pol = conv<ZZ_pX>(c);
  ZZX ppol = conv<ZZX>(pol);
  for(i=0;i<this->b->n;i++){
      ZZ aux = conv<ZZ>(coeff(pol, i));
      aux *= this->b->t;
      aux = (aux + q/2)/q;
      aux %= q;
      SetCoeff(ppol, i, aux);
  }
  ZZ_p::init(q);
  SetCoeff(P, 0, 1);
  SetCoeff(P, this->b->n, 1);
  ZZ_pE::init(P);
  bak.restore();
  ZZ_pX cp = conv<ZZ_pX>(ppol);
  c = conv<ZZ_pE>(cp);
  if (this->b->e > 0){
    ZZ_pX p10;
    ZZ_pE r10;
    SetCoeff(p10, this->b->e, 1);
    r10 = conv<ZZ_pE>(p10);
    c *= inv(r10);
  }
  return c;
}

ZZ_pE ltv::Mul(ZZ_pE a, ZZ_pE b, ZZ q){
  ZZ_pE res = this->Mult(a, b, q);
  return this->KeySwitch(res);
}
