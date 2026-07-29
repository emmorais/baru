// NTL+GMP negacyclic polynomial multiplication — the C++ half of
// `baru::fhe::ntl` (see `src/fhe/ntl.rs` for the Rust wrapper and for why
// this exists alongside the multi-modular NTT in `src/fhe/ntt.rs`).
//
// Contract with the Rust side
// ---------------------------
// Coefficients cross the FFI as fixed-width little-endian limb arrays, so no
// allocation or ownership crosses the boundary:
//
//   input  coefficient: 2 × uint64  = two's-complement i128   (limb[0] = low)
//   output coefficient: 4 × uint64  = two's-complement i256   (limb[0] = low)
//
// The product is *exact in ℤ* — nothing here is reduced modulo anything. The
// only failure mode is a result too wide for 256 bits, which is reported
// through the return code rather than silently truncated.
//
// Algorithm: build the two operands as NTL `ZZX` (dense arbitrary-precision
// integer polynomials, limbs handled by GMP), take the full 2n−1 term product
// via NTL's `mul`, and fold it negacyclically:
//
//   out[k] = full[k] − full[k + n]
//
// NTL's `mul(ZZX,ZZX,ZZX)` picks its own strategy by operand size — plain,
// Karatsuba, or homomorphic imaging (Schönhage–Strassen / multi-modular FFT)
// — which is exactly the point of having this second implementation: it is a
// mature, independently tuned answer to the same question `ntt.rs` answers by
// hand.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <gmp.h>

#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <NTL/version.h>

namespace {

constexpr int kOk = 0;
constexpr int kOverflow = 1;  // |coefficient| did not fit a signed 256-bit int
constexpr int kBadArg = 2;

// Two's-complement i128 (2 limbs, little-endian) → ZZ.
NTL::ZZ FromI128(const uint64_t* limb) {
  uint64_t lo = limb[0], hi = limb[1];
  const bool negative = (hi >> 63) & 1;
  if (negative) {  // negate in place: ~x + 1
    lo = ~lo;
    hi = ~hi;
    if (++lo == 0) ++hi;
  }
  unsigned char bytes[16];
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<unsigned char>(lo >> (8 * i));
    bytes[8 + i] = static_cast<unsigned char>(hi >> (8 * i));
  }
  NTL::ZZ z = NTL::ZZFromBytes(bytes, 16);
  return negative ? -z : z;
}

// ZZ → two's-complement i256 (4 limbs, little-endian). Returns false if the
// value needs more than 255 magnitude bits, i.e. does not fit signed 256.
bool ToI256(const NTL::ZZ& v, uint64_t* out) {
  const bool negative = v < 0;
  const NTL::ZZ magnitude = negative ? -v : v;
  if (NTL::NumBits(magnitude) > 255) return false;

  unsigned char bytes[32];
  NTL::BytesFromZZ(bytes, magnitude, 32);
  for (int w = 0; w < 4; ++w) {
    uint64_t limb = 0;
    for (int i = 0; i < 8; ++i) {
      limb |= static_cast<uint64_t>(bytes[8 * w + i]) << (8 * i);
    }
    out[w] = limb;
  }
  if (negative) {  // ~x + 1 across the four limbs
    for (int w = 0; w < 4; ++w) out[w] = ~out[w];
    for (int w = 0; w < 4; ++w) {
      if (++out[w] != 0) break;
    }
  }
  return true;
}

void LoadPoly(NTL::ZZX& p, const uint64_t* limbs, size_t n) {
  p.SetLength(static_cast<long>(n));
  for (size_t i = 0; i < n; ++i) {
    NTL::SetCoeff(p, static_cast<long>(i), FromI128(limbs + 2 * i));
  }
  p.normalize();
}

// full[k] − full[k+n], written to `out` as i256 limbs.
int FoldNegacyclic(const NTL::ZZX& full, size_t n, uint64_t* out) {
  for (size_t k = 0; k < n; ++k) {
    const NTL::ZZ c =
        NTL::coeff(full, static_cast<long>(k)) - NTL::coeff(full, static_cast<long>(k + n));
    if (!ToI256(c, out + 4 * k)) return kOverflow;
  }
  return kOk;
}

}  // namespace

extern "C" {

// c = a · b in ℤ[X]/(X^n + 1), exact.
//
// `a`, `b`: 2n limbs each (n i128 coefficients). `out`: 4n limbs (n i256
// coefficients), written only on success.
int baru_ntl_negacyclic_mul(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* out) {
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) return kBadArg;
  NTL::ZZX pa, pb, prod;
  LoadPoly(pa, a, n);
  LoadPoly(pb, b, n);
  NTL::mul(prod, pa, pb);
  return FoldNegacyclic(prod, n, out);
}

// The BFV multiply's three tensor components in one call:
//   d0 = a0·b0,  d1 = a0·b1 + a1·b0,  d2 = a1·b1
//
// Mirrors `ntt::exact_bfv_tensor` so the two implementations are swappable at
// the call site. Unlike the NTT version there is no shared-transform saving to
// exploit here — NTL owns the transform internally — so this is purely four
// products with the cross terms added before the fold.
int baru_ntl_bfv_tensor(const uint64_t* a0, const uint64_t* a1, const uint64_t* b0,
                        const uint64_t* b1, size_t n, uint64_t* d0, uint64_t* d1, uint64_t* d2) {
  if (a0 == nullptr || a1 == nullptr || b0 == nullptr || b1 == nullptr || d0 == nullptr ||
      d1 == nullptr || d2 == nullptr || n == 0) {
    return kBadArg;
  }
  NTL::ZZX pa0, pa1, pb0, pb1;
  LoadPoly(pa0, a0, n);
  LoadPoly(pa1, a1, n);
  LoadPoly(pb0, b0, n);
  LoadPoly(pb1, b1, n);

  NTL::ZZX t00, t01, t10, t11;
  NTL::mul(t00, pa0, pb0);
  NTL::mul(t01, pa0, pb1);
  NTL::mul(t10, pa1, pb0);
  NTL::mul(t11, pa1, pb1);

  int rc = FoldNegacyclic(t00, n, d0);
  if (rc != kOk) return rc;
  rc = FoldNegacyclic(t01 + t10, n, d1);
  if (rc != kOk) return rc;
  return FoldNegacyclic(t11, n, d2);
}

// "NTL <version> / GMP <version>", for benchmark provenance. Static storage:
// the Rust side only reads it.
const char* baru_ntl_version(void) {
  static char buf[64];
  std::snprintf(buf, sizeof(buf), "NTL %s / GMP %s", NTL_VERSION, __gmp_version);
  return buf;
}

}  // extern "C"
