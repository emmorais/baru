// A generalized-diagonal linear transform with baby-step/giant-step, checked against a
// plaintext matrix-vector product.
//
// This is the engine Orion (ASPLOS'25) evaluates every layer with, and the reason its
// operation counts are far below a tap-by-tap convolution. It is separated into its own
// program because it is the piece most likely to be subtly wrong: the index arithmetic of
// a rotated diagonal is easy to get off by one in a way that still produces plausible
// numbers, so it is verified against a reference before anything is built on it.
//
// # The method
//
// For an `n × n` matrix `A` and a vector `v` packed in `n` slots, Halevi and Shoup's
// identity is
//
//     A·v = Σ_d  diag_d(A) ⊙ rot(v, d),      diag_d(A)[i] = A[i][(i+d) mod n]
//
// so a transform costs one plaintext multiply per *nonzero* diagonal and one rotation per
// diagonal. The count of nonzero diagonals — not the matrix's size — is what a layer
// costs, which is why the packing that determines it matters so much.
//
// Baby-step/giant-step then removes most of the rotations. Writing `d = g·n1 + b` with
// `b < n1`,
//
//     Σ_d diag_d ⊙ rot(v,d) = Σ_g rot( Σ_b rot(diag_{g·n1+b}, −g·n1) ⊙ rot(v,b), g·n1 )
//
// The inner rotations of `v` are shared across every `g`, so the cost is `n1` baby
// rotations plus `n2` giant ones rather than `n1·n2`. The diagonals themselves are
// pre-rotated, which is free: they are plaintext.
#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/// The nonzero generalized diagonals of `a`, an `n × n` row-major matrix.
static std::map<size_t, std::vector<double>> diagonals(const std::vector<double>& a, size_t n) {
    std::map<size_t, std::vector<double>> out;
    for (size_t d = 0; d < n; d++) {
        std::vector<double> diag(n, 0.0);
        bool any = false;
        for (size_t i = 0; i < n; i++) {
            double v = a[i * n + ((i + d) % n)];
            diag[i] = v;
            if (v != 0.0) any = true;
        }
        if (any) out.emplace(d, std::move(diag));
    }
    return out;
}

int main() {
    // A matrix with a *sparse* diagonal structure, which is what a convolution under a
    // packing that aligns its offsets looks like: few nonzero diagonals in a large matrix.
    const size_t n = 1024;
    const std::vector<size_t> live = {0, 1, 2, 32, 33, 34, 64, 65, 66};  // a 3x3 stencil
    std::vector<double> a(n * n, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < live.size(); k++)
            a[i * n + ((i + live[k]) % n)] = 0.5 + 0.1 * double(k);

    std::vector<double> v(n);
    for (size_t i = 0; i < n; i++) v[i] = std::sin(0.01 * double(i));

    // The reference.
    std::vector<double> want(n, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) want[i] += a[i * n + j] * v[j];

    auto diags = diagonals(a, n);
    const size_t D = diags.size();
    // Balance the two halves: n1 baby rotations and about D/n1 giant ones.
    size_t n1 = 1;
    while (n1 * n1 < D) n1 <<= 1;

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetMultiplicativeDepth(2);
    p.SetScalingModSize(45);
    p.SetFirstModSize(50);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    const size_t slots = n;

    // Keys: the baby rotations of the vector, and the giant rotations of the sum.
    std::vector<int32_t> rots;
    for (size_t b = 1; b < n1; b++) rots.push_back(int32_t(b));
    for (const auto& kv : diags) {
        size_t g = kv.first / n1;
        if (g > 0) rots.push_back(int32_t(g * n1));
    }
    std::sort(rots.begin(), rots.end());
    rots.erase(std::unique(rots.begin(), rots.end()), rots.end());

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, rots);

    auto pt_v = cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, slots);
    auto ct = cc->Encrypt(keys.publicKey, pt_v);

    size_t n_rot = 0, n_mul = 0;
    auto t0 = Clock::now();

    // Baby step: every rotation of the input that any diagonal will need, computed once.
    std::vector<Ciphertext<DCRTPoly>> baby(n1);
    baby[0] = ct;
    for (size_t b = 1; b < n1; b++) {
        baby[b] = cc->EvalRotate(ct, int32_t(b));
        n_rot++;
    }

    // Giant step: group the diagonals by `g`, accumulate each group against the shared
    // baby rotations, then rotate the group's total once.
    std::map<size_t, std::vector<size_t>> groups;
    for (const auto& kv : diags) groups[kv.first / n1].push_back(kv.first);

    Ciphertext<DCRTPoly> acc = nullptr;
    for (const auto& g : groups) {
        Ciphertext<DCRTPoly> inner = nullptr;
        for (size_t d : g.second) {
            const size_t b = d % n1;
            // The diagonal is pre-rotated by the giant offset, which is free because it
            // is plaintext: rotating data left by `g·n1` after multiplying is the same as
            // multiplying by the diagonal rotated *right* by `g·n1` and then rotating.
            const auto& src = diags.at(d);
            std::vector<double> shifted(slots);
            const size_t off = g.first * n1;
            for (size_t i = 0; i < slots; i++) shifted[i] = src[(i + off) % slots];
            auto pd = cc->MakeCKKSPackedPlaintext(shifted, 1, baby[b]->GetLevel(), nullptr, slots);
            auto term = cc->EvalMult(baby[b], pd);
            n_mul++;
            inner = inner ? cc->EvalAdd(inner, term) : term;
        }
        if (g.first > 0) {
            inner = cc->EvalRotate(inner, int32_t(g.first * n1));
            n_rot++;
        }
        acc = acc ? cc->EvalAdd(acc, inner) : inner;
    }
    double eval = secs(t0);

    Plaintext out;
    cc->Decrypt(keys.secretKey, acc, &out);
    out->SetLength(slots);
    auto got = out->GetRealPackedValue();

    double worst = 0.0;
    for (size_t i = 0; i < n; i++) worst = std::max(worst, std::fabs(got[i] - want[i]));

    std::cout << "n " << n << "\n"
              << "nonzero_diagonals " << D << "\n"
              << "bsgs_n1 " << n1 << "\n"
              << "rotations " << n_rot << "\n"
              << "plaintext_mults " << n_mul << "\n"
              << "rotations_naive " << D << "\n"
              << "eval_s " << eval << "\n"
              << "worst_error " << worst << "\n"
              << "verdict " << (worst < 1e-6 ? "agrees" : "WRONG") << std::endl;
    return worst < 1e-6 ? 0 : 1;
}
