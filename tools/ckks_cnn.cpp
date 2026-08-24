// A homomorphic arm for the convolutional benchmark: CKKS on OpenFHE, evaluating
// the same network on the same input as the secret-sharing arm.
//
// Why this exists rather than a citation. Comparing our measured latency against a
// figure published for someone else's hardware, someone else's library and — as
// `docs/experiments.md` §VI.8 records having got wrong once — someone else's network
// under the same name, is not a comparison. This evaluates the network exported by
// `bench::conv_nn::export`, on this machine, and prints the class it decides so the
// caller can refuse the measurement if the two arms disagree.
//
// What it is not: it is not Orion. It is a competent, depth-conscious CKKS
// implementation of the same pipeline — packed feature maps, convolution by
// rotation, `x^2` activation, average pooling folded into the following scale, and a
// bootstrap inserted exactly when the level budget runs out. Where Orion's
// contribution is automating that last decision across arbitrary graphs, here the
// graph is fixed and the placement is computed directly from the layer list, which
// is sufficient for the networks measured and is stated rather than overclaimed.
//
// Reads the flat text fixture on argv[1], prints a report to stdout as `key value`
// lines.
#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ─── The fixture ─────────────────────────────────────────────────────────────

struct Layer {
    std::string kind;   // conv | act | avgpool | maxpool | flatten | dense
    int out_c = 0, k = 0, in_c = 0, stride = 0, out_dim = 0, in_dim = 0;
    std::vector<double> w, b;
};

struct Fixture {
    int scale_bits = 0;
    int c = 0, h = 0, w = 0;
    std::vector<double> input;
    std::vector<Layer> layers;
    std::vector<double> reference;  // plaintext logits, at output scale
};

static Fixture load(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ckks_cnn: cannot open " << path << "\n";
        std::exit(2);
    }
    Fixture fx;
    double unit = 0;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string tag;
        in >> tag;
        if (tag == "scale_bits") {
            in >> fx.scale_bits;
            unit = std::pow(2.0, fx.scale_bits);
        } else if (tag == "input") {
            in >> fx.c >> fx.h >> fx.w;
        } else if (tag == "values") {
            double v;
            while (in >> v) fx.input.push_back(v / unit);
        } else if (tag == "conv") {
            Layer l;
            l.kind = "conv";
            int li;
            in >> li >> l.out_c >> l.k >> l.in_c;
            size_t nw = size_t(l.out_c) * l.in_c * l.k * l.k;
            l.w.resize(nw);
            for (auto& x : l.w) { in >> x; x /= unit; }
            l.b.resize(l.out_c);
            for (auto& x : l.b) { in >> x; x /= unit; }
            fx.layers.push_back(std::move(l));
        } else if (tag == "dense") {
            Layer l;
            l.kind = "dense";
            int li;
            in >> li >> l.out_dim >> l.in_dim;
            l.w.resize(size_t(l.out_dim) * l.in_dim);
            for (auto& x : l.w) { in >> x; x /= unit; }
            l.b.resize(l.out_dim);
            for (auto& x : l.b) { in >> x; x /= unit; }
            fx.layers.push_back(std::move(l));
        } else if (tag == "act") {
            Layer l; l.kind = "act"; fx.layers.push_back(l);
        } else if (tag == "avgpool" || tag == "maxpool") {
            Layer l; l.kind = tag; int li; in >> li >> l.stride;
            fx.layers.push_back(l);
        } else if (tag == "flatten") {
            Layer l; l.kind = "flatten"; int li; in >> li >> l.in_dim;
            fx.layers.push_back(l);
        } else if (tag == "reference") {
            double v;
            while (in >> v) fx.reference.push_back(v / unit);
        }
    }
    return fx;
}

// The multiplicative depth the network needs, which is what decides both the
// parameters and whether a bootstrap is required at all. Every ciphertext-plaintext
// multiply and every ciphertext-ciphertext multiply consumes one level; additions and
// rotations consume none.
static int depth_of(const Fixture& fx) {
    int d = 0;
    for (const auto& l : fx.layers) {
        if (l.kind == "conv" || l.kind == "dense") d += 1;      // weights, plaintext
        else if (l.kind == "act") d += 1;                        // x*x, ciphertext
        else if (l.kind == "avgpool") d += 1;                    // the 1/s^2 scale
        else if (l.kind == "maxpool") d += 1000;                 // not evaluable here
    }
    return d;
}


// ─── The linear-transform evaluator ──────────────────────────────────────────
//
// A second, independent evaluator following Orion (ASPLOS'25): every layer —
// convolution, pooling and dense alike — is one linear transform, evaluated by its
// generalized diagonals with baby-step/giant-step. It exists for two reasons. It is much
// cheaper, for the reason set out below. And it is a *second implementation* of the same
// network, so the two must agree on the class, which is a far stronger check on either
// than a plaintext reference alone.
//
// # Why a matrix is cheaper than taps
//
// By Halevi and Shoup, an `n × n` matrix applied to a vector packed in `n` slots is
//
//     A·v = Σ_d  diag_d(A) ⊙ rot(v, d),      diag_d(A)[i] = A[i][(i+d) mod n]
//
// so the cost is one plaintext multiply per **nonzero diagonal**, not per matrix entry.
// A convolution's matrix is enormous and almost entirely zero, and how many of its
// diagonals are nonzero depends entirely on the slot layout: if the offset between an
// input element and the output element it feeds is the same everywhere, all of that
// convolution's contributions collapse onto a handful of diagonals.
//
// # Which is what the multiplexed layout is for
//
// Under a naive contiguous layout the offset is *not* position-independent. For a 28×28
// input and 24×24 output the difference between an input slot and its output slot is
// `y·4 + a·28 + b − oc·576`, which depends on the row `y`, so each row contributes its own
// diagonal and a single convolution has thousands. Multiplexed packing removes the
// dependence: spatial extent is held fixed and channels are folded into `g × g` tiles,
// with `g` doubling at every stride, so a pooled feature map still occupies the same grid
// and a tap's offset becomes `a·W·g + b·g` — the same at every position.
//
// This implementation does not assume the resulting diagonal count; it builds each layer's
// matrix in the multiplexed layout and reports how many diagonals it actually has, so the
// packing's effectiveness is measured rather than argued.

/// Where element `(c, y, x)` of a feature map lives.
///
/// The **grid** `(h, w)` and the **valid extent** `(vh, vw)` are deliberately separate,
/// and keeping them so is the whole point of this structure.
///
/// A convolution shrinks the data it holds — a 5×5 kernel over 28×28 leaves 24×24 — but if
/// the *layout* shrank with it, the distance from an input element to the output element it
/// feeds would depend on the row: at grid 28 in and grid 24 out the offset carries a term
/// `y·4`, so every row contributes its own generalized diagonal and one convolution has
/// over a thousand of them (measured: 1225, before this separation existed). Holding the
/// grid fixed and letting the valid region shrink inside it makes the offset `a·W·g + b·g`,
/// the same at every position.
///
/// `g` is the multiplexing gap: channels are grouped `g²` at a time into the `g × g` tile
/// at each grid position, so `ceil(c/g²)` blocks of `(h·g) × (w·g)` are used. A stride-`s`
/// pooling halves the grid and doubles the gap, so `h·g` — and therefore the block size —
/// is **invariant** across the whole network. That is what lets the channel count grow
/// without the slot budget growing with it.
struct Layout {
    int c, vh, vw, h, w, g;

    int outer() const { return (c + g * g - 1) / (g * g); }
    size_t block() const { return size_t(h) * g * size_t(w) * g; }
    size_t span() const { return size_t(outer()) * block(); }

    size_t slot(int ch, int y, int x) const {
        const int gg = g * g;
        const int co = ch / gg, r = ch % gg;
        const int i = r / g, j = r % g;
        const int W = w * g;
        return size_t(co) * block() + size_t(y * g + i) * size_t(W) + size_t(x * g + j);
    }
};

/// A layer as a sparse map from generalized-diagonal index to that diagonal's values.
using Diags = std::map<size_t, std::vector<double>>;

/// Accumulate `weight` into the entry taking `in_slot` to `out_slot`.
static void put(Diags& d, size_t slots, size_t out_slot, size_t in_slot, double weight) {
    if (weight == 0.0) return;
    const size_t idx = (in_slot + slots - out_slot % slots) % slots;
    auto it = d.find(idx);
    if (it == d.end()) it = d.emplace(idx, std::vector<double>(slots, 0.0)).first;
    it->second[out_slot % slots] += weight;
}

int main_lt(const Fixture& fx, int batch, int cap, bool want_boot);


/// Build the linear transform for one layer, and the layout its output has.
///
/// Every layer here is linear: a convolution and a dense layer obviously so, and average
/// pooling too, since it is a weighted sum over a window. The activation is the only
/// nonlinear step and is not built here.
static Diags build_layer(const Layer& l, const Layout& in, Layout& out, size_t slots) {
    Diags d;
    if (l.kind == "conv") {
        // The grid and gap are untouched; only the valid region shrinks.
        out = Layout{l.out_c, in.vh - l.k + 1, in.vw - l.k + 1, in.h, in.w, in.g};
        for (int oc = 0; oc < l.out_c; oc++)
            for (int y = 0; y < out.vh; y++)
                for (int x = 0; x < out.vw; x++) {
                    const size_t os = out.slot(oc, y, x);
                    for (int ic = 0; ic < l.in_c; ic++)
                        for (int a = 0; a < l.k; a++)
                            for (int b = 0; b < l.k; b++)
                                put(d, slots, os, in.slot(ic, y + a, x + b),
                                    l.w[((size_t(oc) * l.in_c + ic) * l.k + a) * l.k + b]);
                }
    } else if (l.kind == "avgpool") {
        // The gap grows by the stride, which is what keeps the layout's offsets
        // position-independent for the next convolution without compacting anything.
        out = Layout{in.c,        in.vh / l.stride, in.vw / l.stride,
                     in.h / l.stride, in.w / l.stride,  in.g * l.stride};
        const double s = 1.0 / double(l.stride * l.stride);
        for (int c = 0; c < in.c; c++)
            for (int y = 0; y < out.vh; y++)
                for (int x = 0; x < out.vw; x++) {
                    const size_t os = out.slot(c, y, x);
                    for (int a = 0; a < l.stride; a++)
                        for (int b = 0; b < l.stride; b++)
                            put(d, slots, os,
                                in.slot(c, y * l.stride + a, x * l.stride + b), s);
                }
    } else if (l.kind == "dense") {
        // A contiguous vector: `outer` counts the values and each occupies one slot.
        out = Layout{l.out_dim, 1, 1, 1, 1, 1};
        const size_t per = size_t(in.vh) * in.vw;
        for (int o = 0; o < l.out_dim; o++) {
            const size_t os = out.slot(o, 0, 0);
            for (int c = 0; c < in.c; c++)
                for (int y = 0; y < in.vh; y++)
                    for (int x = 0; x < in.vw; x++) {
                        // The flattened index must agree with `Chw::idx` on the other arm
                        // exactly, or the two evaluate different networks.
                        const size_t fi = size_t(c) * per + size_t(y) * in.vw + x;
                        put(d, slots, os, in.slot(c, y, x),
                            l.w[size_t(o) * l.in_dim + fi]);
                    }
        }
    }
    return d;
}

int main_lt(const Fixture& fx, int batch, int cap, bool want_boot) {
    // Walk the layers once to learn the widest layout, which sets the slot budget, and to
    // report the diagonal counts the packing achieves.
    Layout cur{fx.c, fx.h, fx.w, fx.h, fx.w, 1};
    size_t widest = cur.span();
    {
        Layout l = cur, o{};
        for (const auto& ly : fx.layers) {
            if (ly.kind == "conv") {
                o = Layout{ly.out_c, l.vh - ly.k + 1, l.vw - ly.k + 1, l.h, l.w, l.g};
            } else if (ly.kind == "avgpool") {
                o = Layout{l.c,           l.vh / ly.stride, l.vw / ly.stride,
                           l.h / ly.stride, l.w / ly.stride,  l.g * ly.stride};
            } else if (ly.kind == "dense") {
                o = Layout{ly.out_dim, 1, 1, 1, 1, 1};
            } else {
                continue;
            }
            widest = std::max(widest, std::max(l.span(), o.span()));
            l = o;
        }
    }
    size_t block = 1;
    while (block < widest) block <<= 1;

    int need = depth_of(fx);
    std::vector<uint32_t> budget = {4, 4};
    int usable = (cap > 0 && cap < need) ? cap : need;
    int depth = usable;
    if (want_boot) depth += FHECKKSRNS::GetBootstrapDepth(budget, UNIFORM_TERNARY);

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetScalingModSize(45);
    p.SetFirstModSize(50);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    p.SetMultiplicativeDepth(depth);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    if (want_boot) {
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
    }

    size_t slots = block * size_t(batch);
    if (slots > cc->GetRingDimension() / 2) {
        std::cerr << "ckks_cnn --lt: batch " << batch << " needs " << slots
                  << " slots, ring provides " << cc->GetRingDimension() / 2
                  << "; largest batch is " << (cc->GetRingDimension() / 2 / block) << std::endl;
        return 2;
    }
    { size_t q = 1; while (q < slots) q <<= 1; slots = q; }

    // Build every layer's transform up front, so the diagonal counts — the quantity the
    // packing exists to keep small — are known before any key is generated.
    std::vector<Diags> transforms;
    std::vector<Layout> outs;
    std::vector<double> report_diags;
    {
        Layout l = cur;
        for (const auto& ly : fx.layers) {
            if (ly.kind == "act" || ly.kind == "flatten") {
                transforms.emplace_back();
                outs.push_back(l);
                report_diags.push_back(0);
                continue;
            }
            Layout o{};
            Diags d = build_layer(ly, l, o, block);
            report_diags.push_back(double(d.size()));
            transforms.push_back(std::move(d));
            outs.push_back(o);
            l = o;
        }
    }

    // Baby-step width per transform, and the rotation keys both halves need.
    std::vector<size_t> n1s(transforms.size(), 1);
    std::vector<int32_t> rots;
    for (size_t t = 0; t < transforms.size(); t++) {
        const size_t D = transforms[t].size();
        if (D == 0) continue;
        size_t n1 = 1;
        while (n1 * n1 < D) n1 <<= 1;
        n1s[t] = n1;
        for (size_t b = 1; b < n1; b++) rots.push_back(int32_t(b));
        for (const auto& kv : transforms[t]) {
            const size_t g = kv.first / n1;
            if (g > 0) rots.push_back(int32_t(g * n1));
        }
    }
    std::sort(rots.begin(), rots.end());
    rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
    rots.erase(std::remove(rots.begin(), rots.end(), 0), rots.end());

    auto t_keys = Clock::now();
    std::cerr << "ckks_cnn --lt: ring " << cc->GetRingDimension() << ", depth " << depth
              << ", block " << block << ", slots " << slots << ", " << rots.size()
              << " rotation keys" << std::endl;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, rots);
    if (want_boot) {
        cc->EvalBootstrapSetup(budget, {0, 0}, slots);
        cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }
    double keygen_s = secs(t_keys);

    // Encrypt: one block per inference, the input map in the gap-1 layout.
    Ciphertext<DCRTPoly> ct;
    {
        std::vector<double> v(slots, 0.0);
        for (int n = 0; n < batch; n++)
            for (int c = 0; c < cur.c; c++)
                for (int y = 0; y < cur.vh; y++)
                    for (int x = 0; x < cur.vw; x++)
                        v[size_t(n) * block + cur.slot(c, y, x)] =
                            fx.input[(size_t(c) * cur.vh + y) * cur.vw + x];
        ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, slots));
    }

    auto t_eval = Clock::now();
    size_t n_rot = 0, n_mul = 0, n_boot = 0;
    Layout live = cur;
    for (size_t t = 0; t < fx.layers.size(); t++) {
        const auto& ly = fx.layers[t];
        if (want_boot && usable - int(ct->GetLevel()) <= 1) {
            ct = cc->EvalBootstrap(ct);
            n_boot++;
        }
        if (ly.kind == "act") {
            ct = cc->EvalMult(ct, ct);
            n_mul++;
            continue;
        }
        if (ly.kind == "flatten") continue;

        const Diags& d = transforms[t];
        const size_t n1 = n1s[t];

        // Baby step: each rotation of the input that some diagonal needs, once.
        std::map<size_t, Ciphertext<DCRTPoly>> baby;
        baby[0] = ct;
        for (const auto& kv : d) {
            const size_t b = kv.first % n1;
            if (b != 0 && !baby.count(b)) {
                baby[b] = cc->EvalRotate(ct, int32_t(b));
                n_rot++;
            }
        }
        // Giant step: accumulate each group against the shared baby rotations, then rotate
        // the group's total once.
        std::map<size_t, std::vector<size_t>> groups;
        for (const auto& kv : d) groups[kv.first / n1].push_back(kv.first);

        Ciphertext<DCRTPoly> acc = nullptr;
        for (const auto& g : groups) {
            Ciphertext<DCRTPoly> inner = nullptr;
            const size_t off = g.first * n1;
            for (size_t idx : g.second) {
                const auto& src = d.at(idx);
                // Pre-rotate the diagonal by the giant offset and tile it across the
                // batch. Both are free: it is plaintext.
                std::vector<double> v(slots, 0.0);
                for (int n = 0; n < batch; n++) {
                    const size_t base = size_t(n) * block;
                    for (size_t i = 0; i < block; i++)
                        v[base + i] = src[(i + off) % block];
                }
                auto pd = cc->MakeCKKSPackedPlaintext(v, 1, baby[idx % n1]->GetLevel(),
                                                      nullptr, slots);
                auto term = cc->EvalMult(baby[idx % n1], pd);
                n_mul++;
                inner = inner ? cc->EvalAdd(inner, term) : term;
            }
            if (off > 0) {
                inner = cc->EvalRotate(inner, int32_t(off));
                n_rot++;
            }
            acc = acc ? cc->EvalAdd(acc, inner) : inner;
        }
        // Bias, at the output layout's slots.
        if (!ly.b.empty()) {
            std::vector<double> v(slots, 0.0);
            const Layout& o = outs[t];
            for (int n = 0; n < batch; n++)
                for (int c = 0; c < o.c; c++)
                    for (int y = 0; y < o.vh; y++)
                        for (int x = 0; x < o.vw; x++)
                            v[size_t(n) * block + o.slot(c, y, x)] = ly.b[c];
            Plaintext pb = cc->MakeCKKSPackedPlaintext(v, 1, acc->GetLevel(), nullptr, slots);
            acc = cc->EvalAdd(acc, pb);
        }
        ct = acc;
        live = outs[t];
    }
    double eval_s = secs(t_eval);

    // Decode every block, so a packing bug in the later ones cannot pass as a speedup.
    std::vector<std::vector<double>> all(static_cast<size_t>(batch));
    {
        Plaintext out;
        cc->Decrypt(keys.secretKey, ct, &out);
        out->SetLength(slots);
        auto v = out->GetRealPackedValue();
        for (int n = 0; n < batch; n++) {
            const size_t base = size_t(n) * block;
            all[n].assign(v.begin() + base, v.begin() + base + size_t(live.c));
        }
    }
    const std::vector<double>& logits = all[0];
    int argmax = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[argmax]) argmax = int(i);
    int blocks_agreeing = 0;
    for (int n = 0; n < batch; n++) {
        int am = 0;
        for (size_t i = 1; i < all[n].size(); i++)
            if (all[n][i] > all[n][am]) am = int(i);
        if (am == argmax) blocks_agreeing++;
    }
    int ref_argmax = 0;
    for (size_t i = 1; i < fx.reference.size(); i++)
        if (fx.reference[i] > fx.reference[ref_argmax]) ref_argmax = int(i);
    double margin = 0.0;
    {
        std::vector<double> sorted = fx.reference;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        if (sorted.size() >= 2) margin = sorted[0] - sorted[1];
    }
    double worst = 0.0;
    for (size_t i = 0; i < logits.size() && i < fx.reference.size(); i++)
        worst = std::max(worst, std::fabs(logits[i] - fx.reference[i]));

    std::cout << "evaluator linear_transform\n"
              << "ring_dim " << cc->GetRingDimension() << "\n"
              << "slots " << slots << "\n"
              << "block " << block << "\n"
              << "batch " << batch << "\n"
              << "blocks_agreeing " << blocks_agreeing << "/" << batch << "\n"
              << "security HEStd_128_classic\n"
              << "depth_needed " << need << "\n"
              << "depth_set " << depth << "\n"
              << "levels_usable " << usable << "\n"
              << "bootstraps " << n_boot << "\n"
              << "ct_mults " << n_mul << "\n"
              << "rotations " << n_rot << "\n"
              << "rotation_keys " << rots.size() << "\n"
              << "keygen_s " << keygen_s << "\n"
              << "eval_s " << eval_s << "\n"
              << "eval_s_per_inference " << (eval_s / double(batch)) << "\n"
              << "throughput_per_s " << (double(batch) / eval_s) << "\n"
              << "class " << argmax << "\n"
              << "reference_class " << ref_argmax << "\n"
              << "reference_margin " << margin << "\n"
              << "worst_logit_error " << worst << "\n"
              << "verdict "
              << (argmax == ref_argmax ? "agrees" : (worst >= margin ? "margin_below_error" : "wrong"))
              << "\n";
    std::cout << "diagonals_per_layer";
    for (double v : report_diags) std::cout << " " << size_t(v);
    std::cout << "\nlogits";
    for (double v : logits) std::cout << " " << v;
    std::cout << "\nreference";
    for (double v : fx.reference) std::cout << " " << v;
    std::cout << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ckks_cnn <fixture> [--bootstrap]\n";
        return 2;
    }
    Fixture fx = load(argv[1]);
    bool want_boot = false;
    // A cap on the computational level budget, which is what forces bootstrapping.
    //
    // A deployment does not get to widen its budget indefinitely: the ring dimension
    // must grow with the modulus chain to hold the security level, and each doubling
    // costs. Capping the budget and bootstrapping is the other side of that trade, and
    // it is the regime the comparison against secret sharing turns on, since secret
    // sharing has no budget to exhaust.
    int cap = 0;
    // How many independent inferences share one ciphertext.
    //
    // Latency and throughput are different measurements and an approximate homomorphic
    // scheme separates them further than secret sharing does. A ciphertext holds
    // thousands of slots and this network's layout occupies a few hundred, so further
    // images can ride in the unused slots at *no additional cost in operations at all*:
    // the same rotation shifts every image's map identically, and the weights are shared
    // across images, so a plaintext mask is simply the single-image pattern repeated per
    // block. Batching is therefore where the paradigm's parallelism actually lives, and
    // measuring only latency understates it.
    int batch = 1;
    // Use the linear-transform evaluator rather than the tap-by-tap one.
    bool use_lt = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--bootstrap") want_boot = true;
        else if (a.rfind("--levels=", 0) == 0) { cap = std::stoi(a.substr(9)); want_boot = true; }
        else if (a.rfind("--batch=", 0) == 0) batch = std::stoi(a.substr(8));
        else if (a == "--lt") use_lt = true;
    }

    for (const auto& l : fx.layers) {
        if (l.kind == "maxpool") {
            std::cout << "refused max pooling is a comparison, which CKKS does not "
                         "evaluate without a costly polynomial approximation; export "
                         "the average-pooling variant\n";
            return 1;
        }
    }

    if (use_lt) return main_lt(fx, batch, cap, want_boot);

    int need = depth_of(fx);
    std::vector<uint32_t> budget = {4, 4};

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(UNIFORM_TERNARY);
    // Never pinned, and never relaxed: the only published HEIR ML datapoint was fast
    // because it ran at `HEStd_NotSet`, and an insecure baseline is worse than none.
    p.SetSecurityLevel(HEStd_128_classic);
    p.SetScalingModSize(45);
    p.SetFirstModSize(50);
    p.SetScalingTechnique(FLEXIBLEAUTO);
    int usable = (cap > 0 && cap < need) ? cap : need;
    int depth = usable;
    if (want_boot) depth += FHECKKSRNS::GetBootstrapDepth(budget, UNIFORM_TERNARY);
    p.SetMultiplicativeDepth(depth);

    auto t_keys = Clock::now();
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    if (want_boot) { cc->Enable(ADVANCEDSHE); cc->Enable(FHE); }

    // Encode into the smallest power of two that covers the layout, not the whole
    // ring.
    //
    // The ring dimension is fixed by the security level and the modulus chain, but the
    // number of *slots* a ciphertext uses is ours to choose, and this network's widest
    // feature map spans under a thousand slots. Using the full ring would encode
    // sixty-five thousand slots to carry a few hundred live values, which costs
    // nothing in a levelled multiply — but bootstrapping's cost and its key material
    // both scale with the slot count, and at the full ring the key set alone exhausted
    // this machine's memory. Sparse encoding is what the low utilisation of a
    // channel-per-ciphertext packing buys back.
    size_t span_needed = size_t(fx.h) * fx.w;
    {
        int ch = fx.h, cw = fx.w, rs = fx.w, cs = 1;
        for (const auto& l : fx.layers) {
            if (l.kind == "conv") {
                ch = ch - l.k + 1; cw = cw - l.k + 1;
            } else if (l.kind == "avgpool") {
                ch /= l.stride; cw /= l.stride; rs *= l.stride; cs *= l.stride;
            }
            span_needed = std::max(span_needed, size_t(ch - 1) * rs + size_t(cw - 1) * cs + 1);
        }
        for (const auto& l : fx.layers)
            if (l.kind == "dense") span_needed = std::max(span_needed, size_t(l.out_dim));
    }
    // Each inference occupies `block` slots; the batch is laid out block by block.
    size_t block = 1;
    while (block < span_needed) block <<= 1;
    size_t slots = block * size_t(batch);
    {
        size_t cap_slots = cc->GetRingDimension() / 2;
        if (slots > cap_slots) {
            std::cerr << "ckks_cnn: batch " << batch << " needs " << slots
                      << " slots but the ring provides " << cap_slots
                      << "; the largest batch at this layout is " << (cap_slots / block)
                      << std::endl;
            return 2;
        }
        size_t p = 1;
        while (p < slots) p <<= 1;
        slots = p;
    }
    std::cerr << "ckks_cnn: ring " << cc->GetRingDimension() << ", depth " << depth
              << ", usable " << usable << ", slots " << slots << std::endl;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Rotations needed: every convolution tap offset, every pooling offset, and the
    // power-of-two shifts a rotate-and-sum reduction walks through.
    std::vector<int32_t> rots;
    {
        // Simulate the layout pass to learn exactly which offsets are rotated.
        int ch = fx.h, cw = fx.w, rs = fx.w, cs = 1, out_max = 0;
        bool packed_now = false;
        for (const auto& l : fx.layers) {
            if (l.kind == "conv") {
                for (int a = 0; a < l.k; a++)
                    for (int b = 0; b < l.k; b++) rots.push_back(a * rs + b * cs);
                ch = ch - l.k + 1; cw = cw - l.k + 1;
            } else if (l.kind == "avgpool") {
                for (int a = 0; a < l.stride; a++)
                    for (int b = 0; b < l.stride; b++) rots.push_back(a * rs + b * cs);
                ch /= l.stride; cw /= l.stride; rs *= l.stride; cs *= l.stride;
            } else if (l.kind == "dense") {
                // Rotate-and-sum walks powers of two up to the live span, and the
                // gather that follows shifts slot 0 up to each output position.
                if (packed_now) {
                    for (size_t t = 1; t < size_t(l.in_dim); t <<= 1) rots.push_back(int32_t(t));
                } else {
                    for (int t = 1; t < cw; t <<= 1) rots.push_back(int32_t(t * cs));
                    for (int t = 1; t < ch; t <<= 1) rots.push_back(int32_t(t * rs));
                }
                out_max = std::max(out_max, l.out_dim);
                packed_now = true;
            }
        }
        for (int t = 1; t < out_max; t <<= 1) rots.push_back(-t);
    }
    std::sort(rots.begin(), rots.end());
    rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
    rots.erase(std::remove(rots.begin(), rots.end(), 0), rots.end());
    std::cerr << "ckks_cnn: " << rots.size() << " rotation keys" << std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, rots);
    // Every rotation the evaluator performs must have had its key generated, and the
    // key list is produced by a *separate* simulation of the layout pass. If the two
    // ever diverge, OpenFHE raises an opaque automorphism error deep inside a
    // key switch; this turns that into a statement of which offset was missing.
    std::set<int32_t> have(rots.begin(), rots.end());
    int bootstraps = 0, mults = 0, rotations = 0;

    if (want_boot) {
        cc->EvalBootstrapSetup(budget, {0, 0}, slots);
        cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }
    double keygen_s = secs(t_keys);

    // ─── Evaluation ──────────────────────────────────────────────────────────
    //
    // One ciphertext per channel, with the feature map addressed by an explicit slot
    // stride rather than compacted after pooling. Element `(y, x)` of a map lives at
    // slot `y·rs + x·cs`, and pooling multiplies both strides by its own rather than
    // gathering the survivors into contiguous slots. That matters because a general
    // slot permutation costs a rotation per element, while a stride change costs
    // nothing at all: the following layer simply reads its taps at scaled offsets.
    //
    // A note that used to be wrong here, kept because the correction matters. It read
    // that realising a convolution as a Toeplitz matrix-vector product would be 2.7 M
    // multiply-accumulates against the convolution's 86 400. That is true only of a
    // *dense* matrix-vector product. Evaluating the same matrix by its generalized
    // diagonals costs one plaintext multiply per **nonzero diagonal**, and a convolution
    // under a layout that aligns its offsets has very few — which is exactly what the
    // `--lt` evaluator below does, and why Orion's operation counts are far below a
    // tap-by-tap convolution's.
    struct Spatial { int ch, cw, rs, cs; };

    // A value is either per-channel feature maps, or a single ciphertext holding `n`
    // scalars in slots `0..n`. Dense layers move between the two.
    std::vector<Ciphertext<DCRTPoly>> maps;
    Ciphertext<DCRTPoly> packed = nullptr;
    int packed_n = 0;
    Spatial sp{fx.h, fx.w, fx.w, 1};
    int nch = fx.c;

    auto rotate = [&](const Ciphertext<DCRTPoly>& ct, int32_t by) {
        // The centre tap of a convolution, and any other zero offset, is the
        // identity: no key, no rotation, no cost.
        if (by == 0) return ct;
        if (!have.count(by)) {
            std::cerr << "ckks_cnn: no rotation key for offset " << by
                      << " — the key list and the evaluation disagree\n";
            std::exit(3);
        }
        rotations++;
        return cc->EvalRotate(ct, by);
    };

    auto pt = [&](const std::vector<double>& v, size_t lvl) {
        return cc->MakeCKKSPackedPlaintext(v, 1, lvl, nullptr, slots);
    };

    // Build a plaintext by writing one inference's pattern and replicating it across
    // every batch block. The weights are shared between inferences, so the replication
    // is exact and the batch costs no extra operation — only the same plaintext,
    // encoded once, covering more slots.
    auto tiled = [&](const std::function<void(std::vector<double>&, size_t)>& fill, size_t lvl) {
        std::vector<double> v(slots, 0.0);
        for (int n = 0; n < batch; n++) fill(v, size_t(n) * block);
        return pt(v, lvl);
    };

    for (int c = 0; c < nch; c++) {
        std::vector<double> v(slots, 0.0);
        for (int n = 0; n < batch; n++) {
            size_t base = size_t(n) * block;
            for (int y = 0; y < sp.ch; y++)
                for (int x = 0; x < sp.cw; x++)
                    v[base + size_t(y) * sp.rs + size_t(x) * sp.cs] =
                        fx.input[(size_t(c) * sp.ch + y) * sp.cw + x];
        }
        Plaintext p0 = pt(v, 0);
        maps.push_back(cc->Encrypt(keys.publicKey, p0));
    }

    auto t_eval = Clock::now();

    // Place a bootstrap exactly when the remaining level budget would not cover the
    // next operation. This is the decision Orion automates over arbitrary graphs;
    // here the graph is a list, so it is computed directly, which is stated rather
    // than dressed up as the same contribution.
    auto refresh = [&](Ciphertext<DCRTPoly>& ct, uint32_t need_more) {
        if (!want_boot) return;
        // `usable` levels are available for computation; the rest of the chain is
        // reserved for the bootstrap itself.
        if (usable - int(ct->GetLevel()) <= int(need_more)) {
            ct = cc->EvalBootstrap(ct);
            bootstraps++;
        }
    };

    // Sum every slot of `ct` into slot 0. Slots outside the live range are already
    // zero, because the weight plaintext that produced `ct` is zero there.
    // Sum the live values of a feature map into slot 0, reducing along each axis at
    // that axis's own stride.
    //
    // A contiguous rotate-and-sum over a span of `n` slots costs `log2 n` rotations,
    // and after two poolings this network's maps are four by four on a row stride of
    // 112 — sixteen live values inside a span of 448, which a contiguous reduction
    // walks in nine rotations where four suffice. The saving is available only to a
    // reduction that knows the layout; a span does not.
    auto total_grid = [&](Ciphertext<DCRTPoly> ct, int rows, int cols, int rs, int cs) {
        for (int t = 1; t < cols; t <<= 1) ct = cc->EvalAdd(ct, rotate(ct, int32_t(t * cs)));
        for (int t = 1; t < rows; t <<= 1) ct = cc->EvalAdd(ct, rotate(ct, int32_t(t * rs)));
        return ct;
    };

    // The contiguous case, for a packed vector of scalars.
    auto total = [&](Ciphertext<DCRTPoly> ct, size_t span) {
        for (size_t s = 1; s < span; s <<= 1) {
            ct = cc->EvalAdd(ct, rotate(ct, int32_t(s)));
        }
        return ct;
    };

    // Collect `n` scalars-in-slot-0 into one ciphertext at slots `0..n`. Rotations
    // consume no levels, so this costs nothing in depth.
    //
    // Each value must move from slot 0 to slot `i`, and the direct way asks for a
    // rotation key per destination — 119 of them for this network's widest hidden
    // layer. A rotation key holds two polynomials per modulus limb, so at ring 2¹⁷ with
    // a bootstrapping chain each one is tens of megabytes and the set alone exhausted
    // this machine's memory before any ciphertext existed.
    //
    // Rotations compose additively, so shifting by `-2^j` for each set bit of `i`
    // reaches slot `i` using only the ⌈log₂ n⌉ power-of-two keys. That is more
    // rotations — the population count of `i` rather than one — and asymptotically
    // fewer keys, which is the right trade when the binding constraint is memory
    // rather than time.
    auto gather = [&](const std::vector<Ciphertext<DCRTPoly>>& v) {
        Ciphertext<DCRTPoly> acc = v[0];
        for (size_t i = 1; i < v.size(); i++) {
            Ciphertext<DCRTPoly> ct = v[i];
            for (size_t bit = 0; (size_t(1) << bit) <= i; bit++)
                if (i & (size_t(1) << bit)) ct = rotate(ct, -int32_t(size_t(1) << bit));
            acc = cc->EvalAdd(acc, ct);
        }
        return acc;
    };

    for (const auto& l : fx.layers) {
        if (l.kind == "conv") {
            int oh = sp.ch - l.k + 1, ow = sp.cw - l.k + 1;
            for (auto& ct : maps) refresh(ct, 1);
            // Rotate each input channel once per tap offset and reuse the result
            // across every output channel.
            //
            // The direct transcription of a convolution nests output channel over
            // input channel over tap and rotates in the innermost loop, so this
            // network's second convolution performs 16 x 6 x 25 = 2400 rotations
            // where only 6 x 25 = 150 distinct shifts exist. A rotation is a key
            // switch, the most expensive primitive in the scheme after bootstrapping,
            // and a tap offset does not depend on the output channel at all.
            std::vector<std::vector<Ciphertext<DCRTPoly>>> shifted(maps.size());
            for (size_t ci = 0; ci < maps.size(); ci++) {
                shifted[ci].reserve(size_t(l.k) * l.k);
                for (int a = 0; a < l.k; a++)
                    for (int b = 0; b < l.k; b++)
                        shifted[ci].push_back(rotate(maps[ci], a * sp.rs + b * sp.cs));
            }
            std::vector<Ciphertext<DCRTPoly>> next;
            for (int o = 0; o < l.out_c; o++) {
                Ciphertext<DCRTPoly> acc = nullptr;
                for (int i2 = 0; i2 < l.in_c; i2++) {
                    for (int t = 0; t < l.k * l.k; t++) {
                        double wgt = l.w[(size_t(o) * l.in_c + i2) * l.k * l.k + t];
                        if (wgt == 0.0) continue;
                        // A **scalar** multiply, not a masked plaintext one. The obvious
                        // implementation multiplies by a plaintext holding the weight
                        // inside the valid output window and zero outside, so that the
                        // positions a rotation drags in from beyond the border are
                        // cleared. That costs an encoding — a number-theoretic transform
                        // — per tap.
                        //
                        // The mask is unnecessary, because the garbage it clears is never
                        // read. Every consumer of a feature map addresses only live
                        // positions: a pooling window sums offsets that stay inside the
                        // valid region of the position it keeps, a following convolution's
                        // own output window restricts its taps to valid inputs, and a
                        // dense layer's weight plaintext is zero except at live positions.
                        // So the invalid slots may hold anything, and the activation may
                        // square it, without ever reaching a logit.
                        auto tp = cc->EvalMult(shifted[i2][t], wgt);
                        mults++;
                        acc = acc ? cc->EvalAdd(acc, tp) : tp;
                    }
                }
                Plaintext pb = tiled(
                    [&](std::vector<double>& v, size_t base) {
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                                v[base + size_t(y) * sp.rs + size_t(x) * sp.cs] = l.b[o];
                    },
                    acc->GetLevel());
                acc = cc->EvalAdd(acc, pb);
                next.push_back(acc);
            }
            maps = std::move(next);
            nch = l.out_c;
            sp.ch = oh;
            sp.cw = ow;
        } else if (l.kind == "act") {
            if (packed) {
                refresh(packed, 1);
                packed = cc->EvalMult(packed, packed);
                mults++;
            } else {
                for (auto& ct : maps) {
                    refresh(ct, 1);
                    ct = cc->EvalMult(ct, ct);
                    mults++;
                }
            }
        } else if (l.kind == "avgpool") {
            int s = l.stride;
            for (int c = 0; c < nch; c++) {
                refresh(maps[c], 1);
                Ciphertext<DCRTPoly> acc = maps[c];
                for (int a = 0; a < s; a++) {
                    for (int b = 0; b < s; b++) {
                        if (a == 0 && b == 0) continue;
                        acc = cc->EvalAdd(acc, rotate(maps[c], a * sp.rs + b * sp.cs));
                    }
                }
                // The window mean's divisor rides in the same plaintext multiply that
                // keeps the surviving positions — the scale-absorption the
                // secret-sharing arm gets by deferring the division into the next
                // rescale, here paid as part of a multiply it needed anyway.
                Plaintext pk = tiled(
                    [&](std::vector<double>& v, size_t base) {
                        for (int y = 0; y < sp.ch / s; y++)
                            for (int x = 0; x < sp.cw / s; x++)
                                v[base + size_t(y) * s * sp.rs + size_t(x) * s * sp.cs] =
                                    1.0 / double(s * s);
                    },
                    acc->GetLevel());
                maps[c] = cc->EvalMult(acc, pk);
                mults++;
            }
            // No compaction: widen the stride instead, which costs nothing.
            sp.ch /= s;
            sp.cw /= s;
            sp.rs *= s;
            sp.cs *= s;
        } else if (l.kind == "flatten") {
            // Nothing to do: the dense layer reads the strided maps directly.
        } else if (l.kind == "dense") {
            std::vector<Ciphertext<DCRTPoly>> outs;
            outs.reserve(l.out_dim);
            if (packed) {
                refresh(packed, 1);
                for (int o = 0; o < l.out_dim; o++) {
                    Plaintext pw = tiled(
                        [&](std::vector<double>& v, size_t base) {
                            for (int i = 0; i < l.in_dim; i++)
                                v[base + size_t(i)] = l.w[size_t(o) * l.in_dim + i];
                        },
                        packed->GetLevel());
                    auto acc = total(cc->EvalMult(packed, pw), size_t(l.in_dim));
                    mults++;
                    Plaintext pb = tiled(
                        [&](std::vector<double>& v, size_t base) { v[base] = l.b[o]; },
                        acc->GetLevel());
                    outs.push_back(cc->EvalAdd(acc, pb));
                }
            } else {
                int per = sp.ch * sp.cw;
                for (auto& ct : maps) refresh(ct, 1);
                for (int o = 0; o < l.out_dim; o++) {
                    Ciphertext<DCRTPoly> acc = nullptr;
                    for (int c = 0; c < nch; c++) {
                        Plaintext pw = tiled(
                            [&](std::vector<double>& v, size_t base) {
                                for (int y = 0; y < sp.ch; y++) {
                                    for (int x = 0; x < sp.cw; x++) {
                                        // The flattened index must match `Chw::idx` on
                                        // the other arm exactly, or the two arms
                                        // evaluate different networks.
                                        size_t fi = size_t(c) * per + size_t(y) * sp.cw + x;
                                        v[base + size_t(y) * sp.rs + size_t(x) * sp.cs] =
                                            l.w[size_t(o) * l.in_dim + fi];
                                    }
                                }
                            },
                            maps[c]->GetLevel());
                        auto tp = cc->EvalMult(maps[c], pw);
                        mults++;
                        acc = acc ? cc->EvalAdd(acc, tp) : tp;
                    }
                    acc = total_grid(acc, sp.ch, sp.cw, sp.rs, sp.cs);
                    Plaintext pb = tiled(
                        [&](std::vector<double>& v, size_t base) { v[base] = l.b[o]; },
                        acc->GetLevel());
                    outs.push_back(cc->EvalAdd(acc, pb));
                }
            }
            packed = gather(outs);
            packed_n = l.out_dim;
        }
    }
    double eval_s = secs(t_eval);

    // ─── Report ──────────────────────────────────────────────────────────────
    //
    // The homomorphic protocol returns the whole logit vector and the client takes
    // its maximum in the clear. That is not a detail: the secret-sharing arm computes
    // the argmax *inside* the protocol and reveals only the class, so it does
    // strictly more work and reveals strictly less. Both facts are recorded so that
    // neither arm is credited with the other's accounting.
    // Every inference in the batch is decoded and checked, not just the first: a bug
    // that corrupted the later blocks would otherwise be reported as a speedup.
    std::vector<std::vector<double>> all(static_cast<size_t>(batch));
    {
        Plaintext out;
        cc->Decrypt(keys.secretKey, packed, &out);
        out->SetLength(slots);
        auto v = out->GetRealPackedValue();
        for (int n = 0; n < batch; n++) {
            size_t base = size_t(n) * block;
            all[n].assign(v.begin() + base, v.begin() + base + size_t(packed_n));
        }
    }
    const std::vector<double>& logits = all[0];
    int argmax = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[argmax]) argmax = int(i);
    // Each block carries the same input in this fixture, so every block must decide the
    // same class; a disagreement between blocks is a packing bug, not an approximation.
    int blocks_agreeing = 0;
    for (int n = 0; n < batch; n++) {
        int am = 0;
        for (size_t i = 1; i < all[n].size(); i++)
            if (all[n][i] > all[n][am]) am = int(i);
        if (am == argmax) blocks_agreeing++;
    }
    int ref_argmax = 0;
    for (size_t i = 1; i < fx.reference.size(); i++)
        if (fx.reference[i] > fx.reference[ref_argmax]) ref_argmax = int(i);

    // When the two classes disagree, say *why*, because the two reasons call for
    // opposite responses.
    //
    // A disagreement means either that the evaluation is wrong, which invalidates the
    // measurement, or that the scheme's approximation error exceeded the margin
    // between the network's top two classes, which invalidates the *fixture* — an
    // untrained network can be very nearly undecided, and then no correct
    // implementation of anything would reliably agree. §VI.7 records the first version
    // of a depth sweep being written up with these two confused, so the report
    // separates them: the reference margin, and the largest per-logit deviation.
    double margin = 0.0;
    {
        std::vector<double> sorted = fx.reference;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        if (sorted.size() >= 2) margin = sorted[0] - sorted[1];
    }
    double worst = 0.0;
    for (size_t i = 0; i < logits.size() && i < fx.reference.size(); i++)
        worst = std::max(worst, std::fabs(logits[i] - fx.reference[i]));

    std::cout << "ring_dim " << cc->GetRingDimension() << "\n"
              << "slots " << slots << "\n"
              << "batch " << batch << "\n"
              << "block " << block << "\n"
              << "blocks_agreeing " << blocks_agreeing << "/" << batch << "\n"
              << "span_needed " << span_needed << "\n"
              << "security HEStd_128_classic\n"
              << "depth_needed " << need << "\n"
              << "depth_set " << depth << "\n"
              << "levels_usable " << usable << "\n"
              << "bootstraps " << bootstraps << "\n"
              << "ct_mults " << mults << "\n"
              << "rotations " << rotations << "\n"
              << "keygen_s " << keygen_s << "\n"
              << "eval_s " << eval_s << "\n"
              << "eval_s_per_inference " << (eval_s / double(batch)) << "\n"
              << "throughput_per_s " << (double(batch) / eval_s) << "\n"
              << "class " << argmax << "\n"
              << "reference_class " << ref_argmax << "\n"
              << "reference_margin " << margin << "\n"
              << "worst_logit_error " << worst << "\n"
              << "verdict "
              << (argmax == ref_argmax
                      ? "agrees"
                      : (worst >= margin ? "margin_below_error" : "wrong"))
              << "\n";
    std::cout << "logits";
    for (double v : logits) std::cout << " " << v;
    std::cout << "\nreference";
    for (double v : fx.reference) std::cout << " " << v;
    std::cout << std::endl;
    return 0;
}
