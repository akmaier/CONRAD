/*
 * GPU polychromatic detectors: energy-integrating (EID) and photon-counting
 * (PCD). Both consume a stack of per-material path-length images (one image per
 * material, mm) and the sampled spectrum, and integrate the Beer-Lambert law
 * over energy on the device. Poisson quantum noise is applied PER ENERGY on the
 * photon counts (physically correct: for an EID Var(sum E*n) = sum E^2*n, which
 * is not Poisson(sum E*n), so noise must be injected before energy weighting /
 * binning). RNG is Philox4x32-10 (Random123): stateless, reproducible via seed.
 */

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

static uint4 philox4x32_10(uint4 ctr, uint2 key)
{
    for (int i = 0; i < 10; i++) {
        uint hi0 = mul_hi(PHILOX_M0, ctr.x); uint lo0 = PHILOX_M0 * ctr.x;
        uint hi1 = mul_hi(PHILOX_M1, ctr.z); uint lo1 = PHILOX_M1 * ctr.z;
        ctr = (uint4)(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += PHILOX_W0; key.y += PHILOX_W1;
    }
    return ctr;
}

static float u01(uint x) { return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f); }

/* Poisson(lambda) with an independent stream per (pixel, energy). Knuth for
 * small lambda, Hoermann PTRS transformed rejection for large lambda. */
static int poisson_pe(float lambda, uint pixel, uint energy, uint seed)
{
    uint2 key = (uint2)(seed, 0x9E3779B9u);
    uint draw = 0u;
    if (lambda <= 0.0f) return 0;
    if (lambda < 30.0f) {
        float L = exp(-lambda), p = 1.0f;
        int k = 0;
        do {
            uint4 r = philox4x32_10((uint4)(pixel, draw++, energy, 0u), key);
            p *= u01(r.x); k++;
        } while (p > L);
        return k - 1;
    }
    float smu = sqrt(lambda);
    float b = 0.931f + 2.53f * smu;
    float a = -0.059f + 0.02483f * b;
    float inv_a = 1.1239f + 1.1328f / (b - 3.4f);
    float vr = 0.9277f - 3.6224f / (b - 2.0f);
    float loglam = log(lambda);
    for (int it = 0; it < 128; it++) {
        uint4 r = philox4x32_10((uint4)(pixel, draw++, energy, 0u), key);
        float U = u01(r.x) - 0.5f;
        float V = u01(r.y);
        float us = 0.5f - fabs(U);
        float kf = floor((2.0f * a / us + b) * U + lambda + 0.43f);
        if (us >= 0.07f && V <= vr) return (int) kf;
        if (kf < 0.0f) continue;
        if (us < 0.013f && V > us) continue;
        if (log(V * inv_a / (a / (us * us) + b)) <= kf * loglam - lambda - lgamma(kf + 1.0f))
            return (int) kf;
    }
    return (int) lambda;
}

/* Energy-integrating detector: out[p] = sum_e w_e * E_e, w_e = photon count
 * after attenuation (Poisson-sampled if noise != 0). mu is pre-scaled so that
 * tau = sum_m mu[m,e] * pathlen_mm[m] is dimensionless. */
kernel void energyIntegratingDetector(
        global const float* pathlen,   /* [nMat*npix], material-major        */
        global const float* mu,        /* [nMat*nE],   material-major, /10    */
        global const float* flux,      /* [nE] incident photons per energy    */
        global const float* energies,  /* [nE] keV                            */
        const int nMat, const int nE, const int npix,
        const int noise, const uint seed,
        global float* out)             /* [npix] integrated intensity         */
{
    int p = get_global_id(0);
    if (p >= npix) return;
    float acc = 0.0f;
    for (int e = 0; e < nE; e++) {
        float tau = 0.0f;
        for (int m = 0; m < nMat; m++)
            tau += mu[m * nE + e] * pathlen[m * npix + p];
        float mean = flux[e] * exp(-tau);
        float n = noise ? (float) poisson_pe(mean, (uint) p, (uint) e, seed) : mean;
        acc += n * energies[e];
    }
    out[p] = acc;
}

/* Photon-counting detector: out[b*npix + p] = photon counts in energy bin b
 * (Poisson-sampled per energy if noise != 0). binOfEnergy[e] = bin index or -1. */
kernel void photonCountingDetector(
        global const float* pathlen,       /* [nMat*npix] */
        global const float* mu,            /* [nMat*nE], /10 */
        global const float* flux,          /* [nE] */
        global const int*   binOfEnergy,   /* [nE] -> bin or -1 */
        const int nMat, const int nE, const int npix, const int nBins,
        const int noise, const uint seed,
        global float* out)                 /* [nBins*npix] */
{
    int p = get_global_id(0);
    if (p >= npix) return;
    float acc[16];
    for (int b = 0; b < nBins; b++) acc[b] = 0.0f;
    for (int e = 0; e < nE; e++) {
        int b = binOfEnergy[e];
        if (b < 0) continue;
        float tau = 0.0f;
        for (int m = 0; m < nMat; m++)
            tau += mu[m * nE + e] * pathlen[m * npix + p];
        float mean = flux[e] * exp(-tau);
        acc[b] += noise ? (float) poisson_pe(mean, (uint) p, (uint) e, seed) : mean;
    }
    for (int b = 0; b < nBins; b++) out[b * npix + p] = acc[b];
}
