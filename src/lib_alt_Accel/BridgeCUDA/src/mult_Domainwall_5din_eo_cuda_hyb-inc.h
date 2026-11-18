/*!
      @file    mult_Domainwall_5din_eo_cuda_hyb-inc.h
      @brief   eo 5D hopping with double storage and float-float arithmetic.

  Hybrid ("selective") FP64 emulation of the 4D hopping term.  Spinor and
  gauge fields stay in their native double layout.  Each loaded double is
  split into a float pair (hi, lo) with integer bit operations only, the
  hopping is computed in float-float (dw_add / dw_mul from bridgeACC_AField.h,
  the same primitives as the all-float-float QDW path), and each result is
  recombined to double on store (one DADD per output real).

  The hopping body is the unchanged native include pair
  (mult_Domainwall_eo_openacc2_{xyz,t_dirac}-inc.h); only the scalar type of
  the local variables and the load wrapper differ from
  mult_domainwall_5din_eo_hopb_dirac_5D_kernel.

  Precision: hi carries 24 bits, lo the next 24 bits (rounded from 29), so a
  loaded double keeps about 48 bits.  Values with |x| below about 2^-74 lose
  lo (exponent range of float), and |x| < 2^-126 is flushed to zero;
  lattice fields are O(1).

  With real_t = float the split is (x, 0): float storage, float-float
  arithmetic.  That instantiation only exists so the host template compiles.
*/

#ifndef MULT_DOMAINWALL_5DIN_EO_CUDA_HYB_TYPES
#define MULT_DOMAINWALL_5DIN_EO_CUDA_HYB_TYPES

// float-float scalar used as the local type inside the hopping body.
struct hyb_ff {
  float hi, lo;
};

__device__ __forceinline__ hyb_ff hyb_make(float h, float l)
{
  hyb_ff r;
  r.hi = h;
  r.lo = l;
  return r;
}

__device__ __forceinline__ hyb_ff operator+(hyb_ff a, hyb_ff b)
{
  hyb_ff r;
  dw_add(a.hi, a.lo, b.hi, b.lo, r.hi, r.lo);
  return r;
}

__device__ __forceinline__ hyb_ff operator-(hyb_ff a)
{
  return hyb_make(-a.hi, -a.lo);
}

__device__ __forceinline__ hyb_ff operator-(hyb_ff a, hyb_ff b)
{
  hyb_ff r;
  dw_add(a.hi, a.lo, -b.hi, -b.lo, r.hi, r.lo);
  return r;
}

__device__ __forceinline__ hyb_ff operator*(hyb_ff a, hyb_ff b)
{
  hyb_ff r;
  dw_mul(a.hi, a.lo, b.hi, b.lo, r.hi, r.lo);
  return r;
}

// Scalars in the body are +-1 (boundary sign bc2) and 2.0 (t projection).
// Both are exact in float, so scaling each word is exact.
__device__ __forceinline__ hyb_ff operator*(int s, hyb_ff a)
{
  const float f = (float)s;
  return hyb_make(f * a.hi, f * a.lo);
}

__device__ __forceinline__ hyb_ff operator*(double s, hyb_ff a)
{
  const float f = (float)s;
  return hyb_make(f * a.hi, f * a.lo);
}

__device__ __forceinline__ hyb_ff& operator+=(hyb_ff& a, hyb_ff b)
{
  a = a + b;
  return a;
}

__device__ __forceinline__ hyb_ff& operator-=(hyb_ff& a, hyb_ff b)
{
  a = a - b;
  return a;
}

// Split a double into (hi, lo) floats without FP64 arithmetic.
// hi = x truncated to 24 significant bits (exact in float).
// lo = remaining 29 bits, rounded to 24 bits by the int->float conversion.
__device__ __forceinline__ hyb_ff hyb_split(double x)
{
  const long long b  = __double_as_longlong(x);
  const int       hw = (int)(b >> 32);
  const int       e  = (hw >> 20) & 0x7ff;             // biased exponent
  const int       ef = e - 1023 + 127;                 // float biased exponent
  if (ef <= 0) {                                       // zero and |x| < 2^-126: flush
    return hyb_make(0.0f, 0.0f);
  }
  const unsigned sign = (unsigned)hw & 0x80000000u;
  if (ef >= 255) {                                     // |x| >= 2^128 (or inf/nan): +-inf
    return hyb_make(__uint_as_float(sign | 0x7f800000u), 0.0f);
  }
  const unsigned mhi  = (unsigned)((b >> 29) & 0x7fffff);  // top 23 mantissa bits
  const float    hi   = __uint_as_float(sign | ((unsigned)ef << 23) | mhi);
  const int      mlo  = (int)(b & 0x1fffffff);              // low 29 mantissa bits
  // lo = mlo * 2^(e - 1075); build the float scale 2^(e-1075) from bits.
  const int es = e - 1075 + 127;
  float lo = 0.0f;
  if (es > 0 && es < 255) {
    lo = (float)mlo * __uint_as_float((unsigned)es << 23);
    if (sign) lo = -lo;
  }
  return hyb_make(hi, lo);
}

__device__ __forceinline__ hyb_ff hyb_split(float x)
{
  return hyb_make(x, 0.0f);
}

__device__ __forceinline__ void hyb_store(double& dst, hyb_ff a)
{
  dst = (double)a.hi + (double)a.lo;
}

__device__ __forceinline__ void hyb_store(float& dst, hyb_ff a)
{
  dst = a.hi + a.lo;
}

#endif  // MULT_DOMAINWALL_5DIN_EO_CUDA_HYB_TYPES

// Read-only view of a native array whose operator[] returns the split pair.
struct hyb_view_t {
  const real_t * __restrict__ p;
  __device__ __forceinline__ hyb_ff operator[](int i) const { return hyb_split(p[i]); }
};

//====================================================================
__global__
void mult_domainwall_5din_eo_hopb_hyb_dirac_5D_kernel(
  real_t * __restrict__ vp, const real_t * __restrict__ up, const real_t * __restrict__ wp,
  int Ns,
  int bc_x, int bc_y, int bc_z, int bc_t,
  int Nx, int Ny, int Nz, int Nt,
  int ieo, int jeo,
  int do_comm_x, int do_comm_y, int do_comm_z, int do_comm_t,
  int Nst_pad, int jgm5){

  const int Nxy      = Nx * Ny;
  const int Nst      = Nx * Ny * Nz * Nt;
  const int Nxyz     = Nx * Ny * Nz;
  const int ist      = blockIdx.x * blockDim.x + threadIdx.x;
  const int GridSize = blockDim.x * gridDim.x;

  hyb_view_t u_up; u_up.p = up;
  hyb_view_t u_dn; u_dn.p = up;

  for( int idx = ist; idx < Nst_pad * Ns; idx += GridSize){

    int idx2_wp = idx / NWP;
    int idx_in  = idx % NWP;
    int is      = idx2_wp % Ns;
    int idx_out = idx2_wp / Ns;
    int site    = idx_in + NWP*idx_out;

    int ix   = site % Nx;
    int iyzt = site / Nx;
    int ixy  = site % Nxy;
    int iy   = iyzt % Ny;
    int izt  = site / Nxy;
    int iz   = izt  % Nz;
    int it   = izt  / Nz;
    int ixyz = site % Nxyz;
    int keo  = (jeo + iy + iz + it) % 2;
    int idir;

    hyb_ff u_0, u_1, u_2, u_3, u_4, u_5;
    hyb_ff u_6, u_7, u_8, u_9, u10, u11;
    hyb_ff u12, u13, u14, u15, u16, u17;
    hyb_ff vt1_0, vt1_1, vt1_2, vt1_3, vt1_4, vt1_5;
    hyb_ff vt2_0, vt2_1, vt2_2, vt2_3, vt2_4, vt2_5;
    hyb_ff wt1r, wt1i, wt2r, wt2i;

    hyb_ff v2_01, v2_11, v2_21, v2_31, v2_41, v2_51;
    hyb_ff v2_02, v2_12, v2_22, v2_32, v2_42, v2_52;
    hyb_ff v2_03, v2_13, v2_23, v2_33, v2_43, v2_53;
    hyb_ff v2_04, v2_14, v2_24, v2_34, v2_44, v2_54;

    hyb_view_t v1; v1.p = wp;
    real_t * v2 = vp;

    #include "inc/mult_Domainwall_eo_openacc2_xyz-inc.h"
    #include "inc/mult_Domainwall_eo_openacc2_t_dirac-inc.h"

    hyb_store(v2[IDX2_SP_5D_R(0,0,is,Ns,site)], v2_01);
    hyb_store(v2[IDX2_SP_5D_I(0,0,is,Ns,site)], v2_11);
    hyb_store(v2[IDX2_SP_5D_R(1,0,is,Ns,site)], v2_21);
    hyb_store(v2[IDX2_SP_5D_I(1,0,is,Ns,site)], v2_31);
    hyb_store(v2[IDX2_SP_5D_R(2,0,is,Ns,site)], v2_41);
    hyb_store(v2[IDX2_SP_5D_I(2,0,is,Ns,site)], v2_51);

    hyb_store(v2[IDX2_SP_5D_R(0,1,is,Ns,site)], v2_02);
    hyb_store(v2[IDX2_SP_5D_I(0,1,is,Ns,site)], v2_12);
    hyb_store(v2[IDX2_SP_5D_R(1,1,is,Ns,site)], v2_22);
    hyb_store(v2[IDX2_SP_5D_I(1,1,is,Ns,site)], v2_32);
    hyb_store(v2[IDX2_SP_5D_R(2,1,is,Ns,site)], v2_42);
    hyb_store(v2[IDX2_SP_5D_I(2,1,is,Ns,site)], v2_52);

    hyb_store(v2[IDX2_SP_5D_R(0,2,is,Ns,site)], v2_03);
    hyb_store(v2[IDX2_SP_5D_I(0,2,is,Ns,site)], v2_13);
    hyb_store(v2[IDX2_SP_5D_R(1,2,is,Ns,site)], v2_23);
    hyb_store(v2[IDX2_SP_5D_I(1,2,is,Ns,site)], v2_33);
    hyb_store(v2[IDX2_SP_5D_R(2,2,is,Ns,site)], v2_43);
    hyb_store(v2[IDX2_SP_5D_I(2,2,is,Ns,site)], v2_53);

    hyb_store(v2[IDX2_SP_5D_R(0,3,is,Ns,site)], v2_04);
    hyb_store(v2[IDX2_SP_5D_I(0,3,is,Ns,site)], v2_14);
    hyb_store(v2[IDX2_SP_5D_R(1,3,is,Ns,site)], v2_24);
    hyb_store(v2[IDX2_SP_5D_I(1,3,is,Ns,site)], v2_34);
    hyb_store(v2[IDX2_SP_5D_R(2,3,is,Ns,site)], v2_44);
    hyb_store(v2[IDX2_SP_5D_I(2,3,is,Ns,site)], v2_54);
  }
}

void mult_domainwall_5din_eo_hopb_hyb_dirac_5d(
    real_t *vp, real_t *up, real_t *wp, int Ns, int *bc,
    int *Nsize, int *do_comm, int ieo, int jeo, int jgm5)
{
  int Nx = Nsize[0];
  int Ny = Nsize[1];
  int Nz = Nsize[2];
  int Nt = Nsize[3];
  int Nst = Nx * Ny * Nz * Nt;
  int Nst_pad = ceil_nwp(Nst);

  real_t* vp_dev = (real_t*)dev_ptr(vp);
  real_t* up_dev = (real_t*)dev_ptr(up);
  real_t* wp_dev = (real_t*)dev_ptr(wp);

  int blockSize = VECTOR_LENGTH;
  int gridSize  = (Nst_pad * Ns + blockSize - 1) / blockSize;

  mult_domainwall_5din_eo_hopb_hyb_dirac_5D_kernel<<<gridSize, blockSize>>>(
      vp_dev, up_dev, wp_dev, Ns,
      bc[0], bc[1], bc[2], bc[3],
      Nx, Ny, Nz, Nt,
      ieo, jeo,
      do_comm[0], do_comm[1], do_comm[2], do_comm[3],
      Nst_pad, jgm5);

  cudaDeviceSynchronize();
}
//============================================================END=====
