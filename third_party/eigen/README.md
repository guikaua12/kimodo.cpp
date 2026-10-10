# Eigen 3.4.0 (vendored headers)

Header-only linear algebra library used by the vendored NVIDIA MotionCorrection
post-processor (`third_party/motion_correction`) for its sparse least-squares
trajectory solve.

- Upstream: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
- SHA-256 of that archive: `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`
- Only the `Eigen/` header tree and the archive's license notices
  (`COPYING.*`) are copied here.
- License: MPL-2.0 (`COPYING.MPL2`). The build defines `EIGEN_MPL2_ONLY` so no
  LGPL-licensed Eigen code can be compiled in. As `COPYING.README` explains,
  some headers carry third-party code under other permissive licenses: the
  SparseLU solver is derived from SuperLU (notice in each file and in the
  repository `NOTICE`), `Half.h` and the MKL/BLAS/LAPACKE support headers are
  BSD (`COPYING.BSD`) and `BFloat16.h` is Apache-2.0 (`COPYING.APACHE`).
