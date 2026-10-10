# NVIDIA MotionCorrection (vendored)

The C++ core of NVIDIA Kimodo's post-processor (foot-skate cleanup and
kinematic-constraint enforcement), used by `src/postprocess.cpp`.

- Upstream: https://github.com/nv-tlabs/kimodo, directory
  `MotionCorrection/src/cpp`, commit `58e781898b3d7e328a676a75d3e338c45dce3ad9`.
- Layout: the upstream `src/cpp` tree is copied flat into this directory
  (`Compiler.h`, `Debug.h`, `Platform.h`, `Math/`, `AnimProcessing/`).
- Not vendored: `BindingsPython.cpp` (pybind11 bindings), the Python package,
  `setup.py`, `CMakeLists.txt` and the upstream tests. kimodo.cpp replaces the
  bindings with `src/motion_correction_glue.{hpp,cpp}`, a line-by-line C++17
  re-implementation of `BindingsPython.cpp` `correct_motion` that takes xyzw
  quaternions and calls `Animation::CorrectMotion` directly.
- License: Apache-2.0 (`LICENSE`, the upstream repository license). Every file
  keeps its SPDX header.
- Depends on the vendored Eigen 3.4.0 headers in `third_party/eigen` (MPL-2.0,
  built with `EIGEN_MPL2_ONLY`).

## Build

CMake builds these sources as the static library `kimodo_motion_correction`
(C++17, warnings off, Eigen as a system include) on x86 targets only, because
the math library uses SSE4.1 intrinsics unconditionally. No AVX flags are used
(upstream builds with `/arch:AVX` / `-mavx`; the only AVX intrinsic sits in a
code path the solver never calls). cl.exe accepts the SSE4.1 intrinsics
without any flag; clang-cl and GCC/Clang get `-msse4.1`. Elsewhere, or with
`-DKIMODO_ENABLE_POSTPROCESS=OFF`, post-processing reports that it is
unavailable (`kimodo::post_processing_available()` is false and requests with
post-processing fail before sampling).

## Local modifications

Each modified file carries a `Modified by kimodo.cpp` note below its SPDX
header (Apache-2.0 section 4(b)). No change affects the numerical results.

1. `Compiler.h`: `FORCE_INLINE` is also defined for Clang (including
   clang-cl); upstream only defined it for MSVC and GCC.
2. `Math/Vector.inl`: the runtime `Vector::Swizzle(uint32_t, ...)` falls back
   to a store-and-reload implementation when AVX is not enabled, because only
   MSVC accepts `_mm_permutevar_ps` without AVX code generation. MSVC (not
   clang-cl) and AVX builds keep the original intrinsic.
3. `AnimProcessing/Utility.cpp`: includes `<algorithm>` for `std::min`,
   `std::max` and `std::find_if` instead of relying on a transitive include.
