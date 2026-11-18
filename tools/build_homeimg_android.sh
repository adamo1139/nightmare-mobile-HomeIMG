#!/usr/bin/env bash
# ANDROID_NDK_ROOT=/path/to/android-ndk-r29 tools/build_homeimg_android.sh
# Header sources: Eigen 3.4 and nlohmann/json 3.x, e.g. Debian's libeigen3-dev
# and nlohmann-json3-dev. Neither desktop BLAS nor libtorch is linked on Android.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
headers="$repo/build/homeimg-headers"
mkdir -p "$headers"
cp -R "${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}/Eigen" "$headers/"
cp -R "${JSON_INCLUDE_DIR:-/usr/include}/nlohmann" "$headers/"
ndk=${ANDROID_NDK_ROOT:?set ANDROID_NDK_ROOT to NDK r29}
grep -q 'Pkg.Revision = 29.0.14206865' "$ndk/source.properties"
cmake -S "$repo/backend/homeimg/android" -B "$repo/build/homeimg-arm64" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DHOMEIMG_HEADERS="$headers"
cmake --build "$repo/build/homeimg-arm64" --target homeimg_cpu -j2
