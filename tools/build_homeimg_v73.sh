#!/usr/bin/env bash
# Build the candidate ABI 111 baseline + matching v73 skel and HomeIMG accelerator.
# Input is the new output directory of tools/replay_homeimg_audit.py (85 patches).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
src=${1:?pass replayed local-dream source directory}
build=${2:-"$repo/build/homeimg-dit-abi111"}
ndk=${ANDROID_NDK_ROOT:?set Android NDK r29 root}
sdk=${HEXAGON_SDK_ROOT:?set Hexagon SDK 6.6.0.0 root}
cpp="$src/app/src/main/cpp"
grep -q 'DIT_ENGINE_ABI_VERSION 111' "$cpp/include/DitEngine.h"
grep -q 'Pkg.Revision = 29.0.14206865' "$ndk/source.properties"
cmake -S "$cpp/dit" -B "$build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release -DHEXAGON_SDK_ROOT="$sdk" \
  '-DCMAKE_C_FLAGS=-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE' \
  '-DCMAKE_CXX_FLAGS=-march=armv8.7a+fp16+dotprod+i8mm -D_GNU_SOURCE' \
  -DGGML_CCACHE=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF
cmake --build "$build" --target htp-v73 dit_engine -j"${BUILD_JOBS:-4}"
gg="$build/sdcpp/ggml/src"
stage="$repo/build/homeimg-v73"
mkdir -p "$stage/jniLibs/arm64-v8a" "$stage/assets/homeimg-htp"
bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
"$bin/aarch64-linux-android31-clang++" -std=c++17 -shared -fPIC -O2 -static-libstdc++ \
  -Wl,-z,max-page-size=16384 -Wl,--no-undefined \
  -I"$cpp/3rdparty/stable-diffusion.cpp/ggml/include" "$repo/backend/homeimg/android/v73.cpp" \
  -Wl,--start-group "$gg/ggml-hexagon/libggml-hexagon.a" "$gg/libggml-base.a" "$gg/libggml.a" "$gg/libggml-cpu.a" \
  -Wl,--end-group -llog -ldl -o "$stage/jniLibs/arm64-v8a/libhomeimg_v73.so"
"$bin/llvm-strip" --strip-unneeded "$stage/jniLibs/arm64-v8a/libhomeimg_v73.so"
cp "$gg/ggml-hexagon/libggml-htp-v73.so" "$stage/assets/homeimg-htp/"
sha256sum "$build/lib/arm64-v8a/libdit_engine.so" "$stage/assets/homeimg-htp/libggml-htp-v73.so" "$stage/jniLibs/arm64-v8a/libhomeimg_v73.so"
