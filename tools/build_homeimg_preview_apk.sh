#!/usr/bin/env bash
# Build a separately installable device-test APK with the exact Activity, package
# validator and native library used by the fork. No Nightmare signing key is used.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
: "${ANDROID_JAR:?set ANDROID_JAR to platform 35 android.jar}"
: "${ANDROID_BUILD_TOOLS:?set ANDROID_BUILD_TOOLS to the build-tools 35.0.0 directory}"
: "${JAVA_HOME:?set JAVA_HOME to a JDK (17 or newer)}"
export PATH="$JAVA_HOME/bin:$PATH"
dst="$repo/build/homeimg-preview"
mkdir -p "$dst"
tmp=$(mktemp -d "$dst/work.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
pkg=com.abrah.nightmare.homeimg.preview
mkdir -p "$tmp/assets" "$tmp/lib/arm64-v8a" "$tmp/classes" "$tmp/dex"
ndkbin="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
"$ndkbin/llvm-strip" -o "$tmp/lib/arm64-v8a/libhomeimg_cpu.so" "$repo/build/homeimg-arm64/libhomeimg_cpu.so"
optional="$repo/build/homeimg-v73"
if [ -d "$optional" ]; then
    cp "$optional/jniLibs/arm64-v8a/libhomeimg_v73.so" "$tmp/lib/arm64-v8a/"
    cp -R "$optional/assets/homeimg-htp" "$tmp/assets/"
fi
cat > "$tmp/AndroidManifest.xml" <<MANIFEST
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="$pkg" android:versionCode="1" android:versionName="1">
  <uses-sdk android:minSdkVersion="31" android:targetSdkVersion="35"/>
  <application android:label="HomeIMG preview" android:allowBackup="false" android:theme="@android:style/Theme.Material.Light">
    <uses-native-library android:name="libcdsprpc.so" android:required="false"/>
    <activity android:name="com.abrah.nightmare.homeimg.HomeImgActivity" android:configChanges="orientation|screenSize|keyboardHidden" android:exported="true">
      <intent-filter><action android:name="android.intent.action.MAIN"/><category android:name="android.intent.category.LAUNCHER"/></intent-filter>
    </activity>
  </application>
</manifest>
MANIFEST
javac -source 11 -target 11 -cp "$ANDROID_JAR" -d "$tmp/classes" "$repo"/app/src/main/java/com/abrah/nightmare/homeimg/*.java
"$ANDROID_BUILD_TOOLS/d8" --min-api 31 --lib "$ANDROID_JAR" --output "$tmp/dex" "$tmp"/classes/com/abrah/nightmare/homeimg/*.class
"$ANDROID_BUILD_TOOLS/aapt2" link -I "$ANDROID_JAR" -A "$tmp/assets" --manifest "$tmp/AndroidManifest.xml" -o "$tmp/base.apk"
cp "$tmp/dex/classes.dex" "$tmp/"
(cd "$tmp" && zip -q -0 base.apk classes.dex lib/arm64-v8a/*.so)
"$ANDROID_BUILD_TOOLS/zipalign" -P 16 -f 4 "$tmp/base.apk" "$tmp/aligned.apk"
key="$dst/debug.p12"
if [ ! -f "$key" ]; then
    keytool -genkeypair -keystore "$key" -storepass android -keypass android -alias homeimg-debug -keyalg RSA -keysize 2048 -validity 3650 -dname 'CN=Local HomeIMG Test'
fi
apk="$tmp/homeimg-preview.apk"
"$ANDROID_BUILD_TOOLS/apksigner" sign --ks "$key" --ks-pass pass:android --out "$apk" "$tmp/aligned.apk"
"$ANDROID_BUILD_TOOLS/apksigner" verify "$apk"
"$ANDROID_BUILD_TOOLS/zipalign" -c -P 16 4 "$apk"
mv "$apk" "$dst/homeimg-preview.apk"
sha256sum "$dst/homeimg-preview.apk"
