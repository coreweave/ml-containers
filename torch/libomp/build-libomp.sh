#!/usr/bin/env bash
set -euxo pipefail

WORK=/tmp/libomp
REF_PKG=libomp5
NEW_PKG=libomp5-cw
OUT=/libomp-replacement.deb
: "${LLVM_VERSION:?set LLVM_VERSION to the apt.llvm.org toolchain major version}"

deps() {
  local codename
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates lsb-release software-properties-common wget
  codename="$(lsb_release -cs)"
  wget -qO - 'https://apt.llvm.org/llvm-snapshot.gpg.key' \
    > /etc/apt/trusted.gpg.d/apt.llvm.org.asc
  apt-add-repository \
    "deb https://apt.llvm.org/$codename/ llvm-toolchain-$codename-$LLVM_VERSION main"
  apt-get install -y --no-install-recommends \
    binutils "clang-$LLVM_VERSION" cmake dpkg-dev git ninja-build patchelf python3
}

build() {
  local tag="${1-}"
  local deb version lib libdir soname provides conflicts replaces

  mkdir -p "$WORK/ref/ctrl" "$WORK/stage/DEBIAN" "$WORK/syms"

  cd "$WORK/ref"
  apt-get download "$REF_PKG"
  deb="$(ls "$WORK/ref/${REF_PKG}"_*.deb)"
  dpkg-deb -x "$deb" "$WORK/ref/root"
  dpkg-deb --ctrl-tarfile "$deb" | tar x -C "$WORK/ref/ctrl"

  version="$(dpkg-deb -f "$deb" Version)"
  lib="$(cd "$WORK/ref/root" && find . -type f -name 'libomp.so.*' | head -1)"
  lib="${lib#.}"
  libdir="$(dirname "$lib")"
  soname="$(objdump -p "$WORK/ref/root$lib" | awk '/SONAME/ {print $2}')"
  : "${tag:=llvmorg-$(printf '%s' "$version" \
      | sed -E 's@^[0-9]+:@@; s@^([0-9]+\.[0-9]+\.[0-9]+).*@\1@')}"

  git clone --depth 1 --filter=blob:none --no-checkout --branch "$tag" \
    https://github.com/llvm/llvm-project.git "$WORK/src"
  cd "$WORK/src"
  git sparse-checkout init --cone
  git sparse-checkout set openmp cmake
  git checkout
  git apply --verbose "$WORK/kmp-affinity-no-assert.patch"

  cmake -G Ninja -S "$WORK/src/openmp" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="clang-$LLVM_VERSION" \
    -DCMAKE_CXX_COMPILER="clang++-$LLVM_VERSION" \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DOPENMP_INSTALL_LIBDIR="${libdir#/usr/}" \
    -DLIBOMP_INSTALL_ALIASES=OFF
  ninja -C "$WORK/build"
  DESTDIR="$WORK/install" ninja -C "$WORK/build" install

  mkdir -p "$WORK/stage$libdir" "$WORK/stage/usr/share/doc/$NEW_PKG"
  cp "$WORK/install$libdir/libomp.so" "$WORK/stage$libdir/$soname"
  cp "$WORK/ref/root/usr/share/doc/$REF_PKG/copyright" \
     "$WORK/stage/usr/share/doc/$NEW_PKG/copyright"

  patchelf --set-soname "$soname" "$WORK/stage$libdir/$soname"
  strip --strip-unneeded "$WORK/stage$libdir/$soname"
  test "$(objdump -p "$WORK/stage$libdir/$soname" \
          | awk '/SONAME/ {print $2}')" = "$soname"

  nm -D --defined-only --with-symbol-versions "$WORK/ref/root$lib" \
    | awk '{print $NF}' | sort -u > "$WORK/syms/ref"
  nm -D --defined-only --with-symbol-versions "$WORK/stage$libdir/$soname" \
    | awk '{print $NF}' | sort -u > "$WORK/syms/new"
  comm -23 "$WORK/syms/ref" "$WORK/syms/new" | tee "$WORK/syms/missing"
  test ! -s "$WORK/syms/missing"

  printf 'reference %s bytes, built %s bytes\n' \
    "$(stat -c %s "$WORK/ref/root$lib")" \
    "$(stat -c %s "$WORK/stage$libdir/$soname")"

  provides="$(dpkg-deb -f "$deb" Provides)"
  conflicts="$(dpkg-deb -f "$deb" Conflicts)"
  replaces="$(dpkg-deb -f "$deb" Replaces)"
  printf '%s\n' \
    "Package: $NEW_PKG" \
    "Source: $(dpkg-deb -f "$deb" Source)" \
    "Version: $version+cw1" \
    "Architecture: $(dpkg-deb -f "$deb" Architecture)" \
    "Maintainer: $(dpkg-deb -f "$deb" Maintainer)" \
    "Installed-Size: $(du -ks "$WORK/stage/usr" | cut -f1)" \
    "Depends: $(dpkg-deb -f "$deb" Depends)" \
    "Provides: ${provides:+$provides, }$REF_PKG (= $version)" \
    "Conflicts: ${conflicts:+$conflicts, }$REF_PKG" \
    "Breaks: $(dpkg-deb -f "$deb" Breaks)" \
    "Replaces: ${replaces:+$replaces, }$REF_PKG" \
    "Section: $(dpkg-deb -f "$deb" Section)" \
    "Priority: $(dpkg-deb -f "$deb" Priority)" \
    "Multi-Arch: $(dpkg-deb -f "$deb" Multi-Arch)" \
    "Homepage: $(dpkg-deb -f "$deb" Homepage)" \
    "Description: LLVM OpenMP runtime for hosts with no sysfs CPU topology" \
    " Built from $tag. On aarch64 libomp reads the package id of each CPU" \
    " from /sys/devices/system/cpu/cpuN/topology/. Where those files are" \
    " unreadable, as in some containers, the stock runtime aborts the" \
    " process. This build falls back to a flat topology instead." \
    > "$WORK/stage/DEBIAN/control"
  cp "$WORK/ref/ctrl/shlibs" "$WORK/ref/ctrl/triggers" "$WORK/stage/DEBIAN/"
  cd "$WORK/stage" && find usr -type f | xargs md5sum > DEBIAN/md5sums

  dpkg-deb --build "$WORK/stage" "$OUT"
  cd /
  rm -rf "$WORK"
}

case "${1-}" in
  deps)  shift; deps ;;
  build) shift; build "${1-}" ;;
  *)     echo "usage: ${0##*/} deps | build [llvm-tag]" >&2; exit 2 ;;
esac
