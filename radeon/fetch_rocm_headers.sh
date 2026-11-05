#!/bin/bash
# hipCUB, rocPRIM and rocThrust are header-only. The pip-based ROCm in the
# rocm/pytorch 7.14 image ships none of them, so unpack the -dev packages
# from repo.radeon.com into one include directory.
#   radeon/fetch_rocm_headers.sh <dest>     -> <dest>/include
set -euo pipefail
DEST=$(realpath -m "$1")
BASE=${ROCM_APT:-https://repo.radeon.com/rocm/apt/7.2.4/pool/main}
VER=4.2.0.70204-93~24.04
mkdir -p "$DEST/debs" "$DEST/include"
for pkg in hipcub rocprim rocthrust; do
  deb="$DEST/debs/${pkg}-dev_${VER}_amd64.deb"
  [ -s "$deb" ] || curl -fL --retry 3 -o "$deb" "$BASE/${pkg:0:1}/${pkg}-dev/${pkg}-dev_${VER}_amd64.deb"
  rm -rf "$DEST/x"
  dpkg-deb -x "$deb" "$DEST/x"
  cp -a "$DEST"/x/opt/rocm-*/include/. "$DEST/include/"
done
rm -rf "$DEST/x"
ls "$DEST/include"
