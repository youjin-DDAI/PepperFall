#!/bin/bash
# Download the 20 FallVision Raw Video rars and verify against Dataverse MD5.
# Skip files whose MD5 already matches. No resume (wget -c): repeated runs previously appended to and corrupted files.
# Usage:bash scripts/download_fallvision.sh [out_root=data/raw/FallVision] [jobs=4]
set -u
OUT=${1:-data/raw/FallVision}
JOBS=${2:-4}
B="https://dataverse.harvard.edu/api/access/datafile"

# id  relative_path  md5
MANIFEST="8138607 Fall/Bed/Raw_Video/f_raw_b_1.rar 04d7dbb909da1e95229d5cdf21ca8a09
8138612 Fall/Bed/Raw_Video/f_raw_b_2.rar 1f731d03a5064451488c82697417410d
8138614 Fall/Bed/Raw_Video/f_raw_b_3.rar 323d738656d33463789c2ab33936ab4f
8138617 Fall/Bed/Raw_Video/f_raw_b_4.rar 7aeb65842d8586e7eb78732d791cceff
8138622 Fall/Chair/Raw_Video/f_raw_c_1.rar cdda1329be88284d3bdcc43604416e10
8138623 Fall/Chair/Raw_Video/f_raw_c_2.rar 3ed3f7dc1917255b3989b2eb0d6807e7
8138624 Fall/Chair/Raw_Video/f_raw_c_3.rar f0903484f73495a865997f9bbe4078f2
8138935 Fall/Stand/Raw_Video/f_raw_s_1.rar 1e5ce3d4681d5a3f0e5882d40d187a85
8138937 Fall/Stand/Raw_Video/f_raw_s_2.rar 8b760af329aac621c71866feac0e507e
8138941 Fall/Stand/Raw_Video/f_raw_s_3.rar 5cde910609f902b76bb23267e4d4655f
8140287 No_Fall/Bed/Raw_Video/nf_raw_b_1.rar 0862d0927fefc660392f10eec644392e
8140288 No_Fall/Bed/Raw_Video/nf_raw_b_2.rar ae97fb5dbde920357c3dd6298ed4e698
8140291 No_Fall/Bed/Raw_Video/nf_raw_b_3.rar ddefd1a7e2d5d7461f511209f5c1e526
8140295 No_Fall/Bed/Raw_Video/nf_raw_b_4.rar 50e38e3dd6a06a7b72224d2a466aa4a1
8140307 No_Fall/Chair/Raw_Video/nf_raw_c_1.rar 5dd3bb8f6800799784d24f0a2e00fc80
8140309 No_Fall/Chair/Raw_Video/nf_raw_c_2.rar c62c66aecdb9cc547964b80660e330cd
8140310 No_Fall/Chair/Raw_Video/nf_raw_c_3.rar 5f9c1e03d0396c7da756b70a6346a061
8140315 No_Fall/Stand/Raw_Video/nf_raw_s_1.rar 7fd5d6181ab15d36be37ed5c0bfda1d7
8140316 No_Fall/Stand/Raw_Video/nf_raw_s_2.rar a0ff24af13f10174fea40f7bf1274e94
8140317 No_Fall/Stand/Raw_Video/nf_raw_s_3.rar f3f6f260fb936ab749ebe37f5cea55f9"

fetch() {
  id=$1; rel=${2//_/ }; md5=$3
  # underscore is a space placeholder; restore _ in the filename itself
  rel="$(dirname "$rel")/$(basename "$2")"
  dst="$OUT/$rel"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && [ "$(md5sum "$dst" | cut -d' ' -f1)" = "$md5" ]; then
    echo "OK(skip) $rel"; return 0
  fi
  tmp="$dst.part"
  rm -f "$tmp"
  for try in 1 2 3; do
    curl -sSL --fail -o "$tmp" "$B/$id" && \
    [ "$(md5sum "$tmp" | cut -d' ' -f1)" = "$md5" ] && { mv -f "$tmp" "$dst"; echo "OK $rel"; return 0; }
    echo "retry $try $rel"; rm -f "$tmp"
  done
  echo "FAIL $rel"; return 1
}
export -f fetch; export OUT B

echo "$MANIFEST" | xargs -P "$JOBS" -L 1 bash -c 'fetch "$@"' _
