import sys, shutil, hashlib
sys.stdout.reconfigure(encoding='utf-8')

# PDI de reference VALIDE (0.80) pour les 24 octets
pdi_a = open(r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_A\d68e9a1a-0a93-4061-9b56-5b9c11363a1a.pdi', 'rb').read()
BASE_OFFSETS = [0x0C57, 0x0C90, 0x0C95]
STRIDE = 0x15C0

# xclbin B (72M) a patcher
xb = r'C:\Users\videl\AppData\Local\Temp\opencode\base_seed0_cache2\layer_f3best_build\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_qdump7.xclbin'
data = bytearray(open(xb, 'rb').read())

# Localiser le debut du PDI dans le xclbin : premier motif 44 02 36 40
pat = bytes([0x44, 0x02, 0x36, 0x40])
start = data.find(pat)
print(f'debut PDI (1er core) dans xclbin: 0x{start:X}')
# Le col0 du PDI commence a 0x0C00 (bloc core), les diffs sont a 0x0C57/0C90/0C95
# donc offset xclbin = start + (offset_pdi - 0x0C00)
pdi_base_in_xclbin = start - 0x0C00
print(f'base PDI dans xclbin: 0x{pdi_base_in_xclbin:X}')

patched = []
for c in range(8):
    for i, o in enumerate(BASE_OFFSETS):
        off_in_xclbin = pdi_base_in_xclbin + o + c * STRIDE
        val_a = pdi_a[o + c * STRIDE]
        val_b = data[off_in_xclbin]
        data[off_in_xclbin] = val_a
        patched.append((off_in_xclbin, val_b, val_a))
        print(f'  col{c} @0x{off_in_xclbin:X}: {val_b:02X} -> {val_a:02X}')

out = r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_B_patched.xclbin'
open(out, 'wb').write(bytes(data))
print(f'\npatche {len(patched)} octets -> {out}')
print(f'hash: {hashlib.sha256(bytes(data)).hexdigest()[:16]}')
