import sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')

# xclbin RR réel (190687, rr4) - le kernel avec le bon seed accumulateur
rr_xclbin = r'C:\Users\videl\AppData\Local\Temp\opencode\rr_repro_cache4\layer_f3best_build\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr_qdump7.xclbin'
data = bytearray(open(rr_xclbin, 'rb').read())
print(f'xclbin RR: {len(data)} B')

# PDI de référence VALIDE (pour les valeurs 02/04/02)
pdi_a = open(r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_A\d68e9a1a-0a93-4061-9b56-5b9c11363a1a.pdi', 'rb').read()

PAT = bytes([0x44, 0x02, 0x36, 0x40])
start = data.find(PAT)
print(f'debut 1er core dans xclbin: 0x{start:X}')
pdi_base_in_xclbin = start - 0x0C00

# stride du RR = 0x15F0 (détecté)
STRIDE_RR = 0x15F0
BASE_OFFSETS = [0x0C57, 0x0C90, 0x0C95]

patched = 0
for c in range(8):
    for i, o in enumerate(BASE_OFFSETS):
        off_x = pdi_base_in_xclbin + o + c * STRIDE_RR
        if off_x >= len(data):
            print(f'  col{c} off 0x{off_x:X} DEPASSE')
            continue
        val_a = pdi_a[o + c * 0x15C0]  # la valeur VALIDE au même offset relatif
        val_b = data[off_x]
        data[off_x] = val_a
        patched += 1
        print(f'  col{c} @0x{off_x:X}: {val_b:02X} -> {val_a:02X}')

out = r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_RR_plus_fingerprint.xclbin'
open(out, 'wb').write(bytes(data))
print(f'\npatche {patched} octets -> {out}')
print(f'hash: {hashlib.sha256(bytes(data)).hexdigest()[:16]}')
