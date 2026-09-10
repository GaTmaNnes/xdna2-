import sys, hashlib, os, glob
sys.stdout.reconfigure(encoding='utf-8')
BASE_OFFSETS = [0x0C57, 0x0C90, 0x0C95]
STRIDE = 0x15C0
PAT = bytes([0x44, 0x02, 0x36, 0x40])

# correspondance xclbin -> cache (pour la taille exacte du main.pdi)
mapping = [
    ('tb_qdump7', r'C:\Users\videl\AppData\Local\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_qdump7.xclbin', 'kernel_repro_cache'),
    ('tb_rr_ancien', r'C:\Users\videl\AppData\Local\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.xclbin', 'base_seed0_cache2'),
    ('tb_rr_qdump7 VRAI', r'C:\Users\videl\AppData\Local\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr_qdump7.xclbin', 'rr_repro_cache4'),
    ('tb_cmlfix_qdump7', r'C:\Users\videl\AppData\Local\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_cmlfix_qdump7.xclbin', 'cmlfix2_cache7'),
]

def pdi_size_from_cache(cache_dir):
    p = glob.glob(os.path.join(cache_dir, '**', 'main.pdi'), recursive=True)
    return os.path.getsize(p[0]) if p else None

def extract_pdi(xclbin_path, pdi_size):
    b = open(xclbin_path, 'rb').read()
    start = b.find(PAT)
    if start < 0:
        return None
    pdi_start = start - 0x0C00
    return b[pdi_start : pdi_start + pdi_size]

def pdi_sig(b):
    cols = []
    for c in range(8):
        cols.append([b[o + c*STRIDE] for o in BASE_OFFSETS])
    ok = all(row == [0x02, 0x04, 0x02] for row in cols)
    return cols, ok

print(f'{"label":20s} {"size":>6s} {"PDIhash":>16s} {"col0":>10s} {"col7":>10s} {"FUNC":>4s}')
for label, path, cache in mapping:
    if not os.path.exists(path):
        print(f'{label:20s} xclbin ABSENT')
        continue
    pdi_size = pdi_size_from_cache(rf'C:\Users\videl\AppData\Local\Temp\opencode\{cache}')
    if pdi_size is None:
        print(f'{label:20s} cache absent')
        continue
    pdi = extract_pdi(path, pdi_size)
    if pdi is None:
        print(f'{label:20s} PDI introuvable')
        continue
    cols, ok = pdi_sig(pdi)
    c0 = '/'.join(f'{v:02X}' for v in cols[0])
    c7 = '/'.join(f'{v:02X}' for v in cols[7])
    print(f'{label:20s} {len(pdi):6d} {hashlib.sha256(pdi).hexdigest()[:16]} {c0:10s} {c7:10s} {"YES" if ok else "no "}')
