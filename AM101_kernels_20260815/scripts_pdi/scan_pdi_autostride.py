import sys, hashlib, os, glob
sys.stdout.reconfigure(encoding='utf-8')
PAT = bytes([0x44, 0x02, 0x36, 0x40])

def analyze_pdi(b):
    """Détecte le stride et les flags pour un PDI quelconque."""
    blocks = []
    start = 0
    while True:
        i = b.find(PAT, start)
        if i < 0:
            break
        blocks.append(i)
        start = i + 1
    if len(blocks) < 2:
        return None
    stride = blocks[1] - blocks[0]
    # flags aux offsets 0x57/0x90/0x95 relatifs au bloc col0 (0x0C00)
    col0 = blocks[0]
    rel0 = col0 - 0x0C00  # souvent 0
    flags = [b[rel0 + 0x0C00 + 0x57], b[rel0 + 0x0C00 + 0x90], b[rel0 + 0x0C00 + 0x95]]
    # vérifier les 8 colonnes avec le stride détecté
    cols = []
    for c in range(8):
        off = rel0 + 0x0C00 + c*stride
        if off + 0x95 < len(b):
            cols.append([b[off+0x57], b[off+0x90], b[off+0x95]])
    ok = all(row == [0x02, 0x04, 0x02] for row in cols) if cols else False
    return {'stride': stride, 'flags': flags, 'cols': cols, 'func': ok,
            'nblocks': len(blocks), 'hash': hashlib.sha256(b).hexdigest()[:16]}

# scanner tous les PDI LINEAGE + références
files = {
    'LINEAGE_qdump7': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_qdump7.pdi',
    'LINEAGE_tb_qdump7': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_tb_qdump7.pdi',
    'LINEAGE_tb_rr': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_tb_rr.pdi',
    'LINEAGE_rr_qdump7 (VRAI RR)': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_rr_qdump7.pdi',
    'LINEAGE_cmlfix_qdump7': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_cmlfix_qdump7.pdi',
    'VALIDE': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\VALIDE.pdi',
    'B': r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\B.pdi',
}
print(f'{"label":24s} {"size":>7s} {"stride":>6s} {"col0":>10s} {"col7":>10s} {"FUNC":>4s}')
for label, path in files.items():
    if not os.path.exists(path):
        print(f'{label:24s} ABSENT')
        continue
    b = open(path, 'rb').read()
    r = analyze_pdi(b)
    if r is None:
        print(f'{label:24s} {len(b):7d}  ? pas de blocs')
        continue
    c0 = '/'.join(f'{v:02X}' for v in r['cols'][0])
    c7 = '/'.join(f'{v:02X}' for v in r['cols'][-1])
    print(f'{label:24s} {len(b):7d} 0x{r["stride"]:04X} {c0:10s} {c7:10s} {"YES" if r["func"] else "no "}')
