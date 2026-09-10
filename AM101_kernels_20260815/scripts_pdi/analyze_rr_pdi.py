import sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')

# PDI RR réel (184224) avec stride 0x15F0
rr = open(r'C:\Users\videl\AppData\Local\Temp\opencode\pdi_compare\LINEAGE_rr_qdump7.pdi', 'rb').read()

print(f'PDI RR: {len(rr)} B  hash={hashlib.sha256(rr).hexdigest()[:16]}')

STRIDE_RR = 0x15F0  # 5616
STRIDE_BASE = 0x15C0  # 5568

# Les blocs core RR (début de chaque bloc = motif 44 02 36 40)
PAT = bytes([0x44, 0x02, 0x36, 0x40])
blocks = []
start = 0
while True:
    i = rr.find(PAT, start)
    if i < 0:
        break
    blocks.append(i)
    start = i + 1
print(f'blocs core trouvés: {len(blocks)} à {[hex(b) for b in blocks[:9]]}')
if blocks:
    # vérifier le stride entre les 8 premiers
    strides = [blocks[i+1]-blocks[i] for i in range(min(7, len(blocks)-1))]
    print(f'strides: {[hex(s) for s in strides]}')

# Comparer les flags aux positions critiques du prologue
# Dans VALIDE (base): ins2 mot3 bit24, ins6 mot2 bit2, ins6 mot3 bit8
# Les offsets relatifs au début du bloc col0 (0x0C00):
#   +0x54 (0x0C54) = mot3 ins2, +0x90 = mot2 ins6, +0x94 = mot3 ins6
# Dans le PDI, col0 commence à 0x0C00. Donc offsets absolus:
#   base: 0x0C57/0x0C90/0x0C95 (dans VALIDE)
# Pour RR, les blocs commencent à 0x0C00 mais stride différent
# -> les mêmes offsets RELATIFS au bloc s'appliquent
blk0 = blocks[0]
print(f'\nbloc0 core RR @ 0x{blk0:X}')
rel = 0x0C00  # le bloc "col0" commence à 0x0C00 dans le PDI
# les 3 flags VALIDE sont à offset absolu 0x0C57, 0x0C90, 0x0C95
# = rel(0x0C00) + 0x57, +0x90, +0x95
for name, off in [('flag1(+0x57)', 0x57), ('flag2(+0x90)', 0x90), ('flag3(+0x95)', 0x95)]:
    abs_off = 0x0C00 + off
    print(f'{name}: @0x{abs_off:X} = {rr[abs_off]:02X}')

# Le prologue RR: comparer les 8 premières instructions VLIW (128-bit = 16 octets)
print('\n=== Prologue RR (16 premières instructions VLIW, col0) ===')
code_start = blk0
for i in range(10):
    off = code_start + i*16
    words = [rr[off+j*4:off+(j+1)*4].hex() for j in range(4)]
    print(f'ins{i}: ' + ' '.join(words))
