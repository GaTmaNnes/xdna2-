#!/usr/bin/env python3
"""Decode t17rr/t17ag .bin TXN (grammaire validée 15/08: opcode low byte)."""
import struct, sys

# Grammar from GRAMMAIRE_CTRLCODES_ETAT_COMPLET_15_08.md
OP = {
    0x00: ("WRITE", 6),
    0x01: ("BLOCKWRITE", 12),
    0x03: ("MASKWRITE", 7),
    0x80: ("TCT", 4),
    0x81: ("DDR_PATCH", 12),
    0xFF: ("TERM", 1),
}

def decode(path):
    d = open(path, 'rb').read()
    w = struct.unpack('<%dI' % (len(d)//4), d)
    # Header 16B: [0]=0x06040100 [1]=0x108 [2]=NumOps [3]=TxnSize
    hdr = w[:4]
    numops, txnsz = hdr[2], hdr[3]
    print("== %s == %d B (%d words) header=%s NumOps=%d TxnSize=%d" %
          (path, len(d), len(w), [hex(x) for x in hdr], numops, txnsz))
    i = 4  # word index past header
    n = 0
    bd_writes = []      # BLOCKWRITE to 0x1D000+ (BD content)
    patches = []        # DDR_PATCH
    tct = []
    while i < len(w) and n < numops:
        opcode = w[i] & 0xFF
        if opcode not in OP:
            print("  ??? opcode=0x%02x at word %d (w=0x%08x) — stop" % (opcode, i, w[i]))
            break
        name, size = OP[opcode]
        rec = w[i:i+size]
        if len(rec) < size:
            print("  TRUNCATED %s at word %d (need %d have %d)" % (name, i, size, len(rec)))
            break
        i += size
        n += 1
        # summary per record type
        if opcode == 0x01:  # BLOCKWRITE: w[2]=BD base, w[3]=0x30 slot, w[4..]=BD words
            base = rec[2]
            nbd_words = size - 4
            bd = rec[4:4+nbd_words]
            bd_writes.append((base, bd))
        elif opcode == 0x81:  # DDR_PATCH
            patches.append(rec)
        elif opcode == 0x80:
            tct.append(rec)
    print("  walked %d/%d ops, end word=%d (total %d)" % (n, numops, i, len(w)))
    print("  BLOCKWRITE (BD) records: %d, DDR_PATCH: %d, TCT: %d" %
          (len(bd_writes), len(patches), len(tct)))
    # Summarize BD writes
    print("\n--- BD BLOCKWRITE records (base -> BD words) ---")
    for k, (base, bd) in enumerate(bd_writes):
        print("  BD#%02d base=0x%08x slot=0x%08x | %s" % (
            k, base, bd[0] if bd else 0,
            ' '.join('%08x' % x for x in bd)))
    print("\n--- DDR_PATCH records ---")
    for k, rec in enumerate(patches):
        print("  P#%02d | %s" % (k, ' '.join('%08x' % x for x in rec)))
    print("\n--- TCT records ---")
    for k, rec in enumerate(tct):
        print("  T#%02d | %s" % (k, ' '.join('%08x' % x for x in rec)))
    return bd_writes

if __name__ == '__main__':
    for p in sys.argv[1:]:
        decode(p)
