#!/usr/bin/env python3
"""Print t17 .bin ops in FILE ORDER with semantic interpretation."""
import struct, sys

OP = {0x00: ("WRITE", 6), 0x01: ("BLOCKWRITE", 12), 0x03: ("MASKWRITE", 7),
      0x80: ("TCT", 4), 0x81: ("DDR_PATCH", 12), 0xFF: ("TERM", 1)}

def col_of(reg):
    # col encoded in bits 24..27: 0x0001D000 -> col0, 0x0201D000 -> col2 ...
    c = (reg >> 24) & 0xFF
    return c

def decode(path):
    d = open(path, 'rb').read()
    w = struct.unpack('<%dI' % (len(d) // 4), d)
    i, n = 4, 0
    while i < len(w) and n < w[2]:
        opcode = w[i] & 0xFF
        name, size = OP[opcode]
        rec = w[i:i + size]
        i += size
        n += 1
        if opcode == 0x01:  # BLOCKWRITE 12 words: w[0]hdr w[1] w[2]=BD base w[3] w[4..11]=8 BD words
            base = rec[2]
            bd = rec[4:12]
            col = col_of(base)
            slot = base & 0x1FF
            print("#%02d BLOCKWRITE col=%d slot=0x%03x base=0x%08x" % (n, col, slot, base))
            print("      BD words: %s" % ' '.join('%08x' % x for x in bd))
            # interpret: w0? w1=addr(patched) w2 w3 w4 w5 w6 w7
            print("      [0]=0x%08x [1](addr)=0x%08x [4]=0x%08x [7]=0x%08x" %
                  (bd[0], bd[1], bd[4], bd[7]))
        elif opcode == 0x81:
            regaddr = rec[6]
            argidx = rec[8]
            argplus = rec[10]
            col = col_of(regaddr)
            print("#%02d DDR_PATCH col=%d reg=0x%08x argidx=%d argplus=0x%08x" %
                  (n, col, regaddr, argidx, argplus))
        elif opcode == 0x80:
            print("#%02d TCT         w=%s" % (n, ' '.join('%08x' % x for x in rec)))
        elif opcode == 0x00:
            print("#%02d WRITE       reg=0x%08x val=0x%08x" % (n, rec[2] | (rec[1] << 32), rec[3]))
        elif opcode == 0x03:
            print("#%02d MASKWRITE   reg=0x%08x val=0x%08x mask=0x%08x" % (n, rec[2] | (rec[1] << 32), rec[3], rec[4]))
        else:
            print("#%02d %s" % (n, name))

if __name__ == '__main__':
    decode(sys.argv[1])
