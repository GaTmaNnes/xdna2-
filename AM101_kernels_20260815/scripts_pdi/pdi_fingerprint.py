import sys, os, glob, json, hashlib

BASE_OFFSETS = [0x0C57, 0x0C90, 0x0C95]
STRIDE = 0x15C0
GOOD = [0x02, 0x04, 0x02]

def scan_pdi(pdi_path):
    b = open(pdi_path, 'rb').read()
    cols = []
    for c in range(8):
        row = []
        for o in BASE_OFFSETS:
            off = o + c * STRIDE
            row.append(b[off] if off < len(b) else -1)
        cols.append(row)
    ok = all(row == GOOD for row in cols)
    return cols, ok, hashlib.sha256(b).hexdigest()[:16]

def scan_cache_dir(cache_dir, name_filter=None):
    """Retrouve le main.pdi + xclbin d'un cache de compilation."""
    results = []
    for pdi in glob.glob(os.path.join(cache_dir, '**', 'main.pdi'), recursive=True):
        cols, ok, ph = scan_pdi(pdi)
        # xclbin sibling
        prj_dir = os.path.dirname(pdi)
        build_dir = os.path.dirname(prj_dir)
        xclbins = glob.glob(os.path.join(build_dir, '*.xclbin'))
        xhash = hashlib.sha256(open(xclbins[0], 'rb').read()).hexdigest()[:16] if xclbins else '?'
        results.append({
            'cache': os.path.basename(cache_dir),
            'pdi': pdi,
            'xclbin': xclbins[0] if xclbins else None,
            'xhash': xhash,
            'pdihash': ph,
            'cols': cols,
            'functional': ok,
            'size_pdi': len(open(pdi, 'rb').read()),
        })
    return results

if __name__ == '__main__':
    # Scan tous les caches de compilation connus
    roots = [
        'C:\\Users\\videl\\AppData\\Local\\Temp\\opencode',
    ]
    caches = ['kernel_repro_cache', 'cmlfix2_cache7', 'cmlfix3_cache8', 'rr_repro_cache9',
              'base_seed0_cache2', 'seed0_fixed_1', 'base_repro_cache1', 'cmlfix_cache5']
    allres = []
    for c in caches:
        d = os.path.join(roots[0], c)
        if os.path.isdir(d):
            allres.extend(scan_cache_dir(d))
    # Ajouter les 2 PDI de référence
    for path, label in [(r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_A\d68e9a1a-0a93-4061-9b56-5b9c11363a1a.pdi', 'VALIDE'),
                        (r'C:\Users\videl\AppData\Local\Temp\opencode\xclbin_B\d0897bcf-510b-4aaa-b7db-c22183e401d6.pdi', 'B(72M)')]:
        cols, ok, ph = scan_pdi(path)
        allres.append({'cache': label, 'pdi': path, 'xclbin': None, 'xhash': '?',
                       'pdihash': ph, 'cols': cols, 'functional': ok,
                       'size_pdi': len(open(path, 'rb').read())})

    print(f'{"label":22s} | {"xhash":8s} | {"col0":10s} | {"col7":10s} | FUNC?')
    for r in allres:
        c0 = '/'.join(f'{v:02X}' for v in r['cols'][0])
        c7 = '/'.join(f'{v:02X}' for v in r['cols'][7])
        print(f'{r["cache"]:22s} | {r["xhash"]:8s} | {c0:10s} | {c7:10s} | {"YES" if r["functional"] else "no "}')

    with open(os.path.join(roots[0], 'pdi_fingerprint_report.json'), 'w') as f:
        json.dump(allres, f, indent=2, default=str)
    print('\nreport: ' + os.path.join(roots[0], 'pdi_fingerprint_report.json'))
