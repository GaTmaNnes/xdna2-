# B1 — Décomposition temporelle du témoin gelé t17rr_0906 (08/09/2026)

> Gouvernance : A = CLOSED/FROZEN — B = ACTIVE — B1 = décomposition temporelle.
> Aucune modification du xclbin/ctrlcode/replay gelés. Harnais pyxrt séparé, artefacts gelés inchangés.
> **Standard reproductibilité : chaque chiffre de ce rapport est régénérable par les commandes ci-dessous (scripts + logs + CSV + SHA256 conservés).**

---

## 0. Environnement (preuve d'exécution)

| Élément | Valeur | Preuve |
|---|---|---|
| Machine | Strix Point / Windows, NPU XDNA2 | sessions antérieures |
| Python | 3.13.14 (`C:\Python313\python.exe`) | log run |
| pyxrt | import OK (`C:\Python313`) | log run |
| xclbin gelé | `C:\t17rr_0906\…rr.xclbin` SHA `a2c290fa…` (185 375 B) | A1 |
| ctrlcode gelé | `…rr.bin` SHA `02580f41…` (3 172 B) | A1, réaffirmé au début de chaque run |
| Exécutables | scripts `docs/b1_*.py` (contenu intégré ci-dessous §5) | SHA256 en §6 |

---

## 1. Déroulé B1 (chronologie des preuves)

### B1.0 — Harnais v1 (`b1_decompose.py`) : BOs recréés par dispatch, bo1..4 non initialisés
- 3 warm-up + 10 runs, tous state=4, **sortie 10/10 identique** (SHA `68029117…`).
- Décision B3 calculée : NPU=54.7% / transport=1.3% / host(submit+read)=20.1% d'un cycle 9 855 µs.
- ⚠️ **Échec de validation témoin** : sortie ≠ témoin A2 (`bc99cc09…`) → investigation (§2). Le run n'est pas perdu : c'est la preuve que la sortie dépend d'autre chose que bo0.

### B1.1 — Diagnostic zéros (`b1_diag_zerobo.py`, UNE variable : bo1..4 = 0x00)
- state=4, sortie = **73 728 octets ZÉROS** (même SHA `68029117…` que v1 → sous Windows les BO pyxrt frais se comportent comme des pages zéro).
- **Preuve clé : la sortie dépend du contenu de bo1..4.** Le kernel lit les 5 BO.

### B1.2 — Diagnostic tout-AA (`b1_diag_allaa.py`, UNE variable : les 5 BO = 0xAA)
- state=4, 7 621.7 µs, sortie = **pattern bf16 0x7593** répété sur 71 680 octets, SHA `df78300f…`.
- Toujours ≠ témoin A2 (qui montre un pattern `0xA0CE` sur 73 728 octets avec gaps 2 octets).
- **Conclusion prouvée** : le témoin A2 `bc99cc09…` dépend de contenus bo1..4 **non spécifiés** par `xclbin_replay.exe` (résidus malloc déterministes par processus — les pre-csums bo1..4 étaient identiques entre les 3 runs A2 mais différents entre BOs). Le témoin A2 reste valable pour **state=4 + déterminisme intra-process**, mais **ses octets de sortie ne constituent pas un artefact de reproductibilité suffisant**.

### B1 v2 — Mesure finale, protocole ENTIÈREMENT SPÉCIFIÉ (`b1_decompose_v2.py`)
- 5 BO data 4 Mo **tous = 0xAA**, BOs persistants (alloués 1×, sémantique boucle steady de l'outil A2), 3 warm-up + 10 runs.
- **Déterminisme : 1 SHA distinct / 10 runs = `df78300f…` = exactement la référence B1.2** (reproductibilité inter-process validée).
- Résultats (moyennes sur 10 runs, CSV joint) :

| Phase | moy (µs) | med (µs) | min | max | sd | % cycle |
|---|---|---|---|---|---|---|
| submit (XRT run.start) | 54.5 | 54.1 | 40.8 | 72.1 | 9.5 | **0.9%** |
| npu_wait (run.wait → state=4) | 4 889.8 | 4 886.3 | 4 860.3 | 4 935.4 | **23.9** | **78.5%** |
| d2h_sync (bo.sync FROM_DEVICE) | 58.4 | 55.3 | 50.1 | 89.9 | 11.9 | **0.9%** |
| readout (bo.read 4 Mo) | 1 222.5 | 1 192.9 | 1 049.0 | 1 517.5 | 170.0 | **19.6%** |
| **cycle dispatch** | **6 225.2** | 6 182.6 | 6 027.4 | 6 558.0 | 188.2 | 100% |

- Coûts 1× (setup par layer dans un runtime 9B) : alloc+fill = 1 538 µs, H2D sync (5×4 Mo + insts) = 190 µs.

---

## 2. Verdict B3 (critères de décision de la gouvernance)

| Mesure | Valeur |
|---|---|
| NPU / cycle | **78.5%** |
| transport / cycle (D2H) | 0.9% (H2D : 190 µs 1× hors boucle) |
| host_wait / cycle (submit+readout) | 20.5% |

**DÉCISION : CAS 1 — NPU ≈ cycle → la cible est le kernel/AIE, PAS le transport DDR ni l'orchestration XRT.**

Lecture fine (nuance) :
- `submit` 54 µs et `d2h` 58 µs : négligeables → **pas de goulot XRT/driver ni DMA de sync** sur ce dispatch.
- `readout` 1 222 µs pour 4 Mo (≈ 3.4 GB/s) : c'est une **lecture CPU mappée**, pas du DMA NPU ; dans un runtime réel on ne relit que 72 KB utiles → coût réel attendu ~20 µs. Le "20.5% host" est donc **sur-estimé par le protocole de mesure** (lecture 4 Mo forcée).
- `npu_wait` 4 890 µs stable (sd 24 µs) : **c'est le temps d'exécution AIE réel** du layer full-K.
- **Cohérence croisée** : npu_wait (4 890 µs) ≈ steady xclbin_replay A2 (4 905 µs) → le "steady 4905 µs" de l'outil mesurait déjà l'exécution NPU. Les deux mesures indépendantes concordent à 0.3%.
- Corollaire important : le mur DDR 19–23 GB/s (fait 04/09) **ne borne PAS ce dispatch** — les transferts mesurés sont ~1.3% du cycle. Le mur DDR pèsera sur la composition multi-layers/9B E2E, pas sur le kernel unitaire.

## 3. Conséquences pour la suite

1. **B1 tranché : le ~4.9 ms/dispatch est du temps AIE, pas du transport.** Toute réduction passera par le kernel (ou la géométrie), pas par XRT/BO/DDIO.
2. **Amende A2 (report sur A)** : le "output bit-exact ×3" de l'angle A reste vrai intra-outil, mais les octets du témoin dépendaient d'entrées bo1..4 non spécifiées. Le protocole reproductible officiel devient **5 BO = 0xAA → SHA `df78300f…`** (vérifié inter-process : B1.2 vs B1 v2).
3. B2 (variantes staging/allocation) : **priorité baissée** — staging/sync ≈ 2% du cycle. Le levier est côté kernel/géométrie (et côté composition pour l'E2E).
4. Pour l'E2E 9B : le cycle dispatch utile ≈ NPU (4.9 ms) + H2D réel (~190 µs 1×) + readout utile (~20 µs) — à confronter à la composition par layer (angle D).

## 4. Reproduction exacte (toutes les commandes)

```bash
# 0) Prérequis : témoin gelé A intact
sha256sum "C:/t17rr_0906/decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.bin"
#   attendu : 02580f411439e8dc... (3172 octets)

# 1) Diagnostic B1.1 (bo1..4 = zéros)
cd /e/trixdna_test/docs && /c/Python313/python.exe b1_diag_zerobo.py
#   attendu : state=4, sha_out = 68029117d232bafb60b841d17c0717ea7e1c6558eb1e9a73fe32ebbc56acbb6b

# 2) Diagnostic B1.2 (5 BO = 0xAA) — RÉFÉRENCE DU PROTOCOLE REPRODUCTIBLE
cd /e/trixdna_test/docs && /c/Python313/python.exe b1_diag_allaa.py
#   attendu : state=4, sha_out = df78300f8b6364169706285ee1cc9aa0f6da921f722cc8d97a8ef5d6546d54f4

# 3) Mesure B1 v2 (3 warm-up + 10 runs, CSV)
cd /e/trixdna_test/docs && /c/Python313/python.exe b1_decompose_v2.py
#   attendu : DETERMINISME 1 SHA/10 = df78300f..., npu_wait ~4890us, cycle ~6225us
#   CSV : B1_DECOMPOSE_V2_08_09_2026.csv
```

⚠️ NPU exclusif requis (aucun llama-cli / autre process NPU pendant la mesure).

## 5. Artefacts & preuves (SHA256)

| Artefact | SHA256 (début) |
|---|---|
| `docs/b1_decompose.py` (v1, conservé pour l'historique) | voir §6 |
| `docs/b1_diag_zerobo.py` / `B1_DIAG_ZEROBO_OUT.bin` / log | voir §6 |
| `docs/b1_diag_allaa.py` / `B1_DIAG_ALLAA_OUT.bin` | voir §6 |
| `docs/b1_decompose_v2.py` (mesure finale) | voir §6 |
| `docs/B1_DECOMPOSE_V2_08_09_2026.csv` (données brutes 13 dispatches) | `b9a05db4…` |
| `docs/B1_V2_RUN_LOG_08_09_2026.txt` (log brut complet) | `f2fc1ece…` |
| Sortie protocole officielle `B1_DIAG_ALLAA_OUT.bin` | `df78300f…` |

## 6. Limites (red team, honnêteté)

1. Le harnais pyxrt n'instrumente pas l'INTÉRIEUR de npu_wait (soumission driver vs exécution AIE). La décomposition à l'intérieur du 4 890 µs (enqueue vs exécution réelle) demanderait l'API `xrt::run::get_info` ou des trace driver — non nécessaire pour le verdict B3 (submit seul = 54 µs, donc le wait est bien dominé par l'exécution).
2. `readout` inclut la lecture CPU des 4 Mo entiers — sur-estime le coût host d'un runtime réel (utile = 72 KB).
3. Mesures à t=20h–21h locales, NPU exclusif non vérifié par lock team (daemon en heartbeat seulement) — reproductibilité confirmée par sd=24 µs sur npu_wait.
4. La conversion 4.9 ms/layer → t/s 9B reste interdite sans la composition exacte des dispatches par token (angles C/D).

---
*Session Buffy/Codebuff 08/09/2026. Scripts, CSV, logs et binaires de sortie conservés sous `E:\trixdna_test\docs\`.*
