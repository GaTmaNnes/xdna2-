ET NPU MINIMAL — 9B sur NPU sans FLM (état 04/09/2026)

Dossier **autonome et reproductible** : tout ce qu'il faut pour
rejouer les 2 seuls chemins NPU qui produisent du texte PROPRE sur
cette machine. Aucun chemin absolu requis — binaire + DLLs + kernels
+ modèles sont dans ce dossier (modèles en hardlink → ne pas les
supprimer ici si la source doit rester).

## Contenu

| Élément | Rôle |
|---|---|
| `bin/` | llama-cli.exe (repo_0808, commit 18b583a) + 6 DLLs (ggml-xdna.dll = le runtime NPU per-op) |
| `kernels/` | cache kernels complet (45 paires .xclbin+.insts + sous-dossiers flowkv) — GEMV INT4, SwiGLU INT4, FlowKV |
| `models/` | Llama-3.2-1B-Instruct-Q4_0.gguf (773 Mo) + Qwen3.5-9B-q40-lmhead-f16.gguf (5.3 Go) — hardlinks |
| `run_1B_npu_propre.sh` | ✅ 1B NPU per-op complet — texte propre ~5.2 t/s |
| `run_9B_npu_conservateur.sh` | ✅ 9B NPU conservateur (GEMV+SwiGLU INT4 NPU, attention CPU) — texte propre ~1.8 t/s |
| `INFOS_MANQUANTES.md` | **LA liste des infos précises qui manquent pour finir** le 9B sur NPU |

## Comment lancer (depuis ce dossier)

```bash
# 1B — chemin per-op complet sur NPU (attention incluse), texte propre
bash run_1B_npu_propre.sh

# 9B — GEMV + SwiGLU sur NPU, attention CPU (seul chemin 9B propre)
bash run_9B_npu_conservateur.sh
```

Résultats attendus :
- 1B : « The capital of France is Paris. » — Prompt ~150 t/s, Generation ~5.2 t/s
- 9B : « Thinking Process: ... Capital City: Paris ... » — Prompt 8.9 t/s, Generation 1.8 t/s

## Pourquoi ce dossier existe

Tout le reste (f3best fusionné 33.9 t/s mécaniques, OGA, FLM,
batching runlist multi-couche) est soit cassé numériquement, soit non
reproductible, soit invalidé par la mesure. Ces 2 scripts sont les
SEULS états qui génèrent du texte propre sur NPU, vérifiés depuis
l'archive le 04/09/2026.

## Ce qui manque pour aller plus loin (résumé)

Lire `INFOS_MANQUANTES.md` — en une phrase :
1. **Carte des 8 couches linear-attention** du 9B hybride (position +
   géométrie SSM) → débloque le dispatch des 24 couches full sur
   FlowKV (→ ~3-5 t/s attendus)
2. **Correction du softmax×V du kernel fusionné** (source .s + dump
   bf16 op-par-op) → débloque le chemin 33.9 t/s mécaniques (→ ~30 t/s)
3. Décomposition µs par op du graph_compute (outillage, pas recherche)

## Notes techniques

- Le binaire charge les kernels depuis `GGML_XDNA_CACHE_DIR` (ici
  `kernels/`). Un kernel absent → JIT compilation (lente, ~minutes)
  ou fallback CPU selon le cas.
- `GGML_XDNA_NUM_COLS=8` : les kernels 8col sont dans le cache.
- Le 9B exige `-c 512` et ~20 Go RAM libres (ctx défaut = ~30 Go).
- Modèles en HARDLINK vers leur source : `runtimes_permanents/...` et
  `E:\Qwen3.5-9B-q40-lmhead-f16.gguf`. Supprimer un hardlink ne
  supprime les données que si c'est le dernier.
- Binaire identique aux archives (md5 91f3ead2...) — c'est le même
  llama-cli.exe que `runtimes_permanents/perop_repo0808` et
  `/e/tmp/repo_0808/build/bin/Release`.

## Preuves / historique

- Session 08/08 : backend ggml-xdna validé, 42.8 t/s f3best = faux
  positif (min_prefix_match=1)
- Sessions 03-04/09 : 9B hybride identifié (GateDeltaNet 24+8),
  mode conservateur = texte propre 1.8 t/s ; matrice d'isolement
  kernel×host ; verdict #258 H2 (softmax×V) ; batching runlist
  invalidé par mesure (appels /2 → temps identique)
- Rapport détaillé : `E:\trixdna_test\docs\COMMENT_LES_AUTRES_DEPASSENT_EN_TOKEN_04_09.md`
