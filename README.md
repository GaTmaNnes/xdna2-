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
D’accord — si l’inversion 24/8 est déjà corrigée dans ton code local, le problème principal devient la validation numérique du kernel d’attention fusionné. Le README public affiche encore l’ancienne formulation, donc il faudra aussi pousser cette correction.
1. Corriger softmax × V
Pour les 8 couches full attention du Qwen3.5-9B, le contrat exact est :
- 16 têtes Q ;
- 4 têtes K/V ;
- dimension par tête : 256 ;
- groupement GQA : kv_head = q_head / 4 ;
- échelle : 1 / sqrt(256) = 1/16 ;
- RoPE appliqué uniquement aux 64 premières dimensions ;
- softmax effectué sur la dimension temporelle ;
- gate de sortie appliquée après l’attention et avant o_proj.
La configuration officielle confirme ces dimensions : Qwen3.5-9B config.json.
La référence CPU à reproduire exactement est :
for (int hq = 0; hq < 16; ++hq) {
    const int hkv = hq / 4;

    float max_score = -INFINITY;

    for (int t = 0; t < seq_len; ++t) {
        float dot = 0.0f;

        for (int d = 0; d < 256; ++d) {
            dot += float(q[hq][d]) * float(k[t][hkv][d]);
        }

        score[t] = dot * (1.0f / 16.0f);
        max_score = std::max(max_score, score[t]);
    }

    float denominator = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        probability[t] = expf(score[t] - max_score);
        denominator += probability[t];
    }

    const float inverse_sum = 1.0f / denominator;

    for (int d = 0; d < 256; ++d) {
        float accumulator = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            accumulator += probability[t]
                         * inverse_sum
                         * float(v[t][hkv][d]);
        }

        output[hq][d] = accumulator;
    }
}
Cause la plus probable
Les noms de kernels du dépôt sont déjà cohérents avec H16_KV4_d256. Le problème est donc probablement dans l’un de ces quatre endroits :
1. Mauvais mapping GQA
   Utiliser hkv = hq % 4 est faux. Il faut hkv = hq / 4 :
Q 0–3   → KV 0
Q 4–7   → KV 1
Q 8–11  → KV 2
Q 12–15 → KV 3
2. Mauvais ordre mémoire de V
   Le producteur peut écrire [token][kv_head][dim] tandis que le kernel lit [kv_head][token][dim]. Une telle erreur produit exactement le symptôme « valeurs raisonnables mais texte incohérent ».
3. Réduction softmax incorrecte
   Le maximum et la somme doivent porter sur tous les tokens valides d’une même tête Q — jamais sur les dimensions de tête, les 4 têtes KV ou uniquement un bloc local de 32 valeurs.
4. Conversion BF16 trop tôt
   Il faut conserver en float ou dans un accumulateur large :
   - le produit scalaire Q·K ;
   - le maximum ;
   - la somme des exponentielles ;
   - l’accumulation probability × V.
La conversion BF16 ne devrait intervenir qu’à la sortie du bloc. Les exemples MLIR-AIE officiels fournissent justement un softmax BF16 et des opérations vectorielles utilisables comme référence. Guide MLIR-AIE.
2. Faire un test qui localise la faute en une exécution
Il faut comparer CPU et NPU étape par étape, sur une seule couche full attention et un seul token de décodage.
Utiliser d’abord seq_len = 4, puis 16, 64 et 256. Pour chaque étape, exporter :
Q après norm/RoPE
K après norm/RoPE
V
scores QK avant scale
scores après scale
max(scores)
exp(scores - max)
somme des exp
probabilités normalisées
résultat softmax × V
résultat après gate
résultat après o_proj
Calculer pour chaque tenseur :
max_abs_error
mean_abs_error
cosine_similarity
nombre de NaN/Inf
indice de la première divergence
Critères initiaux raisonnables :
- Q/K/V convertis en BF16 : cosine > 0,999 ;
- scores : cosine > 0,995 ;
- probabilités : somme entre 0,995 et 1,005 ;
- sortie attention : cosine > 0,99 ;
- aucun NaN ou Inf.
Il ne faut pas tester uniquement le prompt « capitale de la France ». Un texte correct peut masquer des erreurs. Utilise des entrées synthétiques déterministes :
- Q et K nuls : softmax uniforme ;
- un score très dominant : sortie ≈ un seul vecteur V ;
- V identique pour tous les tokens : la sortie doit être exactement V ;
- une seule composante non nulle dans chaque V : révèle immédiatement une transposition ;
- valeurs Q/K négatives : détecte les erreurs de conversion BF16/signées.
Le test « V identique » est le plus rapide : si softmax×V != V, le problème est nécessairement dans la normalisation, les strides, le DMA ou l’accumulation.
3. Séparer le kernel fusionné avant de le réparer
Les nombreux kernels rawscore, smax, scopy, sdirect, vexp et fix montrent qu’une isolation a déjà été tentée. Il faut formaliser cela en quatre kernels de référence :
A. QK uniquement
B. QK + softmax
C. softmax × V avec probabilités fournies par le CPU
D. QK + softmax × V
Interprétation :
- A faux → Q/K, RoPE, scale, GQA ou stride ;
- A correct, B faux → réduction max/exp/somme ;
- B correct, C faux → layout de V ou accumulateur ;
- A/B/C corrects, D faux → synchronisation, réutilisation de buffer ou DMA ;
- D correct mais modèle faux → gate, o_proj, cache KV ou intégration GGML.
Il faut obtenir C correct avant de réactiver le kernel fusionné. C’est le chemin le plus court vers la cause réelle.
4. Vérifier le cache KV
Seules les 8 couches full attention utilisent un cache KV conventionnel. Pour chaque couche, vérifier :
K cache : [token][4][256]
V cache : [token][4][256]
Ou documenter précisément toute autre disposition.
Tests indispensables :
- après le token 0, relire K/V depuis le NPU et comparer au CPU ;
- après le token 1, vérifier que le token 0 n’a pas été écrasé ;
- vérifier l’offset en octets, pas seulement en éléments ;
- vérifier l’alignement requis par les DMA ;
- vérifier que seq_len signifie bien nombre de tokens valides et non capacité du buffer ;
- appliquer le masque causal avant le maximum du softmax.
Ne pas utiliser FlowKV pour les 24 couches Gated DeltaNet : elles nécessitent un état récurrent DeltaNet et un état de convolution causale, pas un cache K/V classique.
5. Réparer proprement Gated DeltaNet
Chaque couche linear-attention doit conserver au minimum :
- l’état de la convolution causale de taille 4 ;
- l’état récurrent DeltaNet par tête ;
- les paramètres de decay/gate ;
- l’ordre exact des opérations et normalisations.
La stratégie sûre est :
1. Garder Gated DeltaNet sur CPU.
2. Accélérer seulement ses projections INT4 sur NPU.
3. Comparer la sortie complète de chaque couche au CPU.
4. Porter ensuite la convolution causale.
5. Porter enfin la récurrence DeltaNet.
6. Ne fusionner qu’après validation token par token.
Il faut vérifier l’état après plusieurs tokens, pas seulement le premier : une mauvaise mise à jour récurrente peut être exacte au token 0 puis diverger progressivement.
6. Éviter les faux gains de batching
Le README indique que diviser le nombre d’appels par deux n’a pas réduit le temps. Cela signifie probablement que :
- les poids sont toujours relus ;
- les mêmes DMA sont exécutés ;
- le runtime attend chaque commande ;
- ou les kernels sont encore sérialisés.
Mesurer séparément :
préparation host
copie host → NPU
chargement/reconfiguration
exécution AIE
copie NPU → host
attente/synchronisation
Le correctif n’est pas simplement de regrouper plusieurs commandes. Il faut que la fusion supprime réellement :
- des transferts intermédiaires ;
- des relectures de poids ;
- des changements d’overlay ;
- des attentes host entre opérations.
Le meilleur objectif est un bloc résident :
RMSNorm
→ projections QKV
→ attention ou DeltaNet
→ projection de sortie
→ résiduel
→ RMSNorm
→ FFN/SwiGLU
→ résiduel
avec activations conservées sur le NPU entre les étapes.
7. Ne pas prendre 30 tokens/s comme objectif acquis
Un fichier de poids d’environ 5,3 Go lu une fois par token représente :
5,3 × 30 ≈ 159 Go/s
Et cela ne compte ni les activations, ni les caches, ni les relectures, ni les transferts. Sur certaines machines XDNA2, 30 tokens/s sera proche ou au-delà de la bande passante réellement exploitable par le NPU.
Avant d’annoncer ce chiffre, mesurer :
octets transférés par token
nombre de lectures de chaque matrice
bande passante NPU effective
temps de reconfiguration
temps total par couche
Un objectif initial plus solide serait :
- texte numériquement correct ;
- 100 % des couches validées ;
- moins de dispatchs ;
- amélioration reproductible par rapport aux 1,8 token/s ;
- consommation énergétique mesurée.
8. Ce qui manque pour appliquer réellement le correctif
Le dépôt public ne contient pas :
- la source de ggml-xdna.dll ;
- la source .s/C++ du kernel fusionné ;
- le code hôte qui prépare les buffers ;
- le générateur des .xclbin/.insts ;
- les traces CPU/NPU ;
- le commit source correspondant au binaire 18b583a.
Sans ces éléments, on peut déterminer la méthode de correction, mais pas modifier le bug réel. Les exécutables seuls ne permettent pas de savoir si la faute est dans le kernel, le DMA ou le backend GGML.
La priorité concrète est donc :
1. Publier ou récupérer la source exacte de ggml-xdna.dll et de f3best.
2. Ajouter le test isolé softmax×V avec V identiques.
3. Vérifier hkv = hq / 4.
4. Verrouiller le layout [token][kv_head][dim].
5. Accumuler softmax et P×V en FP32.
6. Tester C seul, puis réactiver la fusion.
7. Ajouter les traces par opération.
8. Optimiser les dispatchs seulement après égalité numérique.
Le correctif le plus probable est une combinaison « mapping GQA + stride/layout de V + réduction softmax par blocs ». C’est là que je concentrerais immédiatement l’investigation.

