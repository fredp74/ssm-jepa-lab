\# HANDOFF : prototype SSM + JEPA + replay (continuation)



\## 0. Consignes pour le nouveau Claude



\- Répondre en français. Utilisateur : développeur, Python (data science, ML/RL), Rust, R, C++, plus web (HTML5, PHP, jQuery, MariaDB sur CyberPanel).

\- Méthode de travail : scripts que l'utilisateur colle dans son terminal. Ici : Windows 11, PowerShell (pas de bash, pas de `cat <<EOF`, pas de `python3`).

\- Écrire les fichiers avec des here-strings : `@'` ... `'@ | Set-Content -Encoding utf8 "$HOME\\ssm\_jepa\\x.py"`. Le `'@` de fin doit être en début de ligne.

\- Rester honnête et prudent : ne pas survendre. À chaque test, penser aux contrôles (encodeur non entraîné, baseline GRU, plusieurs graines, bruit de la sonde).

\- Machine : GMKtec M7, Ryzen 7 6850H, 32 Go RAM, CPU seulement. Python 3.13 (Microsoft Store), torch 2.14.1 installé. Le warning NumPy est inoffensif (`pip install numpy` le supprime).



\## 1. Point de départ



Question initiale : peut-on concevoir une architecture alternative (SSM/Mamba + modèle du monde + JEPA) "bien plus performante" que les Transformers actuels, notamment pour l'apprentissage continu ? Architecture proposée (schéma fait dans la conversation) :



1\. Encodeur SSM hiérarchique (patches, multi-échelle)

2\. Modèle du monde latent (SSM) prédisant `z(t+k)`, perte JEPA contre une cible EMA

3\. Poids rapides (test-time training) pour l'adaptation en ligne

4\. Mémoire épisodique sparse (rappel)

5\. Planificateur latent (RL en imagination)



Seules les briques 1 et 2, plus un replay (proxy de 4), ont été testées. Les briques 3 et 5 ne l'ont pas été.



\## 2. Code de base (réutilisé par tous les scripts)



```python

import math, copy, statistics as st, torch, torch.nn as nn, torch.nn.functional as F



class SelectiveSSM(nn.Module):

&#x20;   def \_\_init\_\_(s, d, n=16, slow=False):

&#x20;       super().\_\_init\_\_()

&#x20;       s.d, s.n = d, n

&#x20;       s.norm = nn.LayerNorm(d)

&#x20;       s.inp = nn.Linear(d, 2 \* d)

&#x20;       s.dt = nn.Linear(d, d)

&#x20;       s.B = nn.Linear(d, n)

&#x20;       s.C = nn.Linear(d, n)

&#x20;       s.A\_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))

&#x20;       s.D = nn.Parameter(torch.ones(d))

&#x20;       s.out = nn.Linear(d, d)

&#x20;       if slow:  # init de dt à la Mamba (0.001 à 0.1)

&#x20;           with torch.no\_grad():

&#x20;               dtv = torch.exp(torch.rand(d) \* (math.log(0.1) - math.log(0.001)) + math.log(0.001))

&#x20;               s.dt.bias.copy\_(torch.log(torch.expm1(dtv)))

&#x20;               s.dt.weight.mul\_(0.1)



&#x20;   def forward(s, x):

&#x20;       r = x

&#x20;       u, g = s.inp(s.norm(x)).chunk(2, -1)

&#x20;       u = F.silu(u)

&#x20;       dt = F.softplus(s.dt(u))

&#x20;       Bm, Cm = s.B(u), s.C(u)

&#x20;       A = -torch.exp(s.A\_log)

&#x20;       h = torch.zeros(x.size(0), s.d, s.n, device=x.device)

&#x20;       ys = \[]

&#x20;       for t in range(x.size(1)):  # scan séquentiel Python : LENT

&#x20;           dA = torch.exp(dt\[:, t].unsqueeze(-1) \* A)

&#x20;           dBu = dt\[:, t].unsqueeze(-1) \* Bm\[:, t].unsqueeze(1) \* u\[:, t].unsqueeze(-1)

&#x20;           h = dA \* h + dBu

&#x20;           ys.append((h \* Cm\[:, t].unsqueeze(1)).sum(-1))

&#x20;       y = torch.stack(ys, 1) + u \* s.D

&#x20;       return r + s.out(y \* F.silu(g))



class SSMEnc(nn.Module):  # Linear(din,d) -> 2 blocs SSM -> LayerNorm

&#x20;   def \_\_init\_\_(s, din, d, depth=2, slow=False):

&#x20;       super().\_\_init\_\_()

&#x20;       s.p = nn.Linear(din, d)

&#x20;       s.blocks = nn.ModuleList(\[SelectiveSSM(d, slow=slow) for \_ in range(depth)])

&#x20;       s.norm = nn.LayerNorm(d)

&#x20;   def forward(s, x):

&#x20;       x = s.p(x)

&#x20;       for b in s.blocks: x = b(x)

&#x20;       return s.norm(x)



class GRUEnc(nn.Module):  # baseline : Linear -> GRU 2 couches -> LayerNorm

&#x20;   def \_\_init\_\_(s, din, d):

&#x20;       super().\_\_init\_\_()

&#x20;       s.p = nn.Linear(din, d)

&#x20;       s.g = nn.GRU(d, d, num\_layers=2, batch\_first=True)

&#x20;       s.norm = nn.LayerNorm(d)

&#x20;   def forward(s, x):

&#x20;       return s.norm(s.g(s.p(x))\[0])



\# JEPA : cible EMA (m=0.99) + prédicteur MLP + terme de variance

\# C=4 canaux, D=64, K=4 (horizon), W=8 (warm-up), AdamW lr=1e-3, clip 1.0

def train\_step(m, x):

&#x20;   enc, tgt, pred, opt = m

&#x20;   z = enc(x)

&#x20;   with torch.no\_grad(): zt = tgt(x)

&#x20;   jepa = F.mse\_loss(pred(z\[:, W:-K]), zt\[:, W + K:])

&#x20;   var = F.relu(1.0 - z\[:, W:].reshape(-1, D).std(0)).mean()

&#x20;   opt.zero\_grad(); (jepa + var).backward()

&#x20;   torch.nn.utils.clip\_grad\_norm\_(enc.parameters(), 1.0); opt.step()

&#x20;   with torch.no\_grad():

&#x20;       for a, b in zip(tgt.parameters(), enc.parameters()):

&#x20;           a.mul\_(0.99).add\_(b.detach(), alpha=0.01)

```



Données synthétiques : somme de 3 sinusoïdes (fréquences aléatoires dans une plage, phases aléatoires, déphasage par canal `c\*0.7`) plus bruit gaussien 0.1. Tâche A : fréquences dans (0.05, 0.20). Tâche B : (0.30, 0.45). Paramètres : L=32, batch 32, 300 pas par phase.



\## 3. Fichiers dans `C:\\Users\\m7\\ssm\_jepa\\`



| Fichier | Rôle |

|---|---|

| `proto.py` | Premier prototype JEPA (800 pas) |

| `continual.py` | Oubli et replay, 1 graine |

| `seeds.py` | Idem, 5 graines, 3 conditions (sans replay, replay 16, replay 2) |

| `patch\_gru.py` puis `seeds\_gru.py` | Dérivé de `seeds.py` avec encodeur GRU |

| `probe.py` | Sonde linéaire gelée (ridge), SSM et GRU |

| `patch\_ctrl.py` puis `probe\_ctrl.py` | Contrôle encodeur non entraîné (dérivé de `probe.py`) |

| `longctx.py` | JEPA sur séquences de 128 pas (cue 16, trou 96, répétition 16) |

| `patch\_cap.py` puis `capacity.py` | Capacité supervisée (dérivé de `longctx.py`) |

| `patch\_curr.py` puis `curriculum.py` | Marqueur plus curriculum de trou (dérivé de `longctx.py`) |



Attention : les scripts `patch\_\*` génèrent les suivants en lisant les précédents. Si un fichier manque, il faut le reconstituer.



Bug connu dans `curriculum.py` (affichage final seulement, les mesures par graine sont correctes). Correctif :



```powershell

(Get-Content "$HOME\\ssm\_jepa\\curriculum.py") -replace 'format\\("trou", 10\\)','"trou".ljust(10)' | Set-Content -Encoding utf8 "$HOME\\ssm\_jepa\\curriculum.py"

```



\## 4. Résultats (chronologique)



\*\*4.1 Prototype (`proto.py`).\*\* `jepa` descend à 0.15-0.27, `copy\_baseline` autour de 1.1-1.16, `std\_z` entre 1.02 et 1.06 (pas d'effondrement). Le pipeline fonctionne.



\*\*4.2 Oubli et replay (ratio à la baseline "copie dans l'espace latent", plus bas = mieux).\*\* Paramètres : SSM 40 320, GRU 50 368. Moyenne de 5 graines :



| | SSM A | SSM B | GRU A | GRU B |

|---|---|---|---|---|

| Après phase A | 0.200 | 0.489 | 0.097 | 0.396 |

| Phase B sans replay | 1.598 | 0.046 | 3.511 | 0.021 |

| Replay 16 | 0.275 | 0.082 | 0.236 | 0.043 |

| Replay 2 | 0.357 | 0.078 | 0.268 | 0.041 |



Le replay corrige l'oubli, quel que soit l'encodeur. Correction faite en cours de route : le replay ne tire que `B//2` = 16 séquences par batch rejoué, donc la mémoire effective est de 256 séquences (replay 16) et 32 (replay 2), pas 512. L'oubli observé est celui du couple encodeur/prédicteur.



\*\*4.3 Sonde linéaire gelée\*\* (`z(t)` vers `x(t+K)`, erreur divisée par celle de la copie brute `x(t)` vers `x(t+K)`, métrique indépendante de l'encodeur). Moyennes :



| | A refit | B refit | A figée |

|---|---|---|---|

| SSM après A | 0.123 | 0.018 | 0.123 |

| SSM sans replay | 0.118 | 0.015 | 17.7 |

| SSM replay 16 | 0.114 | 0.016 | 30.0 |

| SSM replay 2 | 0.142 | 0.016 | 21.7 |

| GRU après A | 0.088 | 0.013 | 0.088 |

| GRU sans replay | 0.080 | 0.011 | 10.6 |

| GRU replay 16 | 0.094 | 0.014 | 26.6 |

| GRU replay 2 | 0.102 | 0.013 | 27.5 |



\- L'information sur A n'est pas perdue (`A refit` stable). L'espace latent dérive massivement entre phases (`A figée` explose), replay compris. Écarts-types énormes sur `A figée`, donc les différences entre conditions ne sont pas fiables.

\- Conséquence de design : une mémoire épisodique qui stocke des `z` deviendra incompatible avec l'encodeur courant. Pistes : stocker les entrées brutes et ré-encoder, ou ajouter une perte d'ancrage `enc(x\_tampon)` proche du `z` d'origine (non testé).

\- Les colonnes A et B de la sonde n'ont pas le même dénominateur : ne pas les comparer entre elles.



\*\*4.4 Contrôle encodeur non entraîné (`A refit` / `B refit`)\*\* : SSM 0.156 / 0.022, GRU 0.057 / 0.009. Le GRU aléatoire fait mieux que le GRU entraîné (0.088) : l'entraînement JEPA n'a rien apporté sur cette tâche courte et facile. Pour le SSM, gain modeste (0.156 vers 0.123). Plancher de bruit non calculé.



\*\*4.5 Contexte long avec JEPA (`longctx.py`)\*\*, L=128 (cue 16, trou 96 de distracteur, répétition 16), R² de décodage de la fréquence cachée `f` par sonde linéaire, 3 graines :



| | Fin du cue (non entraîné → entraîné) | Milieu / fin du trou |

|---|---|---|

| SSM | 0.208 → 0.962 | -0.13 (bruit de sonde, environ -p/(n-p) = -0.15) |

| SSM-lent | 0.052 → 0.977 | -0.09 / -0.18 |

| GRU | 0.991 → 0.984 | -0.11 / -0.14 |



Aucun encodeur ne retient `f` à travers le trou avec l'objectif JEPA (R² nul). Seul effet net de JEPA dans toute la série : le SSM apprend à décoder `f` en fin de cue.



\*\*4.6 Capacité supervisée (`capacity.py`)\*\*, mêmes données, 400 pas, têtes linéaires sur `z`, R² fin de cue / fin de trou : SSM 0.966 / -0.008, SSM-lent 0.961 / -0.037, GRU 0.985 / -0.045. Aucune rétention apprise. Cause identifiée : défaut de conception de la tâche (pas de marqueur d'indice, trou de 96 pas d'emblée).



\*\*4.7 Curriculum avec marqueur (`curriculum.py`)\*\*, canal marqueur (1 pendant le cue) plus trous 8, 16, 32, 64, 96 (150 pas chacun, poids conservés). R² de `f` en fin de trou, moyenne de 3 graines :



| Trou | 8 | 16 | 32 | 64 | 96 |

|---|---|---|---|---|---|

| SSM (init d'origine) | -0.05 | -0.01 | -0.02 | -0.01 | 0.00 |

| SSM-lent | 0.14 | 0.88 | 0.96 | 0.96 | 0.98 |

| GRU | 0.91 | 0.94 | 0.98 | 0.98 | 0.99 |



\- Le SSM à l'init d'origine (`dt` = softplus d'une sortie linéaire, soit environ 0.7) échoue totalement, même à 8 pas, alors qu'il lit bien le cue : l'init lente de `dt` est une condition nécessaire.

\- SSM-lent et GRU retiennent sur 96 pas. Dispersion forte entre graines pour le SSM-lent à 8 pas (de -0.05 à 0.40). Écart SSM-lent / GRU à 32 pas et plus indistinct du bruit (3 graines). Le GRU apprend plus vite au départ.

\- Non isolé : marqueur et curriculum ont été changés ensemble, on ne sait pas lequel a fait la différence.



\## 5. Conclusions honnêtes à ce stade



\- Établi : pipeline JEPA + EMA stable ; oubli encodeur/prédicteur reproductible et corrigé par replay (indépendant de l'architecture) ; dérive latente forte même avec replay ; encodeur non entraîné = baseline forte ; init lente de `dt` nécessaire ; rétention sur 96 pas apprenable (SSM-lent, GRU) avec marqueur et curriculum.

\- Non établi : tout avantage du SSM sur le GRU, donc rien ne soutient "bien plus performant". L'objectif JEPA seul ne fait pas émerger la rétention (non testé avec marqueur et curriculum).

\- Limites : tâches synthétiques faciles, contexte court (≤ 128 pas), hyperparamètres hérités sans réglage, SSM simplifié (pas de convolution locale ni d'init type Mamba complète, scan Python lent), peu de graines (3 à 5).

\- Erreurs corrigées dans la conversation : taille mémoire du replay (256 et 32, pas 512) ; l'avantage apparent du GRU était en grande partie un effet d'initialisation ; la tâche de `capacity.py` était mal posée.



\## 6. Prochaines étapes proposées (par priorité)



1\. \*\*Scan parallèle par blocs en PyTorch pur\*\* (fonctionne sous Windows, sans `mamba-ssm`) pour tester là où le SSM devrait gagner : relancer le curriculum avec trous de 128, 256, 512, 1024. Le GRU peut tenir ou décrocher, c'est ce que le test dira. Alternative : WSL2 avec les vrais kernels `mamba-ssm`.

2\. \*\*Ablation marqueur vs curriculum\*\* pour savoir lequel est nécessaire. Puis tester si l'objectif JEPA (avec marqueur et curriculum) fait émerger la rétention.

3\. \*\*Replay avec perte d'ancrage latente\*\* : mesurer si elle réduit la dérive (`A figée`) sans perdre en plasticité sur B.

4\. \*\*Consolidation sélective\*\* : ne stocker en mémoire que les séquences à forte erreur de prédiction.

5\. \*\*Poids rapides (TTT)\*\* : tester l'adaptation en ligne quand les fréquences changent en cours de séquence.

6\. \*\*Planificateur latent\*\* (RL/MPC en imagination), plus tard.

7\. Hygiène : réglage des hyperparamètres des deux encodeurs, plus de graines, calcul du plancher de bruit des sondes, `pip install numpy`.



\## 7. Première action suggérée demain



Reprendre l'étape 1 : écrire le scan parallèle (ou choisir WSL2), le valider contre le scan séquentiel sur une petite entrée (les sorties doivent coïncider), puis lancer le curriculum à trous longs avec SSM-lent et GRU, plus le contrôle non entraîné.



\---
