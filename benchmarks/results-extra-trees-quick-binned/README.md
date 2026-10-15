# Extra Trees: exato, 63 bins e 255 bins

Medições locais em 2026-10-15. Mesmo gerador de credit_default do notebook Boruta:
100 mil linhas (80 mil treino), 500 features originais + 500 sombras, 64 árvores,
6 jobs, três seeds, max_depth=8, min_samples_leaf=500, max_features=sqrt,
class_weight=balanced. Binned: sampled_select, amostra de 20 mil linhas,
binning_seed=6, extra_seed=6. Com bootstrap: max_samples=0.8; sem: None.

Cada campanha mede novamente Bankai exato otimizado, sklearn ExtraTreesClassifier
e Bankai Extra Trees binned. A ordem roda entre os três modelos por seed. O timer
inclui preparação e binning; exclui geração e métricas. Não há warmup separado.
As campanhas de 63 e 255 bins foram sequenciais, portanto os tempos estão sujeitos
a ruído do hardware/cache. Todas as linhas, métricas preditivas e controles de
cada campanha estão em measurements.csv; medianas em summary.csv. Hash do binário,
configuração e hashes dos fontes estão em manifest.json.

| Bootstrap | Modelo | Fit mediano (s) | AP do ranking | Recall@24 |
| --- | --- | ---: | ---: | ---: |
| Sim | Bankai exato (campanha 255) | 1.507 | 0.7013 | 0.6667 |
| Sim | sklearn (campanha 255) | 1.604 | 0.7757 | 0.7083 |
| Sim | Bankai binned 63 | 0.857 | 0.6678 | 0.6250 |
| Sim | Bankai binned 255 | 0.970 | 0.6169 | 0.5833 |
| Não | Bankai exato (campanha 255) | 2.272 | 0.7538 | 0.7500 |
| Não | sklearn (campanha 255) | 4.911 | 0.7597 | 0.7083 |
| Não | Bankai binned 63 | 1.187 | 0.6111 | 0.5833 |
| Não | Bankai binned 255 | 1.356 | 0.6107 | 0.5833 |

255 bins não recuperaram a qualidade do ranking neste cenário. Com bootstrap,
o AP do ranking caiu em duas das três seeds; sem bootstrap, subiu em uma e caiu
nas outras duas. Os fits medianos binned aumentaram aproximadamente 13–14%.
Isso não é uma conclusão estatística geral: são três seeds e um cenário, sem
rodar o Boruta completo.

O binned sorteia um limiar entre índices dos bins, construídos por quantis. O exato
sorteia no intervalo dos valores originais do nó. Aumentar a resolução dos bins
não elimina essa diferença na distribuição dos cortes. Ela é uma hipótese para a
diferença de ranking, não uma causa isolada demonstrada por este teste.

Reprodução (cada comando leva dezenas de segundos):

```sh
.venv/bin/maturin develop --release --skip-install
.venv/bin/python benchmarks/run_extra_trees_quick.py --include-binned --bins 63 --output /tmp/b63.json
.venv/bin/python benchmarks/run_extra_trees_quick.py --include-binned --bins 255 --output /tmp/b255.json
```

Acrescente `--no-bootstrap` para repetir sem bootstrap. O script carrega o build
local em python/, com max_bins=None no exato e backend binned explícito no binned.
