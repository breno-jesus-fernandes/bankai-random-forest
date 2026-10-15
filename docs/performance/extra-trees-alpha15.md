# Extra Trees exato na alpha15

A preparação do fit denso float32 passa diretamente do NumPy para uma matriz
float32 por coluna, compartilhada pelas árvores. O buffer temporário mantém o
dtype original. A busca binária especializa a presença de NaN fora dos loops e
reutiliza os pesos de cada nó em uma sequência contígua. Limiares, pesos e
impurezas continuam em float64; entradas float64 preservam seus valores.

A comparação alternada contra a etapa anterior, que já usava matriz por coluna,
reduziu o fit mediano de 1.175 para 0.754 s com bootstrap e de 1.751 para 1.335 s
sem bootstrap. As divisões e predições se mantiveram idênticas nos cenários medidos.
O controle binned ficou estável e o exato alcançou tempos próximos ou melhores
nesse teste de credit_default, sem usar max_bins.

O [relatório completo](../../benchmarks/results-extra-trees-hypotheses-20261015/README.md)
contém medições, hashes, hipóteses descartadas, comandos e limites. Os testes do
wheel alpha15 passaram: 269 Python, 57 unitários Rust e 5 CLI. Não há mudança na
API pública nem no formato de serialização.
