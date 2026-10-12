# Extra Trees: resultados da triagem de fraude

Campanha final sem bootstrap: `campaign-extra-trees-20261012T195932241106Z`.
500 mil linhas, 400 mil de treino, 500 features, 100 ?rvores, 12 jobs e tr?s repeti??es.
Bankai e sklearn usam `bootstrap=False`, `max_samples=None` e pesos de classes balanceados.
Os manifests e checkpoints identificam os fontes, bin?rio, seeds e par?metros.
O build anterior ? ?ltima otimiza??o e seus resultados foram preservados separadamente.

## Observa??es individuais

| repeat | model | fit_seconds | feature_average_precision | recall_at_24 | informative_missing |
| --- | --- | --- | --- | --- | --- |
| 0 | BankaiRF (local build, extra_trees=True) | 47.129146 | 0.637589 | 0.583333 | 10 |
| 0 | Extra Trees | 52.165937 | 0.700308 | 0.666667 | 8 |
| 0 | BankaiRF (local build, extra_trees=False) | 28.136396 | 0.609842 | 0.583333 | 10 |
| 1 | BankaiRF (local build, extra_trees=False) | 27.494113 | 0.703781 | 0.666667 | 8 |
| 1 | Extra Trees | 48.916144 | 0.769241 | 0.708333 | 7 |
| 1 | BankaiRF (local build, extra_trees=True) | 43.680944 | 0.696755 | 0.583333 | 10 |
| 2 | BankaiRF (local build, extra_trees=True) | 43.568203 | 0.746187 | 0.625 | 9 |
| 2 | Extra Trees | 49.249632 | 0.710491 | 0.666667 | 8 |
| 2 | BankaiRF (local build, extra_trees=False) | 27.900955 | 0.644922 | 0.625 | 9 |

## Diferen?as pareadas: Bankai Extra Trees menos sklearn

| repeat | fit_seconds | feature_average_precision | informative_missing |
| --- | --- | --- | --- |
| 0 | -5.036791 | -0.062719 | 2.0 |
| 1 | -5.2352 | -0.072486 | 3.0 |
| 2 | -5.681429 | 0.035696 | 1.0 |

Bankai: mediana de fit 43,681 s; sklearn: 49,250 s. A diferen?a entre medianas
? ?5,569 s (Bankai 11,3% mais r?pido). A mediana das diferen?as pareadas ?
?5,235 s. Essas duas medidas n?o s?o intercambi?veis.

A perda pareada mediana de AP ? 0,062719; a diferen?a entre as medianas de AP
? ?0,013736. O Bankai perde uma mediana pareada de duas features informativas
adicionais nas top 24. A meta de tempo foi atendida nesta triagem; as metas de
ranking AP e recall n?o foram atendidas. Tr?s repeti??es n?o demonstram
n?o inferioridade estat?stica; o aceite permanece condicionado ? campanha de dez.

## Perfil separado

O perfil do mesmo build/dataset est? em `campaign-extra-trees-20261012T200902238631Z`.
O fit instrumentado levou 46,45 s, separado das medi??es acima. Constru??o das
?rvores ocupa 45,22 s, prepara??o nativa 1,01 s, ordena??o can?nica 0,044 s e
import?ncia 0,00028 s. A busca acumula 492,48 s entre workers e a parti??o
25,29 s; esses tempos internos se sobrep?em entre threads. Foram 240.843
features visitadas, 168.410 candidatos rejeitados e 15,34 bilh?es de leituras
de amostras nas duas passagens bin?rias. O buffer m?ximo por worker ? 4 MiB.
O custo dominante continua sendo a busca/leitura das features durante as ?rvores.

## Abla??o da import?ncia

A import?ncia global foi calculada sobre as mesmas ?rvores sem refit, isolando
a agrega??o. Sua mediana de AP ? 0,721968, contra 0,696755 para a normaliza??o
por ?rvore. O recall mediano passa de 0,583333 a 0,666667. A normaliza??o por
?rvore foi mantida por corresponder ? sem?ntica do sklearn; ela n?o melhorou
estas tr?s observa??es. Esse resultado precisa constar de qualquer interpreta??o
da mudan?a de qualidade.

## Bootstrap hist?rico

A campanha `campaign-extra-trees-20261012T194428255413Z`, anterior ? ?ltima
otimiza??o, comparou ligado/desligado. No Bankai, bootstrap ligado teve medianas
23,47 s e AP 0,729764; desligado, 47,89 s e AP 0,700527, com recall mediano
igual (0,666667). No sklearn, ligado teve 17,78 s, AP 0,751588 e recall 0,708333;
desligado, 49,95 s, AP 0,710491 e recall 0,666667. Essa compara??o altera tamb?m
a exposi??o ?s classes em sklearn 1.9.1: pesos balanceados definem probabilidades
de sorteio no bootstrap. Por instru??o posterior do usu?rio, futuros benchmarks
retiram bootstrap independentemente desse resultado hist?rico.

## Limita??es e diferen?as restantes

RNGs e streams de amostragem diferem entre bibliotecas. Bankai usa igualdade
exata para constantes, enquanto sklearn usa toler?ncia float32; Bankai avalia
as duas dire??es de NaN. Bankai prev? por votos das classes das folhas, enquanto
sklearn agrega probabilidades, o que afeta ROC/PR diagn?sticos. N?o h? evid?ncia
suficiente para atribuir a diferen?a restante de ranking a uma dessas diferen?as
isoladamente. A variabilidade de AP entre as tr?s repeti??es ? material.

O layout nativo por linha foi mantido. N?o houve compara??o de layout nativo
por coluna nesta etapa; a c?pia inicial float32 ainda passa temporariamente por
float64, e o workspace/estat?sticas de classes de multiclass n?o recebeu a mesma
especializa??o do caminho bin?rio. A instrumenta??o n?o cobre todas as aloca??es
nem separa todos os motivos individuais de rejei??o.

## Baselines conclu?das e pausa

Todas as campanhas seguintes tamb?m usam bootstrap desligado e tr?s repeti??es:

| Implementa??o | Fit mediano (s) | AP mediano do ranking | Recall@24 mediano |
| --- | ---: | ---: | ---: |
| Bin?rio anterior, bins=63 | 70,999213 | 0,748818 | 0,708333 |
| Bin?rio anterior, valores originais | 58,231566 | 0,730968 | 0,625000 |
| Controle RF, bin?rio anterior | 27,163066 | 0,644922 | 0,625000 |

Campanhas: `20261012T201119462531Z`, `20261012T202259708602Z` e
`20261012T203259423529Z`, respectivamente, sob `benchmarks/campaign-extra-trees-*`.
O controle RF atual teve exatamente os mesmos APs nas tr?s observa??es; o fit
mediano anterior foi 27,16 s contra 27,90 s do build medido (diferen?a 2,7%).

O usu?rio pediu pausa dos benchmarks. Nenhum processo de benchmark desta tarefa
permanece ativo. Os demais cen?rios e a campanha completa dos seis modelos n?o
foram iniciados. A corre??o posterior de exclus?o de pesos zero sem bootstrap
foi compilada e validada (223 testes Python e 54 Rust), mas esse ?ltimo bin?rio
ainda n?o recebeu uma nova campanha de performance. Portanto, as medi??es acima
identificam o build anterior a essa ?ltima corre??o, n?o o bin?rio mais recente.
A compara??o de layouts e a instrumenta??o completa de motivos de rejei??o e
aloca??es continuam pendentes.
