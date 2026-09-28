# Investigação da acurácia

**Atualização:** a correção por maior ganho foi implementada e validada; veja [BEST_FIRST.md](BEST_FIRST.md). Os resultados abaixo descrevem a etapa anterior.

Não houve queda de acurácia introduzida pelo MVP de hardware: as previsões coincidiram com a baseline nos benchmarks anteriores. A diferença observada era entre Bankai e LightGBM.

## Causa principal confirmada: orçamento de folhas consumido em profundidade

Em `crates/xrf/src/forest/tree.rs`, `new_rec` verifica um contador global de folhas, incrementa-o ao dividir um nó e constrói inteiramente o filho esquerdo antes do direito. O primeiro ramo pode consumir quase todo o orçamento de `max_leaf_nodes`. Ao visitar o segundo filho da raiz, o contador já atingiu o limite e esse filho vira uma folha, mesmo contendo muitos exemplos de classes diferentes.

O mesmo comportamento consta no commit baseline `964710b`; a arena preservou a ordem de crescimento e a aplicação do limite.

Com 63 folhas, em todas as 120 árvores por cenário (40 árvores × três seeds), o segundo filho da raiz ficou sem subdivisões. Em média ele continha **48,59%** das entradas bootstrap no binário e **39,35%** no multiclasse. O exportador SHAP troca os lados dos filhos: esse ramo aparece como filho esquerdo nos arrays exportados.

## Experimento controlado

Mesmos datasets do benchmark, 40 árvores, 63 bins, profundidade máxima 12, mínimo de 20 amostras por folha, todas as features elegíveis, bootstrap de 80%, uma thread e seeds 42, 43 e 44. Foram alterados somente o limite de folhas ou a distribuição do orçamento. Acurácia média em 4.000 exemplos de validação:

| Cenário | Atual, 63 folhas | Reserva entre irmãos, 63 folhas | Atual, limite 511 | Atual, sem limite de folhas |
| --- | ---: | ---: | ---: | ---: |
| binary | 74.02% | 88.48% | 92.97% | 92.97% |
| wide_multiclass | 57.17% | 73.28% | 80.67% | 80.67% |

O contrafactual reservou ao irmão uma fração das expansões restantes proporcional ao número de amostras, antes de descer no primeiro ramo. Todas as árvores continuaram com exatamente 63 folhas. Nenhuma deixou o segundo filho da raiz sem divisão. Isso demonstra que a distribuição do orçamento tem efeito grande além da capacidade nominal de 63 folhas.

O experimento altera a estrutura da árvore e, consequentemente, a sequência posterior de sorteios. Não isola cada split/RNG, mas mantém dados, seeds e hiperparâmetros constantes. O efeito se repetiu nas três seeds. A reserva proporcional é apenas uma intervenção diagnóstica, não uma implementação final validada.

Com limite 511 ou sem limite, os resultados coincidiram por seed: as árvores tiveram em média aproximadamente 343 folhas no binário e 292 no multiclasse. Nesses casos, profundidade e mínimo de amostras encerraram o crescimento antes do orçamento de 511.

## Implicações

- O resultado com 63 folhas do benchmark anterior sofre de subajuste por distribuição desigual do orçamento. Não é evidência de perda de qualidade causada por histogramas contíguos, subtração in-place ou arena.
- Para dar uma semântica útil a `max_leaf_nodes`, a correção recomendada é manter candidatos de folhas em uma fila de prioridade pelo ganho global de impureza e expandir o melhor até atingir o limite. Isso distribui o orçamento pela utilidade dos splits, sem depender da ordem de recursão.
- O crescimento por melhor ganho exigirá gerir a vida dos caches de folhas pendentes; precisa preservar a subtração e medir o consumo de memória. Após a correção, repetir os benchmarks de tempo e acurácia: o trabalho por árvore muda mesmo com o mesmo número de folhas.
- Remover/aumentar `max_leaf_nodes` contorna o problema nestes dados, mas aumenta o trabalho. Os tempos do experimento não constituem um benchmark de performance com warmups/repetições suficientes.

## Outros fatores encontrados, sem atribuição quantitativa de impacto

`UniformFeatureSampler::random_feature` em `src/dense.rs` sorteia com reposição; `reload` não muda o estado. Assim, `max_features=None` fornece F tentativas, mas não garante avaliar as F features distintas. O número esperado de features distintas é `F × (1 − (1 − 1/F)^F)`, próximo de 63% de F. A observação do relatório anterior de que todas as features são elegíveis continua correta, mas elegibilidade não significa avaliação completa. Isso deve ser investigado/corrigido separadamente.

Bankai agrega votos de classes das folhas, enquanto LightGBM RF tem outro objetivo e representação dos valores das folhas. Os procedimentos de amostragem e discretização também diferem. A diferença restante no multiclasse não foi decomposta; o experimento não prova que a política de folhas explica toda a diferença em relação ao LightGBM.

## Artefatos e estado final

- [Runner](../investigate_fit_accuracy.py): executar `.venv/bin/python benchmarks/investigate_fit_accuracy.py` na raiz do repositório.
- [Resultados atuais](accuracy-investigation.json): 18 fits, variando limite de folhas e seed.
- [Resultados do contrafactual](accuracy-counterfactual.json): seis fits com reserva entre irmãos, mantendo 63 folhas.
- [Patch diagnóstico](accuracy-counterfactual.patch): alteração exata aplicada temporariamente. Para reproduzir, aplicar em checkout descartável do MVP, compilar com as mesmas flags do relatório principal e executar o runner com `--only-63 --output <arquivo>`. Usar apenas o cenário de orçamento finito do experimento; não é patch pronto para produção.

O código Rust e a extensão Python originais do MVP foram restaurados e recompilados depois do contrafactual. Não foi incorporada uma mudança de comportamento ao estimador nesta investigação.
