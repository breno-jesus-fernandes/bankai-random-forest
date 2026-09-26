| type | train rows | validation rows | profile | implementation | fit s | importance s | total s | rank correlation | top-3 overlap | top feature |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gain | 10000 | 10000 | default | sklearn | 1.831695 | 0.003687 | 1.835382 | 1.000000 | 1.000000 | 2 |
| gain | 10000 | 10000 | default | pyO3 | 1.293302 | 0.001074 | 1.294376 | 0.454135 | 1.000000 | 2 |
| gain | 10000 | 10000 | default | rust-cli | 1.335538 | 0.001100 | 1.336638 | 0.508271 | 1.000000 | 2 |
| split | 10000 | 10000 | default | sklearn | 1.764794 | 0.001879 | 1.766674 | 1.000000 | 1.000000 | 1 |
| split | 10000 | 10000 | default | pyO3 | 1.318781 | 0.000881 | 1.319661 | 0.231579 | 1.000000 | 1 |
| split | 10000 | 10000 | default | rust-cli | 1.362645 | 0.000920 | 1.363565 | 0.657143 | 1.000000 | 1 |
| permutation | 10000 | 10000 | default | sklearn | 1.770551 | 6.891050 | 8.661601 | 1.000000 | 1.000000 | 0 |
| permutation | 10000 | 10000 | default | pyO3 | 1.324912 | 0.134355 | 1.459267 | 0.702256 | 1.000000 | 2 |
| permutation | 10000 | 10000 | default | rust-cli | 1.385495 | 0.125437 | 1.510932 | 0.530827 | 1.000000 | 2 |

`gain` compares sklearn impurity decrease with accumulated XRF criterion-weighted split gain. `split` counts feature nodes in each implementation (sklearn counts are derived from fitted trees). `permutation` compares sklearn inspection permutation on the independent validation partition with native XRF per-tree OOB accuracy decrease. Sklearn permutation uses five repeats. All implementations train on the same first partition.
