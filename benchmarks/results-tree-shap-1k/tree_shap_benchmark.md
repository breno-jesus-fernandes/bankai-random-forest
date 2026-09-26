# Experimental TreeSHAP benchmark

Medians of three seeds. Permutation is model-agnostic and is not value-equivalent to tree_path_dependent.

| mode | route | export/adapt s | explainer s | calculate s | total s | s/row | peak RSS KiB | additivity max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | direct | 0.012378 | 0.002594 | 0.277026 | 0.293682 | 0.002937 | 314240 | 4.1522341120980855e-14 |
| exact | adapter | 0.011894 | 0.002481 | 0.269953 | 0.285007 | 0.002850 | 314240 | 4.1522341120980855e-14 |
| exact | native | 0.000000 | 0.000000 | 1.376285 | 1.376285 | 0.013763 | 314240 | 4.163336342344337e-14 |
| exact | permutation | 0.000000 | 0.000091 | 1.815321 | 1.815412 | 0.018154 | 314240 | not comparable |
| max_bins=16 | direct | 0.012832 | 0.002530 | 0.309377 | 0.325443 | 0.003254 | 314240 | 1.7208456881689926e-15 |
| max_bins=16 | adapter | 0.012323 | 0.002476 | 0.294848 | 0.308965 | 0.003090 | 314240 | 1.7208456881689926e-15 |
| max_bins=16 | native | 0.000000 | 0.000000 | 1.709485 | 1.709485 | 0.017095 | 314240 | 2.3314683517128287e-15 |
| max_bins=16 | permutation | 0.000000 | 0.000052 | 1.737786 | 1.737838 | 0.017378 | 314240 | not comparable |
