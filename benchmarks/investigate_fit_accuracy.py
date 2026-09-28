"""Isolate the leaf-budget effect without changing binning or depth."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from bankai_random_forest import BankaiRandomForestClassifier

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--only-63',action='store_true')
parser.add_argument('--output',type=Path,default=Path('benchmarks/results-fit-hardware-mvp/accuracy-investigation.json'))
args=parser.parse_args()
records=[]
for case,rows,features,classes in [('binary',20000,32,2),('wide_multiclass',12000,128,4)]:
    x,y=make_classification(n_samples=rows+4000,n_features=features,n_informative=16,n_classes=classes,random_state=1729)
    for leaves in ([63] if args.only_63 else [63,511,None]):
        for seed in [42,43,44]:
            model=BankaiRandomForestClassifier(n_estimators=40,max_bins=63,max_depth=12,max_leaf_nodes=leaves,min_samples_leaf=20,max_features=None,max_samples=.8,n_jobs=1,random_state=seed)
            start=time.perf_counter(); model.fit(x[:rows],y[:rows]); seconds=time.perf_counter()-start
            arrays=model._forest.shap_tree_arrays()
            root_unsplit=[]; root_cover=[]; leaf_numbers=[]
            for left,right,feature,threshold,cover,values,missing in arrays:
                # Export swaps children: sklearn-left is the Rust-right subtree,
                # which is grown second after the global leaf budget is consumed.
                child=left[0]
                unsplit=child>=0 and left[child]==-1
                root_unsplit.append(unsplit)
                root_cover.append(cover[child]/cover[0] if unsplit else 0.)
                leaf_numbers.append(sum(v==-1 for v in left))
            record=dict(case=case,max_leaves=leaves,seed=seed,fit_seconds=seconds,accuracy=float(np.mean(model.predict(x[rows:])==y[rows:])),mean_leaves=float(np.mean(leaf_numbers)),second_root_child_unsplit_fraction=float(np.mean(root_unsplit)),mean_root_mass_stranded=float(np.mean(root_cover)))
            records.append(record); print(record,flush=True)
args.output.write_text(json.dumps(records,indent=2)+'\n')
