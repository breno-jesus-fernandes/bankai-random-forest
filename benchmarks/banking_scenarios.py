"""Shared synthetic datasets and estimator parameters from the banking notebook."""
import math
import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from bankai_random_forest import BankaiRandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
N_ROWS, N_FEATURES, N_INFORMATIVE = 500_000, 500, 24
N_TREES, N_JOBS, MAX_BINS = 100, 6, 63
BOOTSTRAP = False
SCENARIOS = {
    'credit_default': {'prevalence': 0.015, 'description': 'Inadimplência futura', 'max_depth': 8, 'min_leaf': 500, 'row_fraction': 0.80},
    'transaction_fraud': {'prevalence': 0.002, 'description': 'Fraude confirmada', 'max_depth': 9, 'min_leaf': 750, 'row_fraction': 0.80},
    'customer_churn': {'prevalence': 0.060, 'description': 'Encerramento futuro', 'max_depth': 7, 'min_leaf': 250, 'row_fraction': 0.80},
    'offer_acceptance': {'prevalence': 0.180, 'description': 'Aceitação de oferta', 'max_depth': 6, 'min_leaf': 300, 'row_fraction': 0.80},
}

def calibrate_intercept(signal, prevalence):
    signal = (signal - signal.mean()) / max(signal.std(), 1e-8)
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(signal + mid).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    return signal + (lo + hi) / 2

def make_dataset(scenario, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    # Indicadores e variáveis monetárias com assimetria, comuns em dados bancários.
    X[:, 4:8] = rng.binomial(1, [0.12, 0.08, 0.18, 0.35], size=(N_ROWS, 4)).astype(np.float32)
    X[:, 8:12] = np.log1p(rng.lognormal(mean=2.0, sigma=0.8, size=(N_ROWS, 4))).astype(np.float32)
    X[:, 8:12] = (X[:, 8:12] - X[:, 8:12].mean(axis=0)) / (X[:, 8:12].std(axis=0) + 1e-6)
    # Proxies correlacionadas; todas continuam informativas para a verdade sintética.
    X[:, 16:20] = 0.75 * X[:, 0:4] + 0.66 * rng.normal(size=(N_ROWS, 4)).astype(np.float32)
    z = X

    if scenario == 'credit_default':
        signal = (1.0*z[:,0] + .85*z[:,1] + .65*z[:,2] + .5*z[:,3]
                  + .8*(z[:,4] > .5) + .7*(z[:,5] > .5) + .5*z[:,6] - .4*z[:,7]
                  + .65*z[:,8] - .55*z[:,9] + .45*z[:,10] + .35*z[:,11]
                  + .7*np.maximum(z[:,12]-.8, 0) + .6*(z[:,13]**2-1)
                  + .65*z[:,14]*z[:,15] + .18*z[:,16:24].sum(axis=1))
    elif scenario == 'transaction_fraud':
        signal = (1.0*z[:,0] + .8*np.maximum(z[:,1]-1, 0) + .7*(z[:,2] > 1.2)
                  + .9*z[:,3]*z[:,4] + .7*(z[:,5] > .5) + .5*z[:,6]
                  + .8*np.abs(z[:,7]) + .55*z[:,8] + .5*z[:,9]
                  + .65*(z[:,10]**2-1) + .7*z[:,11]*z[:,12]
                  + .5*z[:,13] + .4*z[:,14] + .35*z[:,15]
                  + .18*z[:,16:24].sum(axis=1))
    elif scenario == 'customer_churn':
        signal = (-.9*z[:,0] - .8*z[:,1] + .7*z[:,2] + .6*z[:,3]
                  + .65*(z[:,4] > .5) + .7*(z[:,5] > .5) + .5*z[:,6]
                  + .45*z[:,7] - .65*z[:,8] + .55*z[:,9] + .45*z[:,10]
                  + .6*np.maximum(z[:,11]-.7, 0) + .65*z[:,12]*z[:,13]
                  + .5*np.tanh(z[:,14]) + .4*z[:,15] + .18*z[:,16:24].sum(axis=1))
    elif scenario == 'offer_acceptance':
        signal = (.8*z[:,0] + .7*z[:,1] - .5*z[:,2] + .65*z[:,3]
                  + .6*(z[:,4] > .5) + .45*(z[:,5] > .5) + .5*z[:,6]
                  + .4*z[:,7] + .75*z[:,8] + .55*z[:,9] - .4*z[:,10]
                  + .6*(z[:,11]**2-1) + .65*z[:,12]*z[:,13]
                  + .5*np.tanh(z[:,14]) + .35*z[:,15] + .18*z[:,16:24].sum(axis=1))
    else:
        raise ValueError(f'Cenário desconhecido: {scenario}')
    logits = calibrate_intercept(signal, SCENARIOS[scenario]['prevalence'])
    y = rng.binomial(1, expit(logits)).astype(np.int8)
    return X, y, np.arange(N_FEATURES) < N_INFORMATIVE


class XGBoostForestEstimator(XGBClassifier):
    # Expose one sklearn n_estimators value to Boruta while fitting that many
    # parallel trees in a single XGBoost boosting round.
    def fit(self, X, y, **kwargs):
        tree_count = self.n_estimators
        self.set_params(n_estimators=1, num_parallel_tree=tree_count)
        return super().fit(X, y, **kwargs)

def estimator_factory(model_name, seed, positive_weight, scenario):
    cfg = SCENARIOS[scenario]
    depth, leaf, row_fraction = cfg['max_depth'], cfg['min_leaf'], cfg['row_fraction']
    p_augmented = N_FEATURES
    node_fraction = min(1.0, math.sqrt(p_augmented) / p_augmented)
    common_seed = int(seed % (2**31 - 1))
    if model_name in ('BankaiRF (local build, extra_trees=False)', 'BankaiRF (local build, extra_trees=True)'):
        return BankaiRandomForestClassifier(
            n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf, max_features='sqrt',
            max_samples=(row_fraction if BOOTSTRAP else None), bootstrap=BOOTSTRAP, max_bins=(None if model_name.endswith('extra_trees=True)') else MAX_BINS),
            importance_type='gain', class_weight='balanced',
            extra_trees=model_name.endswith('extra_trees=True)'), extra_seed=6,
            random_state=common_seed, n_jobs=N_JOBS)
    if model_name == 'Random Forest (sklearn)':
        return RandomForestClassifier(
            n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf, max_features='sqrt',
            max_samples=(row_fraction if BOOTSTRAP else None), bootstrap=BOOTSTRAP, class_weight='balanced',
            random_state=common_seed, n_jobs=N_JOBS)
    if model_name == 'Extra Trees':
        return ExtraTreesClassifier(
            n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf, max_features='sqrt',
            bootstrap=BOOTSTRAP, max_samples=(row_fraction if BOOTSTRAP else None), class_weight='balanced',
            random_state=common_seed, n_jobs=N_JOBS)
    if model_name == 'LightGBM RF (extra_trees=True)':
        return LGBMClassifier(
            boosting_type='rf', objective='binary', n_estimators=N_TREES,
            max_depth=depth, num_leaves=min(2**depth, 256), min_child_samples=leaf, max_bin=MAX_BINS,
            extra_trees=True,
            bagging_fraction=row_fraction, bagging_freq=1,
            feature_fraction=1.0, feature_fraction_bynode=node_fraction,
            scale_pos_weight=positive_weight, importance_type='gain',
            random_state=common_seed, n_jobs=N_JOBS, verbosity=-1)
    if model_name == 'XGBoost RF':
        return XGBoostForestEstimator(
            objective='binary:logistic', n_estimators=N_TREES,
            num_parallel_tree=N_TREES, max_depth=depth, min_child_weight=leaf * 0.25, tree_method='hist',
            max_bin=MAX_BINS, subsample=row_fraction,
            colsample_bytree=1.0, colsample_bynode=node_fraction,
            learning_rate=1.0, scale_pos_weight=positive_weight,
            importance_type='gain', random_state=common_seed,
            n_jobs=N_JOBS, verbosity=0)
    raise ValueError(model_name)

BANKAI_OFF = 'BankaiRF (local build, extra_trees=False)'
BANKAI_ON = 'BankaiRF (local build, extra_trees=True)'
MODEL_NAMES = [BANKAI_OFF, BANKAI_ON, 'Random Forest (sklearn)', 'Extra Trees', 'LightGBM RF (extra_trees=True)', 'XGBoost RF']