#!/usr/bin/env python
"""Script used to obtain results from the PcmRandomnets paper. All
experiments are described in the config folder. Run it with

    python run_qsprpred_bth.py

Output is saved to output/results.csv and output/results_per_target.csv.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import statistics
import time
import warnings

import pandas as pd
import torch
from qsprpred.data import QSPRTable
from qsprpred.data.processing.pipeline import DatasetPipeline
from qsprpred.data.sampling.splits import ManualSplit, TemporalSplit
from qsprpred.extra.gpu.models.dnn import DNNModel
from qsprpred.models.early_stopping import EarlyStoppingMode

# Random forests
from qsprpred.models.scikit_learn import SklearnModel
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from scipy.stats import sem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ShuffleSplit

from qsprpred_ext import (
    HiddenSizesFullyConnected,
    PCMDataFrameDescriptorSet,
    Randomnets,
)
from utils.column_scaler import ColumnScaler
from utils.csv_append import append_row
from utils.mcc_threshold_scorer import MccThresholdScorer, as_1d

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Skip writing debug files
DEBUG = False

TEMPORAL_SPLITS = {
    'original_bth': 2012,
    'chembl35': 2020,
}

PHYSCHEM_SETS = {
    'full': 'fps_cols_full.txt',
    'original': 'fps_cols_original.txt',
}

def get_model(algorithm, i, parameters, task):
    if algorithm == 'randomnets':
        if task == 'SINGLECLASS':
            raise Exception('Classifier not implemented yet for Randomnets')
        rn_params = dict(parameters)
        rn_params['n_hidden_layers'] = len(parameters['layers'])
        rn_params['dim'] = parameters['layers'][0]
        return Randomnets(
            base_dir='./output/models',
            name=f'RandomnetsBthModel_{i}',
            target_column='BIOACT_PCHEMBL_VALUE',
            parameters=rn_params,
            random_state=42 + i,
            patience=-1,
            tol=0.01
        )
    if algorithm == 'randomforest':
        alg = RandomForestRegressor if task == 'REGRESSION' else RandomForestClassifier
        return SklearnModel(
            base_dir='./output/models',
            name=f'RandomforestBthModel_{i}',
            alg=alg,
            parameters={
                'n_estimators': parameters['n_estimators'],
                'max_features': parameters['max_features'],
                'n_jobs': parameters['n_jobs'],
            },
        )
    if algorithm == 'dnn':
        dnn_params = {
            'n_epochs': parameters['n_epochs'],
            'lr': parameters.get('max_lr', 1e-4),
            'hidden_sizes': parameters['layers'],
            'batch_size': parameters['batch_size'],
        }
        return DNNModel(
            base_dir='./output/models',
            name=f'DNNBthModel_{i}',
            alg=HiddenSizesFullyConnected,
            parameters=dnn_params,
            random_state=42 + i,
            patience=parameters['patience'],
            tol=0,
        )
    raise Exception(f'No such model {algorithm}')

def getDataset(data_dir, task='REGRESSION', fp_size=256, physchem='full'):
    if physchem not in PHYSCHEM_SETS:
        raise Exception(f'No such physchem descriptor set {physchem!r}; '
                        f'expected one of {sorted(PHYSCHEM_SETS)}')
    data = pd.read_csv(f'{data_dir}/compounds_with_smiles_and_split2.csv')
    path = f'{data_dir}/parsed_compounds.csv'
    data.to_csv(path)
    # Make sure that datasets are identifiable by name
    name = '_'.join([
        'BeyondTheHypeDataset',
        os.path.basename(data_dir),
        str(fp_size),
        physchem,
        task,
    ])
    if task == 'REGRESSION':
        dataset = QSPRTable.fromTableFile(
            filename=path,
            path="output",
            name=name,
            sep=",",
            target_props=[{"name": "BIOACT_PCHEMBL_VALUE", "task": task}],
        )
    elif task == 'SINGLECLASS':
        dataset = QSPRTable.fromTableFile(
            filename=path,
            path="output",
            name=name,
            sep=",",
            target_props=[{"name": "BIOACT_PCHEMBL_VALUE", "task": task, "th": [6.5]}],
        )
    else:
        raise Exception(f"Task not supported: {task}")
    dataset.nJobs = 1

    dataset.randomState = 42

    with open(f'{data_dir}/{PHYSCHEM_SETS[physchem]}', 'r') as f:
        fps_cols = [line.strip() for line in f]
    fps_filename = f'compound_features_{fp_size}_fix.txt'
    fps_df = pd.read_csv(f'{data_dir}/{fps_filename}', sep=',')
    fps_df.drop_duplicates(inplace=True)
    fps_df['CMP_CHEMBL_ID'] = fps_df['CMP_CHEMBL_ID'].astype(str)
    if not fps_df['CMP_CHEMBL_ID'].iloc[0].startswith('CHEMBL'):
        import sys
        print(f'WARNING: {fps_filename}: CMP_CHEMBL_ID values lack "CHEMBL" prefix '
              f'(e.g. {fps_df["CMP_CHEMBL_ID"].iloc[0]!r}). Regenerate with prepare_data.py --force.',
              file=sys.stderr)
        fps_df['CMP_CHEMBL_ID'] = 'CHEMBL' + fps_df['CMP_CHEMBL_ID']
    pdesc_df = pd.read_csv(f'{data_dir}/Protein_Descriptors_50_sections.txt', sep='\t')
    pdesc_df = pdesc_df[pdesc_df.columns.intersection(fps_cols + ['TGT_CHEMBL_ID'])]
    cdesc_df = pd.read_csv(f'{data_dir}/compound_additional_physchem_features.txt', sep='\t', low_memory=False)
    cdesc_df = cdesc_df[cdesc_df.columns.intersection(fps_cols + ['CMP_CHEMBL_ID'])]
    cdesc_df['CMP_CHEMBL_ID'] = cdesc_df['CMP_CHEMBL_ID'].astype(str)
    if not cdesc_df['CMP_CHEMBL_ID'].iloc[0].startswith('CHEMBL'):
        print(f'WARNING: compound_additional_physchem_features.txt: CMP_CHEMBL_ID values lack "CHEMBL" prefix '
              f'(e.g. {cdesc_df["CMP_CHEMBL_ID"].iloc[0]!r}). Fix the source file.',
              file=sys.stderr)
        cdesc_df['CMP_CHEMBL_ID'] = 'CHEMBL' + cdesc_df['CMP_CHEMBL_ID']
    fps_descriptors = PCMDataFrameDescriptorSet(fps_df, joining_cols=['CMP_CHEMBL_ID'])
    pdesc_descriptors = PCMDataFrameDescriptorSet(pdesc_df, joining_cols=['TGT_CHEMBL_ID'])
    cdesc_descriptors = PCMDataFrameDescriptorSet(cdesc_df, joining_cols=['CMP_CHEMBL_ID'])
    dataset.addDescriptors([fps_descriptors, pdesc_descriptors, cdesc_descriptors])

    return dataset

def mcc_per_target_classification(model, estimator, dataset, split, scaler):
    X = dataset.getDescriptors()
    y = dataset.getTargets()
    split_obj = dataset.getSplit(split)
    tgt_col = dataset.getDF()['TGT_CHEMBL_ID']
    mccs = []
    for _, test_index in dataset.split(split_obj):
        X_test = X.loc[test_index].copy()
        y_test = y.loc[test_index]
        X_test, y_test = scaler.transform(X_test, y_test)
        y_pred_all = pd.Series(model.predict(X_test, estimator).squeeze(), index=X_test.index)
        for tgt, tgt_idx in tgt_col.loc[test_index].groupby(tgt_col).groups.items():
            idx = test_index.intersection(tgt_idx)
            if len(idx) == 0:
                continue
            y_true_class = as_1d(y_test.loc[idx])
            y_pred_class = as_1d(y_pred_all.loc[idx])
            if len(set(y_true_class)) < 2 or len(set(y_pred_class)) < 2:
                continue
            mccs.append((tgt, matthews_corrcoef(y_true_class, y_pred_class)))
    return mccs

def mcc_per_target_regression(model, estimator, dataset, split, scaler):
    X = dataset.getDescriptors()
    y = dataset.getTargets()
    split_obj = dataset.getSplit(split)
    tgt_col = dataset.getDF()['TGT_CHEMBL_ID']
    mccs = []
    for _, test_index in dataset.split(split_obj):
        X_test = X.loc[test_index].copy()
        y_test = y.loc[test_index]
        X_test, y_test = scaler.transform(X_test, y_test)
        y_pred_all = pd.Series(model.predict(X_test, estimator).squeeze(), index=X_test.index)
        for tgt, tgt_idx in tgt_col.loc[test_index].groupby(tgt_col).groups.items():
            idx = test_index.intersection(tgt_idx)
            if len(idx) == 0:
                continue
            y_true_class = as_1d(y_test.loc[idx]) > 6.5
            y_pred_class = as_1d(y_pred_all.loc[idx]) > 6.5
            if len(set(y_true_class)) < 2 or len(set(y_pred_class)) < 2:
                continue
            mccs.append((tgt, matthews_corrcoef(y_true_class, y_pred_class), len(idx)))
    return mccs

def bedroc_per_target_regression(model, estimator, dataset, split, scaler):
    X = dataset.getDescriptors()
    y = dataset.getTargets()
    split_obj = dataset.getSplit(split)
    tgt_col = dataset.getDF()['TGT_CHEMBL_ID']
    bedrocs = []
    for _, test_index in dataset.split(split_obj):
        X_test = X.loc[test_index].copy()
        y_test = y.loc[test_index]
        X_test, y_test = scaler.transform(X_test, y_test)
        y_pred_all = pd.Series(model.predict(X_test, estimator).squeeze(), index=X_test.index)
        for tgt, tgt_idx in tgt_col.loc[test_index].groupby(tgt_col).groups.items():
            idx = test_index.intersection(tgt_idx)
            if len(idx) == 0:
                continue
            y_true_class = as_1d(y_test.loc[idx]) > 6.5
            y_pred_slice = as_1d(y_pred_all.loc[idx])
            bedroc = CalcBEDROC(
                [[y] for _, y in sorted(zip(y_pred_slice, y_true_class), reverse=True)],
                col=0,
                alpha=20,
            )
            bedrocs.append((tgt, bedroc, len(idx)))
    return bedrocs

def run(n_epochs, max_lr=1e-3, layers=None, split='random',
        algorithm='randomnets', task='REGRESSION', n_nns=25,
        mask_thr=0.5, fp_size=256, data_dir='./data/original_bth',
        physchem='full', patience=200, batch_size=None, n_jobs=None):
    if layers is None:
        layers = [1024]
    if batch_size is None:
        batch_size = 8 if algorithm == 'randomnets' else 256
    if n_jobs is None:
        n_jobs = 32
    os.makedirs("./output/models", exist_ok=True)

    print(f'[{time.strftime("%H:%M:%S")}] 1. Prepare the dataset ({data_dir})')
    dataset = getDataset(data_dir, task, fp_size, physchem)

    for i in range(3):
        all_cols = list(dataset.getDescriptors().columns)
        std_cols = [col for col in all_cols if not 'bit_' in col]

        if algorithm == 'randomforest':
            parameters = {
                'n_estimators': 1000,
                'max_features': 0.3,
                'n_jobs': n_jobs,
                'split': split,
                'physchem': physchem,
            }
        else:
            parameters = {
                'n_epochs': n_epochs,
                'max_lr': max_lr,
                'layers': layers,
                'split': split,
                'n_nns': n_nns,
                'mask_thr': mask_thr,
                'physchem': physchem,
                'batch_size': batch_size,
            }
            if algorithm == 'dnn':
                # The Lightning Randomnets is constructed with patience=-1 (no early
                # stopping), so recording patience for it would be misleading.
                parameters['patience'] = patience
        model = get_model(algorithm, i, parameters, task)

        scaler = ColumnScaler(std_cols)
        pipeline = DatasetPipeline(
            steps= {
                "scaler": scaler
            }
        )

        dataset_name = os.path.basename(data_dir)
        scorer = MccThresholdScorer(6.5, model.name, parameters, './output/results.csv',
                                    dataset=dataset_name, fp_size=fp_size, replicate=i)
        if split == '70-30':
            dataset.addSplit(ManualSplit(splitprop='datasplit', trainval='train', testval='test'), name='test')
        elif split == 'temporal':
            dataset_name = os.path.basename(data_dir)
            timesplit = TEMPORAL_SPLITS[dataset_name]
            dataset.addSplit(TemporalSplit(timesplit=timesplit, timeprop='DOC_YEAR'), name='test')
        else:
            dataset.addSplit(ShuffleSplit(test_size=0.3, n_splits=1, random_state=i), name='test')

        torch.set_num_threads(1)

        for _, (X_train, y_train, X_test, y_test) in enumerate(
            pipeline.applyOnDataSet(dataset, 'test')
        ):
            print(f'[{time.strftime("%H:%M:%S")}] 2. Fit the model (replicate {i})')
            # fit model
            model.initFromData(dataset, pipeline)
            estimator = model.loadEstimator(model.parameters)
            start_fit = time.time()
            if algorithm == 'dnn':
                # Two-pass training so the final model uses the whole training set.
                # Pass 1 (RECORDING): early stopping against an internal validation
                # split records the optimal number of epochs. Pass 2 (OPTIMAL):
                # retrain from scratch on the entire training set for that many
                # epochs, with no validation set held out.
                model.fit(
                    X_train,
                    y_train,
                    estimator,
                    EarlyStoppingMode.RECORDING,
                    monitor=None,
                )
                print(f'[{time.strftime("%H:%M:%S")}] Early stopping recorded epochs '
                      f'{model.earlyStopping.trainedEpochs}; retraining on full '
                      f'training set for {model.earlyStopping.optimalEpochs} epochs')
                estimator = model.loadEstimator(model.parameters)
                model_fit = model.fit(
                    X_train,
                    y_train,
                    estimator,
                    EarlyStoppingMode.OPTIMAL,
                    monitor=None,
                )
                estimator = model_fit
            else:
                model_fit = model.fit(
                    X_train,
                    y_train,
                    estimator,
                    None,
                    monitor=None
                )
            end_fit = time.time()
            time_fit = end_fit - start_fit
            print(f"Time to fit the model: {end_fit - start_fit}")
            test_preds = model.predict(X_test, estimator)
            if DEBUG:
                tc_keys = dataset.getDF().loc[X_test.index, 'TC_key']
                pd.DataFrame({
                    'id': X_test.index,
                    'TC_KEY': tc_keys.values,
                    'predicted_value': test_preds.squeeze(),
                }).to_csv(f'output/test_predictions_{i}.csv', index=False)
            if task == 'REGRESSION':
                print(f'[{time.strftime("%H:%M:%S")}] 3. Calculate metrics (replicate {i})')
                _ = scorer(y_test, test_preds)
                print(f'[{time.strftime("%H:%M:%S")}] 4. Calculate metrics per target (replicate {i})')
                mccs = mcc_per_target_regression(model, estimator, dataset, 'test', scaler)
                bedrocs = bedroc_per_target_regression(model, estimator, dataset, 'test', scaler)
            else:
                mccs = mcc_per_target_classification(model, estimator, dataset, 'test', scaler)
                bedrocs = None
            if DEBUG:
                if mccs and len(mccs[0]) == 3:
                    mccs_out = [{"target": t, "mcc": v, "n_compounds": n} for t, v, n in mccs]
                else:
                    mccs_out = [{"target": t, "mcc": v} for t, v in mccs]
                per_target_results = {
                    "model": model.name,
                    "parameters": parameters,
                    "mccs": mccs_out,
                    "bedrocs": [
                        {"target": t, "bedroc": v, "n_compounds": n} for t, v, n in bedrocs
                    ] if bedrocs is not None else None,
                }
                with open(f"output/results_per_target_{i}.json", "w") as f:
                    json.dump(per_target_results, f, indent=2)
            mcc_avg = statistics.mean([x[1] for x in mccs])
            mcc_std = sem([x[1] for x in mccs])

            if bedrocs is not None:
                bedroc_avg = statistics.mean([x[1] for x in bedrocs])
                bedroc_std = sem([x[1] for x in bedrocs])
            else:
                bedroc_avg = bedroc_std = None
            # bedroc_avg/bedroc_std stay empty for classification runs, so the
            # column layout is the same either way.
            append_row('output/results_per_target.csv', {
                'id': model.name,
                'replicate': i,
                'dataset': dataset_name,
                'fp_size': fp_size,
                'params': json.dumps(parameters),
                'n_epochs': n_epochs,
                'layers': json.dumps(layers),
                'mcc_avg': f'{mcc_avg:.3f}',
                'mcc_std': f'{mcc_std:.3f}',
                'bedroc_avg': '' if bedroc_avg is None else f'{bedroc_avg:.3f}',
                'bedroc_std': '' if bedroc_std is None else f'{bedroc_std:.3f}',
                'runtime': f'{time_fit:.0f}',
            })

        if DEBUG:
            model.saveEstimator()


def run_config(cfg, split_override=None):
    cfg = dict(cfg)
    datasets = cfg.pop('data', ['original_bth'])
    splits = cfg.pop('splits', None) or [cfg.pop('split', 'random')]
    if split_override:
        splits = [split_override]
    n_epochs_values = cfg.pop('n_epochs')
    if not isinstance(n_epochs_values, list):
        n_epochs_values = [n_epochs_values]
    fp_size_values = cfg.pop('fp_size', [256])
    if not isinstance(fp_size_values, list):
        fp_size_values = [fp_size_values]
    physchem_values = cfg.pop('physchem', ['full'])
    if not isinstance(physchem_values, list):
        physchem_values = [physchem_values]
    batch_size_values = cfg.pop('batch_size', [None])
    if not isinstance(batch_size_values, list):
        batch_size_values = [batch_size_values]
    n_jobs_values = cfg.pop('n_jobs', [None])
    if not isinstance(n_jobs_values, list):
        n_jobs_values = [n_jobs_values]
    from itertools import product
    for dataset_name, split, n_epochs, fp_size, physchem, batch_size, n_jobs in product(
        datasets, splits, n_epochs_values, fp_size_values, physchem_values,
        batch_size_values, n_jobs_values
    ):
        data_dir = f'./data/{dataset_name}'
        print(f'  dataset: {dataset_name}  split: {split}  n_epochs: {n_epochs}  '
              f'fp_size: {fp_size}  physchem: {physchem}  batch_size: {batch_size}  '
              f'n_jobs: {n_jobs}')
        run(**cfg, n_epochs=n_epochs, split=split, fp_size=fp_size, data_dir=data_dir,
            physchem=physchem, batch_size=batch_size, n_jobs=n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to a JSON config file')
    parser.add_argument('--config-dir', type=str, default=None, help='Path to a folder of JSON config files to run sequentially (default: config)')
    parser.add_argument('--split', type=str, default=None, help='Run only this split, overriding the splits in the config')
    args = parser.parse_args()

    config_dir = args.config_dir
    if config_dir is None and args.config is None:
        config_dir = 'config'

    if config_dir:
        config_files = sorted(f for f in os.listdir(config_dir) if f.endswith('.json'))
        for filename in config_files:
            print(f'Running config: {filename}')
            with open(os.path.join(config_dir, filename)) as f:
                cfg = json.load(f)
            run_config(cfg, args.split)
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        run_config(cfg, args.split)
