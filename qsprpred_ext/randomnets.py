# Integration of RandomNets within QSPRpred.

from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
from pytorch_lightning import loggers as pl_loggers
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.models import QSPRModel
from qsprpred.models.early_stopping import EarlyStoppingMode
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


### QSPRPRED PART ###
class Randomnets(QSPRModel):
    def __init__(
        self,
        base_dir: str,
        name: str | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
        autoload: bool = True,
        patience: int = 50,
        tol: float = 0,
        target_column: str = 'pchembl_value'
    ):
        self.patience = patience
        self.tol = tol
        self.nDim = None
        self.target_column = target_column
        super().__init__(
            base_dir,
            Randomnets,
            name,
            parameters,
            autoload=autoload,
            random_state=random_state,
        )
        self.parameters = parameters

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        estimator: Any | None = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        split: DataSplit | None = None,
        **kwargs,
    ):
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for this model."
            )
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y if isinstance(y, pd.DataFrame) else pd.DataFrame(y)

        X.to_parquet(self.outDir + '/train_X.parquet')
        y.to_parquet(self.outDir + '/train_y.parquet')
        if X.isna().any().any():
            raise ValueError("Randomnets model cannot handle nan values.")

        n_nns = self.parameters.get('n_nns', 25) if self.parameters else 25
        batch_size = self.parameters.get('batch_size', 8) if self.parameters else 8
        seed = getattr(self, "randomState", None)
        seed = seed if seed is not None else 0
        dataModule = QSPRpredDataModule(self.outDir + '/train_X.parquet', self.outDir + '/train_y.parquet', batch_size, n_nns, 0, True, 0.1, seed, self.target_column)
        dataModule.setup(None)

        validate = self.parameters.get('validate', True) if self.parameters else True

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=self.outDir, default_hp_metric=None
        )
        self.trainer = pytorch_lightning.Trainer(
            max_epochs=estimator.max_epochs,
            devices="auto",
            log_every_n_steps=10,
            logger=tb_logger,
            enable_progress_bar=False,
            limit_val_batches=1.0 if validate else 0,
        )

        self.trainer.fit(estimator, dataModule)

        return estimator

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRTable,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        scores = self.predictProba(X, estimator)
        if self.task.isClassification():
            raise ValueError("Classification models not supported at present.")
        else:
            return scores[0]

    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRTable,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        dl = DataLoader(
            InDataset(X),
            batch_size=8,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

        result, _ = estimator.predict_dataloader(dl, std=True)

        return [result.numpy().reshape((-1, 1))]

    def initFromData(self, data: QSPRTable | None, pipeline: Any = None):
        """Initialize the model from dataset and pipeline."""
        super().initFromData(data, pipeline)
        if pipeline is not None and pipeline.featureNames is not None:
            self.nDim = len(pipeline.featureNames)
        else:
            self.nDim = len(data.getDescriptorNames())

    def loadEstimator(self, params: dict | None = None) -> object:
        if self.nDim is None:
            return "Uninitialized model."
        n_epochs = 5
        n_hidden_layers = 1
        dim = 1024
        max_lr = 1e-4
        n_nns = 25
        mask_thr = 0.5
        # QSPRModel.initRandomState stores the seed on self.randomState and also
        # copies it into self.parameters; honour either so the estimator is seeded.
        random_state = getattr(self, "randomState", None)
        if params is not None:
            n_epochs = params['n_epochs'] if 'n_epochs' in params else n_epochs
            n_hidden_layers = params['n_hidden_layers'] if 'n_hidden_layers' in params else n_hidden_layers
            dim = params['dim'] if 'dim' in params else dim
            max_lr = params['max_lr'] if 'max_lr' in params else max_lr
            n_nns = params['n_nns'] if 'n_nns' in params else n_nns
            mask_thr = params['mask_thr'] if 'mask_thr' in params else mask_thr
            random_state = params['random_state'] if 'random_state' in params else random_state
        estimator = RandomNetsModel(max_epochs=n_epochs, mask_thr=mask_thr,
                                n_nns=n_nns, in_dim=self.nDim, c_dim=0,
                                n_hidden_layers=n_hidden_layers, dim=dim, max_lr=max_lr,
                                random_state=random_state)
        return estimator

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        estimator = RandomNetsModel.load_from_checkpoint(self.outDir +"/final_model.ckpt")
        return estimator

    def saveEstimator(self) -> str:
        if self.trainer is not None:
            self.trainer.save_checkpoint(self.outDir +"/final_model.ckpt", weights_only=True)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False

### RANDOMNETS PART ###
# The basic dataset, QSPRpredDataModule that gets converted to this. Taken from the
# RandomNets repository
class FpsDataset(Dataset):
    def __init__(
        self,
        data,
        target_column="pXC50",
        feature_column="fps",
        sample_mask_column="sample_mask",
        invert_mask=False,
    ):
        self.data = data
        self.target_column = target_column
        self.feature_column = feature_column
        self.sample_mask_column = sample_mask_column
        self.invert_mask = invert_mask
        self._max_n = np.max(np.stack(self.data.sample_mask.values)) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp1 = torch.tensor(self.data[self.feature_column].iloc[idx], dtype=torch.float)
        y1 = torch.tensor(self.data[self.target_column].iloc[idx], dtype=torch.float)

        sample_mask = self.data[self.sample_mask_column].iloc[idx]

        full_set = set(range(self._max_n))

        if self.invert_mask:
            sample_mask = list(full_set - set(sample_mask))

        sample_mask = torch.tensor(sample_mask, dtype=torch.int64) + 0

        return fp1, y1, sample_mask

# A wrapper around dataframe or nparray for use in predict
class InDataset(Dataset):
    def __init__(
        self,
        data
    ):
        self.data = data.copy(deep=True)
        fps = self.data.to_numpy().astype(float)
        self.data["fps"] = [arr for arr in fps]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp1 = torch.tensor(self.data["fps"].iloc[idx], dtype=torch.float)
        return fp1, [], []

def create_balanced_mask(n_features, n_ensemble_members, mask_thr=0.5):
    mask = torch.zeros((n_features, n_ensemble_members), dtype=bool)
    n_features_per_member = int(n_features * (1 - mask_thr))

    # Initialize pool of available indices
    available_indices = list(range(n_features))

    # For each ensemble member
    for member in range(n_ensemble_members):
        # Simple if we have more available than needed to sample
        if len(available_indices) >= n_features_per_member:
            sampled_indices = np.random.choice(
                available_indices, size=n_features_per_member, replace=False
            )
            for idx in sampled_indices:
                if idx in available_indices:
                    available_indices.remove(idx)
        else:
            # Otherwise we need to sample more cleverly
            sampled_indices = available_indices.copy()  # Empty the available indices
            # Refill the available indices.
            available_indices = list(range(n_features))

            # Make a sample buffer to sample from
            sample_buffer = available_indices.copy()

            # Remove the already sampled indices from the sample buffer
            for idx in sampled_indices:
                if idx in sample_buffer:
                    sample_buffer.remove(idx)

            # Sample from the sample buffer
            sampled_indices_fillup = np.random.choice(
                sample_buffer,
                size=n_features_per_member - len(sampled_indices),
                replace=False,
            )

            # remove the fillup indices from the available indices
            for idx in sampled_indices_fillup:
                if idx in available_indices:
                    available_indices.remove(idx)

            # Concatenate the two sets of sampled indices
            sampled_indices = np.concatenate([sampled_indices, sampled_indices_fillup])

        # Refill if completely empty
        if len(available_indices) == 0:
            available_indices = list(range(n_features))
        mask[sampled_indices, member] = True

    return mask

# Actual implementation of the RandomNetsModel. Taken verbatim from the RandomNets repository
# TODO: import this from the repo, remove from this file (see refactor/use_randomnets_pip_install)
class RandomNetsModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mask_thr: float = 0.5,  # 0 is no mask, 1.0 is drop all input!
        dim: int = 1024,
        in_dim: int = 4096,
        c_dim: int = 0,
        n_nns: int = 25,
        max_lr: float = 1e-4,
        dropout: float = 0.25,
        n_hidden_layers: int = 1,
        max_epochs: int = 10,
        balanced_mask: bool = False,
        random_state: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mask_thr = mask_thr
        self.dim = dim
        self.in_dim = in_dim
        self.c_dim = c_dim
        self.n_nns = n_nns
        self.max_lr = max_lr
        self.dropout = dropout
        self.n_hidden_layers = n_hidden_layers
        self.max_epochs = max_epochs
        self.balanced_mask = balanced_mask
        self.random_state = random_state
        # Seed before building the model so the random input mask (torch.rand),
        # the weight initialisation and any numpy-based masking are reproducible.
        # Mirrors QSPRpred's convention (STFullyConnected in neural_network.py),
        # so a fixed random_state reproduces a run and distinct seeds give
        # distinct models (needed for triplicate replicates).
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.create_input_mask()
        self.create_layers()
        self.float()

    def create_input_mask(self):
        if self.balanced_mask:
            input_mask = create_balanced_mask(self.in_dim, self.n_nns, self.mask_thr)
        else:
            random_tensor = torch.rand(
                (
                    self.in_dim - self.c_dim,
                    self.n_nns,
                ),
                requires_grad=False,
            )
            constant_tensor = torch.ones(
                (
                    self.c_dim,
                    self.n_nns
                )
            )

            mask_tensor = torch.cat((random_tensor, constant_tensor))
            input_mask = (
                mask_tensor > self.mask_thr
            )  # If mask_the is 0, all is included
        self.register_buffer(
            "input_mask",
            input_mask,
            persistent=True,
        )  # TODO Some features may not be used at all? Can this be done more "intelligent?", e.g. chance of getting selected for next tree is related to if feature is already used?

    def create_layers(self):
        self.dropout_fn = nn.Dropout(self.dropout)
        self.activation_fn = nn.LeakyReLU()  # nn.ReLU()

        # Conv1D is (N,Cin,L), N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        # So if FP is channels, hidden is Cout, L is n_nns, and stride and kernel_size should be 1.
        self.embedding_nn = nn.Conv1d(
            self.in_dim,  # + self.id_embedding_dim,
            self.dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.embedding = nn.Sequential(
            self.embedding_nn, self.activation_fn, self.dropout_fn
        )  # Out is samples, hidden_dim, n_nns

        hidden_stack = []
        for i in range(self.n_hidden_layers - 1):  # Embedding counts for one
            hidden_nn = nn.Conv1d(
                self.dim, self.dim, stride=1, kernel_size=1, padding=0
            )  # samples, hidden_dim, n_nns
            hidden_stack.extend([hidden_nn, self.activation_fn, self.dropout_fn])
        self.FF = nn.Sequential(*hidden_stack)

        self.predict_nn = nn.Conv1d(self.dim, 1, kernel_size=1, padding=0, stride=1)

    def get_fp_masks(self, sample_mask):
        sample_mask_reshaped = sample_mask.unsqueeze(1).expand(
            -1, self.in_dim, -1
        )  # -> samples, fp_size, n_sel_nns

        input_mask_reshaped = self.input_mask.unsqueeze(0).expand(
            sample_mask.shape[0], -1, -1
        )  # -> samples, fp_size, n_nns

        fp_mask_gathered = torch.gather(
            input_mask_reshaped, 2, sample_mask_reshaped
        )  # Gathers according to indices -> samples, fp_size, n_sel_nns

        return fp_mask_gathered

    def forward(self, fp, sample_mask):
        fp_mask = self.get_fp_masks(sample_mask)

        n_ensemble_sel = sample_mask.shape[1]
        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_ensemble_sel
        )  # -> samples, fp_size, n_nns_sel
        fp_masked = fp_reshaped * fp_mask

        emb_o = self.embedding(fp_masked)

        ff_o = self.FF(emb_o)
        y_hats = self.predict_nn(ff_o)

        return y_hats.squeeze(
            dim=1
        )  # Dims are then samples, n_nns_sel #This fails if n_nns_sel is one!

    def get_loss(self, batch):
        fp, y, sample_mask = batch

        # Add n_nns dim and repeat y n_nns_sel times
        ys = torch.unsqueeze(y, 1).expand(-1, sample_mask.shape[1]).detach()

        y_hats = self.forward(fp, sample_mask)

        loss = torch.nn.functional.mse_loss(y_hats, ys)

        y_hat_ensemble = (y_hats).mean(axis=1)
        ensemble_loss = torch.nn.functional.mse_loss(y_hat_ensemble, y).detach()
        ensemble_std = (y_hats).std(axis=1).mean().detach()

        return loss, ensemble_loss, ensemble_std

    def training_step(self, batch, batch_idx):
        self.train()
        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
        self.log("train_mse_loss", loss)
        return loss

#    def validation_step(self, batch, batch_idx):
#        self.eval()
#        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
#        self.log("val_mse_loss", loss)
#        self.log("val_mse_ensemble_loss", ensemble_loss)
#        self.log("val_mse_ensemble_std", ensemble_std)
#        self.log("hp_metric", ensemble_loss)
#        return loss

#    def test_step(self, batch, batch_idx):
#        self.eval()
#        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
#        self.log("mse_loss", loss)
#        self.log("mse_ensemble_loss", ensemble_loss)
#        self.log("mse_ensemble_std", ensemble_std)
#        return loss

    def configure_optimizers(self):
        # OneCycleLR overrides the param-group lr, so the base lr here is just a
        # placeholder until the scheduler takes over.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        dataloader = (
            # self.trainer._data_connector._train_dataloader_source.dataloader()
            self.trainer._data_connector._datahook_selector.datamodule.train_dataloader()
            # dm.train_dataloader()  # TODO, this needs to be better integrated!
        )  # = self.train_dataloader()

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=None,
            epochs=self.max_epochs,
            steps_per_epoch=len(
                dataloader
            ),  # We call train_dataloader, just to get the length, is this necessary?
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def predict(self, fp, std=False):
        self.eval()

        sample_mask = torch.tensor(
            [range(self.n_nns)] * fp.shape[0], dtype=torch.int64
        )  # Use Full Ensemble

        y_hats = self.forward(fp, sample_mask=sample_mask).detach()

        # take the mean and std along n_nns
        if std:
            return y_hats.mean(axis=1), y_hats.std(axis=1)
        else:
            return y_hats.mean(axis=1)

    def predict_dataloader(self, dataloader, std=False):
        self.eval()
        y_hats = []
        y_hat_stds = []
        for batch in dataloader:
            result = self.predict(batch[0], std=std)
            if (std):
                y_hat_batch, y_hat_std_batch = result
                y_hats.append(y_hat_batch)
                y_hat_stds.append(y_hat_std_batch)
            else:
                y_hat_batch = result
                y_hats.append(y_hat_batch)

        if std:
            return torch.cat(y_hats), torch.cat(y_hat_stds)
        else:
            return torch.cat(y_hats)

# QSPRpred version of FpsDataModule. To be used with the wrapper, the
# idea is that fingerprint preparation is done in QSPRpred and passed
# verbatim into this class
class QSPRpredDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        file_X,
        file_y,
        batch_size,
        n_nns=25,
        sample_mask_thr=0.0,
        dedicated_val=True,
        val_sample_size=0.25,
        random_seed = 0,
        target_column = 'pchembl_value'
    ):
        super().__init__()
        self.file_X = file_X
        self.file_y = file_y
        self.batch_size = batch_size
        self.n_nns = n_nns
        self.sample_mask_thr = sample_mask_thr
        self.dedicated_val = dedicated_val
        self.val_sample_size = val_sample_size
        self.random_seed = random_seed
        self.features_column = "fps"
        self.sample_mask_column = "sample_mask"
        self.target_column = target_column
        self.num_worker = 4
        self.random_seed = random_seed
        self.save_hyperparameters()

    def setup(self, stage):
        if hasattr(self, "data"):
            return
        X = pd.read_parquet(self.file_X)
        y = pd.read_parquet(self.file_y)

        self.data = pd.concat([X, y], axis=1)
        if self.data.isnull().values.any():
            raise ValueError("Null values in dataset; not supported by the randomnets algorithm.")
        # Convert columns to numpy arrays, validation stuff
        fps = X.to_numpy().astype(float)
        self.data[self.features_column] = [arr for arr in fps]

        # prepare model_mask
        # This is the indices of the ensemble ids to associate with each sample.
        # Seed numpy first so the per-sample ensemble masks are reproducible for a
        # given random_seed (only introduces variability when sample masking is on).
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.data[self.sample_mask_column] = [
            np.sort(np.random.choice(
               range(self.n_nns),
               max(
                   int(self.n_nns * (1 - self.sample_mask_thr)), 1
               ),  # Ensure always one selected
               replace=False,
            ))
            for _ in range(len(self.data))
        ]
        self.data_train = self.data

    def train_dataloader(self, shuffle=True):
        dataset = FpsDataset(
            self.data_train,
            feature_column=self.features_column,
            target_column=self.target_column,
            sample_mask_column=self.sample_mask_column,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=shuffle,
            drop_last=True,
        )
