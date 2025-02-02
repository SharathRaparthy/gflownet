import pathlib
from typing import Any, Dict, List, NewType, Optional, Tuple

from rdkit.Chem.rdchem import Mol as RDMol
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.utils.multiprocessing_proxy import wrap_model_mp

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType('FlatRewards', Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType('RewardScalar', Tensor)  # type: ignore


class GFNAlgorithm:
    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch,
                             num_bootstrap: Optional[int] = 0) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations
        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).
        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[RewardScalar, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar reward for each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], device: torch.device):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `default_hps` method.
        device: torch.device
            The torch device of the main worker.
        """
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm

        # Override default hyperparameters with the constructor arguments
        self.hps = {**self.default_hps(), **hps}
        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.hps.get('num_data_loader_workers', 0)
        # The ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        # `self.sampling_model`.
        self.offline_ratio = 0.5
        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio = 1
        self.verbose = False
        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers.  """
        model.to(self.device)
        if self.num_workers > 0:
            placeholder = wrap_model_mp(model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical))
            return placeholder, torch.device('cpu')
        return model, self.device

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        iterator = SamplingIterator(self.training_data, model, self.mb_size * 2, self.ctx, self.algo, self.task, dev,
                                    ratio=self.offline_ratio, log_dir=self.hps['log_dir'])
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        iterator = SamplingIterator(self.test_data, model, self.mb_size, self.ctx, self.algo, self.task, dev,
                                    ratio=self.valid_offline_ratio, stream=False)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch, num_bootstrap=self.mb_size)
        self.step(loss)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch, num_bootstrap=batch.num_offline)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def run(self):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        for it, batch in zip(range(1, 1 + self.hps['num_training_steps']), train_dl):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)
            if self.verbose:
                print(it, ' '.join(f'{k}:{v:.2f}' for k, v in info.items()))
            self.log(info, it, 'train')

            if it % self.hps['validate_every'] == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, 'valid')
                torch.save({
                    'models_state_dict': [self.model.state_dict()],
                    'hps': self.hps,
                }, open(pathlib.Path(self.hps['log_dir']) / 'model_state.pt', 'wb'))

    def log(self, info, index, key):
        if not hasattr(self, '_summary_writer'):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.hps['log_dir'])
        for k, v in info.items():
            self._summary_writer.add_scalar(f'{key}_{k}', v, index)
