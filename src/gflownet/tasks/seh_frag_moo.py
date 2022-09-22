import sys
import ast
from typing import Any, Callable, Dict, List, Tuple, Union
import wandb
import yaml
import random
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
import scipy.stats as stats
import torch
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import RewardScalar
from gflownet.utils import metrics
from gflownet.utils import sascore
from gflownet.utils.transforms import thermometer

import warnings
warnings.filterwarnings("ignore")

class SEHMOOTask(GFNTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 const_temp: int, dirichlet_param: float, num_objectives: int, wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.const_temp = const_temp
        self.dirichlet_param = dirichlet_param
        self.num_objectives = num_objectives

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        return {'seh': model}

    def sample_conditional_information(self, n):
        beta = None
        use_funky_dirichlet = False
        upper_bound = None

        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            beta = self.rng.gamma(loc, scale, n).astype(np.float32)
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        elif self.temperature_sample_dist == 'const':
            beta = np.ones(n).astype(np.float32)
            beta = beta * self.const_temp
            upper_bound = self.const_temp
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
        if not use_funky_dirichlet:
            m = Dirichlet(torch.FloatTensor([self.dirichlet_param] * self.num_objectives))
            preferences = m.sample([n])
        else:
            a = np.random.dirichlet([1] * self.num_objectives, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards, mo_baseline: bool) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        if not mo_baseline:
            scalar_reward = (flat_reward * cond_info['preferences']).sum(1)
        else:
            scalar_reward = (torch.abs(-flat_reward - torch.zeros(flat_reward.shape[1])) * cond_info['preferences']).max(1)[0]
        return scalar_reward**cond_info['beta']

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, self.num_objectives))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
        seh_preds[seh_preds.isnan()] = 0

        def safe(f, x, default):
            try:
                return f(x)
            except Exception:
                return default

        qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
        sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
        sas = (10 - sas) / 9  # Turn into a [0-1] reward
        molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
        molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
        objectives = [seh_preds, qeds, sas, molwts][:self.num_objectives]
        flat_rewards = torch.stack(objectives, 1)
        return FlatRewards(flat_rewards), is_valid


class SEHMOOFragTrainer(SEHFragTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            **super().default_hps(),
            'use_fixed_weight': False,
            'num_cond_dim': 32,  # thermometer encoding of beta + 4 preferences
        }

    def setup(self):
        super().setup()
        self.task = SEHMOOTask(
            self.training_data,
            self.hps['temperature_sample_dist'],
            ast.literal_eval(self.hps['temperature_dist_params']),
            dirichlet_param=self.hps['dirichlet_param'],
            const_temp=self.hps['const_temp'],
            num_objectives=self.hps['num_objectives'],
            wrap_model=self._wrap_model_mp
        )
        self.sampling_hooks.append(MultiObjectiveStatsHook(256))


class MultiObjectiveStatsHook:
    def __init__(self, num_to_keep: int):
        self.num_to_keep = num_to_keep
        self.all_flat_rewards: List[Tensor] = []
        self.hsri_epsilon = 0.3

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        self.all_flat_rewards = self.all_flat_rewards + list(flat_rewards)
        if len(self.all_flat_rewards) > self.num_to_keep:
            self.all_flat_rewards = self.all_flat_rewards[-self.num_to_keep:]

        flat_rewards = torch.stack(self.all_flat_rewards).numpy()
        target_min = flat_rewards.min(0).copy()
        target_range = flat_rewards.max(0).copy() - target_min
        hypercube_transform = metrics.Normalizer(
            loc=target_min,
            scale=target_range,
        )
        gfn_pareto = metrics.pareto_frontier(flat_rewards)
        normed_gfn_pareto = hypercube_transform(gfn_pareto)
        hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=True)
        hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=False)
        unnorm_hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=True)
        unnorm_hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=False)

        upper = np.zeros(normed_gfn_pareto.shape[-1]) + self.hsri_epsilon
        lower = np.ones(normed_gfn_pareto.shape[-1]) * -1 - self.hsri_epsilon
        hsr_indicator = metrics.HSR_Calculator(lower, upper)
        try:
            hsri_w_pareto, x = hsr_indicator.calculate_hsr(-1 * gfn_pareto)
        except Exception:
            hsri_w_pareto = 0
        try:
            hsri_on_flat, _ = hsr_indicator.calculate_hsr(-1 * flat_rewards)
        except Exception:
            hsri_on_flat = 0

        return {
            'HV with zero ref': hypervolume_with_zero_ref,
            # 'HV w/o zero ref': hypervolume_wo_zero_ref,
            # 'Unnormalized HV with zero ref': unnorm_hypervolume_with_zero_ref,
            # 'Unnormalized HV w/o zero ref': unnorm_hypervolume_wo_zero_ref,
            # 'hsri_with_pareto': hsri_w_pareto,
            # 'hsri_on_flat_rew': hsri_on_flat,
        }


def set_seed(seed: int):
    # Seed seeds for reproducibility of experiments
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)




def main():
    """Example of how this model can be run outside of Determined"""
    default_hps = {
        'lr_decay': 10000,
        'log_dir': 'logs/seh_frag_moo/run_1/',
        'num_training_steps': 20_000,
        'validate_every': 500,
        'sampling_tau': 0.95,
        'num_layers': 6,
        'num_data_loader_workers': 12,
        'temperature_dist_params': '(0, 32)',
        'const_temp': 32,
        'temperature_sample_dist': 'const',
        'num_emb': 64,
        'global_batch_size': 64,
        'learning_rate': 0.0001,
        'dirichlet_param': 1.5,
        'num_objectives': 2,
        'experiment_name': '',
        'use_wandb': False,
        'hp_search': False,
        'baseline_training': True
    }
    if default_hps['hp_search']:
        with open("config/hp_sweep.yaml", "r") as stream:
            hp_dict = yaml.safe_load(stream)
    else:
        if default_hps['baseline_training']:

            with open("configs/mo_reinforce_hp.yaml", "r") as stream:
                hp_dict = yaml.safe_load(stream)
        else:
            with open("configs/seh_moo_hp.yaml", "r") as stream:
                hp_dict = yaml.safe_load(stream)
    if default_hps['use_wandb']:
        wandb.init(project='mo-gfn', config=hp_dict, name='seh_frag_moo | number of objectives: ' + str(default_hps['num_objectives']))
    hps = {**default_hps, **hp_dict}
    if default_hps['baseline_training']:
        hps['experiment_name'] = f'seh_frag_moo_baseline/{hps["num_objectives"]}_obj/{hps["seed"]}'
    else:
        hps['experiment_name'] = f'seh_frag_moo/{hps["num_objectives"]}_obj/{hps["seed"]}'
    hps['log_dir'] = hps['log_dir'] + hps["experiment_name"] + "/"
    set_seed(hps['seed'])
    trial = SEHMOOFragTrainer(hps, torch.device('cuda'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    main()
    sys.exit()
