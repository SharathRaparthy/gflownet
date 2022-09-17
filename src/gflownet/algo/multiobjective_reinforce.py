import copy
from itertools import count
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.data as gd
from torch_scatter import scatter

from gflownet.envs.graph_building_env import generate_forward_trajectory
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance, TrajectoryBalanceModel


class MultiObjectiveReinforce(TrajectoryBalance):
    """
    Class that inherits from TrajectoryBalance and implements the multi-objective reinforce algorithm
    """
    def __init__(self, env: GraphBuildingEnv, ctx: GraphBuildingEnvContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None):
        super().__init__(env, ctx, rng, hps, max_len, max_nodes)

    def compute_batch_losses(self, model: TrajectoryBalanceModel, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute  multi objective REINFORCE loss over trajectories contained in the batch """
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = batch.rewards
        cond_info = batch.cond_info

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and the optional bootstrap predictions
        fwd_cat, log_reward_preds = model(batch, cond_info[batch_idx])

        # This is the log prob of each action in the trajectory
        log_prob = fwd_cat.log_prob(batch.actions)

        # Take log rewards, and clip
        assert rewards.ndim == 1
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')

        traj_losses = traj_log_prob * (-rewards - (-1) * rewards.mean())

        loss = traj_losses.mean()
        info = {
            'loss': loss.item(),
        }
        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info

    @staticmethod
    def process_reward(reward):
        """Process the reward to be used in the loss function"""
        return reward