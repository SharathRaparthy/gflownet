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


class TrajectoryBalanceModel(nn.Module):
    def forward(self, batch: gd.Batch) -> Tuple[GraphActionCategorical, Tensor]:
        raise NotImplementedError()

    def logZ(self, cond_info: Tensor) -> Tensor:
        raise NotImplementedError()


class TrajectoryBalance:
    """
    """
    def __init__(self, env: GraphBuildingEnv, ctx: GraphBuildingEnvContext, rng: np.random.RandomState,
                 hps: Dict[str, Any], max_len=None, max_nodes=None):
        """TB implementation, see
        "Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
        Emmanuel Bengio, Chen Sun, Yoshua Bengio"
        https://arxiv.org/abs/2201.13259

        Hyperparameters used:
        random_action_prob: float, probability of taking a uniform random action when sampling
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions
        bootstrap_own_reward: bool, if True, uses the .reward batch data to predict rewards for sampled data
        tb_epsilon: float, if not None, adds this epsilon in the numerator and denominator of the log-ratio
        reward_loss_multiplier: float, multiplying constant for the bootstrap loss.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        hps: Dict[str, Any]
            Hyperparameter dictionary, see above for used keys.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        max_nodes: int
            If not None, ends trajectories of graphs with more than max_nodes steps (illegal action).
        """
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.random_action_prob = hps['random_action_prob']
        self.illegal_action_logreward = hps['illegal_action_logreward']
        self.bootstrap_own_reward = hps['bootstrap_own_reward']
        self.sanitize_samples = True
        self.epsilon = hps['tb_epsilon']
        self.reward_loss_multiplier = hps['reward_loss_multiplier']
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.length_normalize_losses = False
        self.sample_temp = 1

    def _corrupt_actions(self, actions: List[Tuple[int, int, int]], cat: GraphActionCategorical):
        """Sample from the uniform policy with probability `self.random_action_prob`"""
        # Should this be a method of GraphActionCategorical?
        if self.random_action_prob <= 0:
            return
        corrupted, = (self.rng.uniform(size=len(actions)) < self.random_action_prob).nonzero()
        for i in corrupted:
            n_in_batch = [int((b == i).sum()) for b in cat.batch]
            n_each = np.array([float(logit.shape[1]) * nb for logit, nb in zip(cat.logits, n_in_batch)])
            which = self.rng.choice(len(n_each), p=n_each / n_each.sum())
            row = self.rng.choice(n_in_batch[which])
            col = self.rng.choice(cat.logits[which].shape[1])
            actions[i] = (which, row, col)

    def create_training_data_from_own_samples(self, model: TrajectoryBalanceModel, n: int, cond_info: Tensor):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        ctx = self.ctx
        env = self.env
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        logZ_pred = model.logZ(cond_info)
        # This will be returned as training data
        data = [{'traj': [], 'reward_pred': None, 'is_valid': True} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        zero = torch.tensor([0], device=dev).float()
        fwd_logprob: List[List[Tensor]] = [[] for i in range(n)]
        bck_logprob: List[List[Tensor]] = [[zero] for i in range(n)]  # zero in case there is a single invalid action

        graphs = [env.new() for i in range(n)]
        done = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        # TODO report these stats:
        mol_too_big = 0
        mol_not_sane = 0
        invalid_act = 0
        logprob_of_illegal: List[Tensor] = []

        illegal_action_logreward = torch.tensor([self.illegal_action_logreward], device=dev)
        if self.epsilon is not None:
            epsilon = torch.tensor([self.epsilon], device=dev).float()
        for t in (range(self.max_len) if self.max_len is not None else count(0)):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            fwd_cat, log_reward_preds = model(ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if self.sample_temp != 1:
                sample_cat = copy.copy(fwd_cat)
                sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                actions = sample_cat.sample()
            else:
                actions = fwd_cat.sample()
            self._corrupt_actions(actions, fwd_cat)
            graph_actions = [ctx.aidx_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = fwd_cat.log_prob(actions)
            for i, j in zip(not_done(range(n)), range(n)):
                # Step each trajectory, and accumulate statistics
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]['traj'].append((graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop or t == self.max_len - 1:
                    done[i] = True
                    if self.sanitize_samples and not ctx.is_sane(graphs[i]):
                        # check if the graph is sane (e.g. RDKit can
                        # construct a molecule from it) otherwise
                        # treat the done action as illegal
                        mol_not_sane += 1
                        data[i]['reward_pred'] = illegal_action_logreward.exp()
                        data[i]['is_valid'] = False
                    elif self.bootstrap_own_reward:
                        # if we're bootstrapping, extract reward prediction
                        data[i]['reward_pred'] = log_reward_preds[j].detach().exp()
                else:  # If not done, try to step the environment
                    gp = graphs[i]
                    try:
                        # env.step can raise AssertionError if the action is illegal
                        gp = env.step(graphs[i], graph_actions[j])
                        if self.max_nodes is not None:
                            assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        if len(gp.nodes) > self.max_nodes:
                            mol_too_big += 1
                        else:
                            invalid_act += 1
                        done[i] = True
                        data[i]['reward_pred'] = illegal_action_logreward.exp()
                        data[i]['is_valid'] = False
                        continue
                    # Add to the trajectory
                    # P_B = uniform backward
                    n_back = env.count_backward_transitions(gp)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    graphs[i] = gp
            if all(done):
                break

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward flows
            data[i]['logZ'] = logZ_pred[i].item()
            data[i]['fwd_logprob'] = sum(fwd_logprob[i])
            data[i]['bck_logprob'] = sum(bck_logprob[i])
            if self.bootstrap_own_reward and False:  # TODO: verify
                if not data[i]['is_valid']:
                    logprob_of_illegal.append(data[i]['fwd_logprob'].item())
                # If we are bootstrapping, we can report the theoretical loss as well
                numerator = data[i]['fwd_logprob'] + logZ_pred[i]
                denominator = data[i]['bck_logprob'] + data[i]['reward_pred'].log()
                if self.epsilon is not None:
                    numerator = torch.logaddexp(numerator, epsilon)
                    denominator = torch.logaddexp(denominator, epsilon)
                data[i]['loss'] = (numerator - denominator).pow(2)
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{'traj': generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        rewards: Tensor
            The transformed reward (e.g. R(x) ** beta) for each trajectory. Shape (N,)

        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj['traj']]
        actions = [
            self.ctx.GraphAction_to_aidx(g, a) for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj['traj']])
        ]
        num_backward = torch.tensor([
            # Count the number of backward transitions from s_{t+1},
            # unless t+1 = T is the last time step
            self.env.count_backward_transitions(tj['traj'][i + 1][0]) if i + 1 < len(tj['traj']) else 1
            for tj in trajs
            for i in range(len(tj['traj']))
        ])
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i['traj']) for i in trajs])
        batch.num_backward = num_backward
        batch.actions = torch.tensor(actions)
        batch.rewards = rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get('is_valid', True) for i in trajs]).float()
        return batch

    def compute_batch_losses(self, model: TrajectoryBalanceModel, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
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

        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = log_reward_preds[final_graph_idx, 0]
        # Compute trajectory balance objective
        Z = model.logZ(cond_info)[:, 0]
        # This is the log prob of each action in the trajectory
        log_prob = fwd_cat.log_prob(batch.actions)
        # The log prob of each backward action
        log_p_B = (1 / batch.num_backward).log()
        # Take log rewards, and clip
        Rp = torch.maximum(rewards.log(), torch.tensor(-100.0, device=dev))
        # This is the log probability of each trajectory
        traj_log_prob = scatter(log_prob, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')
        # Compute log numerator and denominator of the TB objective
        numerator = Z + traj_log_prob
        denominator = Rp + scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce='sum')

        if self.epsilon is not None:
            # Numerical stability epsilon
            epsilon = torch.tensor([self.epsilon], device=dev).float()
            numerator = torch.logaddexp(numerator, epsilon)
            denominator = torch.logaddexp(denominator, epsilon)

        invalid_mask = 1 - batch.is_valid
        if self.mask_invalid_rewards:
            # Instead of being rude to the model and giving a
            # logreward of -100 what if we say, whatever you think the
            # logprobablity of this trajetcory is it should be smaller
            # (thus the `numerator - 1`). Why 1? Intuition?
            denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

        if self.tb_loss_is_mae:
            traj_losses = abs(numerator - denominator)
        elif self.tb_loss_is_huber:
            pass  # TODO
        else:
            traj_losses = (numerator - denominator).pow(2)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens

        if self.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(rewards)
            if self.reward_loss_is_mae:
                reward_losses = abs(rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp())
            else:
                reward_losses = (rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap].exp()).pow(2)
            reward_loss = reward_losses.mean()
        else:
            reward_loss = 0

        loss = traj_losses.mean() + reward_loss * self.reward_loss_multiplier
        info = {
            'offline_loss': traj_losses[:batch.num_offline].mean(),
            'online_loss': traj_losses[batch.num_offline:].mean() if batch.num_online > 0 else 0,
            'reward_loss': reward_loss,
            'invalid_trajectories': invalid_mask.mean() * 2,
            'invalid_logprob': (invalid_mask * traj_log_prob).sum() / (invalid_mask.sum() + 1e-4),
            'invalid_losses': (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            'logZ': Z.mean(),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError('loss is not finite')
        return loss, info
