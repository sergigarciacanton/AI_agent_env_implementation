# DEPENDENCIES
import math
import random
import sys
import time
from collections import deque
from itertools import permutations
from typing import Deque, Dict, List, Tuple, Optional, Any, SupportsFloat
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from Utils.segmenttree import MinSegmentTree, SumSegmentTree
from environment import EnvironmentUPC
from Env_test.env_test import Env_Test
from config import NODES_2_TRAIN, MODEL_PATH, SERGI_PLOTS
from Utils.graph_upc import get_graph
import os
import re

np.set_printoptions(threshold=sys.maxsize)


# FUNCTIONS
def _plot(step: int, scores: List[float], mean_scores: List[float], losses: List[float], mean_losses: List[float],
          mean_ratio: int, save_path: str):
    """Plot the training progresses."""
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('step %s. score: %s' % (step, np.mean(scores[-mean_ratio:])))
    plt.plot(scores, label='Real')
    plt.plot(mean_scores, label='Promig')
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses, label='Real')
    plt.plot(mean_losses, label='Promig')

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def _get_n_step_info(n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool]:
    """
    Calculate n-step return, next observation, and done flag.

    Args:
    - n_step_buffer (Deque): Deque containing n-step transitions.
    - gamma (float): Discount factor for future rewards.

    Returns:
    Tuple[np.int64, np.ndarray, bool]: A tuple containing:
    - The n-step return (rew),
    - The next observation after the n-step transition (next_obs),
    - A boolean flag indicating whether the episode is done (done).
    """

    # Extract the information of the last transition in the n-step buffer
    rew, next_obs, done = n_step_buffer[-1][-3:]

    # Iterate over the n-step buffer in reverse (excluding the last transition)
    for transition in reversed(list(n_step_buffer)[:-1]):
        # Extract the reward, next observation, and done flag from the current transition
        r, n_o, d = transition[-3:]
        # Update the cumulative reward using the n-step return formula
        rew = r + gamma * rew * (1 - d)
        # If the current transition is done, update next_obs and done accordingly
        next_obs, done = (n_o, d) if d else (next_obs, done)

    return rew, next_obs, done


def get_highest_score_model():
    """
    Get the file path of the model with the highest score from a directory.

    Returns:
        str: The file path of the model with the highest score.

    Raises:
        FileNotFoundError: If no model files are found in the specified directory.
        Exception: If an unexpected error occurs during the process.

    Note:
        Make sure to handle the returned exceptions appropriately when calling this method.

    """
    try:
        # Get a list of model files in the directory
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(".pt")]

        if not model_files:
            raise FileNotFoundError(f"No model files found in {MODEL_PATH}")

        # Extract scores from file names and find the highest score
        scores = [float(re.search(r'-?\d+\.\d+', file).group()) for file in model_files]
        highest_score = max(scores)

        # Construct the file name for the model with the highest score
        highest_score_model_file = f"rainbow_{highest_score}.pt"

        # Construct and return the file path
        model_path = os.path.join(MODEL_PATH, highest_score_model_file)

        return model_path

    except FileNotFoundError as e:
        # Handle FileNotFoundError
        print(e)
    except Exception as e:
        # Handle unexpected errors during the process
        print(f"An unexpected error occurred when finding the highest score model: {e}")


def seed_torch(seed: int) -> None:
    """
    Set random seeds for reproducibility in PyTorch.

    Args:
    - seed (int): The seed value to use for randomization.

    Note:
    - This function sets the random seed for CPU and GPU (if available).
    - Disables CuDNN benchmarking and enables deterministic mode for GPU.
    """

    # Set the random seed for the CPU
    torch.manual_seed(seed)

    # Check if the CuDNN (CUDA Deep Neural Network library) is enabled
    if torch.backends.cudnn.enabled:
        # If CuDNN is enabled, set the random seed for CUDA (GPU)
        torch.cuda.manual_seed(seed)

        # Disable CuDNN benchmarking to ensure reproducibility
        torch.backends.cudnn.benchmark = False

        # Enable deterministic mode in CuDNN for reproducibility
        torch.backends.cudnn.deterministic = True


# CLASSES
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool, ) \
            -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make an n-step transition
        rew, next_obs, done = _get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self,
                 obs_dim: int,
                 size: int,
                 batch_size: int = 32,
                 alpha: float = 0.6,
                 n_step: int = 1,
                 gamma: float = 0.99, ):

        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, n_step, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool, ) -> Tuple[
        np.ndarray, np.ndarray, float, np.ndarray, bool]:

        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        priorities_alpha = priorities ** self.alpha  # Precompute priorities raised to alpha

        for idx, priority_alpha in zip(indices, priorities_alpha):
            assert priority_alpha > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority_alpha
            self.min_tree[idx] = priority_alpha

        self.max_priority = max(self.max_priority, priorities.max())

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.6, ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        # Define module parameters
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Parameters for mean values
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Parameters for standard deviations
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Register buffer for noise
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        # Initialize parameters and noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)

        # Initialize mean values
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # Initialize standard deviations
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # Outer product using torch.einsum for efficient matrix multiplication
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # Alternative 1
        # self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))  # Alternative 2
        # self.weight_epsilon.copy_(torch.einsum('i,j->ij', epsilon_out, epsilon_in))  # Alternative 3
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        linear_transform = F.linear(x,
                                    self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                    self.bias_mu + self.bias_sigma * self.bias_epsilon,
                                    )

        return linear_transform

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    #         linear_transform = F.linear(
    #             x,
    #             self.weight_mu + self.weight_sigma * self.weight_epsilon,
    #             self.bias_mu + self.bias_sigma * self.bias_epsilon,
    #         )
    #     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    #
    #     return linear_transform

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 atom_size: int,
                 support: torch.Tensor,
                 in_features: int = 128,
                 out_features: int = 128):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(nn.Linear(in_dim, out_features), nn.ReLU(), )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(in_features, out_features)
        self.advantage_layer = NoisyLinear(in_features, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(in_features, out_features)
        self.value_layer = NoisyLinear(in_features, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


# class Network(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  out_dim: int,
#                  atom_size: int,
#                  support: torch.Tensor,
#                  in_features: int = 128,
#                  out_features: int = 128):
#         """Initialization."""
#         super(Network, self).__init__()
#
#         # Set the random seed for PyTorch
#         torch.manual_seed(42)
#
#         # Set the random seed for NumPy if your code involves NumPy operations
#         np.random.seed(42)
#
#         self.support = support
#         self.out_dim = out_dim
#         self.atom_size = atom_size
#
#         # Set common feature layer
#         self.feature_layer = nn.Sequential(nn.Linear(in_dim, out_features), nn.ReLU(), )
#
#         # Set advantage layer
#         self.advantage_hidden_layer = NoisyLinear(in_features, out_features)
#         self.advantage_layer = NoisyLinear(in_features, out_dim * atom_size)
#
#         # Set value layer
#         self.value_hidden_layer = NoisyLinear(in_features, out_features)
#         self.value_layer = NoisyLinear(in_features, atom_size)
#
#         # Initialize weights
#         self.apply(self.init_weights)
#
#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.constant_(m.weight, 0)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward method implementation."""
#
#         dist = self.dist(x)
#         q = torch.sum(dist * self.support, dim=2)
#         return q
#
#     def dist(self, x: torch.Tensor) -> torch.Tensor:
#         """Get distribution for atoms."""
#
#         feature = self.feature_layer(x)
#         adv_hid = F.relu(self.advantage_hidden_layer(feature))
#         val_hid = F.relu(self.value_hidden_layer(feature))
#
#         advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
#         value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
#         q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
#
#         dist = F.softmax(q_atoms, dim=-1)
#         dist = dist.clamp(min=1e-3)  # for avoiding nans
#
#         return dist
#
#     def reset_noise(self):
#         """Reset all noisy layers."""
#         self.advantage_hidden_layer.reset_noise()
#         self.advantage_layer.reset_noise()
#         self.value_hidden_layer.reset_noise()
#         self.value_layer.reset_noise()

class RAINBOW:
    """DQN Agent interacting with environment."""

    def __init__(
            self,
            env: gym.Env,
            replay_buff_size: int,
            batch_size: int,
            target_update: int,
            in_features: int,
            out_features: int,
            seed: int = None,
            gamma: float = 0.99,
            learning_rate: float = 0.001,
            tau: float = 0.015,
            # PER parameters
            alpha: float = 0.6,
            beta: float = 0.4,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 300,
            atom_size: int = 91,
            # N-step Learning
            n_step: int = 3,

    ):
        """
        Initialize the DQNAgent.

        Parameters:
        - env (): The environment with which the agent interacts.
        - replay_buff_size (int): Size of the replay buffer.
        - batch_size (int): Batch size for training.
        - target_update (int): Frequency of updating the target network.
        - seed (int): Seed for reproducibility.
        - gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        - learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
        - tau (float, optional): Interpolation parameter for target network update. Default is 0.001.
        - alpha (float, optional): Parameter for Prioritized Experience Replay. Default is 0.2.
        - beta (float, optional): Importance sampling weight parameter. Default is 0.6.
        - prior_eps (float, optional): Small constant to avoid division by zero in priorities. Default is 1e-6.
        - v_min (float, optional): Minimum value in the support of the distribution. Default is 0.0.
        - v_max (float, optional): Maximum value in the support of the distribution. Default is 200.0.
        - atom_size (int, optional): Number of atoms in the categorical distribution. Default is 51.
        - n_step (int, optional): Number of steps for N-step learning. Default is 2.

        """

        # Obs and actions spaces
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Attributes
        self.env = env
        self.batch_size = batch_size
        self.replay_buff_size = replay_buff_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.in_features = in_features
        self.out_features = out_features

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Processing device: {self.device}")

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.replay_buffer = PrioritizedReplayBuffer(obs_dim, replay_buff_size, batch_size, alpha=alpha, gamma=gamma)

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, replay_buff_size, batch_size, n_step=n_step, gamma=gamma)

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim, self.atom_size, self.support, self.in_features, self.out_features).to(
            self.device)
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, self.support, self.in_features,
                                  self.out_features).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)

        # transition to store in replay buffer
        self.experience = list()

        # Initialize UPC scenario graph
        self.graph = get_graph()

        # mode: train / test
        self.is_test = False

        # If SEED is available
        if self.seed is not None:
            seed_torch(self.seed)

        # for name , param in self.dqn.named_parameters():
        #     if 'weight' in name:
        #         print(f'Layer: {name}, Shape: {param.shape}')
        #         print(param.data)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Selects an action for the current observation based on available actions within the current state.

        Parameters:
        - state (np.ndarray): The input state representation.

        Returns:
        - np.ndarray: The selected action.

        This method utilizes a Deep Q-Network (DQN) to select an action based on the provided input state.
        It extracts the current node to identify possible actions and retrieves Q-values for each action.
        The action with the maximum Q-value is selected, and the state-action pair is stored in the experience.

        Note: Ensure that the DQN model is initialized before calling this method.
        """

        # Extract current node to get all possible neighbors (actions)
        current_node = obs[5]

        # All possible nodes (actions) to move to
        action_space = list(nx.neighbors(self.graph, current_node))

        # Get state-action pairs (Q-values)
        state_tensor = torch.FloatTensor(obs).to(self.device)
        q_values = self.dqn(state_tensor)

        # Extract Q-values for possible actions from the current node
        possible_q_values = [(idx, q_values[0, idx]) for idx in action_space]

        # Get the action with the maximum Q-value
        selected_action, max_q_value = max(possible_q_values, key=lambda x: x[1])

        # Store state and action in the experience
        if not self.is_test:
            self.experience = [obs, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> tuple[Any, SupportsFloat, bool, dict[str, Any]]:
        """
        Takes a step in the environment based on the provided action.

        Args:
            action (np.ndarray): The action to be taken in the environment.

        Returns:
            Tuple[np.ndarray, np.float64, bool]: A tuple containing the next state,
            reward, and a boolean indicating whether the episode is done.

        Notes:
            This method interacts with the environment using the specified action,
            updates the agent's experience, and stores transitions in the replay buffer.
            If using N-step transitions, it leverages the memory_n module.

        """

        # Send action to env and get response
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Consider terminated or truncated
        done = terminated or truncated

        # Save reward, next_obs and done in experience
        if not self.is_test:
            self.experience += [reward, next_state, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.experience)
        # 1-step transition
        else:
            one_step_transition = self.experience

        # add a single step transition
        if one_step_transition:
            self.replay_buffer.store(*one_step_transition)

        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """
        Update the model using gradient descent.

        Returns:
            torch.Tensor: The computed loss value.

        Notes:
            This method performs a gradient descent update on the model based on
            the sampled batches from the replay buffer, considering 1-step and
            potentially N-step learning losses. It also handles Prioritized Experience Replay (PER)
            and NoisyNet updates.

        """

        # PER: Sample batches and calculate weights
        samples = self.replay_buffer.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: Importance sampling before averaging
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples_n_step = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_step = self._compute_dqn_loss(samples_n_step, gamma)
            elementwise_loss += elementwise_loss_n_step

            # PER: Importance sampling before averaging
            loss = torch.mean(elementwise_loss * weights)

        # Zero out gradients, back-propagate, and clip gradients
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: Update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        # loss_for_prior_gpu = elementwise_loss.cuda()
        # loss_for_prior = loss_for_prior_gpu.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)

        # NoisyNet: Reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """
        Return categorical DQN loss.

        Args:
            samples (Dict[str, np.ndarray]): A dictionary containing sampled data.
            gamma (float): The discount factor for future rewards.

        Returns:
            torch.Tensor: The computed categorical DQN loss.

        Notes:
            This method calculates the categorical DQN loss using the provided samples.

        """

        # Convert arrays to PyTorch tensors and move them to the specified device
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # Categorical DQN algorithm which represents the distribution of
        # possible returns for each state-action pair using a set of discrete
        # probability masses (atoms)
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        # Disable gradient tracking for the following operations, as they are part of inference, not training.
        with torch.no_grad():
            # Double DQN

            # Determine the action with the highest Q-value using the online DQN network.
            next_action = self.dqn(next_state).argmax(1)
            # Obtain the probability distribution over actions for the next state using the target DQN network.
            next_dist = self.dqn_target.dist(next_state)
            # Select the probabilities corresponding to the actions chosen by the online DQN.
            next_dist = next_dist[range(self.batch_size), next_action]

            # Compute the projected distribution for double DQN,
            # The projected distribution is computed to handle the
            # distributional nature of the algorithm. This involves
            # mapping the target distribution onto the current distribution.

            # Calculate the projected distribution support for the Double DQN update.
            t_z = reward + (1 - done) * gamma * self.support
            # Ensure that the projected distribution support is within the specified range.
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            # Calculate the bin indices for the projected distribution.
            b = (t_z - self.v_min) / delta_z
            # Compute the lower and upper bin indices for interpolation in the projected distribution.
            l = b.floor().long()
            u = b.ceil().long()

            # Generate an offset tensor for indexing in the projected distribution.
            offset = (torch.linspace(0,
                                     (self.batch_size - 1) * self.atom_size,
                                     self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(
                self.device))

            # Initialize a tensor to store the projected distribution.
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            # Update the projected distribution using the lower bin indices and interpolation weights.
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            # Update the projected distribution using the upper bin indices and interpolation weights.
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # Compute the categorical DQN loss.
        # The loss is computed by comparing the log probabilities
        # of the selected actions in the current distribution with
        # the projected distribution. The negative sum of these
        # products represents the loss for each sample in the batch
        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """
        Perform a hard update of the target network.

        This method copies the parameters of the local Deep Q-Network (DQN) to the target DQN,
        ensuring an exact replication of the weights and biases.

        Note: The target network is used to provide stable target Q-values during training,
        and a hard update involves directly copying the parameters without any interpolation.

        """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _target_soft_update(self, tau: float = 0.01):
        """
        Perform a soft update of the target network.

        Parameters:
        - tau (float, optional): The interpolation parameter for the soft update.
          A smaller value results in a slower update. Default is 0.01.

        This method updates the target network parameters as a weighted average
        of the online network parameters.

        """

        # Get the state dictionaries of the online and target networks
        online_params = dict(self.dqn.named_parameters())
        target_params = dict(self.dqn_target.named_parameters())

        # Update the target network parameters using a weighted average
        for name in target_params:
            target_params[name].data.copy_(tau * online_params[name].data + (1.0 - tau) * target_params[name].data)

    def evaluate_model(self, environment) -> None:
        """Test the agent."""
        self.is_test = True

        nodes_to_evaluate = list(permutations(NODES_2_TRAIN, 2))

        self.env = ENV

        for i in range(len(nodes_to_evaluate)):
            obs, _ = self.env.reset(seed=self.seed, nodes_to_evaluate=nodes_to_evaluate[i])
            done = False
            score = 0
            step = 0
            print(f"\nEvaluation for the {(obs[0], obs[1])}:")


            while not done:
                action = self.select_action(obs)
                next_obs, reward, done, info = self.step(action)
                obs = next_obs
                score += reward
                print(f"{step} | action: {action} | rew: {reward} | done: {done} |  obs: {list(obs)}")
                step += 1

        self.env.close()

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a pre-trained model's state dictionary.

        Parameters:
        - model_path (str, optional): Path to the pre-trained model. If None, the highest scoring model is retrieved.

        Raises:
        - FileNotFoundError: If the specified model_path is not found.
        - Other relevant exceptions: Add any other exceptions as needed.
        """
        try:
            # If no specific model path is provided, retrieve the highest scoring model
            if model_path is None:
                model_path = get_highest_score_model()

            # Load the state dictionary from the specified or default model path
            state_dict = torch.load(model_path)

            # Update the online network with the loaded state dictionary
            self.dqn.load_state_dict(state_dict)

            # Update the target network with the state dictionary of the online network
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {model_path}.") from e

    def train(self, max_steps: int, warm_up_batches: int, plotting_interval: int = 1000, monitor_training: int = 1000,
              saving_model: int = 1000, ):
        """
        Train the DQNAgent.

        Parameters:
        - max_steps (int): Maximum number of steps for training.
        - plotting_interval (int, optional): Interval for plotting training progress. Default is 1000.
        - monitor_training (int, optional): Interval for monitoring training. Default is 1000.
        """

        assert (self.replay_buffer.batch_size * warm_up_batches) <= self.replay_buff_size, \
            "Assertion failed: Insufficient samples in the replay buffer. Increase the replay_buff_size"

        self.is_test = False

        # Initial observation from the reseted environment
        obs, _ = self.env.reset(seed=self.seed)
        # print(f"RESET: {obs[:2]}")

        # Initialize var
        reached_destination = 0
        update_cnt = 0
        losses = []
        scores = []
        mean_scores = []
        mean_losses = []
        score = 0
        last_mean_reward = 60

        # Start training
        for step in range(1, max_steps + 1):
            # Get action for obs and retrieve env response
            action = self.select_action(obs)
            next_obs, reward, done, info = self.step(action)
            obs = next_obs
            score += reward
            # Statistics
            reached_destination += info['count']
            # print(f"{step} - action: {action}, rew: {reward}, done: {done}, obs: {next_obs[:2]}")

            # PER: Increase beta
            fraction = min(step / max_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # If the episode ends
            if done:
                obs, _ = self.env.reset(seed=self.seed)
                # print(f"\nRESET: {obs[:2]}")
                scores.append(score)
                if len(scores) < 50:
                    mean_scores.append(np.mean(scores[0:]))
                else:
                    mean_scores.append(np.mean(scores[-50:]))
                score = 0

            # Start training after warmup
            min_samples = self.replay_buffer.batch_size * warm_up_batches
            if len(self.replay_buffer) >= min_samples:
                # Update the model and log the loss
                loss = self.update_model()
                losses.append(loss)
                if len(losses) < 50:
                    mean_losses.append(np.mean(losses[0:]))
                else:
                    mean_losses.append(np.mean(losses[-50:]))
                update_cnt += 1

                # Check if it's time for target network update
                if update_cnt % self.target_update == 0:
                    # self._target_hard_update()  # hard update
                    self._target_soft_update(self.tau)  # soft update

            if step % monitor_training == 0 and len(self.replay_buffer) >= min_samples:
                print(
                    f"Step: {step} | "
                    f"Rewards: {round(np.mean(scores[-monitor_training:]), 3)} | "
                    f"Loss: {round(np.mean(losses[-monitor_training:]), 5)} | "
                    f"Reached destination: {reached_destination}"
                )
                # Update count of reached destination
                reached_destination = 0

            if step % saving_model == 0:
                # Calculate the current mean reward over the specified monitoring window
                current_mean_reward = round(np.mean(scores[-monitor_training:]), 3)

                # Check if the current mean reward is greater than the last mean reward
                if current_mean_reward > last_mean_reward:
                    # Update last_mean_reward with the current mean reward
                    last_mean_reward = current_mean_reward

                    # Construct the model path with the current mean reward
                    model_path = MODEL_PATH + "rainbow_" + str(current_mean_reward) + ".pt"

                    # Save the model
                    torch.save(self.dqn.state_dict(), model_path)

            # Plotting (commented out for now)
            if step % plotting_interval == 0:
                _plot(step, scores, mean_scores, losses, mean_losses, plotting_interval,
                      save_path=SERGI_PLOTS + str(step) + '.png')

        # Close env
        # print(self.dqn.state_dict())
        self.env.close()


# MAIN CODE

ENV = Env_Test()  # For debug
# ENV = EnvironmentUPC()

agent = RAINBOW(
    env=ENV,
    replay_buff_size=1000000,
    batch_size=10,
    target_update=1000,
    learning_rate=0.0001,
    tau=0.85,
    gamma=0.85,
    n_step=11,
    in_features=19,
    out_features=19,
    alpha=0.5,
    beta=0.4,
    v_max=300,
    v_min=0,  # NO NEGATIVE VALUES!!!!
)

agent.train(max_steps=200000, warm_up_batches=200)

# ----------------------------------- EVALUATE -----------------------------------
agent = RAINBOW(
    env=ENV,
    replay_buff_size=5000,
    batch_size=20,
    target_update=100,
    learning_rate=0.001,
    tau=1,
    gamma=1,
    n_step=1,
    in_features=19,
    out_features=19,
    alpha=0,
    beta=0,
    v_max=300,
    v_min=0,
)
agent.load_model()
agent.evaluate_model(ENV)
