# sequence_extractor_lstm.py

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMSequenceExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
    ):
        super().__init__(observation_space, features_dim)

        seq_len, obs_dim = observation_space.shape

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print("Extractor input shape:", observations.shape)
        lstm_out, (h_n, c_n) = self.lstm(observations)
        seq_features = lstm_out[:, -1, :]
        return self.projection(seq_features)