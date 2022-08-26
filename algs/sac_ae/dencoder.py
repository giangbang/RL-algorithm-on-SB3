import torch as th
from typing import Type
import gym
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


def create_denconder(
        observation_space: gym.Space,
        features_dim: int,
        num_layers: int,
        num_filters: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
):
    assert is_image_space(observation_space, check_channels=False)
    obs_shape = observation_space.shape
    encoder_layers, decoder_layers = [], []

    encoder_layers += [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2), activation_fn()]
    decoder_fc = nn.ConvTranspose2d(
        num_filters, obs_shape[0], 3, stride=2, output_padding=1
    )

    for i in range(num_layers - 1):
        encoder_layers += [
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            activation_fn()
        ]

        decoder_layers += [
            activation_fn(),
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
        ]
    encoder_cnn = nn.Sequential(*encoder_layers)

    # Compute shape by doing one forward pass
    with th.no_grad():
        cnn_shape = encoder_cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1:]
    n_flatten = th.prod(cnn_shape).item()
    # Flatten feature dim
    encoder_cnn = nn.Sequential(encoder_cnn, nn.Flatten())
    encoder_fc = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LayerNorm(features_dim))

    decoder_layers += [activation_fn(), nn.Linear(
        features_dim, n_flatten
    )]
    decoder_layers.reverse()

    decoder_dcnn = nn.Sequential(*decoder_layers)

    encoder = PixelEncoder(observation_space, features_dim, encoder_cnn, encoder_fc)
    decoder = PixelDecoder(observation_space, features_dim, decoder_dcnn, decoder_fc, cnn_shape)

    return decoder, encoder


class PixelEncoder(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int,
            encoder: nn.Module,
            encoder_fc: nn.Module
    ):
        super().__init__(observation_space, features_dim)

        self.encoder = encoder
        self.fc = encoder_fc

    def forward(self, observations: th.Tensor, detach: bool = False) -> th.Tensor:
        with th.set_grad_enabled(not detach):
            h = self.encoder(observations)
        h = self.fc(h)
        return th.tanh(h)


class PixelDecoder(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int,
            decoder: nn.Module,
            decoder_fc: nn.Module,  # reshape flatten layer before feeding to deconv layers
            latent_feature_shape: th.Tensor
    ):
        super().__init__(observation_space, features_dim)

        self.fc = decoder_fc
        self.latent_feature_shape = latent_feature_shape
        self.decoder = decoder

    def forward(self, feature: th.Tensor) -> th.Tensor:
        h = self.fc(feature)
        h = h.view(-1, *self.latent_feature_shape)
        return self.decoder(h)
