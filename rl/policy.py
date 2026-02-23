"""
policy.py

Custom PPO policy network for supply chain optimisation.

Architecture:
  - Feature extractor: per-component MLP + global state MLP
  - Attention: dot-product attention over component embeddings
    (lets the network weight components by urgency/criticality)
  - Shared trunk: 2-layer MLP on attended features
  - Policy head: linear projection to action logits per component
  - Value head: linear projection to scalar value estimate

The attention mechanism is key: it allows the network to dynamically
focus on the components that are most at risk, without hardcoding
the ordering. This is important because the supply chain state is
naturally variable - some days one component is critical, other days another.

Design choices:
  - No LSTM/GRU: the state is fully observed (Markovian given the state vector)
  - Attention over components rather than transformer: we don't need cross-component
    attention on the action side (actions are per-component)
  - Separate feature extractors for per-component and global state
  - Layer norm for training stability (supply chain rewards can be large magnitude)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ComponentFeatureExtractor(nn.Module):
    """
    Encodes per-component state vectors into embeddings.

    Input:  (batch, n_components, component_state_dim)
    Output: (batch, n_components, embed_dim)
    """

    def __init__(self, component_state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(component_state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_components, component_state_dim)
        return self.net(x)


class GlobalStateEncoder(nn.Module):
    """Encodes the 3-dimensional global state into an embedding."""

    def __init__(self, global_dim: int = 3, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComponentAttention(nn.Module):
    """
    Scaled dot-product attention over component embeddings.

    Query: global state embedding (what does the agent care about now?)
    Key/Value: component embeddings (what information is available per component?)

    Output: (batch, n_components, embed_dim) - attended component features
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,  # (batch, embed_dim) - global state
        kv: torch.Tensor,     # (batch, n_comp, embed_dim) - component features
    ) -> torch.Tensor:
        batch, n_comp, embed = kv.shape

        # Expand query to (batch, 1, embed_dim) as the global query
        q = self.q_proj(query.unsqueeze(1))  # (batch, 1, embed)
        k = self.k_proj(kv)                   # (batch, n_comp, embed)
        v = self.v_proj(kv)                   # (batch, n_comp, embed)

        # Reshape for multi-head attention
        def split_heads(t: torch.Tensor, n_items: int) -> torch.Tensor:
            return t.view(batch, n_items, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q, 1)          # (batch, n_heads, 1, head_dim)
        k = split_heads(k, n_comp)     # (batch, n_heads, n_comp, head_dim)
        v = split_heads(v, n_comp)     # (batch, n_heads, n_comp, head_dim)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, 1, n_comp)
        attn = F.softmax(attn, dim=-1)

        # Attended values: (batch, n_heads, n_comp, head_dim)
        # We want per-component context, so we replicate query attention back
        out = attn.transpose(-2, -1) * v  # element-wise weighted components
        out = out.transpose(1, 2).contiguous().view(batch, n_comp, embed)
        return self.out_proj(out)  # (batch, n_comp, embed)


class SupplyChainFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 feature extractor combining per-component and global state.

    Outputs a flat feature vector of dimension features_dim.
    """

    N_COMPONENT_DIMS = 10  # must match env._STATE_DIMS_PER_COMPONENT
    GLOBAL_DIMS = 3
    EMBED_DIM = 64
    GLOBAL_EMBED_DIM = 32
    N_ATTN_HEADS = 4

    def __init__(self, observation_space: spaces.Box, n_components: int):
        self.n_components = n_components
        features_dim = n_components * self.EMBED_DIM + self.GLOBAL_EMBED_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.component_encoder = ComponentFeatureExtractor(
            self.N_COMPONENT_DIMS, self.EMBED_DIM
        )
        self.global_encoder = GlobalStateEncoder(self.GLOBAL_DIMS, self.GLOBAL_EMBED_DIM)
        self.attention = ComponentAttention(self.EMBED_DIM, self.N_ATTN_HEADS)
        self.layer_norm = nn.LayerNorm(n_components * self.EMBED_DIM + self.GLOBAL_EMBED_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        n_comp_flat = self.n_components * self.N_COMPONENT_DIMS

        # Split observation into per-component and global
        comp_flat = observations[:, :n_comp_flat]
        global_state = observations[:, n_comp_flat:]

        # Reshape to (batch, n_components, component_dims)
        comp_state = comp_flat.view(batch, self.n_components, self.N_COMPONENT_DIMS)

        # Encode
        comp_embeddings = self.component_encoder(comp_state)  # (batch, n_comp, embed_dim)
        global_embedding = self.global_encoder(global_state)  # (batch, global_embed_dim)

        # Attend
        attended = self.attention(global_embedding, comp_embeddings)  # (batch, n_comp, embed_dim)
        attended_flat = attended.view(batch, -1)  # (batch, n_comp * embed_dim)

        # Concatenate with global
        combined = torch.cat([attended_flat, global_embedding], dim=-1)
        return self.layer_norm(combined)


class SupplyChainPolicy(ActorCriticPolicy):
    """
    PPO policy for supply chain optimisation.

    Uses SupplyChainFeaturesExtractor as the feature backbone,
    then separate actor and critic heads.

    Per-component action logits: the actor projects the attended
    component embedding to 5 action logits for each component.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        n_components: int = 15,
        **kwargs,
    ):
        self.n_components = n_components

        kwargs["features_extractor_class"] = SupplyChainFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"n_components": n_components}

        # Net arch: two-layer MLP for shared trunk
        kwargs["net_arch"] = dict(pi=[256, 128], vf=[256, 128])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Standard SB3 MLP extractor works fine with our features."""
        super()._build_mlp_extractor()


def make_policy_kwargs(n_components: int) -> dict:
    """Return policy_kwargs for PPO constructor."""
    return {
        "policy_class": SupplyChainPolicy,
        "policy_kwargs": {
            "n_components": n_components,
        },
    }
