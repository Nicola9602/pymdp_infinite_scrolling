# L1_scrolling_env.py

from jax import numpy as jnp
from jax import random as jr

from config import (
    POST_VALENCE,
    POST_REWARD,
    P_POSITIVE,
    P_NEGATIVE,
    P_NEUTRAL,
    P_REWARD_IF_POSITIVE,
    P_REWARD_IF_NEGATIVE,
    P_REWARD_IF_NEUTRAL,
)


class L1ScrollingEnv:
    """
    Level 1 generative process.

    Returns batched discrete observations for pymdp/JAX:

        obs_l1 = [
            jnp.array([post_valence_obs_idx]),
            jnp.array([post_reward_obs_idx]),
        ]

    Shape is (batch_size,), not (batch_size, time).
    """

    def __init__(
        self,
        p_positive=P_POSITIVE,
        p_negative=P_NEGATIVE,
        p_neutral=P_NEUTRAL,
        p_reward_if_positive=P_REWARD_IF_POSITIVE,
        p_reward_if_negative=P_REWARD_IF_NEGATIVE,
        p_reward_if_neutral=P_REWARD_IF_NEUTRAL,
    ):
        valence_probs = jnp.array(
            [p_positive, p_negative, p_neutral],
            dtype=jnp.float32,
        )

        self.valence_probs = valence_probs / jnp.sum(valence_probs)

        self.reward_probs = jnp.array(
            [
                p_reward_if_positive,
                p_reward_if_negative,
                p_reward_if_neutral,
            ],
            dtype=jnp.float32,
        )

    def step(self, key, action="scroll"):
        if action != "scroll":
            raise ValueError(
                f"L1 only understands action='scroll', got {action}"
            )

        key_valence, key_reward = jr.split(key)

        # 1. Sample hidden post valence.
        valence_idx = jr.categorical(
            key_valence,
            jnp.log(self.valence_probs),
        )

        valence_idx = valence_idx.astype(jnp.int32)

        # 2. Sample reward conditional on valence.
        p_reward = self.reward_probs[valence_idx]

        reward_happened = jr.bernoulli(
            key_reward,
            p=p_reward,
        )

        # POST_REWARD = ["reward", "no_reward"]
        reward_idx = jnp.where(
            reward_happened,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32),
        )

        # Important pymdp/JAX format:
        # one integer per batch element.
        obs_l1 = [
            jnp.array([valence_idx], dtype=jnp.int32),
            jnp.array([reward_idx], dtype=jnp.int32),
        ]

        valence_idx_int = int(valence_idx)
        reward_idx_int = int(reward_idx)

        info = {
            "true_valence_idx": valence_idx_int,
            "true_valence_label": POST_VALENCE[valence_idx_int],
            "post_reward_idx": reward_idx_int,
            "post_reward_label": POST_REWARD[reward_idx_int],
        }

        return obs_l1, info