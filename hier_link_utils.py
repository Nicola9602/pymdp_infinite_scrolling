# hier_link_utils.py
# L1 -> L2 communication utilities.

from jax import numpy as jnp

from config import (
    POST_VALENCE,
    L1_VALENCE_REPORT,
    REWARD_REPORT,
    AFFECT_OBS,
)


def l1_to_l2_valence_label(posterior_L1, feed_active=True):
    """
    Convert L1 posterior over post_valence into a discrete L2 report.

    Perfect precision version:
        take the MAP post_valence from L1.
    """

    if not feed_active:
        return "no_post"

    if posterior_L1 is None:
        raise ValueError(
            "posterior_L1 cannot be None while feed_active=True"
        )

    post_valence_posterior = posterior_L1[0]
    post_valence_posterior = jnp.squeeze(post_valence_posterior)

    if post_valence_posterior.ndim > 1:
        post_valence_posterior = post_valence_posterior[0]

    winning_idx = int(jnp.argmax(post_valence_posterior))

    return POST_VALENCE[winning_idx]


def l1_reward_to_l2_reward_label(l1_reward_label):
    """
    Map L1 reward observation into L2 reward_report.

    L1:
        reward / no_reward

    L2:
        quick_reward / no_reward / delayed_reward
    """

    if l1_reward_label == "reward":
        return "quick_reward"

    if l1_reward_label == "no_reward":
        return "no_reward"

    raise ValueError(f"Unknown L1 reward label: {l1_reward_label}")


def get_l2_obs_from_l1(
    posterior_L1,
    reward_label,
    affect_label,
    feed_active=True,
):
    """
    Build the full L2 observation list.

    L2 observation modalities:
        0. l1_valence_report
        1. reward_report
        2. affect_obs

    Returned observations use pymdp/JAX batched discrete format:

        jnp.array([idx])

    Shape is (batch_size,), not (batch_size, time).
    """

    valence_report_label = l1_to_l2_valence_label(
        posterior_L1=posterior_L1,
        feed_active=feed_active,
    )

    valence_report_idx = L1_VALENCE_REPORT.index(valence_report_label)
    reward_idx = REWARD_REPORT.index(reward_label)
    affect_idx = AFFECT_OBS.index(affect_label)

    obs_l2 = [
        jnp.array([valence_report_idx], dtype=jnp.int32),
        jnp.array([reward_idx], dtype=jnp.int32),
        jnp.array([affect_idx], dtype=jnp.int32),
    ]

    return obs_l2