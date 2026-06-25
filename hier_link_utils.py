# hier_link_utils.py
# L1 -> L2 communication utilities.

from jax import numpy as jnp

from config import (
    POST_VALENCE,
    L1_VALENCE_REPORT,
    REWARD_REPORT,
    AFFECT_OBS,
    L3_BEHAVIOR_EVIDENCE_OBS,
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

def l2_history_to_l3_behavior_evidence(
    l2_action_history,
    l2_belief_history,
    k,
    persistent_scroll_threshold=0.80,
):
    """
    Convert the last K L2 actions into one L3 behavioral evidence observation.

    L3 does not receive "I am in control" as an observation.

    Instead, it receives evidence computed from L2 behavior, such as:
        - persistent scrolling
        - persistent scrolling while L2 believes it is hooked
        - failed opt-out
        - relapse after opt-out

    l2_action_history:
        list of L2 action labels, e.g.
        ["continue_scrolling", "continue_scrolling", "opt_out", ...]

    l2_belief_history:
        list of L2 believed self-state labels, e.g.
        ["scrolling_ok", "scrolling_hooked", ...]
    """

    if len(l2_action_history) < k:
        return "insufficient_history"

    action_window = l2_action_history[-k:]
    belief_window = l2_belief_history[-k:]

    n_continue = sum(
        action == "continue_scrolling"
        for action in action_window
    )

    frac_continue = n_continue / float(k)

    has_optout = any(
        action == "opt_out"
        for action in action_window
    )

    has_hooked_belief = any(
        belief == "scrolling_hooked"
        for belief in belief_window
    )

    # Detect relapse:
    # the agent opted out at some point, then later returned to scrolling.
    last_optout_idx = None
    for idx, action in enumerate(action_window):
        if action == "opt_out":
            last_optout_idx = idx

    if last_optout_idx is not None:
        actions_after_last_optout = action_window[last_optout_idx + 1:]
        returned_to_scrolling = any(
            action == "continue_scrolling"
            for action in actions_after_last_optout
        )

        if returned_to_scrolling:
            return "relapse_after_optout"

    # Detect failed opt-out:
    # there is at least one opt-out, but the window is still dominated by scrolling.
    if has_optout and frac_continue >= 0.60:
        return "failed_optout"

    # Key doomscrolling evidence:
    # persistent scrolling while L2 already believes it is hooked.
    if frac_continue >= persistent_scroll_threshold and has_hooked_belief:
        return "persistent_scrolling_with_hooked_belief"

    if frac_continue >= persistent_scroll_threshold:
        return "persistent_scrolling"

    # Mostly not scrolling / mostly opting out.
    if frac_continue <= 0.25:
        return "brief_voluntary_use"

    return "extended_but_stable_use"


def get_l3_obs_from_l2_history(
    l2_action_history,
    l2_belief_history,
    k,
):
    """
    Build the L3 observation from recent L2 history.

    Returned format uses the same pymdp/JAX-style discrete format:

        [jnp.array([idx], dtype=jnp.int32)]

    This keeps L3 compatible with the same observation style used by L1 and L2.
    """

    evidence_label = l2_history_to_l3_behavior_evidence(
        l2_action_history=l2_action_history,
        l2_belief_history=l2_belief_history,
        k=k,
    )

    evidence_idx = L3_BEHAVIOR_EVIDENCE_OBS.index(evidence_label)

    obs_l3 = [
        jnp.array([evidence_idx], dtype=jnp.int32)
    ]

    return obs_l3, evidence_label