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

import warnings
from jax import numpy as jnp

from config import L3_BEHAVIOR_EVIDENCE_OBS


def l2_history_to_l3_behavior_evidence(
    l2_action_history,
    l2_belief_history=None,
    k=20,
    brief_use_threshold=0.25,
    successful_optout_threshold=0.50,
):
    """
    Convert recent L2 behavior into one simplified L3 evidence label.

    Possible outputs:

        insufficient_history   # warning-only label, NOT an L3 observation
        brief_use
        persistent_scrolling
        failed_optout
        successful_optout

    Notes:
        - relapse_after_optout is collapsed into failed_optout.
        - extended_but_stable_use is collapsed into persistent_scrolling.
        - persistent_scrolling_with_hooked_belief is collapsed into persistent_scrolling.
        - l2_belief_history is kept only for backward compatibility.
    """

    if k <= 0:
        raise ValueError("k must be > 0")

    if len(l2_action_history) < k:
        warnings.warn(
            "Not enough L2 history to build an L3 observation. "
            "Returning warning-only label: insufficient_history.",
            RuntimeWarning,
        )
        return "insufficient_history"

    action_window = l2_action_history[-k:]

    valid_actions = {"continue_scrolling", "opt_out"}

    invalid_actions = [
        action for action in action_window
        if action not in valid_actions
    ]

    if invalid_actions:
        raise ValueError(f"Unknown L2 actions in history: {invalid_actions}")

    n_continue = sum(
        action == "continue_scrolling"
        for action in action_window
    )

    n_optout = sum(
        action == "opt_out"
        for action in action_window
    )

    frac_continue = n_continue / float(k)
    frac_optout = n_optout / float(k)

    first_optout_idx = None

    for idx, action in enumerate(action_window):
        if action == "opt_out":
            first_optout_idx = idx
            break

    # Case 1: there was at least one opt-out.
    if first_optout_idx is not None:
        actions_after_first_optout = action_window[first_optout_idx + 1:]

        returned_to_scrolling = any(
            action == "continue_scrolling"
            for action in actions_after_first_optout
        )

        # Opted out, then returned to scrolling.
        # This collapses relapse_after_optout into failed_optout.
        if returned_to_scrolling:
            return "failed_optout"

        # Mostly opted out and did not return to scrolling.
        # This is evidence of control.
        if frac_optout >= successful_optout_threshold:
            return "successful_optout"

    # Case 2: little or no scrolling.
    if frac_continue <= brief_use_threshold:
        return "brief_use"

    # Case 3: everything else is persistent scrolling.
    return "persistent_scrolling"


def get_l3_obs_from_l2_history(
    l2_action_history,
    l2_belief_history,
    k,
):
    """
    Build the L3 observation from recent L2 history.

    Returned format, when history is sufficient:

        [jnp.array([idx], dtype=jnp.int32)]

    If history is insufficient, return:

        None, "insufficient_history"

    This prevents insufficient_history from being treated as an actual
    L3 observation.
    """

    evidence_label = l2_history_to_l3_behavior_evidence(
        l2_action_history=l2_action_history,
        l2_belief_history=l2_belief_history,
        k=k,
    )

    if evidence_label == "insufficient_history":
        return None, evidence_label

    if evidence_label not in L3_BEHAVIOR_EVIDENCE_OBS:
        raise ValueError(
            f"Evidence label {evidence_label!r} is not in "
            "L3_BEHAVIOR_EVIDENCE_OBS."
        )

    evidence_idx = L3_BEHAVIOR_EVIDENCE_OBS.index(evidence_label)

    obs_l3 = [
        jnp.array([evidence_idx], dtype=jnp.int32)
    ]

    return obs_l3, evidence_label


def choose_l3_meta_action_from_belief(
    qs_l3,
    not_in_control_threshold=0.50,
):
    """
    Rule-based L3 overseer.

    L3 infers whether the system is:

        in_control
        not_in_control

    This function converts that belief into a meta-action:

        stay_in_control
        stay_hooked

    If L3 believes the system is not in control strongly enough,
    it activates the control/awareness intervention.

    Otherwise, it does not intervene and lets the hooked loop continue.
    """

    # qs_l3[0] may have batch dimensions, e.g. shape (2, 1, 1)
    # or similar depending on pymdp/JAX internals.
    # We squeeze it into a flat belief vector:
    #
    #     [p_in_control, p_not_in_control]
    #
    control_belief = jnp.squeeze(qs_l3[0])

    p_in_control = float(control_belief[0])
    p_not_in_control = float(control_belief[1])

    if p_not_in_control >= not_in_control_threshold:
        return "stay_in_control"

    return "stay_hooked"

def apply_l3_meta_action_to_l2_env(
    l2_env,
    meta_action,
    awareness_when_intervening=0.80,
    awareness_when_not_intervening=0.05,
):
    """
    Apply L3's meta-action as a perturbation to the L2 environment.

    L3 does not directly force L2 to opt out.

    Instead, it changes how likely the L2 environment is to generate
    awareness of hookedness.

    stay_in_control:
        increase p_awareness_when_hooked.
        Hooked scrolling is more likely to feel bad / become visible.

    stay_hooked:
        decrease p_awareness_when_hooked.
        Hooked scrolling is more likely to stay hidden as scroll_relief.
    """

    if meta_action == "stay_in_control":
        l2_env.p_awareness_when_hooked = awareness_when_intervening
        return l2_env.p_awareness_when_hooked

    if meta_action == "stay_hooked":
        l2_env.p_awareness_when_hooked = awareness_when_not_intervening
        return l2_env.p_awareness_when_hooked

    raise ValueError(f"Unknown L3 meta-action: {meta_action}")