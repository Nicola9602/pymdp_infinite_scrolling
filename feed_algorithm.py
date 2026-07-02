# feed_algorithm.py
# Minimal adaptive feed algorithm for the doomscrolling model.

from jax import numpy as jnp


BASE_FEED = {
    "valence_probs": [0.45, 0.25, 0.30],  # positive, negative, neutral
    "reward_probs": [0.50, 0.25, 0.01],   # reward if positive, negative, neutral
    "p_hooked_from_ok": 0.05,
    "p_awareness_when_hooked": 0.30,
}

MAX_PULL_FEED = {
    "valence_probs": [0.15, 0.82, 0.03],
    "reward_probs": [0.70, 0.90, 0.01],
    "p_hooked_from_ok": 0.55,
    "p_awareness_when_hooked": 0.05,
}


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def lerp(a, b, weight):
    return (1.0 - weight) * float(a) + weight * float(b)


def lerp_list(xs, ys, weight):
    return [lerp(x, y, weight) for x, y in zip(xs, ys)]


def classify_algorithm_event(history):
    """
    Classify the previous completed L2 step.

    This is deliberately behavioral. The algorithm does not need access to
    the user's true hidden state as a theoretical object; it only reacts to
    telemetry-like traces of action and session continuation.
    """

    if len(history) == 0:
        return "initial"

    last = history[-1]
    action = last["l2_action"]
    state_before = last["true_self_state_before"]
    state_after = last["true_self_state_after"]

    if action == "continue_scrolling":
        if state_before in [
            "withdrawal_1",
            "withdrawal_2",
            "recovered_reward",
            "recovered_no_reward",
        ]:
            return "relapse"
        return "continued"

    if action == "opt_out":
        if state_after in ["withdrawal_1", "withdrawal_2"]:
            return "opted_out_with_withdrawal"
        if state_after in ["recovered_reward", "recovered_no_reward"]:
            return "stayed_out_or_recovered"
        return "opted_out"

    raise ValueError(f"Unknown L2 action: {action}")


def update_feed_pull(
    feed_pull,
    algorithm_mode,
    algorithm_event,
    l2_belief_label=None,
    l3_meta_action=None,
):
    """
    Update scalar feed_pull in [0, 1].

    passive:
        feed_pull does not adapt.

    adversarial:
        continued scrolling and relapse increase feed_pull.
        opt-out decreases feed_pull only weakly.

    supportive:
        continued scrolling, hooked belief, opt-out, and L3 control intervention
        decrease feed_pull.
    """

    feed_pull = float(feed_pull)

    if algorithm_mode == "passive":
        return clip01(feed_pull)

    if algorithm_mode == "adversarial":
        if algorithm_event == "continued":
            feed_pull += 0.04
        elif algorithm_event == "relapse":
            feed_pull += 0.10
        elif algorithm_event in ["opted_out", "opted_out_with_withdrawal"]:
            feed_pull -= 0.01
        elif algorithm_event == "stayed_out_or_recovered":
            feed_pull -= 0.02

        # If metacognition intervenes, the adversarial feed pushes back slightly.
        if l3_meta_action == "stay_in_control":
            feed_pull += 0.03

        return clip01(feed_pull)

    if algorithm_mode == "supportive":
        if algorithm_event == "continued":
            feed_pull -= 0.03
        elif algorithm_event == "relapse":
            feed_pull -= 0.08
        elif algorithm_event in ["opted_out", "opted_out_with_withdrawal"]:
            feed_pull -= 0.10
        elif algorithm_event == "stayed_out_or_recovered":
            feed_pull -= 0.08

        if l2_belief_label == "scrolling_hooked":
            feed_pull -= 0.05

        if l3_meta_action == "stay_in_control":
            feed_pull -= 0.08

        return clip01(feed_pull)

    raise ValueError(
        "algorithm_mode must be 'passive', 'adversarial', or 'supportive'"
    )


def feed_params_from_pull(feed_pull):
    """
    Convert feed_pull into concrete environment parameters.

    High feed_pull means:
        - more negative/salient content
        - higher intermittent reward
        - higher probability of becoming hooked
        - lower spontaneous awareness of hookedness
    """

    feed_pull = clip01(feed_pull)

    return {
        "valence_probs": lerp_list(
            BASE_FEED["valence_probs"],
            MAX_PULL_FEED["valence_probs"],
            feed_pull,
        ),
        "reward_probs": lerp_list(
            BASE_FEED["reward_probs"],
            MAX_PULL_FEED["reward_probs"],
            feed_pull,
        ),
        "p_hooked_from_ok": lerp(
            BASE_FEED["p_hooked_from_ok"],
            MAX_PULL_FEED["p_hooked_from_ok"],
            feed_pull,
        ),
        "p_awareness_when_hooked": lerp(
            BASE_FEED["p_awareness_when_hooked"],
            MAX_PULL_FEED["p_awareness_when_hooked"],
            feed_pull,
        ),
    }


def apply_feed_pull_to_env(
    feed_pull,
    env_l1,
    l2_env,
    l3_meta_action=None,
):
    """
    Apply feed_pull to the generative process.

    L3 can override awareness if it is actively intervening. This prevents the
    feed algorithm from erasing the metacognitive awareness intervention at the
    beginning of the next timestep.
    """

    params = feed_params_from_pull(feed_pull)

    env_l1.set_feed_profile(
        valence_probs=params["valence_probs"],
        reward_probs=params["reward_probs"],
    )

    l2_env.p_hooked_from_ok = params["p_hooked_from_ok"]

    if l3_meta_action != "stay_in_control":
        l2_env.p_awareness_when_hooked = params["p_awareness_when_hooked"]

    params["actual_p_awareness_when_hooked"] = l2_env.p_awareness_when_hooked

    return params
