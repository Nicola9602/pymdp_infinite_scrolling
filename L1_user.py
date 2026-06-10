# L1_user.py

from pymdp.distribution import compile_model

from config import (
    POST_VALENCE,
    POST_REWARD,
    L1_ACTIONS,
    P_REWARD_IF_POSITIVE,
    P_REWARD_IF_NEGATIVE,
    P_REWARD_IF_NEUTRAL,
    P_POSITIVE,
    P_NEGATIVE,
    P_NEUTRAL,
)


def init_L1_model():
    """
    Level 1: passive post perception.

    L1 does not decide whether to scroll or quit. It only receives a post
    when L2 has decided that the feed is active. L1 infers the hidden
    post_valence from two observation modalities:
        1. post_valence_obs
        2. post_reward_obs

    no_post is deliberately absent from L1. no_post is generated at L2
    when the higher-level agent opts out.
    """

    model_description = {
        "observations": {
            "post_valence_obs": {
                "elements": POST_VALENCE,
                "depends_on": ["post_valence"],
            },
            "post_reward_obs": {
                "elements": POST_REWARD,
                "depends_on": ["post_valence"],
            },
        },
        "controls": {
            "post_action": {
                "elements": L1_ACTIONS,
            },
        },
        "states": {
            "post_valence": {
                "elements": POST_VALENCE,
                "depends_on": ["post_valence"],
                "controlled_by": ["post_action"],
            },
        },
    }

    model = compile_model(model_description)

    # ------------------------------------------------------------------
    # A: likelihood mappings
    # A[observation, hidden_state]
    # ------------------------------------------------------------------

    # Perfect observation of post valence for now.
    for valence in POST_VALENCE:
        model.A["post_valence_obs"][valence, valence] = 1.0
    model.A["post_valence_obs"].normalize()

    # Intermittent quick reward conditional on hidden post valence.
    model.A["post_reward_obs"]["reward", "positive"] = P_REWARD_IF_POSITIVE
    model.A["post_reward_obs"]["no_reward", "positive"] = 1.0 - P_REWARD_IF_POSITIVE

    model.A["post_reward_obs"]["reward", "negative"] = P_REWARD_IF_NEGATIVE
    model.A["post_reward_obs"]["no_reward", "negative"] = 1.0 - P_REWARD_IF_NEGATIVE

    model.A["post_reward_obs"]["reward", "neutral"] = P_REWARD_IF_NEUTRAL
    model.A["post_reward_obs"]["no_reward", "neutral"] = 1.0 - P_REWARD_IF_NEUTRAL

    model.A["post_reward_obs"].normalize()

    # ------------------------------------------------------------------
    # B: transition mapping for hidden post valence
    # B["to", "from", "action"]
    # ------------------------------------------------------------------

    # Each scroll samples a new post from the baseline feed distribution.
    for previous_valence in POST_VALENCE:
        model.B["post_valence"]["positive", previous_valence, "scroll"] = P_POSITIVE
        model.B["post_valence"]["negative", previous_valence, "scroll"] = P_NEGATIVE
        model.B["post_valence"]["neutral", previous_valence, "scroll"] = P_NEUTRAL
    model.B["post_valence"].normalize()

    # ------------------------------------------------------------------
    # C: preferences
    # ------------------------------------------------------------------

    # L1 has only one action, so C does not drive any meaningful choice here.
    # Keep it neutral. L2 will handle the marshmallow preferences.
    for valence in POST_VALENCE:
        model.C["post_valence_obs"][valence] = 0.0
    for reward_obs in POST_REWARD:
        model.C["post_reward_obs"][reward_obs] = 0.0

    # ------------------------------------------------------------------
    # D: initial prior over post valence
    # ------------------------------------------------------------------

    model.D["post_valence"]["positive"] = 1.0 / 3.0
    model.D["post_valence"]["negative"] = 1.0 / 3.0
    model.D["post_valence"]["neutral"] = 1.0 / 3.0
    model.D["post_valence"].normalize()

    mappings = {
        "post_valence_names": POST_VALENCE,
        "post_reward_names": POST_REWARD,
        "post_action_names": L1_ACTIONS,
    }

    dimensions = {
        "num_obs": [len(POST_VALENCE), len(POST_REWARD)],
        "num_states": [len(POST_VALENCE)],
        "num_controls": [len(L1_ACTIONS)],
    }

    return model, mappings, dimensions
