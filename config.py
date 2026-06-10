# config.py

# -----------------------------
# Level 1: passive post stream
# -----------------------------
POST_VALENCE = ["positive", "negative", "neutral"]
POST_REWARD = ["reward", "no_reward"]
L1_ACTIONS = ["scroll"]

# Agent's belief about quick reward probability conditional on post valence.
P_REWARD_IF_POSITIVE = 0.7
P_REWARD_IF_NEGATIVE = 0.8
P_REWARD_IF_NEUTRAL = 0.01

# Baseline feed distribution used by the L1 environment and L1/L2 generative models.
P_POSITIVE = 0.40
P_NEGATIVE = 0.59
P_NEUTRAL = 0.01

# -----------------------------
# Level 2: marshmallow level
# -----------------------------
L1_VALENCE_REPORT = ["positive", "negative", "neutral", "no_post"]

SELF_STATE = [
    "scrolling_ok",
    "scrolling_hooked",
    "withdrawal_1",
    "withdrawal_2",
    "recovered_reward",
    "recovered_no_reward"
]

REWARD_REPORT = ["no_reward", "quick_reward", "delayed_reward"]
AFFECT_OBS = ["okay", "scroll_relief", "urge", "bad_feeling", "deep_relief", "boredom"]
L2_ACTIONS = ["continue_scrolling", "opt_out"]
