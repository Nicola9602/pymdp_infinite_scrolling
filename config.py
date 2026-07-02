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



# -----------------------------
# Level 3: metacognitive awareness level
# -----------------------------

# TODO: here I am using some words for a counting of behavioural obs incoming, 
# not sure that's the smartest way

# L3 receives evidence over the last K L2 actions.
# Smaller K = frequent self-checking.
# Larger K = longer flow before awareness kicks in.
L3_UPDATE_EVERY = 20


L3_BEHAVIOR_EVIDENCE_OBS = [
    "brief_use",
    "persistent_scrolling",
    "failed_optout",
    "successful_optout",
]

L3_CONTROL_STATE = [
    "in_control",
    "not_in_control",
]

L3_ACTIONS = [
    "stay_in_control",
    "stay_hooked",
]

# -----------------------------
# Minimal adaptive feed algorithm
# -----------------------------

# Main experimental condition.
# "passive" = fixed feed pull
# "adversarial" = feed pull increases with continued scrolling / relapse
# "supportive" = feed pull decreases when the user seems captured
# ALGORITHM_MODE = "adversarial"
ALGORITHM_MODE = "passive"
# ALGORITHM_MODE = "supportive"

# Initial feed pull.
# 0.0 = low-pull feed
# 1.0 = maximum high-pull feed
FEED_PULL_INITIAL = 0.30