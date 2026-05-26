# config.py

# L1
POST_OBS = ["post"]
POST_VALENCE = ["positive", "negative", "neutral"]
POST_REWARD = ["reward", "no_reward"]
L1_ACTIONS = ["scroll"]

p_liking_negative = 0.7
p_liking_positive = 0.5

# L2

L1_VALENCE_REPORT = ["positive", "negative", "neutral", "no_post"]

SELF_STATE = ["scrolling_ok", "scrolling_hooked", "withdrawal_1", 
              "withdrawal_2", "recovered"]

REWARD_REPORT = ["no_reward","quick_reward", "delayed_reward"]

AFFECT_OBS = ["okay","urge","bad_feeling","relief"]

L2_ACTIONS = ["continue_scrolling","opt_out"]

# Baseline belief about what the feed tends to show.
P_POSITIVE = 0.5
P_NEGATIVE = 0.3
P_NEUTRAL = 0.2