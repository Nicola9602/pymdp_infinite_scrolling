# L1 user 
# 26/05/2026


import pymdp
import jax.tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import Env
from pymdp.envs import rollout


from config import(
    POST_OBS,
    POST_VALENCE,
    POST_REWARD,
    L1_ACTIONS,
    p_liking_negative,
    p_liking_positive,
)

POST_OBS = ["post"]
POST_VALENCE = ["positive", "negative", "neutral"]
POST_REWARD = ["reward", "no_reward"]
L1_ACTIONS = ["scroll"]
p_liking_negative = 0.7
p_liking_positive = 0.5

##!! THIS WILL BE THE ALGORITHM AGENT TASK TO ASSIGN
#AND SWITCH THESE TO KEEP THE USER ENGAGED
#This acutually later will be done by the other agent 
p_positive = 0.5
p_negative = 0.3
p_neutral = 0.2 
 
model_description = { 
    "observations": { 
        "post_obs": { #Is the agent seeing a post or not? Must be seeing one
            "elements": POST_OBS, 
            "depends_on": ["post_action"], # Which in this case is only one so surely seeing it
        },
        "post_valence": { #This is the valence of the post itself, so negative vs 
                          # positive posts this could be interesting later on to model
                          # people being drawn to negative stuff 
            "elements": POST_VALENCE,
            "depends_on": ["post_valence"],
        },
        "post_reward":{# This is whether the agent likes it or not at all
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
            # post_valence_t+1 depends on post_valence_t and post_action_t
            "elements": POST_VALENCE,
            "depends_on": ["post_valence"], 
            "controlled_by": ["post_action"],
        },
    },          
}


model = compile_model(model_description)

# A1: likelihood mapping from hidden post valence to observed post.
# A[observation, hidden_state]
#------------------------------------------------------------------------
model.A["post_obs"]["post", "scroll"] = 1 

# Here later on I can put some uncertainity for now let's keep it certina 100%
model.A["post_valence"]["negative", "negative"] = 1
model.A["post_valence"]["positive", "positive"] = 1
model.A["post_valence"]["neutral", "neutral"] = 1 
model.A["post_valence"].normalize()

model.A["post_reward"]["reward", "negative"] = p_liking_negative 
model.A["post_reward"]["no_reward", "positive"] = 1 - p_liking_negative
model.A["post_reward"]["reward", "positive"] = p_liking_positive
model.A["post_reward"]["no_reward", "negative"] = 1 - p_liking_positive
model.A["post_reward"]["no_reward", "neutral"] = 0.99 # Always receive neutral in no reward
model.A["post_reward"]["no_reward", "neutral"] = 0.01
model.A["post_reward"].normalize()
# B transitions 
# B["to", "from", "action"]
#------------------------------------------------------------------------
# For now I keep it p_like p_dislike but this does not make sense 
# later on I actually want an algorithm adjusting these weights depending on what it 
# observes from the person 

for previous_valence in POST_VALENCE:
    model.B["post_valence"]["positive", previous_valence, "scroll"] = p_positive
    model.B["post_valence"]["negative", previous_valence, "scroll"] = p_negative
    model.B["post_valence"]["neutral", previous_valence, "scroll"] = p_neutral

model.B["post_valence"].normalize()

# Give a D 
#------------------------------------------------------------------------
#Whcih kind of first post is the user expecting
model.D["post_valence"]["positive"] = 0.5
model.D["post_valence"]["negative"] = 0.5
model.D["post_valence"]["neutral"] = 0.5
model.D["post_valence"].normalize()

# What could the platform observe from the person?
#this is a good question 
# Keep it minimal. The platform does not need to observe the person’s internal state directly. It observes behavioural proxies, for example:

# skip          = user scrolls quickly
# dwell         = user stays on the post
# like_click    = user clicks like
# continue      = user keeps scrolling
# quit          = user leaves the session

# A “hooked” user is not directly observed. It should be a hidden state inferred from behaviour.

# For example:

# ENGAGEMENT = ["low", "high"]

# Then:

# high engagement → more likely to dwell / continue
# low engagement  → more likely to skip / leave