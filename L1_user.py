import pymdp
import jax.tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import Env
from pymdp.envs import rollout


from config import Config

POST_OBS = ["post"]
POST_VALENCE = ["positive", "negative", "neutral"]
POST_REWARD = ["reward", "no_reward"]
L1_ACTIONS = ["scroll"]
p_liking_negative = 0.7
p_liking_positive = 0.5
p_like = 0.7
p_dislike = 0.3 
 
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

model.A["post_reward"]["reward", "negative"] = p_liking_negative 
model.A["post_reward"]["no_reward", "negative"] = 1 - p_liking_negative
model.A["post_reward"]["reward", "positive"] = p_liking_positive
model.A["post_reward"]["no_reward", "negative"] = 1 - p_liking_positive

# B transitions 
# B["to", "from", "action"]
#------------------------------------------------------------------------
# For now I keep it p_like p_dislike but this does not make sense 
# later on I actually want an algorithm adjusting these weights depending on what it 
# observes from the person 

model.B["post_valence"]["positive", "positive", "scroll"] = p_like
model.B["post_valence"]["positive", "negative", "scroll"] = p_like
model.B["post_valence"]["negative", "negative", "scroll"] = p_dislike
model.B["post_valence"]["negative", "positive", "scroll"] = p_dislike

#Give a C 
#------------------------------------------------------------------------
# Here I can set the agent preferring negative content rather than positive content 
model.C["post_valence"]["like"] = p_liking_positive
model.C["post_valence"]["dislike"] = p_liking_negative

