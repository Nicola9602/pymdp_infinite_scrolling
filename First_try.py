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
POST_VALENCE = ["like", "dislike"]
L1_ACTIONS = ["scroll"]
p_like = 0.7
p_dislike = 0.3
 
model_description = { 
    "observations": { 
        "post_obs": { #Is the agent seeing a post or not? Must be seeing one
            "elements": POST_OBS, 
            "depends_on": ["post_action"], # Which in this case is only one so surely seeing it
        },
        "post_valence": { #Is the post of its likings or not?
            "elements": POST_VALENCE,
            "depends_on": ["post_valence"]
        },
    },
    "controls": {
        "post_action": {
            "elements": L1_ACTIONS
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
model.A["post_obs"]["post", "post"] = 1 

model.A["post_valence"]["like", "like"] = 1
model.A["post_valence"]["like", "like"] = 1 


# B transitions 
# B["to", "from", "action"]
model.B["post_valence"]["like", "like", "scroll"] = p_like
model.B["post_valence"]["like", "dislike", "scroll"] = p_like
model.B["post_valence"]["dislike", "like", "scroll"] = p_dislike
model.B["post_valence"]["dislike", "dislike", "scroll"] = p_dislike



