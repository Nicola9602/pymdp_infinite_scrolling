import pymdp
import jax.tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import Env
from pymdp.envs import rollout


#This is actually wrong and must be updated later. 
#the positive negative should come from the lower level not be
# hardoced here 
from config import (
    POST_VALENCE,
    L1_VALENCE_REPORT,
    SELF_STATE,
    REWARD_REPORT,
    AFFECT_OBS,
    L2_ACTIONS,
    P_POSITIVE,
    P_NEGATIVE,
    P_NEUTRAL,
)


def init_L2_model():
    """
    Level 2: marshmallow/self-control level.

    This level does not directly generate raw posts.
    It receives a report from L1:
        positive / negative / neutral

    If the agent opts out, L1 is not run and L2 receives:
        no_post
    """


    model_description = {
        "observations": {
            "l1_valence_report": {
                "elements": L1_VALENCE_REPORT,
                "depends_on": ["self_state", "l1_valence_state"],
            },
            "reward_report": {
                "elements": REWARD_REPORT,
                "depends_on": ["self_state"],
            },
            "affect_obs": {
                "elements": AFFECT_OBS,
                "depends_on": ["self_state"],
            },
        },

        "controls": {
            "choice": {
                "elements": L2_ACTIONS,
            },
        },

        "states": {
            "self_state":{
                # self_state_t+1 depends on self_state_t and choice_t
                "elements": SELF_STATE,
                "depends_on":["self_state"],
                "controlled_by": ["choice"],
            },
            "l1_valence_state":{
                "elements": POST_VALENCE,
                "depends_on": ["l1_valence_state"],
                "controlled_by": ["choice"],
            },
        },
    }

    model = compile_model(model_description)

    #========================================================================
    # A likelihood obs mappings
    # A[observation, hidden state]
    #========================================================================


    # A[l1_valence_report][obsrtrvation, self_state, l1_valence_state]
    #------------------------------------------------------------------------
    # TODO: If the agent is scrolling L2 receives the L1 valence report
    # this connects the levels check if this really happens 

    # TODO: For now scrolling_ok and hooked are the same this could change later
    #I keep them in case. 
    for valence in POST_VALENCE:
        model.A["l1_valence_report"][valence, "scrolling_ok", valence] = 1.0
        model.A["l1_valence_report"][valence, "scrolling_hooked", valence] = 1.0

    #If the agent has opted out, the agent recieves no post regardless of valence
    for valence in POST_VALENCE:
        model.A["l1_valence_report"]["no_post", "withdrawal_1", valence] = 1.0
        model.A["l1_valence_report"]["no_post", "withdrawal_2", valence] = 1.0
        model.A["l1_valence_report"]["no_post", "recovered", valence] = 1.0

    model.A["l1_valence_report"].normalize()


    # A["reward_report"][observation, self_state]
    #------------------------------------------------------------------------
    
    #TODO: it must be made clear the difference in the rewards... like small vs big, 
    # this should be temporally dependent... 
    
    # Scrolling gives small immediate rewards.
    # Opting out gives no reward at first, then a larger delayed reward.
    model.A["reward_report"]["quick_reward", "scrolling_ok"] = 1.0
    model.A["reward_report"]["quick_reward", "scrolling_hooked"] = 1.0

    # Opting out gives no reward at first, then a larger delayed reward.
    model.A["reward_report"]["no_reward", "withdrawal_1"] = 1.0
    model.A["reward_report"]["no_reward", "withdrawal_2"] = 1.0

    model.A["reward_report"]["delayed_reward", "recovered"] = 1.0

    model.A["reward_report"].normalize()

    # A["affect_obs"][observation, self_state]
    #------------------------------------------------------------------------
    # This is where the "free-energy spike" after quitting is represented
    # phenomenologically: as urge/discomfort observations.
    model.A["affect_obs"]["okay", "scrolling_ok"] = 1.0
    model.A["affect_obs"]["bad_feeling", "scrolling_hooked"] = 1.0

    model.A["affect_obs"]["urge", "withdrawal_1"] = 1.0
    model.A["affect_obs"]["urge", "withdrawal_2"] = 1.0

    model.A["affect_obs"]["relief", "recovered"] = 1.0

    model.A["affect_obs"].normalize()


    #========================================================================
    # B Transitions
    # B["to", "from", "action"]
    #========================================================================

    # B["self_state"][next_state, previous_state, action]
    #========================================================================

    # Action: continue_scrolling
    #----------------------------------------------------------------------------
    # - scrolling_ok may stay okay or become hooked

    # TODO: this has to be improved. For example there must be a point in which the guy goes into scrolling_hooked
    # so say after 10 times scrolling or 50 times goes in hooked and when in hooked coming back the withdrawal must be more costly
    # that is restarting something must be most costy. 
     
    model.B["self_state"]["scrolling_ok", "scrolling_ok", "continue_scrolling"] = 0.7
    model.B["self_state"]["scrolling_hooked", "scrolling_ok", "continue_scrolling"] = 0.3
    
    # - scrolling_hooked stays hooked
    model.B["self_state"]["scolline_hooked", "scolling_hooked", "continue_scrolling"] = 1.0

    # - continuing during withdrawal is relapse
    model.B["self_state"]["scrolling_hooked", "withdrawal_1", "continue_scrolling"] = 1.0
    model.B["self_state"]["scrolling_hooked", "withdrawal_2", "continue_scrolling"] = 1.0

    # - continuing after recovery restarts scrolling
    model.B["self_state"]["scrolling_ok", "recovered", "continue_scrolling"] = 1.0

    # Action: opt_out
    #----------------------------------------------------------------------------

    # TODO: here do I want to have the same as withdrawal 1 and 2? So basically the agent
    #always goes from both of them? or maybe depending on its state it can only go trhough one 
    # for exmaple, if it scrolled only < 10 times or > 100 then withdrawal_2 is skipped?

    # when opting out first comes withdrawal/urge
    model.B["self_state"]["withdrawal_1", "scrolling_ok", "opt_out"] = 1.0 
    model.B["self_state"]["withdrawal_2", "scrolling_ok", "opt_out"] = 1.0

     # then recovery

    model.B["self_state"]["withdrawal_2", "withdrawal_1", "opt_out"] = 1.0
    model.B["self_state"]["recovered", "withdrawal_2", "opt_out"] = 1.0
    model.B["self_state"]["recovered", "recovered", "opt_out"] = 1.0

    model.B["self_state"].normalize()

    # B["valence_state"]["next_valence", "prtevious_valence", "action"]
    #========================================================================
    #
    # TODO: is this working? I am not sure is this what it should happen? And should the 
    # agent have a belief about which post will be reported? maybe not. It should have a transition
    # in its own valence state given the report? Maybe the l1_valence_state state doesn't make sense?

    # This is L2's belief about what kind of post L1 will report if
    # scrolling continues. The actual observation will still come from L1.
    for previous_valence in POST_VALENCE:
        model.B["l1_valence_state"]["positive", previous_valence, "continue_scrolling"] = P_POSITIVE
        model.B["l1_valence_state"]["negative", previous_valence, "continue_scrolling"] = P_NEGATIVE
        model.B["l1_valence_state"]["neutral", previous_valence, "continue_scrolling"] = P_NEUTRAL

    # If opted out, this hidden valence state is irrelevant because
    # l1_valence_report will be no_post. Keep it stable.
    for valence in POST_VALENCE:
        model.B["l1_valence_state"][valence, valence, "opt_out"] = 1.0

    model.B["l1_valence_state"].normalize()

    #========================================================================
    # C: Preferences 
    # C["what the agent observes"]["which observation"] = how much the agent likes/dislikes it
    #========================================================================

    # Content itself is not yet what drives the marshmallow challenge.
    # TODO: Leave it neutral for now.
    for report in L1_VALENCE_REPORT:
        model.C["l1_valence_report"][report] = 0.0

    # TODO: set this as variables I can change form another file ?
    # No reward is mildly bad.
    model.C["reward_report"]["no_reward"] = -0.5     
    # Small immediate reward is good.
    model.C["reward_report"]["quick_reward"] = 1.0 
    # Delayed reward is better.
    model.C["reward_report"]["delayed_reward"] = 4.0

    # Urge and bad are aversive 
    model.C["affect_obs"]["okey"] = 0.5
    model.C["affect_obs"]["urge"] = -2.0
    model.C["affect_obs"]["bad_feelingd"] = -5.0
    model.C["affect_obs"]["relief"] = 3.0


    #========================================================================
    # D priors
    # D["which hidden state factor"]["which starting state"] =
    # probability that the agent believes it starts in that hidden state before obs
    #========================================================================

    #Priors over self_state
    #--------------------------------------------------------------------------
    # The agent thinks to start from scrolling_ok

    model.D["self_state"]["scrolling_ok"] = 1.0
    model.D["self_state"]["scrolling_hooked"] = 0.0
    model.D["self_state"]["withdrawal_1"] = 0.0
    model.D["self_state"]["withdrawal_2"] = 0.0
    model.D["self_state"]["recovered"] = 0.0
    model.D["self_state"].normalize()

    #Priors over valence state
    #--------------------------------------------------------------------------
    #Equal priors on the valence state doesn't know where it is strating from 

    model.D["l1_valence_state"]["positive"] = 1.0 / 3.0
    model.D["l1_valence_state"]["negative"] = 1.0 / 3.0
    model.D["l1_valence_state"]["neutral"] = 1.0 / 3.0
    model.D["l1_valence_state"].normalize()


    #========================================================================
    # Function closure 
    #========================================================================
    
    mappings = {
        "l1_valence_report_names": L1_VALENCE_REPORT,
        "self_state_names": SELF_STATE,
        "l1_valence_state_names": POST_VALENCE,
        "reward_report_names": REWARD_REPORT,
        "affect_obs_names": AFFECT_OBS,
        "choice_names": L2_ACTIONS,
    }

    dimensions = {
        "num_obs": [
            len(L1_VALENCE_REPORT),
            len(REWARD_REPORT),
            len(AFFECT_OBS),
        ],
        "num_states": [
            len(SELF_STATE),
            len(POST_VALENCE),
        ],
        "num_controls": [
            len(L2_ACTIONS),
        ],
    }

    return model, mappings, dimensions