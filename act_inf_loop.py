# act_inf_loop.py

from pymdp.agent import Agent

from config import L2_ACTIONS
from L1_user import init_L1_model  
from L2_user import init_L2_model
from L1_scrolling_env import L1ScrollingEnv
from scrolling_env import L2ScrollingEnv
from hier_link_utils import get_l2_obs_from_l1


T2 = 12

# ------------------------------------------------------------------
# Initialize L1
# ------------------------------------------------------------------

l1_model, mappings_L1, dimensions_L1 = init_L1_model()

# Depending on your pymdp version, you may need to convert labelled
# distributions to object arrays before passing them to Agent.
l1_agent = Agent(
    A=l1_model.A,
    B=l1_model.B,
    C=l1_model.C,
    D=l1_model.D,
    policy_len=1,
)

# ------------------------------------------------------------
# Initialize L1 environment / generative process
# ------------------------------------------------------------

env_l1 = L1ScrollingEnv(
    p_positive=0.5,
    p_negative=0.3,
    p_neutral=0.2,
    p_reward_if_positive=0.5,
    p_reward_if_negative=0.7,
    p_reward_if_neutral=0.1,
    seed=4,
)

# ------------------------------------------------------------------
# Initialize L2
# ------------------------------------------------------------------

l2_model, mappings_L2, dimensions_L2 = init_L2_model()

# Important: the marshmallow task needs enough policy depth to see
# opt_out → withdrawal_1 → withdrawal_2 → recovered.
l2_agent = Agent(
    A=l2_model.A,
    B=l2_model.B,
    C=l2_model.C,
    D=l2_model.D,
    policy_len=4,
)

# ------------------------------------------------------------------
# Initialize L2 generative process
# ------------------------------------------------------------------

l2_env = L2ScrollingEnv(starting_state="scrolling_ok")
# TODO: here am I not deciding myself what is going to happen? Shouldn't this happen
# by itself?
current_l2_action = "continue_scrolling"

history = []

# ------------------------------------------------------------------
# Main L2 loop
# ------------------------------------------------------------------

for t2 in range(T2):

    # 1. Step the L2 environment using the previous L2 action.
    #
    # This determines whether the feed is active, and what reward/affect
    # observation the higher level receives.
    reward_label, affect_label, true_self_state = l2_env.step(
        current_l2_action
    )

    feed_active = l2_env.is_feed_active()

    # 2. If the feed is active, run one L1 episode.
    if feed_active:

        # Reset L1 for this lower-level episode.
        # In a deep temporal model, one L2 timestep can correspond to
        # a complete lower-level L1 trial.
        l1_agent.reset()

        # The L1 environment generates a post.
        obs_l1 = env_l1.step("scroll")

        # The L1 agent infers the hidden post valence.
        posterior_L1 = l1_agent.infer_states(obs_l1)

    else:
        # If L2 has opted out, the lower-level post stream is not sampled.
        posterior_L1 = None

    # 3. Convert L1 posterior into an L2 observation.
    #
    # This is the L1 -> L2 communication point.
    obs_l2 = get_l2_obs_from_l1(
        posterior_L1 = posterior_L1,
        dimensions_L2 = dimensions_L2,
        reward_label = reward_label,
        affect_label = affect_label,
        feed_active = feed_active)
    
    # 4. L2 updates its posterior beliefs using the L1-derived observation.
    # TODO: this seems wrong? Should I do it mechanically or...?
    posterior_L2 = l2_agent.infer_states(
        obs_l2,
        distr_obs=True,
    )    

    # 5. L2 evaluates policies.
    q_pi_L2, G_L2 = l2_agent.infer_policies()

    # 6. L2 samples next action.
    chosen_action_L2 = l2_agent.sample_action()

    # Depending on your pymdp version, this may need adjustment.
    # Usually the first control factor is the only relevant one here.
    action_idx = int(chosen_action_L2[0])
    current_l2_action = L2_ACTIONS[action_idx]

    history.append(
        {
            "t2": t2,
            "true_self_state": true_self_state,
            "reward": reward_label,
            "affect": affect_label,
            "feed_active": feed_active,
            "l2_action_next": current_l2_action,
            "posterior_L2": posterior_L2,
            "q_pi_L2": q_pi_L2,
            "G_L2": G_L2,
        }
    )