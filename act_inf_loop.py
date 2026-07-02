# act_inf_loop.py

from jax import random as jr
from jax import numpy as jnp
from pymdp.agent import Agent

from L1_user import init_L1_model
from L2_user import init_L2_model
from L3_user import init_L3_model
from L1_scrolling_env import L1ScrollingEnv
from L2_scrolling_env import L2ScrollingEnv
from config import (
    L2_ACTIONS,
    POST_REWARD,
    POST_VALENCE,
    SELF_STATE,
    L3_UPDATE_EVERY,
    L3_CONTROL_STATE,
    L3_ACTIONS,
    ALGORITHM_MODE,
    FEED_PULL_INITIAL,
)
from hier_link_utils import (
    get_l2_obs_from_l1,
    l1_reward_to_l2_reward_label,
    get_l3_obs_from_l2_history,
    choose_l3_meta_action_from_belief,
    apply_l3_meta_action_to_l2_env,
)
from plot_history import plot_history
from feed_algorithm import (
    classify_algorithm_event,
    update_feed_pull,
    apply_feed_pull_to_env,
)

VERBOSE = True
T2 = 100 

SEED = 4
rng_key = jr.PRNGKey(SEED)

# L2 planning horizon 
#L2_POLICY_LEN = 1  # more flow-like / myopic
L2_POLICY_LEN = 4  # more reflective
#L2_POLICY_LEN = 6  # even more reflective

# L2 belief handling
# "reset" = every timestep starts from l2_agent.D
# "posterior" = next timestep starts from previous qs_l2 directly
# "predictive" = next timestep starts from B-predicted prior after chosen action
# L2_PRIOR_MODE = "reset"
# L2_PRIOR_MODE = "posterior"
L2_PRIOR_MODE = "predictive"


# L2 environment parameters 
L2_STARTING_STATE = "scrolling_ok"
L2_P_HOOKED_FROM_OK = 0.3
L2_P_AWARENESS_WHEN_HOOKED = 0.15

# example of already doomscrolling but unaware 
# L2_STARTING_STATE = "scrolling_hooked"
# L2_P_AWARENESS_WHEN_HOOKED = 0.05

L3_UPDATE_EVERY = 20
FEED_PULL_INITIAL = 0.30


# ------------------------------------------------------------------
# Initialize L1 agent and L1 generative process
# ------------------------------------------------------------------

l1_model, mappings_L1, dimensions_L1 = init_L1_model()

l1_agent = Agent(
    **l1_model,
    batch_size=1,
    policy_len=1,
    gamma=10.0, #Policy precision if one policy has higher FE it will not choose that 
)

env_l1 = L1ScrollingEnv()


# ------------------------------------------------------------------
# Initialize L2 agent and L2 generative process
# ------------------------------------------------------------------

l2_model, mappings_L2, dimensions_L2 = init_L2_model()

l2_agent = Agent(
    **l2_model,
    batch_size=1,
    policy_len=L2_POLICY_LEN,
    gamma=10.0,
)

l2_env = L2ScrollingEnv(
    starting_state=L2_STARTING_STATE,
    p_hooked_from_ok=L2_P_HOOKED_FROM_OK,
    p_awareness_when_hooked=L2_P_AWARENESS_WHEN_HOOKED,
)

# This variable stores the prior used by L2 at the next timestep.
# In reset mode, we ignore it and always use l2_agent.D.
# In posterior mode, we update it after each inference step.
l2_empirical_prior = l2_agent.D

# ------------------------------------------------------------------
# Initialize L3 agent
# -----------------------------------------------------------------
# L3 is the metacognitive overseer.
#
# It does not directly choose L2 actions.
# It observes recent L2 behavior and infers whether the system is:
#
#     in_control
#     not_in_control
#
# Then, through a rule-based bridge, it perturbs L2 by changing
# l2_env.p_awareness_when_hooked.
#
# This means L3 acts by changing how likely hookedness is to become visible
# to L2 as "bad_feeling".

l3_model, mappings_L3, dimensions_L3 = init_L3_model()

l3_agent = Agent(
    **l3_model, 
    batch_size=1,
    policy_len=1,
    gamma=10.0,
)

# This stores L3's prior from one L3 update to the next.
# We use it so L3 has temporal continuity instead of resetting every time.
l3_empirical_prior = l3_agent.D

# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

history = []

# These two lists store the recent behavior that L3 will read.
#
# l2_action_history:
#     stores actions such as:
#         "continue_scrolling"
#         "opt_out"
#
# l2_belief_history:
#     stores L2's believed self-state at each timestep, such as:
#         "scrolling_ok"
#         "scrolling_hooked"
#         "withdrawal_1"
#
# Right now the simplified L3 evidence function mostly uses action history.
# We still store belief history because it may be useful again later.
l2_action_history = []
l2_belief_history = []

# ------------------------------------------------------------------
# Minimal adaptive feed algorithm
# ------------------------------------------------------------------
#
# This is not a full platform POMDP.
# It is a simple adaptive environment variable:
#
#     feed_pull in [0, 1]
#
# Higher feed_pull means:
#     - more negative/salient content
#     - more quick reward
#     - higher p_hooked_from_ok
#     - lower spontaneous awareness, unless L3 intervenes

feed_pull = FEED_PULL_INITIAL

# This keeps track of the most recent L3 intervention.
# If L3 chooses stay_in_control, the feed algorithm will not overwrite
# the awareness boost at the next timestep.
current_l3_meta_action = "stay_hooked"

algorithm_event = "initial"

feed_params = apply_feed_pull_to_env(
    feed_pull=feed_pull,
    env_l1=env_l1,
    l2_env=l2_env,
    l3_meta_action=current_l3_meta_action,
)
# ------------------------------------------------------------------
# Main L2 loop
# ------------------------------------------------------------------

for t2 in range(T2):
    # --------------------------------------------------------------
    # 0. Adaptive feed algorithm update
    # --------------------------------------------------------------
    #
    # The algorithm reads the previous completed timestep from history.
    # Then it updates feed_pull and applies the new feed parameters.
    #
    # This happens before the next L1 post is generated.

    algorithm_event = classify_algorithm_event(history)

    previous_feed_pull = feed_pull

    last_l2_belief = None
    if len(history) > 0:
        last_l2_belief = history[-1]["l2_belief_self_state"]

    feed_pull = update_feed_pull(
        feed_pull=feed_pull,
        algorithm_mode=ALGORITHM_MODE,
        algorithm_event=algorithm_event,
        l2_belief_label=last_l2_belief,
        l3_meta_action=current_l3_meta_action,
    )

    feed_params = apply_feed_pull_to_env(
        feed_pull=feed_pull,
        env_l1=env_l1,
        l2_env=l2_env,
        l3_meta_action=current_l3_meta_action,
    )
    # MOdifies the feed before L1 generates next post 

    true_self_state_before = l2_env.self_state
    feed_active = l2_env.is_feed_active()
    rng_key, key_affect = jr.split(rng_key)
    affect_label = l2_env.get_affect_obs(key_affect)

    posterior_L1 = None
    l1_info = None

    # --------------------------------------------------------------
    # 1. If feed is active, run one L1 perceptual episode
    # --------------------------------------------------------------

    if feed_active:

        rng_key, key_l1_env = jr.split(rng_key)

        obs_l1, l1_info = env_l1.step(
            key_l1_env,
            action="scroll",
        )

        # obs_l1 format:
        #     [jnp.array([valence_idx]), jnp.array([reward_idx])]
        qs_l1 = l1_agent.infer_states(
            obs_l1,
            empirical_prior=l1_agent.D,
        )

        # To check qs_l1 posterior uncomment this which tells you what the obs 
        # dict(zip(POST_VALENCE, qs_l1[0].squeeze().tolist())) 
        

        posterior_L1 = qs_l1

        reward_label = l1_reward_to_l2_reward_label(
            l1_info["post_reward_label"]
        )

    else:

        reward_label = l2_env.get_nonfeed_reward_obs()

    # --------------------------------------------------------------
    # 2. Convert L1 posterior plus reward/affect into L2 observation
    # --------------------------------------------------------------

    obs_l2 = get_l2_obs_from_l1(
        posterior_L1=posterior_L1,
        reward_label=reward_label,
        affect_label=affect_label,
        feed_active=feed_active,
    )

    # obs_l2 format:
    #     [
    #       jnp.array([valence_report_idx]),
    #       jnp.array([reward_idx]),
    #       jnp.array([affect_idx]),
    #     ]

    if L2_PRIOR_MODE == "reset":
        empirical_prior_l2 = l2_agent.D

    elif L2_PRIOR_MODE in ["posterior", "predictive"]:
        empirical_prior_l2 = l2_empirical_prior

    else:
        raise ValueError(
            "L2_PRIOR_MODE must be 'reset', 'posterior', or 'predictive'"
        )

    qs_l2 = l2_agent.infer_states(
        obs_l2,
        empirical_prior=empirical_prior_l2,
    )
    # Human-readable summary of the L2 posterior over self_state.
    self_state_belief = jnp.squeeze(qs_l2[0])
    l2_belief_idx = int(jnp.argmax(self_state_belief))
    l2_belief_label = SELF_STATE[l2_belief_idx]
    l2_belief_prob = float(self_state_belief[l2_belief_idx])

    # --------------------------------------------------------------
    # 3. L2 evaluates policies and samples action
    # --------------------------------------------------------------

    q_pi_L2, neg_efe_L2 = l2_agent.infer_policies(qs_l2)

    rng_key, key_action, key_l2_env = jr.split(rng_key, 3)

    action_keys = jr.split(
        key_action,
        l2_agent.batch_size + 1,
    )

    chosen_action_L2 = l2_agent.sample_action(
        q_pi_L2,
        rng_key=action_keys[1:],
    )

    action_idx = int(chosen_action_L2[0][0])
    action_label = L2_ACTIONS[action_idx]

    # --------------------------------------------------------------
    # 4. Transition true L2 environment
    # --------------------------------------------------------------

    true_self_state_after = l2_env.step(
        key_l2_env,
        action_label,
    )

    if L2_PRIOR_MODE == "posterior":
        # Direct posterior carry-over.
        # This remembers beliefs, but does not advance them through B.
        l2_empirical_prior = qs_l2

    elif L2_PRIOR_MODE == "predictive":
        # Action-predictive carry-over.
        # This uses the chosen action and B to predict next timestep's prior.
        l2_empirical_prior = l2_agent.update_empirical_prior(
            chosen_action_L2,
            qs_l2,
        )
    print("L2 inference algorithm:", getattr(l2_agent, "inference_algo", None))

    # --------------------------------------------------------------
    # 5. Store recent L2 behavior for L3
    # --------------------------------------------------------------
    #
    # L3 does not observe posts, rewards, or affect directly.
    # It observes a compressed summary of L2 behavior over the last K steps.
    #
    # So after L2 has acted, we append:
    #
    #     1. what L2 did
    #     2. what L2 believed about its own self-state
    #
    # This becomes the behavioral history that L3 reads.

    l2_action_history.append(action_label)
    l2_belief_history.append(l2_belief_label)


    # --------------------------------------------------------------
    # 6. Optional L3 update
    # --------------------------------------------------------------
    #
    # L3 does not update every L2 timestep.
    #
    # Instead, it updates every L3_UPDATE_EVERY timesteps.
    # For example, if L3_UPDATE_EVERY = 20, then L3 checks behavior at:
    #
    #     t2 = 19, 39, 59, ...
    #
    # because at t2 = 19 we have collected 20 L2 actions.
    #
    # The L3 update affects the NEXT L2 timestep, because it changes
    # l2_env.p_awareness_when_hooked after the current L2 step is complete.

    l3_update_ran = False
    l3_evidence_label= None
    l3_control_state_label = None
    l3_control_state_prob = None
    l3_meta_action = None
    l3_awareness_after = l2_env.p_awareness_when_hooked

    if (t2+1) % L3_UPDATE_EVERY == 0:
        # Build the L3 observation from the last K L2 actions.
        # obs_l3 has pymdp/JAX format:[jnp.array([evidence_idx], dtype=jnp.int32)]

        obs_l3, l3_evidence_label = get_l3_obs_from_l2_history(
            l2_action_history=l2_action_history,
            l2_belief_history=l2_belief_history,
            k=L3_UPDATE_EVERY)

    # L3 infers its hidden control state from behav evidence 
    # if obs_l3 is not none

        if obs_l3 is not None: 
            l3_update_ran = True 

            qs_l3 = l3_agent.infer_states(
                obs_l3,
                empirical_prior=l3_empirical_prior,
            )
        
            # Convert L3 posterior into a human-readable label

            control_state_belief = jnp.squeeze(qs_l3[0])
            l3_control_state_idx = int(jnp.argmax(control_state_belief))

            l3_control_state_label = L3_CONTROL_STATE[l3_control_state_idx]
            l3_control_state_prob = float(control_state_belief[l3_control_state_idx])

            #Chosse l3 action 
            # if l3 thinks the system is not in control returns "stay_in_control"
            # meaning: activate the awareness/control itnervention
            # otherwise it returns: "stay_hooked"+
            # meaning: do not intervene

            l3_meta_action = choose_l3_meta_action_from_belief(qs_l3)
            current_l3_meta_action = l3_meta_action

            # APply the l3 meta action to the l2 environment. 
            # This does not force any actioni it just changes p_awareness_when_hooked
            # High awareness means hookedness is more likely to became a "bad feeling"
            #Low awareness means hookedness is more likely to stay hidden as "scroll_relief"

            l3_awareness_after = apply_l3_meta_action_to_l2_env(
                l2_env = l2_env,
                meta_action=l3_meta_action,
            )

            #Convert this into pymdp action format. 
            # This lets us update l3's empirical prior through its B matrix.

            l3_meta_action_idx = L3_ACTIONS.index(l3_meta_action)

            chosen_action_L3 = [jnp.array([l3_meta_action_idx], dtype=jnp.int32)]

            # Update l3's priors 

            l3_empirical_prior = l3_agent.update_empirical_prior(
                chosen_action_L3,
                qs_l3,
            )

    # --------------------------------------------------------------
    # 7. Store everything
    # --------------------------------------------------------------

    history.append(
        {
            "t2": t2,
            "true_self_state_before": true_self_state_before,
            "true_self_state_after": true_self_state_after,
            "feed_active": feed_active,
            "l1_true_valence": None if l1_info is None else l1_info["true_valence_label"],
            "l1_reward": None if l1_info is None else l1_info["post_reward_label"],
            "l2_reward_report": reward_label,
            "l2_affect_obs": affect_label,
            "l2_action": action_label,
            "l2_prior_mode": L2_PRIOR_MODE,
            "l2_belief_self_state": l2_belief_label,
            "l2_belief_self_prob": l2_belief_prob,
            "qs_l2": qs_l2,
            "q_pi_L2": q_pi_L2,
            "neg_efe_L2": neg_efe_L2,
            "l3_update_ran": l3_update_ran,
            "l3_evidence": l3_evidence_label,
            "l3_control_state": l3_control_state_label,
            "l3_control_state_prob": l3_control_state_prob,
            "l3_meta_action": l3_meta_action,
            "l2_awareness_when_hooked": l3_awareness_after,
            "algorithm_mode": ALGORITHM_MODE,
            "algorithm_event": algorithm_event,
            "feed_pull": feed_pull,
            "previous_feed_pull": previous_feed_pull,
            "feed_valence_probs": feed_params["valence_probs"],
            "feed_reward_probs": feed_params["reward_probs"],
            "feed_p_hooked_from_ok": feed_params["p_hooked_from_ok"],
            "feed_p_awareness_when_hooked": feed_params["actual_p_awareness_when_hooked"],
        }
    )

    # --------------------------------------------------------------
    # 6. Live print for this timestep
    # --------------------------------------------------------------

    if VERBOSE:
        l1_valence = None if l1_info is None else l1_info["true_valence_label"]
        l1_reward = None if l1_info is None else l1_info["post_reward_label"]

        print(
            f"t2={t2:02d} | "
            f"state_before={true_self_state_before} | "
            f"feed_active={feed_active} | "
            f"L1_valence={l1_valence} | "
            f"L1_reward={l1_reward} | "
            f"L2_reward={reward_label} | "
            f"affect={affect_label} | "
            f"L2_belief={l2_belief_label}({l2_belief_prob:.2f}) | "
            f"action={action_label} | "
            f"state_after={true_self_state_after} | "
            f"alg={ALGORITHM_MODE}:{algorithm_event} | "
            f"feed_pull={feed_pull:.2f} | "
            f"p_hook={feed_params['p_hooked_from_ok']:.2f} | "
            f"p_aware={feed_params['actual_p_awareness_when_hooked']:.2f}"
        )
        if l3_update_ran:
            print(
                f"      L3 update | "
                f"evidence={l3_evidence_label} | "
                f"L3_belief={l3_control_state_label}"
                f"({l3_control_state_prob:.2f}) | "
                f"meta_action={l3_meta_action} | "
                f"p_awareness_when_hooked={l3_awareness_after:.2f}"
            )


# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------

print("\nSimulation finished.")
print(f"Number of L2 timesteps: {len(history)}")

print("\nFinal timestep:")

final_row = history[-1]

print(
    {
        "t2": final_row["t2"],
        "true_self_state_before": final_row["true_self_state_before"],
        "true_self_state_after": final_row["true_self_state_after"],
        "feed_active": final_row["feed_active"],
        "l1_true_valence": final_row["l1_true_valence"],
        "l1_reward": final_row["l1_reward"],
        "l2_reward_report": final_row["l2_reward_report"],
        "l2_affect_obs": final_row["l2_affect_obs"],
        "l2_action": final_row["l2_action"],
    }
)

print("\nAction sequence:")

for row in history:
    print(
        f"t2={row['t2']:02d} | "
        f"state_before={row['true_self_state_before']} | "
        f"state_after={row['true_self_state_after']} | "
        f"reward={row['l2_reward_report']} | "
        f"affect={row['l2_affect_obs']} | "
        f"action={row['l2_action']}"
    )

plot_history(
    history,
    filename=(
        f"scrolling_flow_"
        f"{ALGORITHM_MODE}_"
        f"{L2_PRIOR_MODE}_"
        f"h{L2_POLICY_LEN}_"
        f"seed{SEED}.png"
    ),
    show=True,
)