# act_inf_loop.py

from jax import random as jr
from pymdp.agent import Agent

from config import L2_ACTIONS
from L1_user import init_L1_model
from L2_user import init_L2_model
from L1_scrolling_env import L1ScrollingEnv
from L2_scrolling_env import L2ScrollingEnv
from hier_link_utils import (
    get_l2_obs_from_l1,
    l1_reward_to_l2_reward_label,
)
from plot_history import plot_history

VERBOSE = True
T2 = 12
rng_key = jr.PRNGKey(4)


# ------------------------------------------------------------------
# Initialize L1 agent and L1 generative process
# ------------------------------------------------------------------

l1_model, mappings_L1, dimensions_L1 = init_L1_model()

l1_agent = Agent(
    **l1_model,
    batch_size=1,
    policy_len=1,
    gamma=10.0,
)

env_l1 = L1ScrollingEnv()


# ------------------------------------------------------------------
# Initialize L2 agent and L2 generative process
# ------------------------------------------------------------------

l2_model, mappings_L2, dimensions_L2 = init_L2_model()

l2_agent = Agent(
    **l2_model,
    batch_size=1,
    policy_len=4,
    gamma=10.0,
)

l2_env = L2ScrollingEnv(starting_state="scrolling_ok")


# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------

history = []


# ------------------------------------------------------------------
# Main L2 loop
# ------------------------------------------------------------------

for t2 in range(T2):

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
    qs_l2 = l2_agent.infer_states(
        obs_l2,
        empirical_prior=l2_agent.D,
    )

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

    # --------------------------------------------------------------
    # 5. Store everything
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
            "qs_l2": qs_l2,
            "q_pi_L2": q_pi_L2,
            "neg_efe_L2": neg_efe_L2,
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
            f"action={action_label} | "
            f"state_after={true_self_state_after}"
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
    filename="scrolling_flow.png",
    show=True,
)