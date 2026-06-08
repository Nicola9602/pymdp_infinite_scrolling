config.py
    Shared labels and parameter values.

L1_user.py
    Your lower-level passive scrolling model.
    This already exists.

L2_user_agent_init.py
    Only the L2 generative model:
        observations
        states
        controls
        A, B, C, D

scrolling_env.py
    The generative process for the higher level:
        true self_state
        reward/affect observations after continue_scrolling or opt_out

hier_link_utils.py
    Communication functions:
        L1 posterior → L2 observation

act_inf_loop.py
    The actual simulation loop:
        run L1 if scrolling
        convert L1 posterior into L2 observation
        run L2 inference
        sample L2 action
        step environment