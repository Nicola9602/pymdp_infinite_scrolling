# Fixed JAX-style hierarchical scrolling model

Files:

- `config.py`: shared labels and parameters.
- `L1_user.py`: Level 1 generative model, returned via `init_L1_model()`.
- `L1_scrolling_env.py`: Level 1 generative process. Samples posts and intermittent quick reward with explicit JAX PRNG keys.
- `L2_user.py`: Level 2 marshmallow/self-control generative model.
- `scrolling_env.py`: Level 2 generative process for self-state transitions and affect/delayed reward.
- `hier_link_utils.py`: L1-to-L2 communication utilities.
- `act_inf_loop.py`: manual two-level active inference loop using current JAX-style `pymdp.agent.Agent` API.

This version intentionally does not use `rollout()` yet. The two-level custom loop is easier to debug manually. Later, you can wrap the whole hierarchy into a custom JAX `Env` and use `rollout()`.
