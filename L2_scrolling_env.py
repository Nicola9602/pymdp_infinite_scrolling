# L2_scrolling_env.py

from jax import random as jr

from config import SELF_STATE


class L2ScrollingEnv:
    """
    Level 2 generative process for the marshmallow task.

    This environment tracks the true self_state.

    It does NOT generate positive/negative/neutral posts. Those come from L1.
    It does generate:
        - whether the feed is active
        - affect observations from self_state
        - delayed/no reward when the feed is inactive
    """

    def __init__(self, starting_state="scrolling_ok", p_hooked_from_ok=0.3, p_awareness_when_hooked=0.15):
        if starting_state not in SELF_STATE:
            raise ValueError(f"Unknown starting state: {starting_state}")

        self.state_idx = SELF_STATE.index(starting_state)
        self.p_hooked_from_ok = p_hooked_from_ok
        self.p_awareness_when_hooked = p_awareness_when_hooked

    @property
    def self_state(self):
        return SELF_STATE[self.state_idx]

    def reset(self, starting_state="scrolling_ok"):
        if starting_state not in SELF_STATE:
            raise ValueError(f"Unknown starting state: {starting_state}")
        self.state_idx = SELF_STATE.index(starting_state)
        return self.self_state

    def is_feed_active(self):
        return self.self_state in ["scrolling_ok", "scrolling_hooked"]

    # Now the true environment can sometimes generate bad_feeling, 
    # but most hooked states still feel like scroll_relief.
    def get_affect_obs(self, key = None):
        if self.self_state == "scrolling_ok":
            return "scroll_relief"
        if self.self_state == "scrolling_hooked":
            if key is None: 
                return "scroll_relief"
            realizes_hooked = bool(
                jr.bernoulli(key, p=self.p_awareness_when_hooked)
            )
            if realizes_hooked: 
                return "bad_feeling"
            return "scroll_relief"
    
        if self.self_state in ["withdrawal_1", "withdrawal_2"]:
            return "urge"
        if self.self_state == "recovered_reward":
            return "deep_relief"
        if self.self_state == "recovered_no_reward":
            return "boredom"
        raise RuntimeError(f"Unhandled self_state: {self.self_state}")

    def get_nonfeed_reward_obs(self):
        """
        Reward observation when the feed is not active.
        If the feed is active, quick reward/no_reward comes from L1.
        """
        if self.self_state in ["scrolling_ok", "scrolling_hooked"]:
            raise RuntimeError("Feed is active; quick reward should come from L1.")
        if self.self_state in ["withdrawal_1", "withdrawal_2"]:
            return "no_reward"
        if self.self_state == "recovered_reward":
            return "delayed_reward"
        if self.self_state == "recovered_no_reward":
            return "no_reward"
        raise RuntimeError(f"Unhandled self_state: {self.self_state}")

    def step(self, key, action):
        """
        Apply the chosen L2 action and update true self_state.
        Randomness is explicit through a JAX PRNG key.
        """
        if action == "continue_scrolling":
            self._continue_scrolling(key)
        elif action == "opt_out":
            self._opt_out(key)
        else:
            raise ValueError(f"Unknown L2 action: {action}")

        return self.self_state

    def _continue_scrolling(self, key):
        if self.self_state == "scrolling_ok":
            hooked = bool(jr.bernoulli(key, p=self.p_hooked_from_ok))
            self.state_idx = SELF_STATE.index("scrolling_hooked" if hooked else "scrolling_ok")
        elif self.self_state == "scrolling_hooked":
            self.state_idx = SELF_STATE.index("scrolling_hooked")
        elif self.self_state in ["withdrawal_1", "withdrawal_2"]:
            # Relapse during withdrawal.
            self.state_idx = SELF_STATE.index("scrolling_hooked")
        elif self.self_state in ["recovered_reward", "recovered_no_reward"]:
            # Restarting after recovery.
            self.state_idx = SELF_STATE.index("scrolling_ok")

    def _opt_out(self, key):
        if self.self_state in ["scrolling_ok", "scrolling_hooked"]:
            self.state_idx = SELF_STATE.index("withdrawal_1")

        elif self.self_state == "withdrawal_1":
            progresses = bool(jr.bernoulli(key, p= 0.40))
            if progresses:
                self.state_idx = SELF_STATE.index("withdrawal_2")
            else:
                self.state_idx = SELF_STATE.index("withdrawal_1")
        
        elif self.self_state == "withdrawal_2":
            recovers = bool(jr.bernoulli(key, p=0.50))

            if recovers: 
                self.state_idx = SELF_STATE.index("recovered_reward")
            else:
                self.state_idx = SELF_STATE.index("withdrawal_2")

        elif self.self_state == "recovered_reward":
            self.state_idx = SELF_STATE.index("recovered_no_reward")
        elif self.self_state == "recovered_no_reward":
            self.state_idx = SELF_STATE.index("recovered_no_reward")
