# scrolling_env.py

from config import SELF_STATE

class L2ScrollingEnv: 
    """
    Generative process for the higher-level marshmallow task.

    It tracks the true self_state and returns direct L2 observations:
        reward_report
        affect_obs

    It does not generate positive/negative posts.
    Those come from L1 when the feed is active.
    """
    def __init__(self, starting_state="scrolling_ok"):
        if starting_state not in SELF_STATE:
            raise ValueError(f"Unknown starting state: {starting_state}")
        
        self.self_state = starting_state 

    def step(self,action):
        """
        Apply the previous L2 action to the true environment state.
        """
        if action == "continue_scrolling":
            self._continue_scrolling() 

        elif action == "oput_out":
            self._opt_out()

        else:
            raise ValueError(f"Unknown L2 action:{action}")        
        
        # get the obs from L1
        reward_label = self._get_reward_obs()
        affect_label = self._get_affect_obs()

        return reward_label, affect_label, self.self_state 

    def is_feed_active(self):
        return self.self_state in ["scrolling_ok", "scrolling_hooked"]

    # Here I define the continue scrolling and opt later opt out actions 
    def _cotinue_scrolling(self):
        # If it continues scrolling from scorlling ok it goes in to scrolling hooked
        if self.self_state == "scrolling_ok":
            # TODO: Deterministic version for now: eventually you may sample this.
            self.self_state = "scrolling_hooked"

        #If it is in scrolling_hooked stay there 
        elif self.self_state == "scrolling_hooked":
            self.self_state = "scrolling_hooked"
        
        elif self.self_state in ["withdrawal_1", "withdrawal_2"]:
            # Relapse
            self.self_state = "scrolling_ok"

        elif self.self_state == "recovered":
            # Relapse again here is basically as saying that in continuous 
            # scrolling is impossible to recover
            self.self_state = "scrolling_ok"
        
    def _opt_out(self):
        if self.self_state in ["scrolling_ok", "scrolling_hooked"]:
            self.self_state = "withdrawal_1"
        
        elif self.self_state == "withdrawal_1":
            self.self_state = "withdrawal_2"

        elif self.self_state == "withdrawal_2":
            self.self_state = "recovered"

        elif self.self_state == "recovered":
            self.self_state = "recovered"

    def _get_reward_obs(self):
        if self.self_state in ["scrolling_ok", "scrolling_hooked"]:
            return "quick_reward"

        if self.self_state in ["withdrawal_1", "withdrawal_2"]:
            return "no_reward"
        
        raise RuntimeError(f"Unhandled self state: {self.self_state}")
    
    def _get_affect_obs(self):
        if self.self_state == "scrolling_ok":
            return "okay"
        
        # TODO: here it should be a mix between a bad feeling and a good feeling
        if self.self_state == "scrolling_hooked":
            return "bad_feeling"
        
        if self.self_state in ["withdrawal_1", "withdrawal_2"]:
            return "urge"
        
        if self.self_state == "recovered":
            return "relief"
        
        raise RuntimeError(f"Unhandeled self state : {self.selft_state}")