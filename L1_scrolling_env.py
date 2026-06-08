# L1_scrolling_env.py

import numpy as np

from config import POST_VALENCE


POST_OBS = ["post"]
POST_REWARD = ["reward", "no_reward"]

class L1ScrollingEnv:
    """
    Level 1 generative process.

    This environment generates one post at a time.

    It does NOT generate no_post.
    no_post belongs to L2, because it happens when the higher level opts out.
    """

    def __init__(
            self, 
            p_positive = 0.5,
            p_negative = 0.3,
            p_neutral = 0.2,
            p_reward_if_positive = 0.5,
            p_reward_if_negative = 0.7,
            p_reward_if_neutral = 0.1,
            seed=None,
    ): 
        self.rng = np.random.default.rng(seed)

        self.valence_probs = {
            "positive": p_positive,
            "negative": p_negative,
            "neutral": p_neutral,
        }
        
        self.reward_probs = {
            "positive": p_reward_if_positive,
            "negative": p_reward_if_negative,
            "neutral": p_reward_if_neutral,
        }

        self.current_valence=None
        self.current_reward=None


    def step(self, action="scroll"):
        """
        Generate one L1 observation.

        Returns observation indices for the L1 agent:
            obs_l1 = [
                post_obs_idx,
                post_valence_obs_idx,
                post_reward_obs_idx,
            ]

        1. Check that the action is "scroll".

        2. Randomly choose the next post type:
            positive / negative / neutral

        3. Based on that post type, randomly decide whether reward happens.

        4. Convert the labels into indices.

        5. Return the observation to the L1 agent.
        """

        if action != "scroll":
            raise ValueError(f"L1 only understands action='scroll', got {action}")

        valences = list(self.valence_probs.keys())
        probs = np.array([self.valence_probs[v] for v in valences], dtype=float)
        # Normalise them
        probs = probs / probs.sum()

        #This samples the valences randomly, 
        # TODO: here I can add the second agent that samples the stuff 
        # looking at what keeps the first agent engaged which we know being 
        # some kind of stuff. 
        sampled_valence = self.rng.choice(valences, p=probs)
        self.current_valence = sampled_valence

        # Given the kind of post we sampled, what is the probability that the user experiences reward?
        p_reward = self.reward_probs[sampled_valence]
        # TODO: this is a weird way to decide whether reward happened 
        # it basically samples a random from 0 to 1 and is true if it is < then p_reward..
        reward_happened = self.rng.random() < p_reward

        sampled_reward = "reward" if reward_happened else "no_reward"
        self.current_reward = sampled_reward

        post_obs_idx = POST_OBS.index("post")
        valence_obs_idx = POST_VALENCE.index(sampled_valence)
        reward_obs_idx = POST_REWARD.index(sampled_reward)

        return [post_obs_idx, valence_obs_idx, reward_obs_idx]

    def reset(self):
        self.current_valence = None
        self.current_reward = None