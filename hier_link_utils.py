# hier_link_utils.py
#This is where L1 communicates with L2

import numpy as np
from pymdp import utils

from config import (
    POST_VALENCE,
    L1_VALENCE_REPORT,
    REWARD_REPORT,
    AFFECT_OBS,
)

def l1_to_l2_valence_label(posterior_L1, feed_active=True):
    """
    Convert L1 posterior over post_valence into a categorical L2 report.

    posterior_L1[0] is assumed to be the posterior over:
        POST_VALENCE = ["positive", "negative", "neutral"]
    """
    if not feed_active:
        return "no_post"
    
    post_valence_posterior = posterior_L1[0]
    # TODO: this passess only the winner not the full distribtion... 
    # should i change it?
    winning_idx = int(np.argmax(post_valence_posterior))

    return POST_VALENCE[winning_idx]

def get_l2_obs_from_l1(
        posterior_L1,
        dimensions_L2,
        reward_label,
        affect_label,
        feed_active = True):
    """
    Build the full L2 observation object array.

    L2 observation modalities:
        0. l1_valence_report
        1. reward_report
        2. affect_obs

    If feed_active=True:
        l1_valence_report is derived from L1 posterior.

    If feed_active=False:
        l1_valence_report is no_post.
    """

    obs_to_L2 = utils.obj_array_zeros(dimensions_L2["numb_obs"])

    valence_report_label = l1_to_l2_valence_label(
        posterior_L1 = posterior_L1,
        feed_active = feed_active)

    valence_report_idx = L1_VALENCE_REPORT.index(valence_report_label)
    reward_idx = REWARD_REPORT.index(reward_label)
    affect_idx = AFFECT_OBS.index(affect_label)

    obs_to_L2[0] = utils.onehot(
        valence_report_idx,
        len(L1_VALENCE_REPORT))     
    
    obs_to_L2[1] = utils.onehot(
        reward_idx,
        len(REWARD_REPORT))
    
    obs_to_L2[2] = utils.onehot(
        affect_idx,
        len(AFFECT_OBS)
    )

    return obs_to_L2