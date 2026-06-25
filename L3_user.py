from pymdp.distribution import compile_model

from config import (
    L3_BEHAVIOR_EVIDENCE_OBS,
    L3_CONTROL_STATE,
    L3_ACTIONS,
)


def init_L3_model():
    """
    Level 3: metacognitive awareness level.

    L3 does not observe "I am in control" directly.

    Instead, it receives a behavioral evidence observation computed from
    the recent L2 action history, for example:

        persistent_scrolling
        persistent_scrolling_with_hooked_belief
        failed_optout
        relapse_after_optout
        succesfull_optout

    From this, it infers a hidden control_state:

        in_control
        flow_absorbed
        losing_control
        aware_of_hookedness
        committed_to_recovery
        safe

    Then it chooses a meta-action:

        stay_in_flow
        increase_awareness
        commit_to_recovery
        be_proud

    The meta-action does not directly force L2 to opt out.
    Instead, it modulates L2's awareness and preferences.
    """