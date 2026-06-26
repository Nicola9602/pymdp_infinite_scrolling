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

    model_description = {
        "observations":{
            "behavior_evidence_obs": {
                "elements": L3_BEHAVIOR_EVIDENCE_OBS,
                "depends_on": ["control_state"],
            },
        },
        "controls": {
            "meta_choice": {
                "elements": L3_ACTIONS,
            },
        },
        "states": {
            "control_state": {
                "elements": L3_CONTROL_STATE,
                "depends_on": ["control_state"],
                "controlled_by": ["meta_choice"],
            },
        },
    }

    model = compile_model(model_description)

    # ------------------------------------------------------------------
    # A: likelihood mapping
    # A["behavior_evidence_obs"][observation, hidden_control_state]
    # ------------------------------------------------------------------
    
    # Evidence compatible with control.
    model.A["behavior_evidence_obs"]["brief_use", "in_control"] = 0.55
    model.A["behavior_evidence_obs"]["successful_optout", "in_control"] = 0.99
    model.A["behavior_evidence_obs"]["persistent_scrolling", "in_control"] = 0.01
    model.A["behavior_evidence_obs"]["failed_optout", "in_control"] = 0.01

    # Evidence compatible with loss of control.
    model.A["behavior_evidence_obs"]["brief_use", "not_in_control"] = 0.01
    model.A["behavior_evidence_obs"]["successful_optout", "not_in_control"] = 0.01
    model.A["behavior_evidence_obs"]["persistent_scrolling", "not_in_control"] = 0.99
    model.A["behavior_evidence_obs"]["failed_optout", "not_in_control"] = 0.99

    model.A["behavior_evidence_obs"].normalize()

        # ------------------------------------------------------------------
    # B: transition mapping
    #
    # B["control_state"][next_state, previous_state, action]
    # ------------------------------------------------------------------

    # Action: stay_in_control
    # This means: L3 engages the oversight mechanism.
    # In the full hierarchy, this should perturb L2 by increasing awareness.
    model.B["control_state"]["in_control", "in_control", "stay_in_control"] = 0.95
    model.B["control_state"]["not_in_control", "in_control", "stay_in_control"] = 0.05

    model.B["control_state"]["in_control", "not_in_control", "stay_in_control"] = 0.99
    model.B["control_state"]["not_in_control", "not_in_control", "stay_in_control"] = 0.01
        model.B["control_state"]["not_in_control", "not_in_control", "stay_in_control"] = 0.01

    # Action: stay_hooked
    # This means: L3 does not intervene / allows the hooked loop to continue.
    model.B["control_state"]["in_control", "in_control", "stay_hooked"] = 0.01
    model.B["control_state"]["not_in_control", "in_control", "stay_hooked"] = 0.99

    model.B["control_state"]["in_control", "not_in_control", "stay_hooked"] = 0.01
    model.B["control_state"]["not_in_control", "not_in_control", "stay_hooked"] = 0.99

    model.B["control_state"].normalize()
