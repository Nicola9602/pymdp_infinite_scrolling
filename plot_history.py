# plot_history.py

import matplotlib.pyplot as plt


def _collect_categories(history, key):
    values = []

    for row in history:
        value = row.get(key)

        if value is not None and value not in values:
            values.append(value)

    return values


def _make_code_mapping(values):
    return {value: idx for idx, value in enumerate(values)}


def plot_history(history, filename="scrolling_flow.png", show=True):
    """
    Plot the simulation as a readable timeline.

    It saves a PNG file and optionally opens a matplotlib window.

    Expected history keys:
        - t2
        - true_self_state_before
        - true_self_state_after
        - feed_active
        - l1_true_valence
        - l1_reward
        - l2_reward_report
        - l2_affect_obs
        - l2_action
    """

    if len(history) == 0:
        print("No history to plot.")
        return

    t = [row["t2"] for row in history]

    # ------------------------------------------------------------------
    # Define what we want to plot
    # ------------------------------------------------------------------

    tracks = [
        ("true_self_state_before", "State before"),
        ("true_self_state_after", "State after"),
        ("l2_belief_self_state", "L2 believed state"),
        ("l2_action", "L2 action"),
        ("l2_affect_obs", "L2 affect"),
        ("l2_reward_report", "L2 reward"),
        ("l1_true_valence", "L1 valence"),
        ("l1_reward", "L1 reward"),

        ("algorithm_mode", "Algorithm mode"),
        ("algorithm_event", "Algorithm event"),
        ("feed_pull", "Feed pull"),
        ("feed_p_hooked_from_ok", "Feed p hooked"),
        ("feed_p_awareness_when_hooked", "Feed/L3 awareness p"),

        # L3 tracks.
        # These will only have non-None values on L3 update timesteps.
        ("l3_evidence", "L3 evidence"),
        ("l3_control_state", "L3 control belief"),
        ("l3_meta_action", "L3 meta-action"),

        # This one is numeric, but the current plotter treats all values
        # categorically. That is okay for now because awareness usually
        # only takes a few values, such as 0.05, 0.15, or 0.80.
        ("l2_awareness_when_hooked", "L2 awareness p"),
    ]

    n_tracks = len(tracks)

    fig, axes = plt.subplots(
        n_tracks,
        1,
        figsize=(14, 2.2 * n_tracks),
        sharex=True,
    )

    if n_tracks == 1:
        axes = [axes]


    # ------------------------------------------------------------------
    # Plot each categorical variable as a timeline
    # ------------------------------------------------------------------

    for ax, (key, title) in zip(axes, tracks):
        categories = _collect_categories(history, key)

        if len(categories) == 0:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            continue

        mapping = _make_code_mapping(categories)

        y = []
        labels = []

        for row in history:
            value = row.get(key)

            if value is None:
                y.append(None)
                labels.append("None")
            else:
                y.append(mapping[value])
                labels.append(value)

        # Plot only non-None values.
        valid_t = []
        valid_y = []
        valid_labels = []

        for ti, yi, label in zip(t, y, labels):
            if yi is not None:
                valid_t.append(ti)
                valid_y.append(yi)
                valid_labels.append(label)

        ax.plot(valid_t, valid_y, marker="o", linewidth=1)

        for ti, yi, label in zip(valid_t, valid_y, valid_labels):
            ax.text(
                ti,
                yi + 0.08,
                label,
                rotation=30,
                ha="left",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(title)
        ax.set_yticks(list(mapping.values()))
        ax.set_yticklabels(list(mapping.keys()))
        ax.grid(True, axis="x", alpha=0.3)

    # ------------------------------------------------------------------
    # Add feed-active shading
    # ------------------------------------------------------------------

    for ax in axes:
        for row in history:
            if not row.get("feed_active", False):
                ax.axvspan(
                    row["t2"] - 0.5,
                    row["t2"] + 0.5,
                    alpha=0.08,
                )

    axes[-1].set_xlabel("L2 timestep")

    fig.suptitle("Doomscrolling active-inference simulation flow", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(filename, dpi=200)
    print(f"Saved plot to {filename}")

    if show:
        plt.show()

    plt.close(fig)