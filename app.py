"""Streamlit implementation of DA and DASE for finite storage.

The implementation follows the LP formulations in the paper:

1. DA computes each agent's individual fair share using supply S / n and
   storage capacity C / n.
2. DASE starts from the DA allocation.
3. At each DASE iteration, all active agents receive the largest common
   proportional increment that preserves feasibility.
4. A time point is saturated when its residual-slack LP has value zero.
   Every active agent with positive demand at a saturated time is finalized.

Only DA and DASE outcomes are displayed in the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pulp
import streamlit as st
st.set_page_config(
    page_title="DASE Allocation Demo",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None,
    },
)


NUMERIC_TOL = 1e-8
SATURATION_TOL = 1e-7


# -----------------------------------------------------------------------------
# Parsing and validation
# -----------------------------------------------------------------------------


def parse_matrix(text: str) -> np.ndarray:
    """Parse a whitespace/comma-separated matrix from a text area."""
    rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
    if not rows:
        raise ValueError("Please enter the demand matrix.")

    data: list[list[float]] = []
    expected_width: int | None = None

    for line_number, row in enumerate(rows, start=1):
        cleaned = row.replace(",", " ").replace(";", " ")
        try:
            values = [float(value) for value in cleaned.split()]
        except ValueError as exc:
            raise ValueError(
                f"Could not parse row {line_number}. Use only numbers separated "
                "by spaces or commas."
            ) from exc

        if not values:
            raise ValueError(f"Demand row {line_number} is empty.")

        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ValueError("All demand rows must have the same number of entries.")

        data.append(values)

    return np.asarray(data, dtype=float)


def parse_vector(text: str) -> np.ndarray:
    """Parse a single-row or single-column vector."""
    matrix = parse_matrix(text)
    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        return matrix.reshape(-1)
    raise ValueError("Supply must be entered as one row or one column.")


def validate_instance(demands: np.ndarray, supply: np.ndarray, capacity: float) -> None:
    """Validate the CATS instance used by DA and DASE."""
    if demands.ndim != 2:
        raise ValueError("Demands must be a matrix: one row per agent.")
    if supply.ndim != 1:
        raise ValueError("Supply must be a vector.")
    if demands.shape[0] == 0 or demands.shape[1] == 0:
        raise ValueError("The instance must contain at least one agent and one time step.")
    if demands.shape[1] != supply.size:
        raise ValueError(
            "The supply length must equal the number of columns in the demand matrix."
        )
    if not np.all(np.isfinite(demands)) or not np.all(np.isfinite(supply)):
        raise ValueError("Demands and supply must contain only finite numbers.")
    if not np.isfinite(capacity):
        raise ValueError("Storage capacity must be a finite number.")
    if capacity < 0:
        raise ValueError("Storage capacity must be non-negative.")
    if np.any(demands < -NUMERIC_TOL) or np.any(supply < -NUMERIC_TOL):
        raise ValueError("Demands and supply must be non-negative.")

    row_sums = demands.sum(axis=1)
    if np.any(row_sums <= NUMERIC_TOL):
        raise ValueError("Every agent must have positive total demand.")
    if not np.allclose(row_sums, row_sums[0], rtol=1e-7, atol=1e-9):
        raise ValueError(
            "Demand vectors must be normalized to the same total: all rows must "
            "have the same sum."
        )


# -----------------------------------------------------------------------------
# LP utilities
# -----------------------------------------------------------------------------


def _solver() -> pulp.PULP_CBC_CMD:
    return pulp.PULP_CBC_CMD(msg=False)


def _require_optimal(problem: pulp.LpProblem, label: str) -> None:
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"{label} did not solve optimally. Solver status: {status}.")


def _value(variable: pulp.LpVariable, label: str) -> float:
    result = pulp.value(variable)
    if result is None or not np.isfinite(result):
        raise RuntimeError(f"The solver did not return a finite value for {label}.")
    return float(result)


# -----------------------------------------------------------------------------
# Feasibility and storage paths
# -----------------------------------------------------------------------------


def maximal_storage_path(
    supply: np.ndarray,
    global_allocation: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Return the physical storage path R[W].

    R(0) = 0 and, for t = 1,...,T,
        R(t) = min(C, R(t-1) + S(t) - W(t)).

    Any amount above capacity is discarded. A negative available amount means
    the allocation is infeasible because it borrows from future supply.
    """
    storage = np.zeros(supply.size, dtype=float)
    previous = 0.0

    scale = max(1.0, float(np.max(supply, initial=0.0)), float(capacity))
    tolerance = NUMERIC_TOL * scale

    for t in range(supply.size):
        available = previous + float(supply[t]) - float(global_allocation[t])
        if available < -tolerance:
            raise RuntimeError(
                f"Allocation is infeasible at time {t + 1}: it exceeds the "
                f"available resource by {-available:.6g}."
            )
        available = max(0.0, available)
        storage[t] = min(capacity, available)
        previous = storage[t]

    return storage


# -----------------------------------------------------------------------------
# DA: decentralized allocation with finite storage
# -----------------------------------------------------------------------------


def da_single_agent(
    demand: np.ndarray,
    supply: np.ndarray,
    capacity: float,
    number_of_agents: int,
) -> tuple[np.ndarray, float]:
    """Compute one agent's DA allocation.

    The agent receives individual supply S / n and individual capacity C / n.
    The LP maximizes alpha subject to feasibility of alpha * demand.
    """
    time_steps = supply.size
    individual_supply = supply / number_of_agents
    individual_capacity = capacity / number_of_agents

    problem = pulp.LpProblem("DA_individual_fair_share", pulp.LpMaximize)
    alpha = pulp.LpVariable("alpha", lowBound=0)
    gamma = pulp.LpVariable.dicts(
        "gamma",
        range(time_steps + 1),
        lowBound=0,
        upBound=individual_capacity,
    )

    problem += alpha
    problem += gamma[0] == 0

    for t in range(1, time_steps + 1):
        problem += (
            gamma[t]
            <= gamma[t - 1]
            + float(individual_supply[t - 1])
            - alpha * float(demand[t - 1])
        )

    problem.solve(_solver())
    _require_optimal(problem, "DA individual-fair-share LP")

    alpha_value = max(0.0, _value(alpha, "DA scaling factor"))
    allocation = alpha_value * demand
    return allocation, alpha_value


def DA(
    demands: np.ndarray,
    supply: np.ndarray,
    capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the complete DA allocation and individual fair shares."""
    number_of_agents = demands.shape[0]
    allocation = np.zeros_like(demands, dtype=float)
    fair_shares = np.zeros(number_of_agents, dtype=float)

    for agent in range(number_of_agents):
        allocation[agent], fair_shares[agent] = da_single_agent(
            demands[agent], supply, capacity, number_of_agents
        )

    # This also verifies that the sum of the individual certificates is globally
    # feasible under capacity C.
    maximal_storage_path(supply, allocation.sum(axis=0), capacity)
    return allocation, fair_shares


# -----------------------------------------------------------------------------
# DASE LPs
# -----------------------------------------------------------------------------


def equalizing_increment(
    active_agents: Iterable[int],
    demands: np.ndarray,
    supply: np.ndarray,
    capacity: float,
    global_allocation: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Solve the Equalize LP from the DASE algorithm."""
    active = sorted(active_agents)
    time_steps = supply.size
    aggregate_active_demand = demands[active].sum(axis=0)

    problem = pulp.LpProblem("DASE_equalize", pulp.LpMaximize)
    delta = pulp.LpVariable("delta", lowBound=0)
    gamma = pulp.LpVariable.dicts(
        "gamma", range(time_steps + 1), lowBound=0, upBound=capacity
    )

    problem += delta
    problem += gamma[0] == 0

    for t in range(1, time_steps + 1):
        problem += (
            gamma[t]
            <= gamma[t - 1]
            + float(supply[t - 1])
            - float(global_allocation[t - 1])
            - delta * float(aggregate_active_demand[t - 1])
        )

    problem.solve(_solver())
    _require_optimal(problem, "DASE Equalize LP")

    delta_value = max(0.0, _value(delta, "DASE equalizing increment"))
    certificate = np.array(
        [_value(gamma[t], f"equalizing certificate gamma[{t}]") for t in range(1, time_steps + 1)],
        dtype=float,
    )
    return delta_value, certificate


def residual_slack(
    theta: int,
    supply: np.ndarray,
    capacity: float,
    global_allocation: np.ndarray,
) -> float:
    """Solve ResidualSlack(theta, W) from the DASE algorithm.

    theta is zero-based in Python. The LP maximizes an additional amount eta at
    time theta while keeping every other time point unchanged.
    """
    time_steps = supply.size

    problem = pulp.LpProblem(f"DASE_residual_slack_t{theta + 1}", pulp.LpMaximize)
    eta = pulp.LpVariable("eta", lowBound=0)
    gamma = pulp.LpVariable.dicts(
        "gamma", range(time_steps + 1), lowBound=0, upBound=capacity
    )

    problem += eta
    problem += gamma[0] == 0

    for t in range(1, time_steps + 1):
        extra = eta if (t - 1) == theta else 0.0
        problem += (
            gamma[t]
            <= gamma[t - 1]
            + float(supply[t - 1])
            - float(global_allocation[t - 1])
            - extra
        )

    problem.solve(_solver())
    _require_optimal(problem, f"DASE residual-slack LP for time {theta + 1}")

    return max(0.0, _value(eta, f"residual slack at time {theta + 1}"))


@dataclass(frozen=True)
class DASEIteration:
    iteration: int
    active_before: tuple[int, ...]
    delta: float
    residual_slacks: tuple[float, ...]
    saturated_times: tuple[int, ...]
    finalized_agents: tuple[int, ...]


def DASE(
    demands: np.ndarray,
    supply: np.ndarray,
    capacity: float,
    da_allocation: np.ndarray | None = None,
) -> tuple[np.ndarray, list[DASEIteration]]:
    """Run finite-storage DASE exactly as stated in the paper."""
    if da_allocation is None:
        da_allocation, _ = DA(demands, supply, capacity)

    allocation = np.array(da_allocation, dtype=float, copy=True)
    global_allocation = allocation.sum(axis=0)
    maximal_storage_path(supply, global_allocation, capacity)

    active: set[int] = set(range(demands.shape[0]))
    history: list[DASEIteration] = []

    for iteration in range(1, demands.shape[0] + 1):
        if not active:
            break

        active_before = tuple(sorted(active))
        delta, _certificate = equalizing_increment(
            active, demands, supply, capacity, global_allocation
        )

        for agent in active_before:
            allocation[agent] += delta * demands[agent]

        aggregate_active_demand = demands[list(active_before)].sum(axis=0)
        global_allocation = global_allocation + delta * aggregate_active_demand
        maximal_storage_path(supply, global_allocation, capacity)

        slacks = np.array(
            [
                residual_slack(theta, supply, capacity, global_allocation)
                for theta in range(supply.size)
            ],
            dtype=float,
        )

        scale = max(
            1.0,
            float(np.max(supply, initial=0.0)),
            float(np.max(global_allocation, initial=0.0)),
            float(capacity),
        )
        saturation_threshold = SATURATION_TOL * scale
        saturated = set(np.flatnonzero(slacks <= saturation_threshold).tolist())

        finalized = {
            agent
            for agent in active
            if any(demands[agent, theta] > NUMERIC_TOL for theta in saturated)
        }

        if not finalized:
            # CBC can occasionally return a tiny positive eta at a theoretically
            # saturated time. We permit a slightly wider numerical threshold, but
            # never silently finalize an agent when all residual slacks are clearly
            # positive.
            minimum_slack = float(np.min(slacks))
            wider_threshold = 100.0 * saturation_threshold
            if minimum_slack <= wider_threshold:
                saturated = set(np.flatnonzero(slacks <= wider_threshold).tolist())
                finalized = {
                    agent
                    for agent in active
                    if any(
                        demands[agent, theta] > NUMERIC_TOL for theta in saturated
                    )
                }

        if not finalized:
            raise RuntimeError(
                "DASE made no progress. This usually indicates a numerical solver "
                "issue. Minimum residual slack: "
                f"{float(np.min(slacks)):.6g}."
            )

        active.difference_update(finalized)
        history.append(
            DASEIteration(
                iteration=iteration,
                active_before=tuple(agent + 1 for agent in active_before),
                delta=delta,
                residual_slacks=tuple(float(value) for value in slacks),
                saturated_times=tuple(theta + 1 for theta in sorted(saturated)),
                finalized_agents=tuple(agent + 1 for agent in sorted(finalized)),
            )
        )

    if active:
        raise RuntimeError("DASE did not terminate after at most n iterations.")

    maximal_storage_path(supply, allocation.sum(axis=0), capacity)
    return allocation, history


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------


def leontief_utilities(allocation: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Compute u_i(w_i) = min_{t:d_i(t)>0} w_i(t)/d_i(t)."""
    utilities = np.zeros(demands.shape[0], dtype=float)
    for agent in range(demands.shape[0]):
        positive = demands[agent] > NUMERIC_TOL
        utilities[agent] = float(
            np.min(allocation[agent, positive] / demands[agent, positive])
        )
    return utilities


def allocation_frame(allocation: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        allocation,
        index=[f"Agent {agent + 1}" for agent in range(allocation.shape[0])],
        columns=[f"t={time + 1}" for time in range(allocation.shape[1])],
    )


def utility_frame(utilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"Leontief utility": utilities},
        index=[f"Agent {agent + 1}" for agent in range(utilities.size)],
    )




# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------


def show_outcome(
    title: str,
    allocation: np.ndarray,
    demands: np.ndarray,
) -> None:
    st.subheader(title)
    st.markdown("**Allocation**")
    st.dataframe(allocation_frame(allocation), use_container_width=True)

    st.markdown("**Agent utilities (scaling factors)**")
    st.dataframe(
        utility_frame(leontief_utilities(allocation, demands)),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="DASE",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("DASE")
    st.caption("Decentralized Allocation with Sequential Equalizing")

    st.markdown(
        r"""
Enter a finite-storage instance below.

- **Demands:** one agent per row and one time step per column.
- **Supply:** one row or one column, with one value per time step.
- **Capacity:** the finite central-storage capacity \(C\).
- All demand rows must be non-negative and normalized to the same total.
        """
    )

    left, right = st.columns([2, 1])

    with left:
        demand_text = st.text_area(
            "Demand matrix",
            placeholder="1 0 1 0\n0 0 1 1\n1 1 0 0\n0 0 0 2",
            height=190,
        )
        supply_text = st.text_area(
            "Supply vector",
            placeholder="16 0 84 0",
            height=90,
        )

    with right:
        capacity = st.number_input(
            "Storage capacity C",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.8g",
        )
        st.info(
            "Numbers may be separated by spaces or commas. Fractions should be "
            "entered as decimals, for example 1.5 rather than 3/2."
        )

    if st.button("Compute DA and DASE", type="primary", use_container_width=True):
        try:
            demands = parse_matrix(demand_text)
            supply = parse_vector(supply_text)
            validate_instance(demands, supply, float(capacity))

            da_allocation, _fair_shares = DA(demands, supply, float(capacity))
            dase_allocation, _history = DASE(
                demands, supply, float(capacity), da_allocation=da_allocation
            )

            da_tab, dase_tab = st.tabs(["DA outcome", "DASE outcome"])

            with da_tab:
                show_outcome(
                    "Decentralized Allocation (DA)",
                    da_allocation,
                    demands,
                )

            with dase_tab:
                show_outcome(
                    "Decentralized Allocation with Sequential Equalizing (DASE)",
                    dase_allocation,
                    demands,
                )

        except Exception as exc:  # Streamlit should show a readable error, not a stack trace.
            st.error(str(exc))


if __name__ == "__main__":
    main()
