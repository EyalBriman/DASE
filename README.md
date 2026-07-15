# DASE

This repository contains a Streamlit implementation of **Decentralized Allocation with Sequential Equalizing (DASE)** for fair division over time with **finite central storage**.

The app accepts:

- a supply vector \(S\),
- one normalized demand vector \(d_i\) for each agent, and
- a finite storage capacity \(C\).

It displays only the outcomes of:

1. **DA** — Decentralized Allocation;
2. **DASE** — Decentralized Allocation with Sequential Equalizing.

## Method implemented

### DA

For \(n\) agents, agent \(i\) receives the individual instance

\[
S_i(t)=\frac{S(t)}{n},
\qquad
C_i=\frac{C}{n}.
\]

The app solves the individual-fair-share LP

\[
\max \alpha
\]

subject to feasibility of the tight allocation

\[
w_i(t)=\alpha d_i(t).
\]

The DA outcome is

\[
w_i^{DA}(t)=u_i^*d_i(t),
\]

where \(u_i^*\) is the optimal LP value.

### DASE

DASE starts from the DA allocation. At every iteration:

1. all active agents receive the largest common proportional increment \(\Delta\) that preserves feasibility;
2. for every time \(\theta\), a residual-slack LP computes the largest additional amount \(\eta_\theta\) that can be allocated only at \(\theta\);
3. a time is saturated when \(\eta_\theta=0\);
4. every active agent with positive demand at a saturated time is finalized.

The implementation uses the finite-capacity linear feasibility certificates from the paper.

## Input format

### Demand matrix

Enter one agent per row and one time step per column. Values may be separated by spaces or commas.

Example:

```text
1 0 1 0
0 0 1 1
1 1 0 0
0 0 0 2
```

All rows must:

- contain non-negative values;
- have positive total demand;
- sum to the same total.

### Supply vector

Enter one row or one column with one value per time step.

Example:

```text
16 0 84 0
```

### Storage capacity

Enter a finite non-negative number, for example:

```text
12
```

Fractions should be written as decimals in the web interface, for example `1.5` rather than `3/2`.

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Files

- `app.py` — Streamlit interface and the complete DA/DASE implementation.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — minimal Streamlit toolbar configuration.
