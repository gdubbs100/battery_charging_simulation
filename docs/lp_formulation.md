# Battery Dispatch Linear Program

Used by both the Oracle and Forecast LP policies via `solve_lp` in
[`src/battery_sim/agents/policies/_utils.py`](../src/battery_sim/agents/policies/_utils.py).

$$\begin{aligned}
\max_{c, d} \quad & \sum_{t=1}^{T} p_{t+1} \,\Delta t \left(\eta\, d_t - c_t\right) \\[6pt]
\text{s.t.} \quad
& -\eta\,\Delta t \sum_{s=1}^{t} (c_s - d_s) \leq E_0 - E_{\min}, && t = 1,\ldots,T \\
& \eta\,\Delta t \sum_{s=1}^{t} (c_s - d_s) \leq E_{\max} - E_0, && t = 1,\ldots,T \\
& 0 \leq c_t \leq \bar{c}, \quad 0 \leq d_t \leq \bar{d}, && t = 1,\ldots,T
\end{aligned}$$

| Symbol | Definition |
|---|---|
| $c_t,\, d_t$ | Commanded charge and discharge power (MW). Both non-negative; net action is $u_t = c_t - d_t$. Only $\eta c_t$ and $\eta d_t$ actually transfer — the rest is lost to inefficiency. |
| $\Delta t$ | Interval duration (hours). |
| $\eta \in (0,1)$ | One-way efficiency, applied symmetrically to both directions. Battery gains $\eta c_t \Delta t$ MWh when charging; releases $\eta d_t \Delta t$ MWh to the grid when discharging. Energy balance is $E_t = E_0 + \eta\,\Delta t\sum_{s=1}^{t}(c_s - d_s)$. The objective has $\eta$ on $d_t$ (revenue = $\eta d_t p$, what the grid receives) but not on $c_t$ (cost = $c_t p$, what the grid provides). |
| $E_0$ | Battery energy at the start of the horizon (MWh). |
| $E_{\min},\, E_{\max}$ | Energy bounds: $E_{\min} = 0$, $E_{\max} = C$ (battery capacity in MWh). The two inequality pairs enforce these at every prefix $t$. |
| $\bar{c},\, \bar{d}$ | Maximum charge and discharge rates (MW). |

**Oracle** passes the full known price series and solves once — a perfect-foresight upper bound. **Forecast LP** solves the same LP at each step over a rolling `horizon`-length window of model-predicted prices, using only the first action (receding-horizon MPC).
