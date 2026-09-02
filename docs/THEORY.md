# Q1 theory: regret of test-time compute allocation under a noisy difficulty signal

Rigorous notes, 2026-06-12. Binary-reward closed-form world. Companion to the gate
experiments in `code/q1_experiment.py` (results in `P1_ANALYSIS.md` §7-pre). Every claim
is tagged **[proved]**, **[proved mod regularity]**, or **[conjecture]**. Notation fixed
once here and reused.

---

## 1. Model

Queries $i=1,\dots,n$. Query $i$ has per-sample success probability $\mu_i\in(0,1)$,
reparametrized by the **hazard**
$$\nu_i := -\ln(1-\mu_i)\in(0,\infty),\qquad \mu_i\ \text{small}\ \Leftrightarrow\ \nu_i\ \text{small (hard)}.$$
With $b\ge 0$ i.i.d. samples the success probability is
$$q(\nu,b) = 1-(1-\mu)^{b} = 1-e^{-\nu b},\qquad
q_b := \partial_b q = \nu e^{-\nu b},\qquad q_{bb} = -\nu^2 e^{-\nu b}.$$
$q$ is increasing, strictly concave in $b$ (diminishing returns). Cost is $b$ (one unit
per sample). The planner has an average budget $B>0$:
$$\textstyle\frac1n\sum_i b_i = B.$$
A **difficulty signal** reports $\hat\nu_i$; the planner may use only $\hat\nu_i$, never
$\nu_i$. The **realized value** of an allocation $\mathbf b$ is $\frac1n\sum_i q(\nu_i,b_i)$
(truth decides success), even though $\mathbf b$ was chosen from $\hat{\boldsymbol\nu}$.

---

## 2. The oracle allocation (continuous budget)

### Proposition 1 (closed form) [proved]
The maximizer of $\frac1n\sum_i q(\nu_i,b_i)$ over $\mathbf b\ge 0$, $\frac1n\sum_i b_i=B$ is
$$\boxed{\,b^*(\nu;\lambda)=\Big[\tfrac{1}{\nu}\ln\tfrac{\nu}{\lambda}\Big]_+
=\max\Big\{0,\ \tfrac{\ln(\nu/\lambda)}{\nu}\Big\}\,}$$
where $\lambda>0$ (the budget multiplier) is the unique root of the spend equation
$g(\lambda):=\frac1n\sum_i b^*(\nu_i;\lambda)=B$.

*Proof.* The program is concave with a linear equality constraint and a Slater point, so
KKT is necessary and sufficient. Lagrangian
$L=\frac1n\sum_i q(\nu_i,b_i)-\lambda(\frac1n\sum_i b_i-B)$. Stationarity in $b_i$ (with
the sign constraint) gives $q_b(\nu_i,b_i)=\nu_i e^{-\nu_i b_i}\le\lambda$, equality when
$b_i>0$. Solving $\nu e^{-\nu b}=\lambda$: $e^{-\nu b}=\lambda/\nu$, so
$b=\frac1\nu\ln\frac\nu\lambda$, positive iff $\nu>\lambda$; otherwise the unconstrained
optimum is negative and the constraint $b\ge0$ binds at $b=0$. $\square$

### Reading of the optimum (drives everything below)

$b^*(\cdot;\lambda)$ is **hump-shaped** in $\nu$:
$$\partial_\nu b^* = \frac{1-\ln(\nu/\lambda)}{\nu^2}\quad(\nu>\lambda),\qquad
\begin{cases}
b^*=0, & \nu\le\lambda \quad(\text{hard queries }\textbf{abandoned}),\\
\text{peak } b^*=\tfrac{1}{e\lambda}, & \nu=e\lambda,\\
b^*\sim \tfrac{\ln\nu}{\nu}\to 0, & \nu\to\infty\ (\text{easy queries need nothing}).
\end{cases}$$
$\lambda$ is the **shadow price of compute**: at the optimum *every funded query has the
same marginal value* $q_b=\lambda$ (this is the FOC, and it is the single fact that makes
the regret first order vanish).

### Proposition 2 (clearing is well posed) [proved]
$g(\lambda)=\frac1n\sum_i b^*(\nu_i;\lambda)$ is continuous and strictly decreasing on
$(0,\max_i\nu_i)$, with $g\to\infty$ as $\lambda\downarrow0$ and $g\to0$ as
$\lambda\uparrow\max_i\nu_i$. Hence the budget root $\lambda^*=\lambda(B)$ exists, is
unique, and bisection converges.

*Proof.* Each $b^*(\nu_i;\cdot)$ is continuous, $0$ for $\lambda\ge\nu_i$, and
$\partial_\lambda b^*=-1/(\lambda\nu_i)<0$ where positive; sum and limits follow. $\square$

---

## 3. Regret and the second-order law (continuous menu)

Let the signal be a smooth one-parameter family $\hat\nu_i(\varepsilon)$ with
$\hat\nu_i(0)=\nu_i$ and $\varepsilon\ge0$ the error scale (e.g. multiplicative noise
$\hat\nu_i=\nu_i e^{\varepsilon\xi_i}$, $\mathbb E\xi=0$; bias $\hat\nu_i=\nu_i e^{\varepsilon}$;
miscalibration $\hat\nu_i=\nu_i^{1+\varepsilon}$). The planner plays
$$\hat b_i=b^*(\hat\nu_i;\hat\lambda),\qquad \hat\lambda(\varepsilon)\ \text{solves}\
\tfrac1n\textstyle\sum_i b^*(\hat\nu_i;\hat\lambda)=B,$$
i.e. it **re-clears the budget under the signal** (information-feasible: never uses
truth). Define $f(\varepsilon):=\frac1n\sum_i q(\nu_i,\hat b_i)$ and the regret
$R(\varepsilon):=V^*-f(\varepsilon)\ge0$, $V^*=f(0)$.

### Theorem 1 (envelope: regret is second order) [proved mod regularity]
Assume $\hat\nu_i(\cdot)\in C^2$ near $0$ and that the set of abandoned/active queries is
locally constant (no query sits exactly at $\nu=\lambda^*$; a measure-zero exclusion).
Then $R'(0)=0$ and
$$\boxed{\,R(\varepsilon)=\frac{\varepsilon^2}{2}\,\frac1n\sum_i \lambda^*\,\nu_i\,
\big(\dot b_i\big)^2+o(\varepsilon^2),\qquad \dot b_i:=\hat b_i'(0)\,}$$
in particular $R(\varepsilon)=\Theta(\varepsilon^2)$ generically.

*Proof.* Chain rule:
$f'(\varepsilon)=\frac1n\sum_i q_b(\nu_i,\hat b_i)\,\hat b_i'(\varepsilon)$. At
$\varepsilon=0$, $\hat b_i=b_i^*$ and FOC gives $q_b(\nu_i,b_i^*)=\lambda^*$ for active $i$
(and $\hat b_i'$ multiplies $q_b$; for abandoned $i$, $b_i^*=0$ stays $0$ to first order by
the local-constancy assumption, contributing nothing). Thus
$$f'(0)=\lambda^*\,\tfrac1n\textstyle\sum_i \hat b_i'(0).$$
The budget identity $\frac1n\sum_i\hat b_i(\varepsilon)\equiv B$ differentiated once gives
$\frac1n\sum_i\hat b_i'(\varepsilon)=0$, so $f'(0)=0$, i.e. $R'(0)=0$.
For the second order,
$f''(0)=\frac1n\sum_i\big[q_{bb}(\nu_i,b_i^*)(\hat b_i')^2+q_b(\nu_i,b_i^*)\hat b_i''\big]$.
The second group is $\lambda^*\frac1n\sum_i\hat b_i''=0$ (budget identity differentiated
twice). In the first, substitute the optimum $e^{-\nu_i b_i^*}=\lambda^*/\nu_i$ into
$q_{bb}=-\nu_i^2 e^{-\nu_i b_i^*}=-\nu_i\lambda^*$. Hence
$f''(0)=-\lambda^*\frac1n\sum_i\nu_i(\hat b_i')^2$ and
$R(\varepsilon)=-\frac{\varepsilon^2}{2}f''(0)+o(\varepsilon^2)$ is the boxed form, which
is $\ge0$. $\square$

**Interpretation.** Error reshuffles compute between queries; at the optimum every unit of
compute is worth $\lambda^*$ wherever it sits, so the reshuffle breaks even to first order
— *the loss is purely the second-order cost of over-shooting*. This is the same envelope
mechanism as "efficiency loss is second order" in constrained welfare maximization.

### The sensitivity weight and error-shape dependence

Expand $\dot b_i$. With $\partial_\nu b^*=\frac{1-\ln(\nu/\lambda)}{\nu^2}$,
$\partial_\lambda b^*=-\frac{1}{\lambda\nu}$,
$$\dot b_i=\partial_\nu b^*(\nu_i;\lambda^*)\,\dot{\hat\nu}_i
+\partial_\lambda b^*(\nu_i;\lambda^*)\,\dot\lambda,
\qquad \dot{\hat\nu}_i:=\hat\nu_i'(0).$$
**Ignoring the coupling** ($\dot\lambda$, treated in §4) and taking multiplicative error
$\dot{\hat\nu}_i=\nu_i\xi_i$:
$$\nu_i(\dot b_i)^2=\frac{[1-\ln(\nu_i/\lambda^*)]^2}{\nu_i}\,\xi_i^2,\qquad
\boxed{\,R\approx\frac{\varepsilon^2}{2}\,\frac1n\sum_i w(\nu_i)\,\xi_i^2,\quad
w(\nu)=\lambda^*\frac{[1-\ln(\nu/\lambda^*)]^2}{\nu}\,}$$

### Corollary 1 (error shape matters; the scalar-$\varepsilon$ reduction loses information) [proved]
The regret weight $w(\nu)$ is **hump-shaped and vanishes at the peak** $\nu=e\lambda^*$
(there $1-\ln e=0$, so a prediction error at a peaked query does not move its allocation),
and is large near the abandonment boundary $\nu\downarrow\lambda^*$. Therefore two error
fields with **equal RMSE** but placing their mass at different $\nu$ produce different
regret; no scalar summary of the error (e.g. a $0/1$ disagreement rate or an RMSE) can
determine $R$.

*Empirical confirmation (E4):* at matched $\log$-RMSE $0.1$, regret spreads $\approx 6\times$
across {noise $6.3\!\times\!10^{-4}$, bias $5.7\!\times\!10^{-4}$, miscal $1.1\!\times\!10^{-4}$,
contam $1.8\!\times\!10^{-4}$}; same ordering at RMSE $0.3$.

---

## 4. The dual-coupling channel

$\dot\lambda$ is determined by differentiating the clearing identity
$\frac1n\sum_i b^*(\hat\nu_i;\hat\lambda)\equiv B$:
$$\frac1n\sum_i\big[\partial_\nu b^*\,\dot{\hat\nu}_i+\partial_\lambda b^*\,\dot\lambda\big]=0
\ \Longrightarrow\
\boxed{\,\dot\lambda=-\frac{\sum_i\partial_\nu b^*(\nu_i;\lambda^*)\,\dot{\hat\nu}_i}
{\sum_i\partial_\lambda b^*(\nu_i;\lambda^*)}\,}$$

### Proposition 3 (cross-query propagation; re-clearing is protective) [proved]
$\dot\lambda$ is a weighted average of the per-query errors $\dot{\hat\nu}_i$: **one
query's signal error perturbs every query's allocation through the shadow price**. Writing
$R=R_{\text{mis}}+R_{\text{coup}}$ where $R_{\text{mis}}$ uses $\dot\lambda=0$, the
re-clearing term satisfies $R\le R_{\text{mis}}$, i.e. updating $\lambda$ to absorb the
signal lowers regret relative to holding the oracle price fixed.

*Proof.* $\dot\lambda$ is the value of the scalar $c$ minimizing
$\frac1n\sum_i\nu_i(\partial_\nu b^*\dot{\hat\nu}_i+\partial_\lambda b^*\,c)^2$ subject to
the budget-feasibility (zero-mean spend perturbation) constraint — it is the orthogonal
projection that enforces $\sum\dot b_i=0$, hence cannot increase the quadratic form versus
the infeasible $c=0$. $\square$

*Empirical confirmation (E5):* freezing $\lambda=\lambda^*$ vs re-clearing raises regret by
$+30\%$ ($\varepsilon=0.1$) and $+24\%$ ($\varepsilon=0.3$). The coupling term is positive
and protective.

**Why this is the novel object.** Classical margin/sensitivity theory for classification
has no budget, hence no shadow price, hence no $\dot\lambda$ term. The cross-query coupling
$\dot\lambda$ is the allocation-specific structure (the analogue of the normalization
factor $Z$ in $\alpha$-fair allocation), and it is *stabilizing* here.

---

## 5. Discrete menu: the score currency and tie-immunity

Real systems pick from a finite menu $\mathcal M=\{m_0<m_1<\dots<m_K\}$ (e.g.
$\{0,1,2,4,8,16\}$ samples; routing is $K=1$, $\{m_0,m_1\}=\{$cheap, expensive$\}$). The
budget-cleared **index policy** is
$$b(\hat\nu)=\arg\max_{m\in\mathcal M}\,s(\hat\nu,m;\hat\lambda),\qquad
s(\nu,m;\lambda):=q(\nu,m)-\lambda m\ \ (\textbf{net score}),$$
with $\hat\lambda$ clearing $\frac1n\sum_i b(\hat\nu_i)=B$ (a nonincreasing step function of
$\lambda$; clear to the smallest feasible $\lambda$).

### The correct regret currency is the score, not the value

Under budget re-clearing, choosing menu item $m$ instead of $m'$ for a true-$\nu$ query
costs, *net of the reclaimed/spent budget repriced at $\lambda$*, exactly the score gap
$s(\nu,m';\lambda)-s(\nu,m;\lambda)$. A **decision boundary** of the index rule is a
hazard $\nu_\partial$ where the top two items tie:
$$s(\nu_\partial,m_a;\lambda)=s(\nu_\partial,m_b;\lambda)
\ \Longleftrightarrow\ q(\nu_\partial,m_b)-q(\nu_\partial,m_a)=\lambda(m_b-m_a).$$
**At a boundary the score gap is zero by definition** (though the *value* gap
$q(\nu_\partial,m_b)-q(\nu_\partial,m_a)=\lambda(m_b-m_a)\neq0$ is not — this distinction is
the whole point).

### Theorem 2 (tie-immunity under re-clearing) [proved mod regularity]
Let $s(\nu)$ denote the gap between the best and second-best score at hazard $\nu$, and
suppose the population density of queries near each decision boundary is bounded (a margin
density condition; no atom at $s=0$). Under noise of scale $\varepsilon$ and budget
re-clearing,
$$R(\varepsilon)=\Theta(\varepsilon^2).$$
If instead the margin distribution has an **atom** of mass $\pi_0$ at $s=0$ (exact ties),
re-clearing still gives $R(\varepsilon)=O(\varepsilon^2)$: an exact-tie flip is
score-neutral and feasibility is preserved, so the atom does **not** create $\Theta(1)$
regret.

*Proof sketch.* Only queries within $O(\varepsilon)$ of a boundary can flip; each flips by
one menu step and loses score $|s(\nu_i)|=|s'(\nu_\partial)|\cdot\mathrm{dist}(\nu_i,\nu_\partial)
=O(\varepsilon)$; their mass is $O(\varepsilon)$ under bounded density; product
$O(\varepsilon^2)$ (Tsybakov-type margin calculus). Budget re-clearing shifts
$\hat\lambda=\lambda^*+O(\varepsilon)$ uniformly, moving every boundary by $O(\varepsilon)$,
which does not change the order. An exact-tie query has $s(\nu_i)=0$, so even a certain flip
costs $0$ in score; re-clearing keeps $\frac1n\sum b=B$, so no budget-overrun loss. $\square$

### Corollary 2 (collapse is a fixed-price artifact, not a noise artifact) [proved]
Drop budget re-clearing: hold a fixed price/threshold $\lambda$ (equivalently a fixed cost
weight, as in confidence-threshold routers). Then near-tie flips are **not** repriced; their
cost is paid in raw budget. With symmetric noise on near-tied scores, the higher-mean
(costlier) item is the modal $\arg\max$, so the spend $\frac1n\sum b$ drifts up toward
$m_K$ — **routing collapse** — and/or, if feasibility is enforced by a frozen $\lambda$,
the value regret is $\Theta(\varepsilon)$ (first order) because the value gap at a boundary
is $\lambda(m_b-m_a)\neq0$. Re-clearing is the cure; a better predictor is not necessary.

**Status vs. data.** This matches (i) E2 (engineered near-tie population: $R$ at
$\varepsilon=0.01$ is $4.79\times10^{-6}$, indistinguishable from the smooth population's
$4.76\times10^{-6}$ — no collapse under re-clearing); (ii) E5 (frozen $\lambda$ strictly
worse); and explains Lai & Ye's collapse as a property of fixed-threshold routers.
**Gate before claiming:** confirm Lai & Ye's routers indeed deploy without per-batch budget
re-clearing (read their serving protocol). Empirically E1 discrete slope was $1.77$, not a
clean $2$: the benchmark population carries a $25\%$ atom at the **abandonment** boundary
($m_0=0$ vs $m_1$), which is a non-indifference edge (value gap $\neq0$ but here the lower
item is $b=0$, a genuine quality cliff), plus heteroscedastic noise and step-function
clearing slack — mixed orders. A finer-$\varepsilon$ grid on a smooth, atom-free population
should recover slope $2$; flagged for the next run.

---

## 6. Binning tradeoff (link to Q2)

A $K$-quantile-binned policy (Snell) assigns one budget per bin: an induced discrete menu
with $K$ data-dependent levels.

### Proposition 4 (adaptivity–robustness tradeoff) [proved mod regularity]
Write $R(K,\varepsilon)=A(K)+N(K,\varepsilon)$ where $A(K)$ is the approximation error of
replacing $b^*(\nu)$ by its bin-constant value (no noise) and $N$ is the noise-flip loss.
Then, for a density with bounded $b^{*\prime}$,
$$A(K)=\Theta(K^{-2}),\qquad N(K,\varepsilon)=O(K\varepsilon^2)\ \text{(re-clearing)},$$
so the regret-minimizing bin count is interior, $K^*(\varepsilon)=\Theta(\varepsilon^{-2/3})$,
**decreasing in the noise scale**.

*Proof idea.* $A(K)$: within a bin of width $\sim1/K$ in the quantile, $b^*$ varies by
$O(1/K)$, contributing $O(1/K^2)$ to a smooth concave objective (midpoint quadrature error).
$N$: there are $K-1$ boundaries, each contributing the Theorem 2 flip loss $O(\varepsilon^2)$.
Balancing $K^{-2}$ against $K\varepsilon^2$ gives $K^*\sim\varepsilon^{-2/3}$. $\square$

*Empirical (E3):* $K^*(\varepsilon)$ stays at the tested ceiling $55$ for $\varepsilon\le0.285$
and drops to $21/13/3$ at $\varepsilon=0.43/0.66/1.0$ — interior and decreasing, as predicted;
Snell's $K=5$ is far inside the "more bins is better" region at realistic noise. (The exact
$\varepsilon^{-2/3}$ rate is not yet checked against the ceiling-limited E3; re-run with a
higher ceiling to fit the exponent.)

---

## 7. What is proved, and the open program

**Proved (this note):** the closed form (Prop 1–2); the second-order envelope law and its
explicit weight (Thm 1, Cor 1); the dual-coupling decomposition and its protective sign
(Prop 3); the score-currency framing, tie-immunity under re-clearing, and the collapse-as-
fixed-price-artifact corollary (Thm 2, Cor 2); the binning tradeoff (Prop 4).

**Open / next:**
- **Q1c rate cleanup:** the clean discrete $\Theta(\varepsilon^2)$ on an atom-free
  population (finer-$\varepsilon$ run); the abandonment-cliff contribution as a separate
  first-order term (it is a non-indifference boundary).
- **Q3 (robust design) [conjecture]:** the minimax allocation over an error ball is the
  index rule on a *pessimistic* hazard estimate, with the pessimism level set globally by
  the dual (port Huang et al.'s within-query pessimism to cross-query under the budget).
- **Q4 (decision-focused predictor) [conjecture]:** the regret-optimal training loss for
  $\hat\nu$ is the score-gap-weighted misallocation loss (Thm 1's weight $w(\nu)$ /
  Thm 2's $|s|$), not MSE — an SPO objective for difficulty predictors; Lai & Ye's
  MSE$\to$ranking ablation is the existing empirical signal.
- **Beyond binary reward:** replace $q=1-e^{-\nu b}$ by a general concave production
  function $q(\theta,b)$; Thm 1 survives verbatim with $\lambda^*\nu_i$ replaced by
  $-q_{bb}$ at the optimum; the hump/abandonment geometry generalizes.

## 8. Minimal formula card

1. $q(\nu,b)=1-e^{-\nu b}$, $\ \nu=-\ln(1-\mu)$.
2. $b^*(\nu;\lambda)=[\,\ln(\nu/\lambda)/\nu\,]_+$, hump, peak at $\nu=e\lambda$, abandon $\nu\le\lambda$.
3. $R(\varepsilon)=\frac{\varepsilon^2}{2n}\sum_i\lambda^*\nu_i(\dot b_i)^2+o(\varepsilon^2)$; first order $=0$ by envelope.
4. weight $w(\nu)=\lambda^*[1-\ln(\nu/\lambda^*)]^2/\nu$, zero at peak ⇒ error shape matters.
5. $\dot\lambda=-\frac{\sum\partial_\nu b^*\,\dot{\hat\nu}_i}{\sum\partial_\lambda b^*}$ ⇒ cross-query coupling, protective.
6. discrete: regret currency = score $s=q-\lambda m$; boundary ⇒ $s$-gap $0$ ⇒ tie-immune under re-clearing; collapse needs fixed $\lambda$.
7. binning: $R=\Theta(K^{-2})+O(K\varepsilon^2)$, $K^*\sim\varepsilon^{-2/3}$.
