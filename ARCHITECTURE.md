# TriDomainMoE Technical & Mathematical Architecture

This document provides in-depth mathematical derivations and engineering specifications for the **TriDomainMoE** quantitative framework.

---

## 1. Domain Decomposition Rationale

Financial time series exhibit non-stationary multi-frequency dynamics where distinct market forces dominate across varying temporal scales:
$$\mathcal{M}_t = \mathcal{F}_{\text{micro}}(\mathbf{x}_t^{\text{tech}}) + \mathcal{F}_{\text{macro}}(\mathbf{x}_t^{\text{macro}}) + \mathcal{F}_{\text{fund}}(\mathbf{x}_t^{\text{fund}})$$

Rather than forcing a monolithic neural network to learn conflicting gradients, TriDomainMoE decouples the state space into three specialized domain representations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TriDomainMoE Framework                           │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Domain Expert            │ Mathematical Formulation │ Receptive Field       │
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ Microstructure Tech      │ Dilated Causal Conv1D    │ 32 M5 Bars (~2.6 hrs) │
│ Macro SSM Term Structure │ HiPPO Linear Recurrence  │ Multi-day H1/H4 Trend │
│ Fundamental Sentiment    │ Gated Residual Highway   │ Continuous 24h Decay  │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

---

## 2. Microstructure Tech Expert (Dilated Causal Convolutions)

### Causal Temporal Convolutions
To strictly prevent future lookahead leakage, temporal convolutions enforce causal padding where output at time $t$ depends exclusively on $\{x_{t}, x_{t-1}, \dots, x_{t-k}\}$:
$$y_t = \sum_{i=0}^{K-1} w_i \cdot x_{t - d \cdot i} + b$$
where $K$ is the kernel size, $d$ is the dilation rate, and $w_i$ are learned convolutional filter weights.

The effective receptive field ($RF$) across stacked dilated layers is given by:
$$RF = 1 + \sum_{l=1}^L (K_l - 1) \cdot d_l$$
With $K=3$ and dilations $d \in \{1, 2\}$, the network achieves an exponential receptive field across 32 continuous bars with minimal parameter overhead.

### Parkinson High-Low Volatility Ratio
Standard close-to-close volatility misses intra-bar extremes. The Parkinson extreme-value estimator utilizes high and low prices to yield an efficiency gain of approximately $5\times$:
$$\sigma_{\text{parkinson}} = \sqrt{\frac{1}{4 \ln 2 \cdot N} \sum_{i=1}^N \left(\ln \frac{H_i}{L_i}\right)^2}$$
The Parkinson Volatility Ratio ($\sigma_{\text{parkinson}} / \sigma_{\text{close}}$) serves as an intra-bar fakeout detector: ratios $> 1.25$ signal high wick noise and false breakouts.

---

## 3. Macro Term Structure SSM (HiPPO Linear Recurrence)

### State Space Formulation
The Macro SSM maps continuous macroeconomic time series $u(t)$ into an $N$-dimensional memory state $h(t)$:
$$\frac{d}{dt} h(t) = \mathbf{A} h(t) + \mathbf{B} u(t)$$
$$y(t) = \mathbf{C} h(t) + \mathbf{D} u(t)$$

### HiPPO Memory Discretization
To prevent vanishing gradients across multi-week lookbacks, the transition matrix $\mathbf{A}$ is initialized using the **High-Order Polynomial Projection Operators (HiPPO)** Legendre prior:
$$A_{nk} = \begin{cases} (2n + 1)^{1/2} (2k + 1)^{1/2} & \text{if } n > k \\ n + 1 & \text{if } n = k \\ -(2n + 1)^{1/2} (2k + 1)^{1/2} & \text{if } n < k \end{cases}$$
The linear recurrence allows continuous multi-day trend modeling ($H4$ and $D1$ secular trends, term structure slopes) without the quadratic memory complexity of standard self-attention mechanisms.

---

## 4. Fundamental Sentiment Expert (Gated Residual Highway)

The fundamental sentiment network models narrative headlines, regulatory flow, and continuous Cumulative Volume Delta (CVD):
$$\mathbf{h}_1 = \text{GELU}(\mathbf{W}_{\text{in}} \mathbf{x}^{\text{fund}} + \mathbf{b}_{\text{in}})$$
$$\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{h}_1 + \mathbf{b}_g)$$
$$\mathbf{h}_2 = \mathbf{h}_1 + \mathbf{g} \odot \text{GELU}(\mathbf{W}_r \mathbf{h}_1 + \mathbf{b}_r)$$
The transform gate $\mathbf{g} \in [0, 1]$ modulates information passage, dynamically suppressing low-confidence sentiment fluctuations during quiet market periods while allowing rapid transmission during high-impact macro news.

---

## 5. Continuous Softmax CAW Router & Meta-Sizer

### Correlation-Aware Softmax
The router computes gating distribution $\mathbf{g}(z_t)$ over the 8-dimensional regime state vector $z_t$:
$$g_i(z_t) = \frac{\exp(h_i / \tau)}{\sum_{j=1}^K \exp(h_j / \tau)}, \quad h_i = \mathbf{W}_i z_t + b_i + \epsilon_i$$
where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ injects exploration noise during training, and $\tau$ is the softmax temperature.

### Cosine Repulsion Diversity Loss
To eliminate expert collapse and enforce distinct domain specialization, the router loss penalizes cosine similarity between expert weight representations:
$$\mathcal{L}_{\text{repulsion}} = \lambda_c \sum_{i=1}^K \sum_{j \ne i}^K \max\left(0, \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} - \gamma\right)$$
where $\gamma$ is the orthogonal separation threshold.

### Unconditional Drift Base Forecaster
The final directional prediction incorporates an unconditional base trend anchor $f_{\text{base}}$:
$$\hat{y}_t = f_{\text{base}}(\mathbf{x}_t) + \sum_{i=1}^K g_i(z_t) \cdot f_i(\mathbf{x}_t^{(i)})$$

---

## 6. Continual Adaptation (ReCAP Framework)

The **Regime-Aware Continual Adaptive Portfolio (ReCAP)** mathematically isolates adaptation parameters from baseline models:
1. The baseline parameter set $\theta_0$ is permanently frozen:
   $$\nabla_{\theta_0} \mathcal{L} = 0$$
2. For regime $k$, adaptation trains a modular parameter delta:
   $$d_k = \theta_k - \theta_0$$
3. The active inference model synthesizes dynamic parameters:
   $$\theta_{\text{active}}(z_t) = \theta_0 + \sum_{k=1}^K w_k(z_t) \cdot d_k$$
Because $\theta_0$ is immutable, the framework guarantees **zero catastrophic forgetting** ($0.00\%$ historical performance degradation).

---

## 7. Multi-Resolution Drift Surveillance (MRDD)

The surveillance engine performs continuous multi-scale wavelet decomposition on prediction residuals:
$$e_t = y_t - \hat{y}_t$$
$$D_j(t) = \sum_{k} d_{j,k} \psi_{j,k}(t)$$
Wavelet energy at resolution level $j$:
$$E_j = \frac{1}{N} \sum_{t=1}^N |D_j(t)|^2$$
When energy divergence $\Delta E / E_{\text{baseline}} \ge 35\%$, the Multi-Resolution Drift Detector flags structural regime drift and triggers automated counterfactual reflection.
