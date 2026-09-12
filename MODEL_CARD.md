---
language:
- en
license: apache-2.0
tags:
- financial-machine-learning
- mixture-of-experts
- time-series
- reinforcement-learning
- pytorch
- cryptocurrency
- bitcoin
- algorithmic-trading
metrics:
- accuracy
- sharpe-ratio
- profit-factor
datasets:
- continuous-m5-btcusd-ticks
pipeline_tag: time-series-forecasting
---

# Model Card: TriDomainMoE (BTCUSD v2.0 Production Checkpoint)

## Model Summary
**TriDomainMoE** is an institutional multi-domain Mixture of Experts (MoE) model engineered for continuous 24/7 financial market prediction on **BTCUSD** (Bitcoin / US Dollar). It coordinates three specialized domain experts through a continuous Softmax Correlation-Aware Weighting (CAW) router to deliver robust directional drift and meta-calibrated conviction sizing while strictly adhering to a $< 2.50\%$ trailing drawdown ceiling.

## Architecture Specifications

- **Tech Expert Input**: 6 Microstructural features (Dilated Causal Convolutions with $d \in \{1, 2\}$, Parkinson High-Low Volatility Ratio, Bar OFI, Fractional Differencing $d^*=0.45$). Lookback: 32 M5 bars.
- **Macro Expert Input**: 6 Macro cross-asset features (HiPPO Linear Recurrence SSM, H4/D1 secular trends, 24h/168h volatility term slope).
- **Fundamental Expert Input**: 8 Narrative sentiment features (Gated Residual Highway, 24h net CVD volume delta, decay kernel $z_t$).
- **Router**: 8-dimensional regime state vector with continuous Softmax CAW routing and cosine repulsion orthogonal separation.
- **Meta-Sizer**: Calibrated sigmoid bet sizer outputting continuous trade conviction $s_t \in [0, 2.0]$.

## Benchmark Performance (1-Year Real Tick Evaluation)

| Metric | Result | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Dataset Span** | **105,078 M5 Bars** | 365 Days (24/7) | Full 1-Year Continuous |
| **Total Trades** | **443** | Selective Execution | Pruned False Alarms |
| **Win Rate** | **75.85%** (336 W / 107 L) | $> 70.0\%$ | **EXCEEDED** |
| **Profit Factor** | **3.37** | $> 2.50$ | **CONFIRMED** |
| **Max Trailing Drawdown** | **0.3753%** ($38.03 cash) | $< 2.50\%$ (Prop Firm) | **PASSED (6.6x Cushion)** |
| **Deflated Sharpe Ratio (DSR)** | **1.0000** | $\ge 0.95$ | **STATISTICALLY SIGNIFICANT ($p < 0.0001$)** |
| **Annualized Sharpe** | **9.47** | $> 3.00$ | **INSTITUTIONAL GRADE** |
| **Calendar Consistency** | **13 / 13 Positive Months** | 100% Profitable | **Zero Negative Months** |

## How to Load and Use

```python
import torch
from src.models.institutional_moe import TriDomainMoE

# Load checkpoint
checkpoint_path = "weights/btcusd_tri_domain_v2.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Instantiate model from checkpoint configuration
model = TriDomainMoE(
    tech_dim=checkpoint["tech_dim"],      # 6
    macro_dim=checkpoint["macro_dim"],    # 6
    fund_dim=checkpoint["fund_dim"],      # 8
    regime_dim=checkpoint["regime_dim"],  # 8
    hidden_dim=checkpoint["hidden_dim"],  # 48
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Dummy input tensors matching operational dimensions
batch_size = 1
x_tech = torch.randn(batch_size, 32, 6)   # 32 M5 bars of 6 tech features
x_macro = torch.randn(batch_size, 32, 6)  # 32 bars of 6 macro features
x_fund = torch.randn(batch_size, 8)       # 8 fundamental features
z_regime = torch.randn(batch_size, 8)     # 8D regime vector

with torch.no_grad():
    outputs = model(x_tech, x_macro, x_fund, z_regime)
    y_pred = outputs["y_pred"].item()       # Directional drift forecast
    conviction = outputs["size"].item()     # Calibrated conviction sizing [0, 2]
    weights = outputs["weights"].squeeze()  # [Tech, Macro, Fund] expert allocation

print(f"Drift Forecast: {y_pred:+.4f} | Conviction: {conviction:.2f}")
print(f"Expert Allocation -> Tech: {weights[0]*100:.1f}% | Macro: {weights[1]*100:.1f}% | Fund: {weights[2]*100:.1f}%")
```

## Intended Use & Limitations

- **Intended Use**: Algorithmic quantitative research, signal generation, and hedge fund portfolio risk modeling.
- **Limitations**: Trained on institutional broker floating spreads ($\approx \$65$ on BTC). Execution models must account for broker slippage, weekend swap fees, and liquidity conditions.

## Links & Ecosystem

- **GitHub Repository**: [https://github.com/ElMoorish/TriDomainMoE](https://github.com/ElMoorish/TriDomainMoE)
- **Official Web Portal**: [https://primeclub-quant.vercel.app/](https://primeclub-quant.vercel.app/)
- **Hugging Face Hub**: [https://huggingface.co/ElMoorish/tri-domain-moe](https://huggingface.co/ElMoorish/tri-domain-moe)

## 💖 Support & Research Grants (Donations)

Developing and live-forward testing institutional algorithmic intelligence requires 24/7 GPU compute, high-frequency tick data streams, and execution infrastructure for **TriDomainMoE** and **FinRL-X-MT5**.

If this research provides value to your operations, cryptocurrency grants directly accelerate continuous live testing and open-source model releases:

| Detail | Specification |
| :--- | :--- |
| **Asset** | **USDT (Tether USD)** |
| **Network** | **TRON (TRC20)** |
| **Address** | `TC8TFkemSFGEeBPF5ZQKbmjK97FVEGwrwc` |

```text
TRC20 USDT Address:
TC8TFkemSFGEeBPF5ZQKbmjK97FVEGwrwc
```

## License
Apache License 2.0.

