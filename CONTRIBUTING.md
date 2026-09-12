# Contributing to TriDomainMoE

Thank you for your interest in contributing to **TriDomainMoE**! We welcome pull requests, feature enhancements, bug fixes, and quantitative research discussions.

---

## Code of Conduct & Quantitative Standards

This project adheres to rigorous mathematical and institutional engineering standards:
1. **Zero Future Lookahead Bias**: No feature or indicator may reference $t+k$ observations. All convolutions must be causal (`CausalConv1d`).
2. **Deterministic Risk Controls**: Pre-trade filters, volatility targeting, and trailing drawdown circuit breakers must be preserved in all pull requests.
3. **No Hardcoded Credentials**: Pull requests containing hardcoded credentials, server IPs, account numbers, or personal user paths will be rejected immediately.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TriDomainMoE.git
   cd TriDomainMoE
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install editable package with test dependencies**:
   ```bash
   pip install -e ".[test]"
   ```

---

## Running Unit Tests

All 25 automated unit tests must pass before submitting a pull request:
```bash
pytest tests/ -v
```

---

## Submitting Pull Requests

1. Create a feature branch (`git checkout -b feature/my-enhancement`).
2. Implement your changes with accompanying unit tests in `tests/`.
3. Verify test passing and formatting (`pytest tests/`).
4. Commit your changes using semantic messages (`git commit -m "feat(router): add entropy regularization penalty"`).
5. Push to your fork and open a Pull Request against `main`.
