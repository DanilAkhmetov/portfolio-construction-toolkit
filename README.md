# Portfolio Construction Toolkit

This is a Python research toolkit for portfolio construction, developed alongside my study of portfolio management and quantitative finance.

This small project brings together reusable implementations of portfolio construction theory. The toolkit is intended to support both learning and research projects using real financial data.

> **Status:** active development. This is an evolving research toolkit rather than a finished production library. The current version is being published as a working foundation and will be polished and extended as new projects are developed.

## What is included

### Risk and performance analytics

- Return compounding and annualization
- Annualized volatility and Sharpe ratio
- Drawdown analysis
- Semideviation
- Skewness and kurtosis
- Jarque-Bera normality testing
- Historical VaR
- Gaussian VaR and Cornish-Fisher modified VaR
- Historical CVaR / Expected Shortfall
- Summary performance statistics

### Portfolio optimization

- Portfolio return and volatility
- Two-asset and multi-asset efficient frontiers
- Minimum-volatility portfolios
- Maximum Sharpe Ratio portfolio
- Global Minimum Variance portfolio
- Long-only constrained optimization using SLSQP

### Dynamic allocation and simulation

- Constant Proportion Portfolio Insurance (CPPI)
- Drawdown-based floors
- Geometric Brownian Motion simulation
- Monte Carlo analysis of terminal wealth
- Fixed-mix allocation
- Glide-path allocation
- Floor-based dynamic allocation
- Drawdown-controlled allocation

### Fixed income and Asset-Liability Management

- Discount factors and present value
- Funding-ratio analysis
- CIR interest-rate simulation
- Zero-coupon bond pricing under CIR
- Coupon bond cash flows and pricing
- Macaulay duration
- Duration matching
- Bond total returns
- Performance-Seeking Portfolio / Goal-Hedging Portfolio allocation workflows

### Sharpe Style analysis and backtesting

- Return-based style analysis
- Portfolio fitting through constrained tracking-loss minimization
- Equal-weighted and capitalization-weighted portfolios
- Rolling / walk-forward portfolio construction
- Lagged portfolio weights to avoid using estimated weights in the same return period

### Covariance estimation

- Sample covariance
- Constant-correlation covariance estimator
- Covariance shrinkage toward a constant-correlation prior
- GMV portfolio construction using interchangeable covariance estimators

### Black-Litterman Implementation

- Market-implied equilibrium returns
- View matrices and view uncertainty
- Proportional-prior Omega construction
- Black-Litterman posterior expected returns and covariance
- Mean-variance and Maximum Sharpe portfolio weights from posterior estimates

### Risk budgeting & Risk Parity Portfolios

- Asset risk contributions
- Target risk-contribution portfolios
- Equal Risk Contribution / Risk Parity portfolios

## Repository structure

```text
portfolio-construction-toolkit/
├── portfolio_toolkit.py   # portfolio construction & analytics functions
├── data.py                # research-data loading
├── README.md
├── requirements.txt
└── .gitignore
```

`portfolio_toolkit.py` contains the reusable portfolio and risk methodology. Dataset-specific loading logic is kept separately in `data.py` so that the quantitative functions remain independent of any particular data source.

## Quick example

```python
import data as dt
import portfolio_toolkit as pt

# Load monthly Fama-French industry returns from a local data directory 
# (NOTE: this data should be downloaded from public sources before usage of data functions - check Data Utilities)
returns = dt.get_ind_returns(
    weighting="vw",
    n_inds=30,
    data_dir="/path/to/local/data"
)

# Estimate a shrinkage covariance matrix
cov = pt.shrinkage_cov(returns, delta=0.5)

# Construct the Global Minimum Variance portfolio
weights = pt.gmv(cov)

print(weights)
```

The same portfolio functions can also be used with any appropriately formatted `pandas` Series or DataFrame; the bundled data helpers are optional convenience functions for the research examples.

## Data utilities

`data.py` provides convenience loaders for several publicly available research datasets used in portfolio-analysis workflows. It is **not intended to be a general market-data API** and does not download market data automatically.

The underlying datasets should be obtained from their original providers. The main sources used by the current loaders are:

- **Kenneth R. French Data Library:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **EDHEC Hedge Fund Indices:** EDHEC-Risk Institute / EDHEC Business School

## Current research direction

The toolkit is being developed further through applied portfolio-management projects. 

## Python stack

The current implementation uses:

- NumPy
- pandas
- SciPy
- Matplotlib

Related research work may also use packages such as `statsmodels` for factor-regression analysis.

## Background

This wonderful toolkit grew out of my finance coursework, with the goal of turning course implementations into a reusable base for research & projects with real financial data. It is still evolving and will be refined as new projects add requirements and expose opportunities for better design.

## Disclaimer

This repository is for educational and research purposes only. It does not constitute investment advice or a recommendation to trade or invest in any security or strategy.
