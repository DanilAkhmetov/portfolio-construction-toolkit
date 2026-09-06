"""
Reusable portfolio construction and quantitative finance tools.

Dataset-specific loading and preparation helpers live in ``data.py``.
"""

import pandas as pd
import scipy.stats
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# RETURN AND RISK ANALYTICS
# -----------------------------------------------------------------------------

def drawdown(return_series: pd.Series):
    """
    Takes a time SERIES of asset returns.
    Computes and returns a DataFrame that contains:
    - the wealth index
    - the previous peaks
    - percent drawdowns
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def semideviation(r):
    "negative SemiDeviation"
    is_negative = r<0

    return r[is_negative].std()


def semideviation3(r):
    """
    Returns the semideviation (downside volatility relative to the mean return).
    r must be a Series or a DataFrame, else raises a TypeError.
    """
    excess = r - r.mean()
    excess_negative = excess[excess < 0]
    excess_negative_square = excess_negative ** 2
    n_negative = (excess < 0).sum()

    return (excess_negative_square.sum() / n_negative) ** 0.5


def skewness(r):

    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()

    return exp / sigma_r **3


def kurtosis(r):

    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()

    return exp / sigma_r **4


def is_normal(r, p = 0.01):
    """
    The Null Hypothesis: Normallity.
    If p-value small enough -> reject

    """

    stats, p_value = scipy.stats.jarque_bera(r)

    return bool(p_value > p)


def var_historic(r, level = 5):

    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)

    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)

    else: raise TypeError


def var_gaussian(r, level = 5, CF = False):
    """
    Returns the Parametric VaR with absolute values;
    If CF True, returns with adjustments for skewness and kurtosis - Cornish - Fisher

    """

    z = scipy.stats.norm.ppf(level / 100)

    if CF:
        s = skewness(r)
        k = kurtosis(r)
        z = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3*z) * (k - 3) / 24
        - (2*z**3 - 5*z) * (s**2) / 36
    )

    return -(r.mean() + z*r.std(ddof = 0))


def cvar_historic(r, level = 5):
    """
    Computes econditional VaR

    """

    if isinstance(r, pd.Series):

        is_beyond = r <= - var_historic(r,level =level)

        return -r[is_beyond].mean()

    elif isinstance(r, pd.DataFrame):

        return r.aggregate(cvar_historic, level = level)

    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.
    """
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized Sharpe ratio of a set of returns.
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1

    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)

    return ann_ex_ret / ann_vol


# -----------------------------------------------------------------------------
# PORTFOLIO OPTIMIZATION
# -----------------------------------------------------------------------------

def portfolio_return(weights, rets):
    return weights.T @ rets


def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier.
    """
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })

    return ef.plot.line(x="Volatility", y="Returns", style=style)


def minimize_vol(target_return, er, cov):
    """
    Volatility Optimizer
    """
    n = er.shape[0]

    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq' ,
        'args': (er,) ,
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(
        portfolio_vol, init_guess,
        args = (cov,),
        method = 'SLSQP',
        options = {'disp': False},
        constraints = (return_is_target, weights_sum_to_1),
        bounds = bounds
    )
    if not results.success:
        raise RuntimeError(f"Minimum-volatility optimization failed: {results.message}")
    return results.x


def optimal_weights(n_points, er , cov):
    """
    Optimimal weights finder
    """

    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target,er,cov) for target in target_rs ]

    return weights


def plot_ef(n_points, er, cov, show_cml = False, show_ew = False,
            show_gmv = False,
            rf = 0.03, style=".-"):
    """
    Plots the N-asset efficient frontier.
    With Naive D. Portfolio / Global Min Var Portfolio / Max SR
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })

    ax = ef.plot.line(x="Volatility", y="Returns", style=style)

    if show_cml :

        w_msr = msr(rf, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [rf, r_msr]

        ax.plot(cml_x, cml_y, marker = 'o', linestyle = "dashed", label = 'CML')

    if show_ew :

        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color = 'red', marker = 'o', label = 'Equal Weights')

    if show_gmv:

        '''
        weights that minimize the portfolio vol-y
        '''
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n,n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
        }

        res = minimize(
            portfolio_vol, init_guess,
            args = (cov,),
            method = 'SLSQP',
            options = {'disp': False},
            constraints = (weights_sum_to_1,),
            bounds = bounds
        )
        weights_gmv = res.x
        """
        # OR we can simply use MSR (minimizing vol) with same ER for each asset
        weights_gmv = msr(0, np.repeat(1, cov.shape[0]), cov)

        r_gmv = portfolio_return(weights_gmv, er)
        vol_gmv = portfolio_vol(weights_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color = 'green', marker = 'o', label = 'GMV')


    ax.set_xlim(left=0)
    ax.legend()

    return ax


def msr(rf, er, cov):
    """
    Returns the weights that maximize SR
    """
    n = er.shape[0]

    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sr(weights, rf,er,cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        neg_sr = - (r-rf)/vol
        return neg_sr

    results = minimize(
        neg_sr, init_guess,
        args = (rf, er, cov,),
        method = 'SLSQP',
        options = {'disp': False},
        constraints = (weights_sum_to_1,),
        bounds = bounds
    )
    if not results.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {results.message}")
    return results.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


# -----------------------------------------------------------------------------
# DYNAMIC ALLOCATION AND SIMULATION
# -----------------------------------------------------------------------------

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset.
    Returns a dictionary containing:
    - Asset Value History
    - Risk Budget History
    - Risky Weight History
    """
    # Set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = risky_r.to_frame(name="R")

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12

    # Set up DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):

        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1-drawdown)

        cushion = (account_value - floor_value) / account_value

        risky_w = np.clip(m * cushion,0,1)

        safe_w = 1 - risky_w

        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        # Recompute the new account value at the end of this step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        # Save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth = start * (risky_r+1).cumprod()

    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }

    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats
    for the returns in the columns of r.
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, CF=True)
    hist_cvar5 = r.aggregate(cvar_historic)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15,
        steps_per_year=12, s_0=100.0, prices=True):
    """
    Simulates asset price paths using Geometric Brownian Motion.

    If prices=True, returns simulated price paths.
    If prices=False, returns simulated simple returns.

    Rows are time steps and columns are scenarios.
    """

    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year)

    rets_plus_1 = np.random.normal(
        loc=1 + mu * dt,
        scale=sigma * np.sqrt(dt),
        size=(n_steps+1, n_scenarios)
    )

    rets_plus_1[0] = 1

    rets_plus_1 = pd.DataFrame(rets_plus_1)

    if prices:
        return s_0 * rets_plus_1.cumprod()
    else:
        return rets_plus_1 - 1


def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3,
              floor=0.0, riskfree_rate=0.03, y_max=100, steps = 12):
    """
    Plot the results of a Monte Carlo simulation of CPPI.
    """
    start = 100

    sim_rets = gbm(
        n_scenarios=n_scenarios,
        mu=mu,
        sigma=sigma,
        prices=False,
        steps_per_year=steps
    )

    risky_r = pd.DataFrame(sim_rets)

    # Run the backtest
    btr = run_cppi(
        risky_r=risky_r,
        riskfree_rate=riskfree_rate,
        m=m,
        start=start,
        floor=floor
    )

    wealth = btr["Wealth"]

    y_max = wealth.values.max() * y_max / 100

    terminal_wealth = wealth.iloc[-1]

    # Calculate terminal wealth stats

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()

    failure_mask = np.less(terminal_wealth, start * floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures / n_scenarios

    e_shortfall = np.dot(
        start * floor - terminal_wealth,
        failure_mask
    ) / n_failures if n_failures > 0 else 0


    # Plot
    fig, (wealth_ax, hist_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
        figsize=(24, 9)
    )

    plt.subplots_adjust(wspace=0.0)

    wealth.plot(
        ax=wealth_ax,
        legend=False,
        alpha=0.3,
        color="indianred"
    )

    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start * floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)

    terminal_wealth.plot.hist(
        ax=hist_ax,
        bins=50,
        ec="w",
        fc="black",
        color = 'indianred',
        orientation="horizontal",
        rwidth = 0.9
    )

    hist_ax.axhline(y=start, ls=":", color="black")

    hist_ax.axhline(y=start, ls=":", color="black")

    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")

    bst=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.8)

    hist_ax.annotate(
        f"Mean: ${int(tw_mean)}",
        xy=(0.7, 0.90),
        bbox=bst,
        xycoords="axes fraction"
    )

    hist_ax.annotate(
        f"Median: ${int(tw_median)}",
        xy=(0.7, 0.85),
        bbox= bst,
        xycoords="axes fraction"
    )

    if floor > 0.01:
        hist_ax.axhline(
            y=start * floor,
            ls="--",
            color="red",
            linewidth=3
        )

        hist_ax.annotate(
            f"Violations: {n_failures} ({p_fail * 100:.2f}%)\nE Shortfall: ${int(e_shortfall)}",
            xy=(0.7, 0.7),
            xycoords="axes fraction",
            bbox= bst
        )

## Then, using widgets, create controls and build an interactive Monte Carlo

    """

    import ipywidgets as widgets
    from IPython.display import display

    cppi_controls = widgets.interactive(
        {fun},
        n_scenarios=widgets.IntSlider(min=1, max=1000, step=5, value=50),
        mu=(0.0, 0.2, 0.01),
        sigma=(0.0, 0.3, 0.05),
        floor=(0.0, 2.0, 0.1),
        m=(1, 5, 0.5),
        riskfree_rate=(0.0, 0.05, 0.01),
        y_max=widgets.IntSlider(
            min=0,
            max=100,
            step=1,
            value=100,
            description="Zoom Y Axis"
        )
    )

    display(cppi_controls)

    """


# -----------------------------------------------------------------------------
# FIXED INCOME AND ASSET-LIABILITY MANAGEMENT
# -----------------------------------------------------------------------------

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t.

    r is the per-period interest rate.
    Returns a DataFrame indexed by t.

    t can be a list/array/Index of time periods.
    r can be a float, Series, or DataFrame.
    """
    discounts = pd.DataFrame([(1 + r) ** -i for i in t])
    discounts.index = t

    return discounts


def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by
    the time as an index and amounts as values.

    r can be a scalar, Series, or DataFrame with the number of rows
    matching the number of rows in flows.
    """
    dates = flows.index
    discounts = discount(dates, r)

    return discounts.multiply(flows, axis="rows").sum()


def funding_ratio(assets, liabilities,r):
    """
    assets can be cash flows themselves
    """
    return pv(assets, r)/ pv(liabilities, r)


def inst_to_ann(r):
    """
    converts short rates to annualised
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    converts annualised rates to short rates
    """
    return np.log1p(r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,
        steps_per_year=12, r_0=None):
    """
    Simulates interest rate paths using the CIR model and also returns
    zero-coupon bond prices.

    Inputs b and r_0 are assumed to be annualized rates.
    Inside the model, they are converted to instantaneous rates.
    Returns:
    - rates: simulated annualized interest rates
    - prices: zero-coupon bond prices
    """

    if r_0 is None:
        r_0 = b

    r_0 = ann_to_inst(r_0)
    b = ann_to_inst(b)

    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1

    shock = np.random.normal(
        loc=0,
        scale=np.sqrt(dt),
        size=(num_steps, n_scenarios)
    )

    rates = np.empty_like(shock)
    rates[0] = r_0

    # For price generation
    h = np.sqrt(a**2 + 2 * sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        """
        Computes the price of a zero-coupon bond under the CIR model.
        ttm = time to maturity
        r = current instantaneous short rate
        """
        _A = (
            (2 * h * np.exp((h + a) * ttm / 2)) /
            (2 * h + (h + a) * (np.exp(h * ttm) - 1))
        ) ** (2 * a * b / sigma**2)

        _B = (
            2 * (np.exp(h * ttm) - 1) /
            (2 * h + (h + a) * (np.exp(h * ttm) - 1))
        )

        _P = _A * np.exp(-_B * r)

        return _P

    prices[0] = price(n_years, r_0)

    for step in range(1, num_steps):
        r_t = rates[step - 1]

        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]

        rates[step] = abs(r_t + d_r_t)

        # Price of a zero-coupon bond with remaining maturity
        prices[step] = price(n_years - step * dt, rates[step])

    rates = pd.DataFrame(
        data=inst_to_ann(rates),
        index=range(num_steps)
    )

    prices = pd.DataFrame(
        data=prices,
        index=range(num_steps)
    )

    return rates, prices


def show_cir_rates(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,
                   steps_per_year=12, r_0=0.03):

    rates, prices = cir(n_years, n_scenarios, a, b, sigma, steps_per_year, r_0)

    ax = rates.plot(figsize=(12, 6), legend=False)
    ax.axhline(0, color="black")
    ax.set_title("CIR Interest Rate Simulations")
    ax.set_ylabel("Annualized Rate")

    plt.show()

## Then, using widgets, create controls and build an interactive Monte Carlo

"""
rate_controls = widgets.interactive(
    show_cir_rates,
    n_years=widgets.IntSlider(min=1, max=20, value=10),
    n_scenarios=widgets.IntSlider(min=1, max=50, value=10),
    a=(0.01, 1.0, 0.01),
    b=(0.01, 0.2, 0.01),
    sigma=(0.01, 0.20, 0.01),
    steps_per_year=widgets.IntSlider(min=1, max=48, value=12),
    r_0=(0.0, 0.2, 0.01)
)

display(rate_controls)
"""


def show_cir_prices(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,
                    steps_per_year=12, r_0=0.03):

    rates, prices = cir(n_years, n_scenarios, a, b, sigma, steps_per_year, r_0)

    ax = prices.plot(figsize=(12, 6), legend=False)
    ax.axhline(1, color="black", linestyle=":")
    ax.set_title("CIR Zero-Coupon Bond Price Simulations")
    ax.set_ylabel("Zero-Coupon Bond Price")

    plt.show()

## Then, using widgets, create controls and build an interactive Monte Carlo

"""
price_controls = widgets.interactive(
    show_cir_prices,
    n_years=widgets.IntSlider(min=1, max=20, value=10),
    n_scenarios=widgets.IntSlider(min=1, max=50, value=10),
    a=(0.01, 1.0, 0.01),
    b=(0.01, 0.2, 0.01),
    sigma=(0.01, 0.20, 0.01),
    steps_per_year=widgets.IntSlider(min=1, max=48, value=12),
    r_0=(0.0, 0.2, 0.01)
)

display(price_controls)
"""


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a series of cash flows generated by a bond,
    indexed by coupon number.
    """
    n_coupons = round(maturity * coupons_per_year)
    if n_coupons < 1:
        raise ValueError("maturity must imply at least one coupon payment")
    coupon_amt = principal * coupon_rate / coupons_per_year

    coupon_times = np.arange(1, n_coupons + 1)

    cash_flows = pd.Series(
        data=coupon_amt,
        index=coupon_times
    )

    cash_flows.iloc[-1] += principal

    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=0.03,
               coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity,
    at which time the principal and the final coupon are returned.

    This is not designed to be efficient; rather, it illustrates
    the underlying principle behind bond pricing.

    If discount_rate is a DataFrame, then this is assumed to be the rate
    on each coupon date and the bond value is computed over time.
    The index of the discount_rate DataFrame is assumed to be the coupon number.
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)

        for t in pricing_dates:
            prices.loc[t] = bond_price(
                maturity - t / coupons_per_year,
                principal,
                coupon_rate,
                coupons_per_year,
                discount_rate.loc[t]
            )

        return prices

    else:
        # Base case: single time period
        if maturity <= 0:
            return principal + principal * coupon_rate / coupons_per_year

        cash_flows = bond_cash_flows(
            maturity,
            principal,
            coupon_rate,
            coupons_per_year
        )

        return pv(cash_flows, discount_rate / coupons_per_year)


def macaulay_duration(cf, r=0.03):
    """
    Computes the Macaulay duration of a sequence of cash flows.

    cf is a Series where:
    - index = payment times
    - values = cash flow amounts
    """
    dates = cf.index
    discounts = discount(dates, r)

    discounted_cf = discounts.multiply(cf, axis="rows")
    weights = discounted_cf.divide(discounted_cf.sum(), axis="columns")
    weighted_times = weights.multiply(np.asarray(dates), axis="rows")

    duration = weighted_times.sum()
    return duration.iloc[0] if len(duration) == 1 else duration


def match_durations(cf_t, cf_s, cf_l, r=0.03):
    """
    Computes the weight in the short-duration asset needed
    to match the duration of the target liability.

    cf_t = target liability cash flows
    cf_s = short-duration asset cash flows
    cf_l = long-duration asset cash flows
    r = discount rate
    """
    d_t = macaulay_duration(cf_t, r)
    d_s = macaulay_duration(cf_s, r)
    d_l = macaulay_duration(cf_l, r)

    w_s = (d_l - d_t) / (d_l - d_s)

    return w_s


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """

    """
    coupons = pd.DataFrame(data = 0.0, index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year * t_max/12),
                          dtype = int)
    coupons.iloc[pay_date] = principal * coupon_rate/coupons_per_year

    total_returns = (monthly_prices + coupons)/ monthly_prices.shift()-1

    return total_returns.dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a backtest simulation of allocating between two sets of returns.

    r1 and r2 are T x N DataFrames of returns, where:
    - T is the time-step index
    - N is the number of scenarios

    allocator is a function that takes two sets of returns and allocator-specific
    parameters, and produces an allocation to the first portfolio as a T x 1 DataFrame.

    Returns a T x N DataFrame of portfolio returns.
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")

    weights = allocator(r1, r2, **kwargs)

    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that don't match r1")

    r_mix = weights * r1 + (1 - weights) * r2

    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP
    across N scenarios.

    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
    - each column is a scenario
    - each row is the price for a timestep

    Returns a T x N DataFrame of PSP weights.
    """
    return pd.DataFrame(
        data=w1,
        index=r1.index,
        columns=r1.columns
    )


def terminal_values(rets):
    """
    Computes the terminal wealth for each return series.

    rets should be a Series or DataFrame of periodic returns.
    The function compounds returns over time and returns the final
    value of 1 unit invested at the beginning.
    """
    return (rets + 1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produces summary statistics on the terminal wealth per invested dollar
    across a range of N scenarios.

    rets is a T x N DataFrame of returns, where:
    - T is the time-step index
    - N is the number of scenarios

    Returns a 1-column DataFrame of summary statistics indexed by stat name.
    """
    terminal_wealth = (rets + 1).prod()

    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap

    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan

    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (terminal_wealth[reach] - cap).mean() if reach.sum() > 0 else np.nan

    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])

    return sum_stats


def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2.

    Returns a T x N DataFrame of weights in r1.
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]

    path = pd.Series(
        data=np.linspace(start_glide, end_glide, num=n_points)
    )

    paths = pd.concat([path] * n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns

    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal of providing exposure to the upside
    of the PSP without going below the floor.

    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.

    Returns a DataFrame with the same shape as psp_r/ghp_r representing
    the weights in the PSP.
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")

    n_steps, n_scenarios = psp_r.shape

    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)

    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)

    for step in range(n_steps):
        floor_value = floor * zc_prices.iloc[step]

        cushion = (account_value - floor_value) / account_value

        psp_w = (m * cushion).clip(0, 1)
        ghp_w = 1 - psp_w

        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w

        account_value = (
            psp_alloc * (1 + psp_r.iloc[step]) +
            ghp_alloc * (1 + ghp_r.iloc[step])
        )

        w_history.iloc[step] = psp_w

    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal of providing exposure to the upside
    of the PSP without violating the drawdown floor.

    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.

    Returns a DataFrame with the same shape as psp_r/ghp_r representing
    the weights in the PSP.
    """
    n_steps, n_scenarios = psp_r.shape

    account_value = np.repeat(1.0, n_scenarios)
    floor_value = np.repeat(1.0, n_scenarios)
    peak_value = np.repeat(1.0, n_scenarios)

    w_history = pd.DataFrame(
        index=psp_r.index,
        columns=psp_r.columns,
        dtype=float
    )

    for step in range(n_steps):
        floor_value = (1 - maxdd) * peak_value

        cushion = (account_value - floor_value) / account_value

        psp_w = (m * cushion).clip(0, 1)
        ghp_w = 1 - psp_w

        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w

        account_value = (
            psp_alloc * (1 + psp_r.iloc[step]) +
            ghp_alloc * (1 + ghp_r.iloc[step])
        )

        peak_value = np.maximum(peak_value, account_value)

        w_history.iloc[step] = psp_w

    return w_history


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


# -----------------------------------------------------------------------------
# STYLE ANALYSIS AND WALK-FORWARD BACKTESTING
# -----------------------------------------------------------------------------

def tracking_error(r_a, r_b):
    """
    Returns the root sum of squared active returns between two return series.

    This loss is used by ``style_analysis``. It is not the annualized
    standard deviation of active returns often called tracking error in
    performance reporting.
    """
    return np.sqrt(((r_a - r_b)**2).sum())


def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))


def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    if not solution.success:
        raise RuntimeError(f"Style-analysis optimization failed: {solution.message}")
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights


def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[-1]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew


def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights.
    We pass the full cap-weight history, but for each rolling window we only use the cap weights
    available at the end of that window. Those become the portfolio weights for the next period
    after the shift in backtest_ws !
    """
    w = cap_weights.loc[r.index[-1]]
    return w/w.sum()


def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    if not 1 <= estimation_window < n_periods:
        raise ValueError("estimation_window must be at least 1 and smaller than the number of observations")
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    # !!!!! Window 1 covers … up to Jan → weight is stamped Jan
    ## iloc[60] - January already !!!!
    ## TO BE CLEAR:   I am using weights result in 12th month for the returns of the 1st month - so i shift weights down by one date
    weights = weights.shift(1).dropna()
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns


# -----------------------------------------------------------------------------
# COVARIANCE ESTIMATION
# -----------------------------------------------------------------------------

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance
    """
    return r.cov()


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix using the Elton-Gruber
    constant correlation model.

    Steps:
    1. Compute sample correlations.
    2. Compute the average off-diagonal correlation.
    3. Build a correlation matrix where every off-diagonal element
       equals that average correlation.
    4. Keep the original asset volatilities.
    5. Convert the constant-correlation matrix into a covariance matrix.
    """

    rhos = r.corr()
    n = rhos.shape[0]
    if n < 2:
        raise ValueError("cc_cov requires returns for at least two assets")

    # Average correlation excluding diagonal of 1s
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))

    # Constant correlation matrix
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.0)

    # Asset volatilities
    sd = r.std()

    # Covariance = correlation * vol_i * vol_j
    # np.outer(sd, sd) takes two vectors and creates a matrix of all pairwise products.
    ccov = ccor * np.outer(sd, sd)

    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)


def weight_gmv(r, cov_estimator = sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns
    """
    est = cov_estimator(r, **kwargs)

    return gmv(est)


def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the sample covariance
    and the constant correlation covariance estimator.

    Formula:
        Sigma_shrink = delta * Sigma_CC + (1 - delta) * Sigma_sample

    Inputs:
    - r: returns DataFrame
    - delta: shrinkage intensity

    If delta = 0:
        use pure sample covariance

    If delta = 1:
        use pure constant-correlation covariance

    If delta = 0.5:
        average both estimates
    """
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")

    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)

    return delta * prior + (1 - delta) * sample


# -----------------------------------------------------------------------------
# BLACK-LITTERMAN ALLOCATION
# -----------------------------------------------------------------------------

def as_colvec(x):
    """
    Converts a 1D NumPy array with shape (n,) into a column vector with shape (n, 1),
    making matrix operations easier and more consistent with mathematical notation.
    """
    if x.ndim == 2:
        return x
    else:
        return np.expand_dims(x, axis=1)


def implied_rets(delta, sigma, w):
    """
    Computes implied equilibrium returns:

        Pi = delta * Sigma * w

    This is reverse optimization: we use market weights w from the market,
    covariance matrix sigma, and risk aversion delta to infer the returns that make the market
    portfolio optimal.
    """
    ir = delta * sigma.dot(w).squeeze()
    ir.name = 'Implied Returns'

    return ir


def proportional_prior(sigma, tau, p):
    """
    Computes the simplified Black-Litterman Omega matrix.

    Formula:

        Omega = diag(P * tau * Sigma * P.T)

    where:
    - sigma is the N x N covariance matrix of asset returns
    - tau is a scalar measuring uncertainty in the prior
    - p is the K x N view/linkage matrix
    - Omega is the K x K view uncertainty matrix

    This assumes that view uncertainty is proportional to the prior covariance
    of the view portfolios, and that view errors are uncorrelated.
    """

    # Covariance matrix of the view portfolios: P * tau * Sigma * P.T
    helit_omega = p.dot(tau * sigma).dot(p.T)

    # Keep only diagonal elements: assume view errors are uncorrelated
    omega = pd.DataFrame(
        np.diag(np.diag(helit_omega.values)),
        index=p.index,
        columns=p.index
    )

    return omega


def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=.02):
    """
    Computes Black-Litterman posterior expected returns and effective covariance.

    Returns
    -------
    mu_bl : pandas.DataFrame
        Posterior expected returns.
    sigma_bl_total : pandas.DataFrame
        Effective covariance matrix of future returns.
    """
    pi = implied_rets(delta, sigma_prior, w_prior)
    pi_vec = as_colvec(pi.values)
    q_vec = as_colvec(q.values)

    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)

    sigma_prior_values = sigma_prior.values
    p_values = p.values
    omega_values = omega.values

    tau_sigma_inv = inv(tau * sigma_prior_values)
    omega_inv = inv(omega_values)

    A = tau_sigma_inv + p_values.T @ omega_inv @ p_values
    B = tau_sigma_inv @ pi_vec + p_values.T @ omega_inv @ q_vec

    sigma_bl = inv(A)
    mu_bl_values = sigma_bl @ B
    sigma_bl_total_values = sigma_prior_values + sigma_bl

    mu_bl = pd.DataFrame(
        mu_bl_values,
        index=sigma_prior.index,
        columns=["rets"]
    )
    sigma_bl_total = pd.DataFrame(
        sigma_bl_total_values,
        index=sigma_prior.index,
        columns=sigma_prior.columns
    )

    return mu_bl, sigma_bl_total


def inverse(d):
    """
    Invert a pandas DataFrame while preserving row/column labels.
    """
    return pd.DataFrame(
        inv(d.values),
        index=d.columns,
        columns=d.index
    )


def w_msr(sigma, mu, scale=True):
    """
    Computes unconstrained Maximum Sharpe Ratio portfolio weights.

    Formula:

        w_MSR = Sigma^(-1) * mu / (1.T * Sigma^(-1) * mu)

    Inputs:
    - sigma: N x N covariance matrix as a DataFrame
    - mu: expected excess returns as a Series or one-column DataFrame
    - scale: if True, normalize weights to sum to 1

    Returns:
    - portfolio weights
    """
    w = inverse(sigma).dot(mu)

    if scale:
        w = w / w.sum()

    return w


def w_star(delta, sigma, mu):
    """
    Computes mean-variance optimal risky weights:

        w* = (1 / delta) * Sigma^(-1) * mu
    """
    return inverse(sigma).dot(mu) / delta


# -----------------------------------------------------------------------------
# RISK BUDGETING AND EQUAL RISK CONTRIBUTION
# -----------------------------------------------------------------------------

def risk_contribution(w, cov):
    """
    Computes percentage contribution of each asset to portfolio variance.

    Formula:
        RC_i = w_i * (Sigma w)_i
        Percent RC_i = RC_i / sum(RC_i)

    Inputs:
    - w: portfolio weights
    - cov: covariance matrix

    Returns:
    - percentage risk contributions, summing to 1
    """
    total_contrib = w * cov.dot(w)
    pct_contrib = total_contrib / total_contrib.sum()

    return pct_contrib


def target_risk_contribution(target, cov):

    def ssd(weights, target, cov):
        return ((risk_contribution(weights, cov) - target)**2).sum()

    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)

    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = ((0.0, 1.0),) * n

    results = minimize(
        ssd,
        init_guess,
        args=(target, cov),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1,),
        bounds=bounds
    )
    if not results.success:
        raise RuntimeError(f"Risk-contribution optimization failed: {results.message}")

    return results.x


def risk_parity_weights(cov):
    """
    Computes Equal Risk Contribution / Risk Parity weights.

    Target risk contribution for each asset:

        1 / n
    """

    n = cov.shape[0]

    return target_risk_contribution(target = np.repeat(1/n, cov.shape[0]),cov=cov)
