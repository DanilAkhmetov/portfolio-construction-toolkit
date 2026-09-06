"""Research-data loading utilities for the portfolio construction toolkit.

This module provides small convenience functions for loading and formatting
financial datasets used in the accompanying my portfolio-construction notebooks
and research examples. It is not intended to be a general market-data API and
it does not download data automatically.

The loaders are designed around well-known research datasets, primarily:

- Kenneth R. French Data Library
  Fama-French factors, size portfolios, and 30/49 industry portfolios.
  Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

- EDHEC Hedge Fund Indices
  Hedge-fund strategy index returns developed by EDHEC-Risk Institute.
  Source/origin: EDHEC-Risk Institute / EDHEC Business School.

Users should obtain the required source files from the relevant provider and
store them locally. By default, the functions look for files in a ``data``
directory next to this module. Every public loader accepts ``data_dir`` so the
same functions can be used with data stored anywhere else.
"""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

KEN_FRENCH_DATA_LIBRARY = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
)
EDHEC_SOURCE = "EDHEC-Risk Institute / EDHEC Business School"


def _as_data_dir(data_dir):
    """Return ``data_dir`` as an expanded, absolute :class:`pathlib.Path`."""
    return Path(data_dir).expanduser().resolve()


def get_hfi_returns(data_dir=DEFAULT_DATA_DIR):
    """Load monthly EDHEC Hedge Fund Index returns.

    Expected file
    -------------
    ``edhec-hedgefundindices.csv``

    Source
    ------
    EDHEC-Risk Institute / EDHEC Business School.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing the source CSV file. Defaults to the local
        ``data`` directory next to this module.

    Returns
    -------
    pandas.DataFrame
        Monthly hedge-fund index returns expressed as decimals, indexed by a
        monthly ``PeriodIndex``.

    Notes
    -----
    The source file used in the course stores returns in percent, so values
    are divided by 100.
    """
    data_dir = _as_data_dir(data_dir)
    hfi = pd.read_csv(
        data_dir / "edhec-hedgefundindices.csv",
        header=0,
        index_col=0,
        parse_dates=True,
    )
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    return hfi


def get_ffme_returns(data_dir=DEFAULT_DATA_DIR):
    """Load small- and large-cap Fama-French size-portfolio returns.

    Expected file
    -------------
    ``Portfolios_Formed_on_ME_monthly_EW.csv``

    Source
    ------
    Kenneth R. French Data Library.

    The loader extracts the lowest and highest market-cap deciles (``Lo 10``
    and ``Hi 10``) and renames them ``SmallCap`` and ``LargeCap``.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing the source CSV file.

    Returns
    -------
    pandas.DataFrame
        Monthly SmallCap and LargeCap returns expressed as decimals, indexed
        by a monthly ``PeriodIndex``.
    """
    data_dir = _as_data_dir(data_dir)
    me_m = pd.read_csv(
        data_dir / "Portfolios_Formed_on_ME_monthly_EW.csv",
        header=0,
        index_col=0,
        na_values=-99.99,
    )

    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets


def get_ind_file(filetype, weighting="vw", n_inds=30, data_dir=DEFAULT_DATA_DIR):
    """Load and format a Fama-French industry-portfolio dataset.

    Expected filename pattern
    -------------------------
    ``ind{n_inds}_m_{name}.csv``

    Examples
    --------
    ``ind30_m_vw_rets.csv``
        30 value-weighted industry portfolio returns.
    ``ind49_m_ew_rets.csv``
        49 equal-weighted industry portfolio returns.
    ``ind30_m_nfirms.csv``
        Number of firms in each 30-industry portfolio.
    ``ind30_m_size.csv``
        Average firm size in each 30-industry portfolio.

    Source
    ------
    Kenneth R. French Data Library.

    Parameters
    ----------
    filetype : {"returns", "nfirms", "size"}
        Type of industry data to load.
    weighting : {"ew", "vw"}, default "vw"
        Equal-weighted or value-weighted returns. Used only when
        ``filetype="returns"``.
    n_inds : int, default 30
        Number of industry portfolios, normally 30 or 49 for the datasets
        used in the research examples.
    data_dir : str or pathlib.Path, optional
        Directory containing the source CSV files.

    Returns
    -------
    pandas.DataFrame
        Monthly industry data with cleaned column labels and a monthly
        ``PeriodIndex``. Return files are converted from percent to decimal;
        size and firm-count datasets retain their source units.
    """
    data_dir = _as_data_dir(data_dir)

    if filetype == "returns":
        name = f"{weighting}_rets"
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError("filetype must be one of: returns, nfirms, size")

    filepath = data_dir / f"ind{n_inds}_m_{name}.csv"
    ind = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        na_values=-99.99,
    ) / divisor

    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_returns(weighting="vw", n_inds=30, data_dir=DEFAULT_DATA_DIR):
    """Load monthly Fama-French industry-portfolio returns.

    Convenience wrapper around :func:`get_ind_file` for the industry return
    files available from the Kenneth R. French Data Library.

    Parameters
    ----------
    weighting : {"ew", "vw"}, default "vw"
        Equal-weighted or value-weighted industry returns.
    n_inds : int, default 30
        Number of industry portfolios, normally 30 or 49.
    data_dir : str or pathlib.Path, optional
        Directory containing the source CSV files.

    Returns
    -------
    pandas.DataFrame
        Monthly industry returns expressed as decimals.
    """
    return get_ind_file(
        "returns",
        weighting=weighting,
        n_inds=n_inds,
        data_dir=data_dir,
    )


def get_ind_nfirms(n_inds=30, data_dir=DEFAULT_DATA_DIR):
    """Load the number of firms in each Fama-French industry portfolio.

    Expected file: ``ind{n_inds}_m_nfirms.csv``.
    Source: Kenneth R. French Data Library.
    """
    return get_ind_file("nfirms", n_inds=n_inds, data_dir=data_dir)


def get_ind_size(n_inds=30, data_dir=DEFAULT_DATA_DIR):
    """Load average firm size for each Fama-French industry portfolio.

    Expected file: ``ind{n_inds}_m_size.csv``.
    Source: Kenneth R. French Data Library.
    """
    return get_ind_file("size", n_inds=n_inds, data_dir=data_dir)


def get_ind_market_caps(n_inds=30, weights=False, data_dir=DEFAULT_DATA_DIR):
    """Derive industry market capitalizations from Fama-French data.

    The approximation used in the course is

    ``industry market cap = number of firms * average firm size``.

    Parameters
    ----------
    n_inds : int, default 30
        Number of industry portfolios.
    weights : bool, default False
        If ``False``, return estimated market capitalizations. If ``True``,
        normalize each row to return industry market-cap weights.
    data_dir : str or pathlib.Path, optional
        Directory containing the industry size and firm-count datasets.

    Returns
    -------
    pandas.DataFrame
        Estimated industry market capitalizations or market-cap weights.
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds, data_dir=data_dir)
    ind_size = get_ind_size(n_inds=n_inds, data_dir=data_dir)
    ind_mktcap = ind_nfirms * ind_size

    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        return ind_mktcap.divide(total_mktcap, axis="rows")

    return ind_mktcap


def get_total_market_index_rets(
    weighting="vw",
    n_inds=30,
    data_dir=DEFAULT_DATA_DIR,
):
    """Construct an estimated market return series from industry portfolios.

    Industry market capitalizations are approximated from the Fama-French
    firm-count and average-size datasets. These values are converted into
    market-cap weights and used to aggregate the industry return series.

    Parameters
    ----------
    weighting : {"ew", "vw"}, default "vw"
        Industry return dataset used in the aggregation.
    n_inds : int, default 30
        Number of industry portfolios.
    data_dir : str or pathlib.Path, optional
        Directory containing the required industry datasets.

    Returns
    -------
    pandas.Series
        Monthly estimated market returns.

    Notes
    -----
    The final series is rounded to three decimal places to preserve the
    behavior of the original course implementation.
    """
    size = get_ind_size(n_inds=n_inds, data_dir=data_dir)
    n_firms = get_ind_nfirms(n_inds=n_inds, data_dir=data_dir)
    rets = get_ind_returns(
        weighting=weighting,
        n_inds=n_inds,
        data_dir=data_dir,
    )

    cap = n_firms * size
    total = cap.sum(axis=1)
    weights = cap.divide(total, axis=0)

    return (rets * weights).sum(axis="columns").round(3)


def get_fff_returns(data_dir=DEFAULT_DATA_DIR):
    """Load monthly Fama-French research factors.

    Expected file
    -------------
    ``F-F_Research_Data_Factors_m.csv``

    Source
    ------
    Kenneth R. French Data Library.

    The course file contains the standard monthly series ``Mkt-RF``, ``SMB``,
    ``HML`` and ``RF``.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing the source factor CSV file.

    Returns
    -------
    pandas.DataFrame
        Monthly Fama-French factor returns expressed as decimals, indexed by a
        monthly ``PeriodIndex``.
    """
    data_dir = _as_data_dir(data_dir)
    rets = pd.read_csv(
        data_dir / "F-F_Research_Data_Factors_m.csv",
        header=0,
        index_col=0,
        na_values=-99.99,
    ) / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets
