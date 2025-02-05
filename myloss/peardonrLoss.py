from xarray.core.dataarray import DataArray
from nhWrap.neuralhydrology.evaluation.metrics import _validate_inputs, _mask_valid
import numpy as np
from scipy import stats

def pearsonr(obs: DataArray, sim: DataArray) -> float:
    """Calculate pearson correlation coefficient (using scipy.stats.pearsonr)

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Pearson correlation coefficient

    """

    print("this is my pearson")

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 2:
        return np.nan

    r, _ = stats.pearsonr(obs.values, sim.values)

    return float(r)