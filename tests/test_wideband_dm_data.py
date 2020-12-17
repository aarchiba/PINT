""" Various of tests on the wideband DM data
"""

import io
import os
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.time import TimeDelta
from pinttestdata import datadir
from numpy.testing import assert_allclose

from pint.models import get_model
from pint.residuals import Residuals, WidebandTOAResiduals
from pint.toa import get_TOAs, make_fake_toas
from pint.fitter import WidebandTOAFitter

os.chdir(datadir)


par_base = """
PSR J1234+5678
ELAT 0
ELONG 0
F0 1
DM 10
PEPOCH 57000
"""
par = (
    par_base
    + """
DMJUMP -fe L-wide 0
"""
)

tim = """
FORMAT 1
fake 999999 57000 1 @
fake 999999 57001 1 @
fake 1400 57002 1 ao
fake 1400 57003 1 ao
fake 1400 57004 1 ao -fe L-wide -pp_dm 20 -pp_dme 1e-6
fake 1400 57005 1 ao -fe L-wide -pp_dm 20 -pp_dme 1e-6
fake 1400 57006 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
fake 1400 57007 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
"""

tim_all = """
FORMAT 1
fake 1400 57002 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
fake 1400 57003 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
fake 1400 57004 1 ao -fe L-wide -pp_dm 20 -pp_dme 1e-6
fake 1400 57005 1 ao -fe L-wide -pp_dm 20 -pp_dme 1e-6
fake 1400 57006 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
fake 1400 57007 1 ao -fe Rcvr1_2 -pp_dm 10 -pp_dme 1e-6
"""


@pytest.fixture
def wb_model(tmpdir):
    parfile = tmpdir / "file.par"
    with open(parfile, "wt") as f:
        f.write(par)
    return get_model(str(parfile))


@pytest.fixture
def wb_toas(wb_model):
    toas = get_TOAs(io.StringIO(tim))
    for i in range(9):
        r = Residuals(toas, wb_model)
        if np.all(r.time_resids < 1 * u.ns):
            break
        toas.adjust_TOAs(TimeDelta(-r.time_resids))
    else:
        raise ValueError
    return toas


@pytest.fixture
def wb_toas_all(wb_model):
    toas = get_TOAs(io.StringIO(tim_all))
    for i in range(9):
        r = Residuals(toas, wb_model)
        if np.all(r.time_resids < 1 * u.ns):
            break
        toas.adjust_TOAs(TimeDelta(-r.time_resids))
    else:
        raise ValueError
    return toas


class TestDMData:
    def setup(self):
        self.model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        self.toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim")

    def test_data_reading(self):
        dm_data_raw, valid = self.toas.get_flag_value("pp_dm")
        # For this input, the DM number should be the same with the TOA number.
        dm_data = np.array(dm_data_raw)[valid]
        assert len(valid) == self.toas.ntoas
        assert len(dm_data) == self.toas.ntoas
        assert dm_data.mean != 0.0

    def test_dm_modelcomponent(self):
        assert "DispersionJump" in self.model.components.keys()
        assert "ScaleDmError" in self.model.components.keys()
        assert "SolarWindDispersion" in self.model.components.keys()

    def test_dm_jumps(self):
        # First get the toas for jump
        toa_backends, valid_flags = self.toas.get_flag_value("fe")
        toa_backends = np.array(toa_backends)
        all_backends = list(set(toa_backends))
        dm_jump_value = self.model.jump_dm(self.toas)
        dm_jump_params = [
            getattr(self.model, x)
            for x in self.model.params
            if (x.startswith("DMJUMP"))
        ]
        dm_jump_map = {}
        for dmj in dm_jump_params:
            dm_jump_map[dmj.key_value[0]] = dmj
        for be in all_backends:
            assert all(dm_jump_value[toa_backends == be] == -dm_jump_map[be].quantity)

        r = WidebandTOAResiduals(
            self.toas, self.model, dm_resid_args=dict(subtract_mean=False)
        )

        model2 = deepcopy(self.model)
        for i, be in enumerate(all_backends):
            dm_jump_map[be].value += i + 1

        r2 = WidebandTOAResiduals(
            self.toas, model2, dm_resid_args=dict(subtract_mean=False)
        )

        delta_dm = r2.dm.resids_value - r.dm.resids_value
        delta_dm_intended = np.zeros_like(delta_dm)
        for i, be in enumerate(all_backends):
            delta_dm_intended[toa_backends == be] = -(i + 1)
        assert np.allclose(delta_dm, delta_dm_intended)


def test_wideband_residuals(wb_model, wb_toas):
    r = WidebandTOAResiduals(wb_toas, wb_model, dm_resid_args=dict(subtract_mean=False))
    assert len(r.toa.time_resids) == len(wb_toas)
    assert len(r.dm.dm_data) < len(wb_toas)


def test_wideband_residuals_dmjump(wb_model, wb_toas):
    r = WidebandTOAResiduals(wb_toas, wb_model, dm_resid_args=dict(subtract_mean=False))
    model = deepcopy(wb_model)
    assert wb_model.DMJUMP1.value == 0
    model.DMJUMP1.value = 10
    assert model.DMJUMP1.value == 10
    with pytest.raises(AttributeError):
        model.DMJUMP0
    with pytest.raises(AttributeError):
        model.DMJUMP2
    r2 = WidebandTOAResiduals(wb_toas, model, dm_resid_args=dict(subtract_mean=False))
    assert 0 < np.sum(r.dm.resids_value != r2.dm.resids_value) < len(r.dm.resids_value)


def test_wideband_residuals_dof(wb_model, wb_toas_all):
    wb_model.free_params = ["DMJUMP1"]
    r = WidebandTOAResiduals(
        wb_toas_all, wb_model, dm_resid_args=dict(subtract_mean=False)
    )
    assert r.dof == 12 - 2
    assert_allclose(r.chi2, 2e14)
    assert_allclose(r.reduced_chi2, r.chi2 / r.dof)


@pytest.mark.xfail(reason="All TOAs must have DMs, currently")
def test_wideband_fit_dmjump(wb_model, wb_toas):
    wb_model.free_params = ["DMJUMP1"]
    fitter = WidebandTOAFitter(wb_toas, wb_model)
    fitter.fit_toas()
    assert_allclose(fitter.model.DMJUMP1.value, -10, atol=1e-3)


def test_wideband_fit_dmjump_all(wb_model, wb_toas_all):
    wb_model.free_params = ["DMJUMP1"]
    fitter = WidebandTOAFitter(wb_toas_all, wb_model)
    fitter.fit_toas()
    print(fitter.print_summary())
    assert_allclose(fitter.model.DMJUMP1.value, -10, atol=1e-3)


def test_dmjump_uncertainty():
    model = get_model(io.StringIO("\n".join([par_base, "DMJUMP mjd 56000 58000 0"])))
    toas = make_fake_toas(57000, 59000, 32, model=model, obs="ao", error=0.1 * u.s)
    for f in toas.table["flags"]:
        f["pp_dm"] = 10.0
        f["pp_dme"] = 4e-5

    model.free_params = ["DM"]
    fitter = WidebandTOAFitter(toas, model)
    fitter.fit_toas()
    cov = fitter.get_covariance_matrix()
    dm_std = np.sqrt(cov[0, 0])

    model.free_params = ["DMJUMP1"]
    fitter = WidebandTOAFitter(toas, model)
    fitter.fit_toas()
    cov = fitter.get_covariance_matrix()
    assert_allclose(np.sqrt(cov[0, 0]), np.sqrt(2) * dm_std)

    model.free_params = ["DM", "DMJUMP1"]
    fitter = WidebandTOAFitter(toas, model)
    fitter.fit_toas()
    cov = fitter.get_covariance_matrix()
    assert_allclose(np.sqrt(cov[0, 0]), np.sqrt(2) * dm_std)
    assert_allclose(np.sqrt(cov[1, 1]), 2 * dm_std)

    model.free_params = ["F0", "DMJUMP1"]
    fitter = WidebandTOAFitter(toas, model)
    fitter.fit_toas()
    cov = fitter.get_covariance_matrix()
    assert_allclose(np.sqrt(cov[1, 1]), np.sqrt(2) * dm_std)
