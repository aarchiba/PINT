""" Various of tests on the wideband DM data
"""

import os
import numpy as np
import pytest
from copy import deepcopy

from pint.models import get_model
from pint.toa import get_TOAs
from pinttestdata import datadir
from pint.residuals import WidebandTOAResiduals


os.chdir(datadir)


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

        r = WidebandTOAResiduals(self.toas, self.model)

        model2 = deepcopy(self.model)
        for i, be in enumerate(all_backends):
            dm_jump_map[be].value += i + 1

        r2 = WidebandTOAResiduals(self.toas, model2)

        delta_dm = (
            r2.residual_objs["dm"].resids_value - r.residual_objs["dm"].resids_value
        )
        delta_dm_intended = np.zeros_like(delta_dm)
        for i, be in enumerate(all_backends):
            delta_dm_intended[toa_backends == be] = i + 1
        assert np.allclose(delta_dm, delta_dm_intended)

    def test_dm_noise(self):
        pass
