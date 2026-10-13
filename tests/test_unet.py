import warnings

import pytest

from diana.models.unet import _group_norm


class TestGroupNormFallback:
    @pytest.mark.parametrize("channels", [8, 16, 32, 64, 128, 512])
    def test_standard_widths_are_silent(self, channels):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gn = _group_norm(channels)
        assert gn.num_groups == min(32, channels)

    @pytest.mark.parametrize("channels,groups", [(7, 1), (12, 4), (30, 2)])
    def test_degenerate_widths_warn(self, channels, groups):
        with pytest.warns(UserWarning, match="group"):
            gn = _group_norm(channels)
        assert gn.num_groups == groups
