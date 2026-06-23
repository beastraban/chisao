"""Public-API surface: the package imports and exposes what it promises."""

import chisao


def test_version_present():
    assert isinstance(chisao.__version__, str)
    assert chisao.__version__


def test_public_symbols_exist():
    expected = [
        "sticky_hands",
        "optimize",
        "carry_tiger_seed",
        "carry_tiger_rays",
        "random_seed",
        "lbfgs_batch",
        "gradient_ascent_batch",
        "deduplicate_peaks_L_infinity",
        "estimate_peak_width",
        "SampleBank",
        "SingleWhip",
        "randcoord_line_search_batch",
        "get_gpu_info",
        "get_array_module",
    ]
    for name in expected:
        assert hasattr(chisao, name), f"missing public symbol: {name}"
