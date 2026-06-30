"""L1 check for ts-refactor-03: behavior preserved + currency logic extracted.

The behavior assertions pass on the seed already (the formatting is correct, just
duplicated). The gap is structural: there is no shared ``common.py``, so the
extraction assertions fail until the duplicated ``_format_currency`` is moved into
``common.format_currency`` and imported by all three report modules.

``common`` is imported lazily inside the structural tests (not at module top)
so its absence on the seed fails only those tests, leaving the behavior tests to
demonstrate that rendering is unchanged.
"""

import importlib
from pathlib import Path

import report_a
import report_b
import report_c

_REPORTS = (report_a, report_b, report_c)


def test_reports_render_expected_output():
    assert report_a.render("Revenue", 1234.5) == "Revenue: $1,234.50"
    assert report_b.render("Cost", 1000000) == "Cost: $1,000,000.00"
    assert report_c.render("Net", 0.05) == "Net: $0.05"


def test_currency_logic_extracted_to_common():
    assert Path("common.py").is_file(), "expected a shared helper module common.py"
    common = importlib.import_module("common")
    assert callable(common.format_currency)
    assert common.format_currency(1234.5) == "$1,234.50"


def test_all_reports_use_the_shared_helper():
    common = importlib.import_module("common")
    for module in _REPORTS:
        assert module.format_currency is common.format_currency, (
            f"{module.__name__} must import format_currency from common, not redefine it"
        )
