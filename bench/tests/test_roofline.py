from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from roofline.roofline import report_measured_h100_gate


class MeasuredGateLabelTests(unittest.TestCase):
    def test_historical_cuda_event_is_not_labeled_browser_visible(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            report_measured_h100_gate()

        report = output.getvalue()
        self.assertIn("first GPU RGB", report)
        self.assertIn("not CPU-ready or browser-visible", report)
        self.assertNotIn("first visible RGB", report)


if __name__ == "__main__":
    unittest.main()
