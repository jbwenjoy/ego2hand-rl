import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dex_supermimic.assets.pen import get_tracking_pen_spec, get_pen_cfg


class PenAssetPathTests(unittest.TestCase):
    def test_explicit_xml_works_with_missing_bundled_resources_and_preserves_inertia(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'pen.xml'
            p.write_text('<mujoco><worldbody><body name="cube"><freejoint/>'
                         '<geom type="capsule" size=".01 .1" density="500"/>'
                         '</body></worldbody></mujoco>', encoding='utf8')
            source = mujoco.MjModel.from_xml_path(str(p))
            with patch('dex_supermimic.assets.pen.PEN_XML', Path(directory) / 'missing.xml'):
                model = get_tracking_pen_spec(mass=.03, xml_path=p).compile()
                configured = get_pen_cfg(.03, xml_path=p).spec_fn().compile()
            b, original = model.body('cube'), source.body('cube')
            np.testing.assert_allclose(b.mass, [.03], rtol=0, atol=1e-15)
            np.testing.assert_allclose(b.inertia, original.inertia * .03 / original.mass[0], rtol=1e-14)
            for field in ['body_ipos', 'body_iquat', 'geom_pos', 'geom_quat', 'geom_size', 'geom_friction', 'jnt_type']:
                np.testing.assert_array_equal(getattr(model, field), getattr(source, field))
            for field in ['body_mass', 'body_inertia', 'geom_size']:
                np.testing.assert_array_equal(getattr(model, field), getattr(configured, field))


if __name__ == '__main__':
    unittest.main()
