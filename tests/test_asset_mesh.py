"""Frame/triangle-mass/void primitives against analytic geometric cases."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.assets.mesh import bake_scene,exact_weld,surface_mass_prior,union_membership,winding_numbers
from supermimic_research.assets.validation import plane_lower_bound,corridor_clearance
from supermimic_research.assets.compound import scene_xml


class AssetMeshTests(unittest.TestCase):
    def test_scene_transform_applied_once_and_bounds_are_authoritative(self):
        import trimesh
        scene=trimesh.Scene();T=np.eye(4);T[:3,:3]*=.001;T[:3,3]=[.1,.2,.3]
        scene.add_geometry(trimesh.creation.box(extents=[1000.,1000.,1000.]),transform=T)
        expected=np.array([[-.4,-.3,-.2],[.6,.7,.8]])
        v,f,meta=bake_scene(scene,expected);self.assertLess(meta['bounds_error_m'],1e-14)
        with self.assertRaises(ValueError):bake_scene(scene,expected*.001)
        w,g=exact_weld(np.r_[v,v],np.r_[f,f+len(v)])
        np.testing.assert_array_equal(w[g],np.r_[v,v][np.r_[f,f+len(v)]])

    def test_triangle_lamina_inertia_and_translation_invariance(self):
        v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]);f=np.array([[0,1,2]])
        prior=surface_mass_prior(v,f,12.)
        np.testing.assert_allclose(prior['COM_model_m'],[1/3,1/3,0],atol=1e-14)
        np.testing.assert_allclose(prior['inertia_model_kg_m2'],[[2/3,1/3,0],[1/3,2/3,0],[0,0,4/3]],atol=1e-14)
        moved=surface_mass_prior(v+[.2,-.3,.5],f,12.)
        np.testing.assert_allclose(moved['inertia_model_kg_m2'],prior['inertia_model_kg_m2'],atol=1e-14)

    def test_union_does_not_fill_gap_like_whole_hull(self):
        from scipy.spatial import ConvexHull
        from itertools import product
        cube=np.array(list(product((-.2,.2),repeat=3)));left=cube+[-1.,0.,0.];right=cube+[1.,0.,0.]
        points=np.array([[0.,0.,0.],[-1.,0.,0.]])
        np.testing.assert_array_equal(union_membership(points,[ConvexHull(left).equations,ConvexHull(right).equations]),[False,True])
        self.assertTrue(union_membership(points[:1],[ConvexHull(np.r_[left,right]).equations])[0])

    def test_winding_is_descriptor_with_known_closed_cube(self):
        import trimesh
        cube=trimesh.creation.box()
        w=winding_numbers([[0.,0.,0.],[3.,0.,0.]],cube.vertices,cube.faces)
        self.assertAlmostEqual(abs(w[0]),1.);self.assertLess(abs(w[1]),1e-12)

    def test_plane_bound_is_conservative_at_cube_corners(self):
        import trimesh
        from scipy.spatial import ConvexHull
        box=trimesh.creation.box(extents=[2.,2.,2.]);bound=plane_lower_bound([[2.,2.,2.]], [ConvexHull(box.vertices).equations])
        self.assertAlmostEqual(bound[0],1.);self.assertLess(bound[0],np.sqrt(3))

    def test_corridor_needs_half_step_margin_not_only_empty_samples(self):
        points=np.array([[0.,0.,0.],[0.,0.,1.]])
        self.assertFalse(corridor_clearance([.4,.4],points,.1)['clear'])
        self.assertTrue(corridor_clearance([.7,.7],points,.1)['clear'])

    def test_native_compound_void_is_not_filled_by_visual_hull(self):
        import trimesh,mujoco
        visual=trimesh.creation.box(extents=[.05,.03,.03]);parts=[]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);visual.export(folder/'visual.obj')
            for i,x in enumerate((-.015,.015)):
                piece=trimesh.creation.box(extents=[.01,.01,.01]);piece.apply_translation([x,0,0]);piece.export(folder/f'part{i}.obj')
                parts.append(dict(index=i,obj=dict(path=str(folder/f'part{i}.obj'))))
            meta=dict(visual=dict(path=str(folder/'visual.obj')),physical_prior=surface_mass_prior(visual.vertices,visual.faces,.25),bake=dict(bounds_model_m=visual.bounds.tolist()))
            xml=folder/'fixture.xml';xml.write_text(scene_xml(meta,dict(components=parts),folder,probe_radius=.003,contact_time=.002))
            m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=[0,0,0,1,0,0,0];mujoco.mj_forward(m,d)
            self.assertEqual(d.ncon,0)
            # Surface overlap is the positive control. Fully centered interior
            # containment can be degenerate in native convex dispatch; zero
            # native contacts alone is therefore not a void certificate.
            d.qpos[0]=-.0205;mujoco.mj_forward(m,d);self.assertGreater(d.ncon,0)
            self.assertEqual(m.geom_contype[m.geom('mug9_visual').id],0)


if __name__=='__main__':unittest.main()
