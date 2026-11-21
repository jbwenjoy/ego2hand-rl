"""Export the existing calibrated Wuji Hand 2 collision model for a Newton floating wrist."""
import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from supermimic_research.assets.hand2_scene import fixed_hand
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def export(out):
    out.mkdir(parents=True, exist_ok=False)
    spec, native, _, _ = fixed_hand()
    root=ET.fromstring(spec.to_xml())
    compiler=root.find("compiler")
    compiler.set("meshdir","meshes")
    meshdir=out/"meshes";meshdir.mkdir()
    palm=root.find("worldbody/body")
    assert palm is not None and palm.get("name")=="right_palm_link"
    palm.attrib.pop("pos",None);palm.attrib.pop("quat",None)
    arm=root.find("worldbody");arm.remove(palm);current=arm
    dummy_mass=.0001
    for axis,vector in zip("xyz",("1 0 0","0 1 0","0 0 1")):
        current=ET.SubElement(current,"body",name=f"right_wrist_p{axis}_body")
        ET.SubElement(current,"inertial",pos="0 0 0",mass=str(dummy_mass),diaginertia="1e-7 1e-7 1e-7")
        ET.SubElement(current,"joint",name=f"right_wrist_{axis}",type="slide",axis=vector,limited="false",armature="0")
    current.append(palm)
    palm.insert(0,ET.Element("joint",name="right_wrist_rotvec",type="ball",limited="false",armature="0"))
    # All collision-enabled shapes come directly from the native model. Drop visual-only
    # meshes, inertial and FK information remain explicit in the retained body tree.
    native_geoms={native.geom(i).name:i for i in range(native.ngeom) if native.geom_contype[i] or native.geom_conaffinity[i]}
    used=set()
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("name") not in native_geoms:
                body.remove(geom)
                continue
            if geom.get("mesh"): used.add(geom.get("mesh"))
            geom.attrib.pop("material",None)
        for site in body.findall("site"): body.remove(site)
    assets=root.find("asset")
    bytes_by_name={Path(k).name:v for k,v in spec.assets.items()}
    for member in list(assets):
        if member.tag!="mesh" or member.get("name") not in used:
            assets.remove(member)
            continue
        n=Path(member.get("file")).name
        member.set("file",n);(meshdir/n).write_bytes(bytes_by_name[n])
    root.remove(root.find("actuator"))   # Newton maps the calibrated torque/gains to its control vectors.
    joints={}
    for i in range(native.nu):
        j=int(native.actuator_trnid[i,0]); dof=int(native.jnt_dofadr[j]);name=native.joint(j).name
        assert np.isclose(native.actuator_biasprm[i,1],-native.actuator_gainprm[i,0])
        joints[name]=dict(kp=native.actuator_gainprm[i,0],kd=-native.actuator_biasprm[i,2],effort=native.actuator_forcerange[i,1],
                          range=native.jnt_range[j].tolist(),ctrl_range=native.actuator_ctrlrange[i].tolist(),
                          friction=native.dof_frictionloss[dof],armature=native.dof_armature[dof],damping=native.dof_damping[dof])
    tips=[]
    for finger in range(1,6):
        site=native.site(f"right_finger{finger}_tip").id
        tips.append(dict(name=native.site(site).name,parent=native.body(native.site_bodyid[site]).name,
                         pos=native.site_pos[site].tolist(),quat_xyzw=native.site_quat[site,[1,2,3,0]].tolist()))
    xml=out/"wuji_right.xml";ET.indent(root)
    ET.ElementTree(root).write(xml,encoding="utf-8",xml_declaration=True)
    check=mujoco.MjModel.from_xml_path(str(xml))
    assert check.ngeom==len(native_geoms)==21 and check.nq==27 and check.nv==26
    mass=np.array([check.body(native.body(i).name).mass[0] for i in range(1,native.nbody)])
    np.testing.assert_allclose(mass,native.body_mass[1:],atol=5e-6)
    geom_meta={}
    for name,g in native_geoms.items():
        b=check.geom(name).id
        assert native.geom_type[g]==check.geom_type[b]
        np.testing.assert_allclose(native.geom_size[g],check.geom_size[b],atol=2e-6)
        geom_meta[name]={"body":native.body(native.geom_bodyid[g]).name,"type":int(native.geom_type[g]),
                         "friction":native.geom_friction[g].tolist()}
    write_json(out/"adapter.json",dict(robot="Wuji Hand2 right, upstream calibrated pen-task factory",xml="wuji_right.xml",
               model_pin=file_pin(xml),mesh_pins=[file_pin(p) for p in sorted(meshdir.iterdir())],joints=joints,
               geometry=geom_meta,tips=tips,robot_mass_kg=float(mass.sum()),dummy_link_mass_kg=dummy_mass,
               native_body_masses={native.body(i).name:float(native.body_mass[i]) for i in range(1,native.nbody)},
               native_exclusions=[dict(c.attrib) for c in root.findall("contact/exclude")],source=file_pin(__file__)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("out",type=Path)
    export(p.parse_args().out.resolve())
