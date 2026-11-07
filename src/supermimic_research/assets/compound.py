"""Explicit-inertia compound asset XML, probe fixture and isolated drop scene."""
from pathlib import Path
import os,xml.etree.ElementTree as ET
import numpy as np


def vec(value):return ' '.join(format(float(x),'.17g') for x in value)


def scene_xml(mesh_report,decomposition,out,*,drop=False,probe_radius=None,contact_time=.02):
    if not .001<=contact_time<=.05:raise ValueError('Explicit resolvable contact time prior')
    out=Path(out);root=ET.Element('mujoco',model='hot3d_mug9_compound')
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    ET.SubElement(root,'option',timestep='.0003125',integrator='implicitfast',solver='Newton',iterations='100',ls_iterations='20',tolerance='1e-6',ls_tolerance='.01',
        cone='pyramidal',gravity='0 0 0' if probe_radius is not None else '0 0 -9.81')
    asset=ET.SubElement(root,'asset')
    def rel(path):return Path(os.path.relpath(path,out)).as_posix()
    ET.SubElement(asset,'mesh',name='mug_visual',file=rel(mesh_report['visual']['path']))
    for record in decomposition['components']:ET.SubElement(asset,'mesh',name=f"mug_part{record['index']:03d}",file=rel(record['obj']['path']))
    world=ET.SubElement(root,'worldbody')
    if drop:ET.SubElement(world,'geom',name='asset_table',type='box',size='.25 .25 .025',pos='0 0 -.025',friction='.6 .005 .0001',condim='3')
    bounds=np.array(mesh_report['bake']['bounds_model_m'])
    body=ET.SubElement(world,'body',name='hot3d_mug9',pos=vec([0,0,.1-bounds[0,1]]) if drop else '0 0 0',quat=vec([np.sqrt(.5),np.sqrt(.5),0,0]) if drop else '1 0 0 0')
    if probe_radius is None:ET.SubElement(body,'freejoint',name='mug9_free')
    prior=mesh_report['physical_prior'];I=np.array(prior['inertia_model_kg_m2'])
    ET.SubElement(body,'inertial',pos=vec(prior['COM_model_m']),mass=str(prior['mass_kg']),fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
    ET.SubElement(body,'geom',name='mug9_visual',type='mesh',mesh='mug_visual',contype='0',conaffinity='0',density='0',group='2',rgba='.88 .86 .82 1')
    for record in decomposition['components']:
        name=f"mug9_collision_{record['index']:03d}"
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=f"mug_part{record['index']:03d}",density='0',group='3',friction='.7 .005 .0001',condim='3',solref=vec([contact_time,1]),solimp='.9 .95 .001 .5 2',rgba='.2 .5 .8 .25')
    if drop:
        contact=ET.SubElement(root,'contact')
        for record in decomposition['components']:ET.SubElement(contact,'pair',geom1=f"mug9_collision_{record['index']:03d}",geom2='asset_table',condim='3',friction='.6 .6 .005 .0001 .0001',solref=vec([contact_time,1]),solimp='.9 .95 .001 .5 2')
    if probe_radius is not None:
        probe=ET.SubElement(world,'body',name='geometry_probe',pos='0 .15 0');ET.SubElement(probe,'freejoint')
        ET.SubElement(probe,'inertial',pos='0 0 0',mass='.001',diaginertia=vec(np.full(3,.4*.001*probe_radius**2)))
        ET.SubElement(probe,'geom',name='probe',type='sphere',size=str(probe_radius),density='0',condim='3')
    ET.indent(root,space='  ')
    return ET.tostring(root,encoding='unicode')+'\n'
