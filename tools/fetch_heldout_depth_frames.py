"""Download only predeclared source depth views and calibration from the official ZIPs."""
import argparse
import json
from pathlib import Path
import struct
import zlib

import requests
import yaml


def fetch(source_meta, stages, dataset, index_dir, urls, output):
    window=json.loads(source_meta.read_bytes())
    if window["role"]!="evaluation after recipe freeze": raise ValueError("Source is not released under a frozen recipe")
    cfg=json.loads(stages.read_bytes()); first,last=window["frames"]; seq=window["sequence"]
    subject=seq.split("/")[0]
    wanted={subject:[f"{seq}/meta.yaml"]+
            [f"{seq}/{camera}/depth_{i:06}.png" for camera in cfg["depth_camera_serials"]
             for i in (first,(first+last)//2,last)]}
    archive_urls=yaml.safe_load(urls.read_bytes())
    rows=[]
    with requests.Session() as session:
        def obtain(key,names):
            entries={r[0]:r for r in json.loads((index_dir/f"{key}_index.json").read_bytes()) if r[0] in names}
            for name in names:
                dest=dataset/name
                if name not in entries:
                    rows.append({"member":name,"available":False,"reason":"absent in source ZIP"});continue
                _,size,compressed,offset,crc,method=entries[name]
                def valid(content):return len(content)==size and zlib.crc32(content)==crc
                if dest.exists():
                    if not valid(dest.read_bytes()):raise ValueError(f"Invalid cached source {name}")
                else:
                    with session.get(archive_urls[key],headers={"Range":f"bytes={offset}-{offset+compressed+4095}",
                                     "Accept-Encoding":"identity"},stream=True,timeout=(15,60)) as resp:
                        if resp.status_code!=206 or not resp.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                            raise IOError(f"Source server did not honor ZIP range for {name}")
                        block=resp.content
                    length,extra=struct.unpack_from("<HH",block,26)
                    if block[:4]!=b"PK\x03\x04" or block[30:30+length].decode()!=name:
                        raise ValueError("Source ZIP member mismatch")
                    i=30+length+extra;data=block[i:i+compressed]
                    if method==8:data=zlib.decompress(data,-15)
                    if not valid(data):raise IOError("Downloaded source member failed byte/CRC check")
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    part=dest.with_suffix(dest.suffix+".partial");part.write_bytes(data);part.replace(dest)
                rows.append(dict(member=name,available=True,bytes=size,crc32=crc))
        obtain(subject,wanted[subject])
        meta=yaml.safe_load((dataset/seq/"meta.yaml").read_bytes())
        obtain("calibration",[f"calibration/extrinsics/{meta['extrinsics']}"]+
               [f"calibration/intrinsics/{s}.yaml" for s in cfg["depth_camera_serials"]])
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(dict(sequence=seq,members=rows),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("source","stages","dataset","index_dir","urls","report"):
        parser.add_argument(key,type=Path)
    a=parser.parse_args()
    fetch(a.source,a.stages,a.dataset,a.index_dir,a.urls,a.report)
