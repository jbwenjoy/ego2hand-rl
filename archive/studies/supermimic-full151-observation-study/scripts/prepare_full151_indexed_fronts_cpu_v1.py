"""Create new metadata fronts; retain the executed v2 sources verbatim."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)
def save(name,text):
    ast.parse(text)
    p=ROOT/'scripts'/name
    with p.open('x',encoding='utf-8',newline='') as f:f.write(text)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    worker=(ROOT/'scripts/execute_full151_learning_v2.py').read_text(encoding='utf-8')
    worker=replace_once(worker,"return any(Path(v['path']).resolve()==Path(value['path']).resolve() and v['sha256']==value['sha256'] for v in p['source_pins'].values())", "# The first full source validation still hashes every pin. This index only\n    # avoids repeated Windows filesystem resolution in membership queries.\n    key=id(p)\n    if key not in _ARCHIVE_INDEX:\n        _ARCHIVE_INDEX[key]=(p,{(str(Path(v['path']).resolve()).casefold(),v['sha256']) for v in p['source_pins'].values()})\n    return (str(Path(value['path']).resolve()).casefold(),value['sha256']) in _ARCHIVE_INDEX[key][1]")
    worker=replace_once(worker,"METHODS=('A1L','P_ONLY','A2')","METHODS=('A1L','P_ONLY','A2')\n_ARCHIVE_INDEX={}")
    worker=worker.replace("ROOT/'scripts/run_full151_learning_v2.py'","ROOT/'scripts/run_full151_learning_v3.py'")
    supervisor=(ROOT/'scripts/run_full151_learning_v2.py').read_text(encoding='utf-8')
    supervisor=replace_once(supervisor,"ENTRY=ROOT/'scripts/execute_full151_learning_v2.py'","ENTRY=ROOT/'scripts/execute_full151_learning_v3.py'")
    builder=(ROOT/'scripts/build_full151_three_arm_plan_cpu_v2.py').read_text(encoding='utf-8')
    builder=builder.replace('full151-v2-three-arm-{tag}-v1.json','full151-v3-three-arm-{tag}-v1.json').replace('full151-v2-{tag}-{m}-seed0-v1.json','full151-v3-{tag}-{m}-seed0-v1.json').replace('full151_v2_{tag}_{method}_seed0_v1','full151_v3_{tag}_{method}_seed0_v1')
    builder=builder.replace("ROOT/'scripts/execute_full151_learning_v2.py'","ROOT/'scripts/execute_full151_learning_v3.py'").replace("ROOT/'scripts/run_full151_learning_v2.py'","ROOT/'scripts/run_full151_learning_v3.py'")
    pins=[save('execute_full151_learning_v3.py',worker),save('run_full151_learning_v3.py',supervisor),save('build_full151_three_arm_plan_cpu_v3.py',builder)]
    print(json.dumps({'created':pins,'AST':'PASS','run':False,'algorithm_changes':False}))
if __name__=='__main__':main()
