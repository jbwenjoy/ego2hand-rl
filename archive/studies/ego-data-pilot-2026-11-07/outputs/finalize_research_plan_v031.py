"""Finalize the reviewed new Markdown plan; preserve all preceding versions."""
from pathlib import Path
import hashlib,json,re
M=Path(r'D:\workspace\supermimic-chord-study'); draft=M/'docs/research-plan-v0.31.DRAFT.md'; out=M/'docs/research-plan-v0.31.md'
raw=draft.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='bdfab4d68b9ee41cf8a680a68ef2f2419e7c9d81c1238f52e36be95d965a0c07'
text=raw.decode('utf8')
text=text.replace('# 研究方案 v0.31 草稿：','# 研究方案 v0.31：',1)
start='2026-11-07，承接 [v0.30](research-plan-v0.30.md) 和完整 Goal。'
end='\n\n## 本轮改变了什么'
begin=text.index(start); finish=text.index(end)
text=text[:begin]+start+'新世界模型已实际在首个健康checkpoint250后退出，保存内容检查14/14、日志与读回证据检查9/9均通过。首健康点及ETA已具备，完成本交接后按用户既定条件请求暂停Goal。完整目标仍未完成；5000步、Actor和真实ego任务收益继续待验证。'+text[finish:]
text=text.replace('根代理另核保存payload后收口里程碑。','另有保存payload核验14/14及保存文件核验9/9，均无额外NN或原生物理调用。')
text=text.replace('根代理完成保存核验、ETA与交接后暂停Goal；不以草稿声明整个研究或ego重定向完成。','保存核验、ETA与本交接完成后暂停Goal；整个研究及ego重定向仍未完成。')
text+='\n\n本轮完整证据索引与冻结训练记录见[里程碑v12](D:/workspace/ego-data-pilot-2026-11-07/outputs/goal_extension_tripod_momentum_milestone_v12.json)。训练worker共1次own checkpoint读回，保存payload审另增1次CPU读取；合计2次own读取，不增加模型构造、forward、训练或原生仿真。\n'
assert '{{' not in text and '草稿' not in text
links=re.findall(r'\]\(([^)]+)\)',text)
for link in links:
    if link.startswith('D:/'):
        if link.endswith('milestone_v12.json'): continue
        assert Path(link).exists(),link
    elif not link.startswith(('http://','https://')): assert (out.parent/link).exists(),link
assert not out.exists()
with out.open('x',encoding='utf8',newline='\n') as f:f.write(text)
assert hashlib.sha256((M/'docs/research-plan-v0.30.md').read_bytes()).hexdigest()=='cf92e33ef9e00b798cb5d903951962b0f6dd05394325e361f84328a53355bde1'
print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),links_checked=len(links))))
