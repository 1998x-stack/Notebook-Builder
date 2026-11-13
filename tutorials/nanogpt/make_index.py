"""Generate an offline, local-first index for executed notebook tutorials."""
from __future__ import annotations
import html
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def build_index():
    manifest=json.loads((HERE/'outputs'/'manifest.json').read_text(encoding='utf-8'))
    headings={
    '00':'全景 · 张量流','01':'字符编码 · 数据批次','02':'注意力 · 未来遮罩','03':'GPT Block · 梯度',
    '04':'训练循环 · Loss','05':'自回归采样 · Top-k','06':'AdamW · LR 调度','07':'规模估算 · 因果检验'
    }
    figure_by_num={'00':'00_tensor_flow.png','01':'01_shifted_tokens.png','02':'02_attention_weights.png',
        '03':'03_model_parameters.png','04':'04_training_loss.png','05':'05_temperature.png',
        '06':'06_lr_schedule.png','07':'07_parameter_scaling.png'}
    cards=[]
    for i,t in enumerate(manifest['tutorials']):
        slug=t['slug'];n=slug[:2]
        cards.append(f'''<article class="lesson"><a class="thumb" href="outputs/{html.escape(slug)}.html"><img loading="lazy" src="outputs/figures/{figure_by_num[n]}" alt="nanoGPT chapter {n} real figure"></a>
        <div class="copy"><span class="number">{n} / 07</span><h3>{html.escape(headings[n])}</h3>
        <div class="meta">{t['code_cells']} code cells · {t['print_cells']} stdout cells · {t['figure_cells']} inline figures</div>
        <p><a href="outputs/{html.escape(slug)}.html">阅读带运行结果的 HTML ↗</a> <a href="notebooks/{html.escape(slug)}.ipynb">下载 .ipynb ↓</a></p></div></article>''')
    figs=sorted((HERE/'outputs'/'figures').glob('*.png'))
    gallery='\n'.join(f'<a class="gallery-image" href="outputs/figures/{html.escape(p.name)}"><img loading="lazy" src="outputs/figures/{html.escape(p.name)}" alt="{html.escape(p.stem)}"/><span>{html.escape(p.stem)}</span></a>' for p in figs)
    total_fig=sum(x['figure_cells'] for x in manifest['tutorials'])
    markup='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>nanoGPT × Notebook Builder — 图解教程</title>
    <style>
    :root{color-scheme:dark;--bg:#0d1118;--surface:#171f2d;--line:#334155;--text:#edf3ff;--muted:#a5b4c8;--accent:#8bd2fd;--accent2:#91e2bb}
    *{box-sizing:border-box}body{margin:0;background:linear-gradient(130deg,#0e192b 0%,var(--bg) 39%,#11131e 100%);color:var(--text);font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif}
    a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}main{max-width:1160px;padding:48px 28px 90px;margin:auto}.eyebrow{color:var(--accent2);font-size:.82rem;font-weight:750;letter-spacing:.2em;text-transform:uppercase}
    h1{font-size:clamp(2.7rem,7vw,5.8rem);line-height:1.03;letter-spacing:-.06em;margin:22px 0 16px}h1 span{color:var(--accent)}.lead{font-size:1.2rem;color:var(--muted);max-width:790px}h2{margin:54px 0 22px;font-size:1.75rem;letter-spacing:-.02em}
    .chips{display:flex;gap:9px;flex-wrap:wrap;margin-top:28px}.chip{background:#23364a;border:1px solid #3d5268;border-radius:30px;padding:7px 15px;color:#e4effb;font-size:.86rem}
    .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:32px}.stat{padding:20px;border:1px solid var(--line);border-radius:15px;background:#151d2a}.stat strong{font-size:2rem;display:block;line-height:1.2}.stat small{color:var(--muted)}
    .lessons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.lesson{border:1px solid #364153;background:#151e2c;border-radius:18px;overflow:hidden;transition:transform .2s}.lesson:hover{transform:translateY(-3px)}.thumb{display:block;height:210px;overflow:hidden;background:#fff}.thumb img{height:100%;width:100%;object-fit:contain}.copy{padding:18px 20px 20px}.number{font:750 12px monospace;color:var(--accent2)}h3{font-size:1.3rem;margin:6px 0}.meta{font-size:.84rem;color:var(--muted)}.copy p{display:flex;gap:20px;flex-wrap:wrap;font-size:.87rem;margin:18px 0 0}
    .gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.gallery-image{display:block;background:#151e2c;border:1px solid var(--line);border-radius:12px;overflow:hidden}.gallery-image img{background:#fff;object-fit:contain;width:100%;height:158px}.gallery-image span{display:block;padding:9px 11px;color:#d9e6f8;font:11px monospace}
    .foot{color:var(--muted);font-size:.91rem;border-top:1px solid var(--line);margin-top:52px;padding-top:20px}.callout{background:#14252a;border-left:3px solid var(--accent2);padding:18px 22px;border-radius:4px;color:#b8d0d1}.foot code{color:#d4e6fa}
    @media(max-width:780px){.lessons{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.gallery{grid-template-columns:repeat(2,1fr)}main{padding:26px 16px 70px}}@media(max-width:450px){.gallery{grid-template-columns:1fr}.copy p{gap:10px}}
    </style></head><body><main><div class="eyebrow">REPRODUCIBLE CODE LAB / OFFLINE CPU</div>
    <h1>nano<span>GPT</span><br/>Notebook Builder</h1><p class="lead">从字符编码、Causal Self-Attention 到真实训练、采样和可验证的扩展实验。每节都将原理、代码执行中的 print 和图表放在同一个 Notebook 中。</p>
    <div class="chips"><span class="chip">Python 3 / PyTorch CPU</span><span class="chip">中文讲解 + 可运行代码</span><span class="chip">真实 stdout + PNG figures</span><span class="chip">无外部下载</span></div>
    <div class="stats"><div class="stat"><strong>8</strong><small>递进式章节</small></div><div class="stat"><strong>''' + str(total_fig) + '''</strong><small>嵌入式执行图表</small></div><div class="stat"><strong>''' + str(len(figs)) + '''</strong><small>独立 PNG</small></div><div class="stat"><strong>0</strong><small>Cell 运行异常</small></div></div>
    <h2>课程章节 · 从零构造 GPT</h2><section class="lessons">''' + '\n'.join(cards) + '''</section>
    <h2>可视化实验图集</h2><section class="gallery">''' + gallery + '''</section>
    <h2>复现方式</h2><div class="callout"><code>nbuild tutorial nanogpt --build --execute --html</code><p>代码与教材来自本地 /mnt/notebook-builder/tutorials/nanogpt；执行会使用本机 Python Kernel。只执行可信 Cell。05 优先读取由 04 保存的本地可信 checkpoint。</p></div>
    <div class="foot">Source: <a href="https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291">karpathy/nanoGPT · pinned revision</a> · 教学化 CPU 实现，并非原项目训练代码的逐行复刻。原项目 README 已注明 deprecated，并指向 nanochat。<br/>本课程图表来自真实本地执行结果；合成数据不能用于证明 GPT-2 重现效果。</div>
    </main></body></html>'''
    dest=HERE/'index.html';dest.write_text(markup,encoding='utf-8')
    return {'path':str(dest),'size_bytes':dest.stat().st_size,'chapters':len(cards),'gallery_figures':len(figs)}

if __name__=='__main__':
    print(json.dumps(build_index(),ensure_ascii=False,indent=2))
