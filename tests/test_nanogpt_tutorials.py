"""nanoGPT tutorial generator + executed-notebook contract tests (CPU, offline)."""
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TUT = ROOT / 'tutorials' / 'nanogpt'
sys.path.insert(0, str(TUT))
from nanogpt_edu import GPT, GPTConfig, char_dataset, get_batch, lr_schedule, parameter_groups
from build_tutorials import chapters, run_series, UPSTREAM_SHA


def test_eight_chapters_with_goals_and_bounded_specs():
    cs=chapters()
    assert len(cs)==8
    assert [c['slug'][:2] for c in cs]==[f'{i:02d}' for i in range(8)]
    assert all(c['goals'] and sum(x['cell_type']=='code' for x in c['cells'])>=2 for c in cs)
    assert len(UPSTREAM_SHA)==40
    assert run_series(list_only=True)['count']==8


def test_model_forward_tied_weights_and_backward():
    cfg=GPTConfig(vocab_size=20,block_size=8,n_layer=2,n_head=2,n_embd=16)
    model=GPT(cfg)
    assert model.transformer.wte.weight.data_ptr()==model.lm_head.weight.data_ptr()
    x=torch.randint(0,20,(3,8))
    logits,loss=model(x,x)
    assert logits.shape==(3,8,20) and torch.isfinite(loss)
    loss.backward()
    assert model.transformer.wte.weight.grad is not None
    model.eval()
    logits_last,_=model(x)
    assert logits_last.shape==(3,1,20)
    generated=model.generate(x[:1,:2],max_new_tokens=5,top_k=4)
    assert generated.shape==(1,7)
    assert sum(sum(p.numel() for p in group['params']) for group in parameter_groups(model))==sum(p.numel() for p in model.parameters())


def test_causality_of_attention():
    torch.manual_seed(66)
    model=GPT(GPTConfig(vocab_size=20,block_size=12,n_layer=1,n_head=2,n_embd=16)).eval()
    a=torch.randint(0,20,(1,10)); b=a.clone(); b[:,5:]=(b[:,5:]+3)%20
    with torch.no_grad():
        x,_=model(a,a)
        y,_=model(b,b)
    assert torch.allclose(x[:,:5],y[:,:5],atol=1e-6)


def test_token_target_shift_and_scheduler():
    ids,stoi,itos=char_dataset('abracadabra '*10)
    x,y=get_batch(ids,5,4)
    assert torch.equal(x[:,1:],y[:,:-1])
    assert lr_schedule(10,50,10,.01,.001)==pytest.approx(.01)
    assert lr_schedule(50,50,10,.01,.001)==pytest.approx(.001)
    with pytest.raises(ValueError): GPTConfig(n_embd=15,n_head=2)


def test_notebooks_valid_and_real_print_and_figures(tmp_path):
    for c in chapters():
        path=TUT/'notebooks'/f"{c['slug']}.ipynb"
        assert path.exists(), f'Missing built tutorial {path}'
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        code=[cell for cell in nb.cells if cell.cell_type=='code']
        assert len(code)>=3
        assert all(c.execution_count is not None for c in code),path
        assert sum(any(o.output_type=='stream' and o.get('name')=='stdout' for o in cell.outputs) for cell in code)>=2
        assert sum(any(o.output_type in ('display_data','execute_result') and 'image/png' in o.get('data',{}) for o in cell.outputs) for cell in code)>=1
        assert not any(o.output_type=='error' for cell in code for o in cell.outputs)
    # HTML is a generated artifact, not a mandatory committed build product.
    # Verify one actual export in a fresh checkout instead of depending on stale output files.
    from notebook_builder import export_html
    html=tmp_path/'smoke-export.html'
    export_html(TUT/'notebooks'/'00_map_environment.ipynb',html)
    assert html.exists() and html.stat().st_size>10000


def test_cli_list_without_execution():
    proc=subprocess.run([sys.executable,str(ROOT/'nbuild.py'),'tutorial','nanogpt','--list'],
                        capture_output=True,text=True,timeout=20)
    assert proc.returncode==0,proc.stderr
    assert '00_map_environment' in proc.stdout and '07_scaling_causality_checks' in proc.stdout
