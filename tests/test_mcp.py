"""MCP protocol smoke tests without requiring the optional official SDK."""
import json
import os
from pathlib import Path
import subprocess
import sys

import notebook_builder as b
import mcp_tools

SERVER = Path(__file__).resolve().parents[1] / 'mcp_server.py'


def req(method, params=None, id=1):
    item={'jsonrpc':'2.0','method':method,'params':params or {}}
    if id is not None:
        item['id']=id
    return item


def rpc(tmp_path, queries, *, readonly='1', execute='0'):
    env={**os.environ, 'NB_ROOT':str(tmp_path), 'NB_MCP_READONLY':readonly,
         'NB_MCP_ALLOW_EXEC':execute, 'NB_MCP_TRANSPORT':'native'}
    result = subprocess.run([sys.executable,str(SERVER)],
                            input=''.join(json.dumps(x)+'\n' for x in queries),
                            text=True,capture_output=True,env=env,timeout=80)
    assert result.returncode==0, result.stderr
    # Modern ipykernel may legitimately warn on stderr about local TCP/ZMQ
    # transport; stderr is diagnostic, not the MCP JSON-RPC stdout channel.
    # Accept only this specific warning; fail on unexpected stderr output.
    err_lines=[line for line in result.stderr.splitlines() if line.strip()]
    assert all(line.startswith('[IPKernelApp] WARNING | Kernel is running over TCP without encryption.')
               for line in err_lines), result.stderr
    return [json.loads(s) for s in result.stdout.strip().splitlines()]


def unpack(response):
    return json.loads(response['result']['content'][0]['text'])


def test_protocol_handshake_and_readonly(tmp_path):
    data=rpc(tmp_path,[req('initialize',{'protocolVersion':'2025-06-18'},1),
                       req('notifications/initialized',id=None),
                       req('tools/list',id=2),req('resources/list',id=3),
                       req('resources/templates/list',id=4),
                       req('tools/call',{'name':'notebook_create','arguments':{'path':'no.ipynb'}},5),
                       req('tools/call',{'name':'nebula_health','arguments':{}},6)])
    assert len(data)==6
    assert data[0]['result']['protocolVersion']=='2025-06-18'
    assert len(data[1]['result']['tools'])==len(mcp_tools.REGISTRY)>=40
    assert data[4]['result']['isError'] is True
    assert unpack(data[5])['read_only'] is True


def test_mcp_authoring_protocol_and_security(tmp_path):
    result=rpc(tmp_path,[
        req('tools/call',{'name':'notebook_create','arguments':{'path':'p.ipynb','title':'Demo'}},1),
        req('tools/call',{'name':'cell_add','arguments':{'path':'p.ipynb','source':'print(6*7)',
                                                  'expected_version':1,'operation_id':'retry-a'}},2),
        req('tools/call',{'name':'cell_add','arguments':{'path':'p.ipynb','source':'print(6*7)',
                                                  'expected_version':1,'operation_id':'retry-a'}},3),
        req('tools/call',{'name':'notebook_get','arguments':{'path':'p.ipynb'}},4),
        req('resources/read',{'uri':'nbuild://notebooks/p.ipynb'},5),
        req('tools/call',{'name':'notebook_create','arguments':{'path':'../outside.ipynb'}},6),
        req('tools/call',{'name':'code_execute','arguments':{'path':'p.ipynb','code':'print(1)'}},7),
    ],readonly='0',execute='0')
    assert len(result)==7
    assert unpack(result[0])['version']==1
    assert unpack(result[1])['version']==2
    assert unpack(result[2])['replayed'] is True
    assert len(unpack(result[3])['cells'])==2
    assert 'p.ipynb' in result[4]['result']['contents'][0]['text']
    assert result[5]['result']['isError'] is True
    assert result[6]['result']['isError'] is True
    assert not (tmp_path.parent/'outside.ipynb').exists()


def test_mcp_live_kernel(tmp_path):
    b.create(tmp_path/'run.ipynb')
    result=rpc(tmp_path,[
        req('tools/call',{'name':'runtime_start','arguments':{'path':'run.ipynb'}},1),
        req('tools/call',{'name':'code_execute','arguments':{'path':'run.ipynb','code':'z=123', 'use_live_kernel':True}},2),
        req('tools/call',{'name':'code_execute','arguments':{'path':'run.ipynb','code':'print(z)', 'use_live_kernel':True}},3),
        req('tools/call',{'name':'kernel_variables','arguments':{'path':'run.ipynb','names':['z']}},4),
        req('tools/call',{'name':'runtime_stop','arguments':{'path':'run.ipynb'}},5)],
               readonly='0',execute='1')
    assert len(result)==5 and all(x['result']['isError'] is False for x in result)
    assert unpack(result[0])['status']=='ready'
    assert unpack(result[2])['outputs'][0]['text'].strip() == '123'
    assert unpack(result[3])['variables'][0]['name'] == 'z'
    assert unpack(result[4])['status'] == 'stopped'


def test_tool_input_schema_and_extra_arguments(tmp_path):
    schema=mcp_tools.REGISTRY['cell_add'].schema
    assert {'path','source'}.issubset(set(schema['required']))
    assert schema['additionalProperties'] is False
    r=rpc(tmp_path,[req('tools/call',{'name':'nebula_health','arguments':{'unexpected':1}})])
    assert r[0]['result']['isError'] is True


def test_mcp_job_survives_client_connection(tmp_path):
    p=tmp_path/'job.ipynb'
    b.create(p,cells=[{'source':'print(100+23)'}])
    cell=b.describe(p)['cells'][0]['id']
    reply=rpc(tmp_path,[req('tools/call',{'name':'run_submit',
                'arguments':{'path':'job.ipynb','cell_id':cell,'idempotency_key':'job-mcp-01','timeout':60}})],
                readonly='0',execute='1')
    assert reply[0]['result']['isError'] is False
    run_id=unpack(reply[0])['run_id']
    # A fresh MCP server subprocess should see the same SQLite run after the
    # calling server disconnected.
    report=rpc(tmp_path,[req('tools/call',{'name':'run_wait',
                  'arguments':{'run_id':run_id,'wait_seconds':20}})])
    result=unpack(report[0])
    assert result['status']=='succeeded',result
    assert b.describe(p)['version']==2