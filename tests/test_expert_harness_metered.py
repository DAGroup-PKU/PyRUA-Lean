import json,tempfile,unittest,urllib.request,urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datafarm.expert.session import MeteredCodexBackend,SessionSpec,ParsedSession,make_agent_backend,classify_errors
from datafarm.agent import metered_provider as mp
# Test-only price table (no built-in default exists): USD per million tokens.
PRICING_CFG={'model':'metered-test-model','source':'test fixture','usd_per_million_tokens':{'input':10,'cached_input':1,'cache_write':12.5,'output':50},'long_context':{'above_input_tokens':272000,'input_factor':2,'output_factor':1.5}}
PRICING=mp.load_pricing(PRICING_CFG)

class Response:
 status_code=200
 headers={}
 def json(self):
  return {'id':'fixture','status':'completed','output':[], 'usage':{'input_tokens':100,'output_tokens':10,'input_tokens_details':{'cached_tokens':20}}}
 @property
 def content(self):return json.dumps(self.json()).encode()
 def iter_lines(self,chunk_size):
  yield b'data: {"type":"response.completed","response":{"id":"fixture","usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":20}}}}'
 def close(self):pass
class Remote:
 def get(self,*args,**kwargs):return Response()
 def post(self,*args,**kwargs):
  assert kwargs['json']['stream'] is False
  return Response()
class MeteredTests(unittest.TestCase):
 def test_reservation_covers_all_usage_mixtures_and_context_tiers(self):
  from decimal import Decimal
  # Build exact conservative maxima on both sides of the billing threshold.
  overhead=len(json.dumps({'input':''},ensure_ascii=False).encode())+16384
  for maximum in (200000,272000,272001,400000):
   payload={'input':'x'*(maximum-overhead)}
   reserve=mp.bound(payload,PRICING)
   expected=(Decimal(maximum)*Decimal('0.0000125')+Decimal('0.2048')
             if maximum<=272000 else Decimal(maximum)*Decimal('0.000025')+Decimal('0.3072'))
   self.assertEqual(reserve,expected)
   for n in {0,maximum//2,min(maximum,272000),maximum}:
    for cached,writes in ((0,0),(n,0),(0,n),(n//3,n//3)):
     actual=mp.usd({'input_tokens':n,'output_tokens':mp.MAX_OUTPUT,
                   'input_tokens_details':{'cached_tokens':cached,'cache_write_tokens':writes}},PRICING)
     self.assertGreaterEqual(reserve,actual)
  # Unknown media still reserves maximum context, never the smaller tier.
  self.assertEqual(mp.bound({'input':[{'type':'input_file'}]},PRICING),
                   Decimal(mp.MAX_INPUT)*Decimal('0.000025')+Decimal('0.3072'))
 def test_receipt_retries_only_readonly_transport_failures(self):
  receipt={'request_id':'exact','usage':Response().json()['usage']}
  with patch.object(mp,'_openai_compatible_receipt_once',side_effect=[mp.requests.exceptions.ReadTimeout(),mp.requests.exceptions.ConnectionError(),receipt]) as lookup:
   self.assertEqual(mp.openai_compatible_receipt('https://fixture','secret','exact','metered-test-model'),receipt)
   self.assertEqual(lookup.call_count,3)
  with patch.object(mp,'_openai_compatible_receipt_once',side_effect=mp.requests.exceptions.ReadTimeout()) as lookup:
   with self.assertRaises(mp.requests.exceptions.ReadTimeout):mp.openai_compatible_receipt('https://fixture','secret','exact','metered-test-model')
   self.assertEqual(lookup.call_count,3)
  with patch.object(mp,'_openai_compatible_receipt_once',side_effect=mp.requests.exceptions.HTTPError()) as lookup:
   with self.assertRaises(mp.requests.exceptions.HTTPError):mp.openai_compatible_receipt('https://fixture','secret','exact','metered-test-model')
   self.assertEqual(lookup.call_count,1)
 def test_pricing_is_explicit_configuration(self):
  with self.assertRaisesRegex(ValueError,'not configured'):mp.load_pricing(None)
  with self.assertRaisesRegex(ValueError,'lacks'):mp.load_pricing({'model':'m','usd_per_million_tokens':{'input':1}})
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'not configured'):mp.Meter(Path(d)/'journal',1)
   self.assertFalse((Path(d)/'journal').exists())
   status,body=mp.Meter(Path(d)/'j2',1,pricing=PRICING).request({'model':'other-model'},'must-not-call','k')
   self.assertEqual(status,400)
  b=make_agent_backend({'backend':'codex-metered','model':'metered-test-model','credentials_file':'/protected/x'},{})
  with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(Exception,'agent.pricing'):
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):pass
 def test_buffered_items_and_usage_are_preserved(self):
  obj={'id':'r','status':'completed','usage':{'input_tokens':4,'output_tokens':8},'output':[
   {'type':'function_call','id':'f','call_id':'c','name':'exec_command','arguments':'{"cmd":"true"}'},
   {'type':'custom_tool_call','id':'p','call_id':'d','name':'apply_patch','input':'patch text'},
   {'type':'message','id':'m','role':'assistant','content':[{'type':'output_text','text':'DONE','annotations':[]}]}]}
  events=[json.loads(line[6:]) for line in mp.completed_response_events(obj).splitlines() if line.startswith(b'data: ')]
  self.assertEqual([e['item'] for e in events if e['type']=='response.output_item.done'],obj['output'])
  self.assertEqual(events[-1]['response'],obj)
  with self.assertRaises(ValueError):mp.completed_response_events(dict(obj,status='in_progress'))
 def backend(self):return make_agent_backend({'backend':'codex-metered','model':'metered-test-model','credentials_file':'/protected/not-exposed','pricing':PRICING_CFG}, {})
 def request(self,b):
  r=urllib.request.Request(b.gateway.url+'/responses',data=json.dumps({'model':'metered-test-model','stream':True}).encode(),headers={'Content-Type':'application/json'})
  return urllib.request.urlopen(r).read()
 def creds(self):return patch('datafarm.agent.credentials.KeysFileCredentials',return_value=lambda:SimpleNamespace(endpoint='https://fixture',api_key='test-only-secret'))
 def test_upstream_receipt_overrides_forged_cli_cost_and_config_hides_key(self):
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=Remote()):
   b=self.backend();spec=SessionSpec(1,'brief',d,max_budget_usd=2)
   b.cfg['reasoning_effort']='high'
   with b.session(spec,Path(d)):
    argv=b.command(spec);self.assertNotIn('test-only-secret',' '.join(argv));self.assertNotIn('/protected/not-exposed',' '.join(argv))
    self.assertIn('project_doc_max_bytes=0',argv);self.request(b)
    self.assertIn('model_reasoning_effort="high"',argv)
   parsed=b.settle(ParsedSession(cost_usd=999,input_tokens=999))
   self.assertAlmostEqual(parsed.cost_usd,0.00132);self.assertEqual(parsed.input_tokens,80)
   self.assertEqual(parsed.cache_read_tokens,20);self.assertTrue(b.receipt['complete'])
 def test_denial_makes_no_upstream_request_and_settles_zero(self):
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',side_effect=AssertionError('no network')):
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=.001),Path(d)):
    with self.assertRaises(urllib.error.HTTPError) as cm:self.request(b)
    self.assertEqual(cm.exception.code,402)
   self.assertEqual(b.settle(ParsedSession()).cost_usd,0);self.assertTrue(b.receipt['complete'])
 def test_disconnect_is_unknown_even_if_cli_claims_success(self):
  class Broken(Remote):
   def post(self,*a,**kw):raise mp.requests.exceptions.ReadTimeout('fixture test-only-secret')
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=Broken()):
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    with self.assertRaises(urllib.error.HTTPError):self.request(b)
   self.assertIsNone(b.settle(ParsedSession(cost_usd=0,saw_result=True)).cost_usd)
   self.assertFalse(b.receipt['complete'])
   journal=(Path(d)/'session-001.provider.jsonl').read_text()
   self.assertIn('reserve',journal)
   self.assertNotIn('test-only-secret',journal)
   failure=next(json.loads(line) for line in journal.splitlines() if json.loads(line)['event']=='unknown')
   self.assertEqual(failure['stage'],'awaiting_headers')
   self.assertGreaterEqual(failure['elapsed_s'],0)
   self.assertIn('[redacted]',failure['error_detail'])
 def test_readonly_preflight_failure_has_no_billable_call(self):
  class Offline(Remote):
   def get(self,*a,**kw):raise mp.requests.exceptions.ReadTimeout('fixture')
   def post(self,*a,**kw):raise AssertionError('must not send generation')
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=Offline()):
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    with self.assertRaises(urllib.error.HTTPError):self.request(b)
   self.assertEqual(b.settle(ParsedSession()).cost_usd,0)
   self.assertTrue(b.receipt['complete'])
   journal=(Path(d)/'session-001.provider.jsonl').read_text()
   self.assertIn('preflight-failed',journal)
   self.assertNotIn('"event": "reserve"',journal)
 def test_connection_is_reused_after_readonly_preflight(self):
  remote=Remote()
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=remote) as factory,patch.object(remote,'get',wraps=remote.get) as get:
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    self.request(b);self.request(b)
   self.assertEqual(factory.call_count,1)
   self.assertEqual(get.call_count,1)
   self.assertAlmostEqual(b.settle(ParsedSession()).cost_usd,.00264)
 def test_explicit_upstream_streaming_preserves_receipt(self):
  class Streaming(Remote):
   def post(self,*a,**kw):
    assert kw['json']['stream'] is True
    return Response()
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=Streaming()):
   b=self.backend();b.cfg['upstream_streaming']=True
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    body=self.request(b)
   self.assertIn(b'response.completed',body)
   self.assertAlmostEqual(b.settle(ParsedSession()).cost_usd,.00132)
 def test_failed_tls_before_http_write_is_known_zero(self):
  from urllib3.connection import HTTPSConnection
  class TLSFailure(Remote):
   def mount(self,prefix,adapter):self.adapter=adapter
   def post(self,*a,**kw):
    cls=self.adapter.poolmanager.pool_classes_by_scheme['https'].ConnectionCls
    conn=cls('fixture')
    with patch.object(HTTPSConnection,'connect',side_effect=mp.requests.exceptions.ReadTimeout('TLS timeout')):
     conn.send(b'never transmitted')
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=TLSFailure()):
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    with self.assertRaises(urllib.error.HTTPError):self.request(b)
   parsed=b.settle(ParsedSession(tool_uses=2))
   self.assertEqual(parsed.cost_usd,0)
   self.assertEqual(classify_errors(parsed,'',1),(True,False,'api_outage'))
   self.assertTrue(b.receipt['complete'])
   self.assertEqual(b.receipt['provider_calls'],0)
   self.assertEqual(b.receipt['connection_attempts_not_sent'],1)
   self.assertIn('not-sent',(Path(d)/'session-001.provider.jsonl').read_text())
 def test_sse_terminal_without_optional_space_is_settled(self):
  class Compact(Response):
   def iter_lines(self,chunk_size):
    for line in super().iter_lines(chunk_size):yield line.replace(b'data: ',b'data:')
    raise AssertionError('must stop at terminal event')
  with tempfile.TemporaryDirectory() as d:
   meter=mp.Meter(Path(d)/'journal',2,pricing=PRICING)
   meter._session=SimpleNamespace(post=lambda *a,**kw:Compact())
   status,_=meter.request({'model':'metered-test-model','stream':True},'https://fixture','test-only-secret')
   self.assertEqual(status,200);self.assertFalse(meter.unknown)
   self.assertAlmostEqual(float(meter.spent),.00132)
 def test_heartbeat_cannot_keep_stream_alive_indefinitely(self):
  now=[0.0];closed=[]
  class Heartbeats(Response):
   headers={'X-Oneapi-Request-Id':'heartbeat-request'}
   def iter_lines(self,chunk_size):
    for value in (10.,20.,31.):
     now[0]=value;yield b': heartbeat'
    raise AssertionError('deadline should stop this stream')
   def close(self):closed.append(True)
  receipt={'request_id':'heartbeat-request','usage':Response().json()['usage'],'provider_row_sha256':'fixture'}
  with tempfile.TemporaryDirectory() as d,patch.object(mp.time,'monotonic',side_effect=lambda:now[0]),patch.object(mp,'openai_compatible_receipt',return_value=receipt):
   meter=mp.Meter(Path(d)/'journal',2,pricing=PRICING,receipt_lookup=True,stream_max_s=30)
   meter._session=SimpleNamespace(post=lambda *a,**kw:Heartbeats())
   status,_=meter.request({'model':'metered-test-model','stream':True},'https://fixture','test-only-secret')
   self.assertEqual(status,503);self.assertTrue(closed);self.assertFalse(meter.unknown)
   self.assertAlmostEqual(float(meter.spent),.00132)
 def test_http_write_failure_keeps_unknown_liability(self):
  class WriteFailure(Remote):
   def mount(self,prefix,adapter):self.adapter=adapter
   def post(self,*a,**kw):
    cls=self.adapter.poolmanager.pool_classes_by_scheme['https'].ConnectionCls
    conn=cls('fixture')
    class Socket:
     def sendall(self,data):raise OSError('partial write')
    conn.sock=Socket()
    conn.send(b'possibly transmitted')
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=WriteFailure()):
   b=self.backend()
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    with self.assertRaises(urllib.error.HTTPError):self.request(b)
   self.assertIsNone(b.settle(ParsedSession()).cost_usd)
   self.assertFalse(b.receipt['complete'])
   self.assertEqual(b.receipt['provider_calls'],1)
   self.assertEqual(b.receipt['connection_attempts_not_sent'],0)
 def test_exact_server_request_receipt_resolves_lost_stream(self):
  class Interrupted(Response):
   headers={'X-Oneapi-Request-Id':'exact-id'}
   def iter_lines(self,chunk_size):
    yield b'data: {"type":"response.created","response":{"id":"r"}}'
    raise mp.requests.exceptions.ReadTimeout('lost body')
  class Streaming(Remote):
   def post(self,*a,**kw):return Interrupted()
  receipt={'request_id':'exact-id','usage':Response().json()['usage'],'provider_row_sha256':'fixture-digest'}
  with tempfile.TemporaryDirectory() as d,self.creds(),patch.object(mp.requests,'Session',return_value=Streaming()),patch.object(mp,'openai_compatible_receipt',return_value=receipt) as lookup:
   b=self.backend();b.cfg.update(upstream_streaming=True,provider_receipt_lookup='openai_compatible')
   with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):
    self.request(b)
   self.assertAlmostEqual(b.settle(ParsedSession()).cost_usd,.00132)
   self.assertTrue(b.receipt['complete'])
   self.assertEqual(lookup.call_args.args[2],'exact-id')
   rows=[json.loads(l) for l in (Path(d)/'session-001.provider.jsonl').read_text().splitlines()]
   paid=next(x for x in rows if x['event']=='settled')
   self.assertFalse(paid['output_received'])
   self.assertEqual(paid['cost_source'],'exact-provider-request-receipt')
 def test_receipt_lookup_rejects_wrong_request_and_model(self):
  class LogResponse:
   def raise_for_status(self):pass
   def json(self):return {'success':True,'data':[row]}
  class LogSession:
   def get(self,*a,**kw):return LogResponse()
   def close(self):pass
  row={'request_id':'other','model_name':'metered-test-model','type':2,'prompt_tokens':100,'completion_tokens':10,'other':json.dumps({'matched_tier':'standard','cache_tokens':20})}
  with patch.object(mp.requests,'Session',return_value=LogSession()):
   self.assertIsNone(mp.openai_compatible_receipt('https://gateway.example/v1/responses','key','wanted','metered-test-model'))
   row['request_id']='wanted';row['model_name']='other-model'
   self.assertIsNone(mp.openai_compatible_receipt('https://gateway.example/v1/responses','key','wanted','metered-test-model'))
   row['model_name']='metered-test-model'
   self.assertEqual(mp.openai_compatible_receipt('https://gateway.example/v1/responses','key','wanted','metered-test-model')['usage']['input_tokens'],100)
 def test_exception_still_closes_and_records_receipt(self):
  with tempfile.TemporaryDirectory() as d,self.creds():
   b=self.backend()
   with self.assertRaises(RuntimeError):
    with b.session(SessionSpec(1,'brief',d,max_budget_usd=2),Path(d)):raise RuntimeError('runner')
   self.assertTrue((Path(d)/'session-001.cost.json').exists())
   self.assertFalse(b.gateway.thread.is_alive())

class RoundMeteredTests(unittest.TestCase):
 def test_retained_liability_reduces_budget_without_claiming_actual_cost(self):
  from xh_helpers import write_adapter
  from datafarm.expert.round import prepare_round,RoundRunner
  from datafarm.expert.layout import RoundPaths
  from datafarm.expert.state import RoundState
  from xh_helpers import FakeClock
  from datafarm.expert.util import HarnessError
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);adapter=write_adapter(root/'task');paths=RoundPaths(root/'round');clock=FakeClock()
   prepare_round(adapter,paths.root,clock=clock)
   runner=RoundRunner(paths,adapter,clock=clock)
   runner.agent_backend=SimpleNamespace(name='codex-metered')
   runner.broker=SimpleNamespace(counted=lambda:0)
   runner.state=RoundState.load(paths.state);data=runner.state.data
   data.update(cost_usd_known=1.25,sessions_unknown_cost=1,
               sessions=[{'index':1,'ended_at':'2026-10-24T00:00:00Z','cost_usd':None}],
               bounded_unknown_costs=[{'session_index':1,'reserved_usd':3,'evidence':'audited-receipt'}])
   self.assertEqual(runner._cost(),4.25)
   self.assertEqual(runner._known_cost(),1.25)
   self.assertIsNone(runner._stop_reason())
   data['agent_session_id']='same-conversation'
   data['sessions'][0]['api_outage']=True
   runner.broker.status_view=lambda now:{'messages':[]}
   runner.broker.budget_view=lambda now:{'counted_rollouts':{'used':0,'in_flight':0,'max':6},
       'concurrency':{'max':2},'wall_s':{'active':0}}
   prompt=runner._prompt(99,False,60)
   self.assertIn('Session 1 of at most 4',prompt)
   self.assertIn('used or reserved',prompt)
   data['cost_usd_known']=2.25
   self.assertEqual(runner._stop_reason(),'usd_exhausted')
   data['sessions_unknown_cost']=2
   self.assertEqual(runner._stop_reason(),'cost_unknown')
   data['bounded_unknown_costs'].append(data['bounded_unknown_costs'][0])
   with self.assertRaises(HarnessError):runner._cost()

 def test_unknown_usage_stops_round_after_one_session(self):
  import sys
  from xh_helpers import write_adapter,write_plan,expert
  from datafarm.expert.round import prepare_round,RoundRunner
  from datafarm.expert.layout import RoundPaths
  from datafarm.expert.rollout import FakeRolloutBackend
  from xh_helpers import FakeClock
  class Broken(Remote):
   def post(self,*a,**kw):raise mp.requests.exceptions.ReadTimeout('fixture test-only-secret')
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);adapter=write_adapter(root/'task');write_plan(root/'task',[[]])
   paths=RoundPaths(root/'round');clock=FakeClock();prepare_round(adapter,paths.root,clock=clock)
   (paths.workspace/'expert.py').write_text(expert())
   script=root/'fake_cli.py';script.write_text('''import sys,json,urllib.request,urllib.error
sys.stdin.read()
print(json.dumps({'type':'thread.started','thread_id':'fixture-thread'}),flush=True)
r=urllib.request.Request(sys.argv[1]+'/responses',data=json.dumps({'model':'metered-test-model','stream':True}).encode(),headers={'Content-Type':'application/json'})
try:urllib.request.urlopen(r).read()
except urllib.error.HTTPError:pass
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':0,'output_tokens':0}}),flush=True)
''')
   class Backend(MeteredCodexBackend):
    def command(self,spec):return [sys.executable,str(script),self.gateway.url]
   backend=Backend({'model':'metered-test-model','credentials_file':'/protected/fake','pricing':PRICING_CFG})
   with patch('datafarm.agent.credentials.KeysFileCredentials',return_value=lambda:SimpleNamespace(endpoint='https://fixture',api_key='test-only-secret')),patch.object(mp.requests,'Session',return_value=Broken()):
    runner=RoundRunner(paths,adapter,agent_backend=backend,rollout_backend=FakeRolloutBackend(),approve_agent_spend=True,clock=clock,tick_interval_s=.01,poll_interval_s=0)
    result=runner.run()
   state=json.loads(paths.state.read_text())
   self.assertEqual(len(state['sessions']),1)
   self.assertEqual(state['end_reason'],'cost_unknown')
   self.assertEqual(state['sessions_unknown_cost'],1)
   self.assertEqual(state['submissions'],[])
   self.assertIsNone(state.get('stop'))
