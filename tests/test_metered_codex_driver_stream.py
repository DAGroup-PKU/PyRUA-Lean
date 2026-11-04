import requests
import importlib.util,json
from pathlib import Path
from unittest.mock import patch
from decimal import Decimal
import unittest,tempfile

spec=importlib.util.spec_from_file_location('driver',Path(__file__).resolve().parents[1]/'examples/runtime/metered_codex_driver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PRICING=m.load_pricing({'model':'metered-test-model','source':'test fixture','usd_per_million_tokens':{'input':10,'cached_input':1,'cache_write':12.5,'output':50},'long_context':{'above_input_tokens':272000,'input_factor':2,'output_factor':1.5}})

class Response:
    status_code=200
    headers={'x-request-id':'fake-provider-id'}
    def __init__(self, terminal=True):self.terminal=terminal;self.closed=False
    def iter_lines(self,chunk_size):
        yield b'data: {"type":"response.created","response":{"id":"r1"}}'
        yield b''
        if self.terminal:
            yield b'data: {"type":"response.completed","response":{"id":"r1","usage":{"input_tokens":9,"output_tokens":5}}}'
            raise AssertionError('must stop reading after terminal usage; do not wait for EOF')
        raise requests.exceptions.ReadTimeout('fixture')
    def close(self):self.closed=True

class Session:
    def __init__(self,r):self.r=r
    def get(self,*a,**kw):return Response()
    def post(self,*a,**kw):
        assert kw['stream'] is True and kw['timeout']==(30,900)
        return self.r

def test_terminal_event_streamed_and_settled_before_eof(tmp_path):
    meter=m.Meter(tmp_path/'journal',5,pricing=PRICING);r=Response();chunks=[]
    with patch.object(requests,'Session',return_value=Session(r)):
        status,body=meter.request({'model':'metered-test-model','stream':True},'https://fixture','not-a-key',emit=chunks.append)
    assert status==200 and chunks[0] is None and b'response.created' in chunks[1]
    assert meter.spent==Decimal('.00034') and not meter.unknown and r.closed

def test_partial_disconnect_stays_unknown_and_blocks_retry(tmp_path):
    meter=m.Meter(tmp_path/'journal',5,pricing=PRICING);r=Response(False)
    with patch.object(requests,'Session',return_value=Session(r)):
        status,_=meter.request({'model':'metered-test-model','stream':True},'https://fixture','not-a-key',emit=lambda p:None)
    assert status==502 and meter.unknown
    status,_=meter.request({'model':'metered-test-model'},'must-not-call','not-a-key')
    assert status==402 and meter.calls==1

class StreamTests(unittest.TestCase):
    def test_terminal(self):
        with tempfile.TemporaryDirectory() as d:test_terminal_event_streamed_and_settled_before_eof(Path(d))
    def test_disconnect(self):
        with tempfile.TemporaryDirectory() as d:test_partial_disconnect_stays_unknown_and_blocks_retry(Path(d))

class AdmissionTests(unittest.TestCase):
    def test_local_denial_is_known_zero_and_never_contacts_provider(self):
        with tempfile.TemporaryDirectory() as d:
            meter=m.Meter(Path(d)/'journal',0.01,pricing=PRICING)
            with patch.object(requests,'Session',side_effect=AssertionError('no upstream call')):
                status,_=meter.request({'model':'metered-test-model'},'https://fixture','not-a-key')
            self.assertEqual(status,402)
            self.assertEqual(meter.calls,0)
            self.assertFalse(meter.unknown)
            self.assertTrue(meter.denied)
            rows=[json.loads(l) for l in meter.path.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'denied-local')
