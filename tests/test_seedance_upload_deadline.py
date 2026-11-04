import hashlib
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from datafarm.render.seedance.upload_transport import curl, exchange, UploadError


class UploadDeadlineTest(unittest.TestCase):
    def test_stalled_http_peer_is_bounded(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        port=listener.getsockname()[1]
        def peer():
            connection,_=listener.accept()
            with connection:time.sleep(1)
            listener.close()
        thread=threading.Thread(target=peer,daemon=True);thread.start()
        started=time.monotonic()
        with patch.dict('os.environ',{'SEEDANCE_API_PROXY':''}):
            with self.assertRaises(UploadError):curl([],f'http://127.0.0.1:{port}/',.2)
        self.assertLess(time.monotonic()-started,2)
        thread.join(2)

    def test_mismatching_download_is_rejected_and_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'source.mp4';source.write_bytes(b'original')
            def fake(args,url,remaining):
                if '--form' in args:return b'{"files":[{"id":"test"}]}'
                Path(args[1]).write_bytes(b'changed');return b''
            with patch('datafarm.render.seedance.upload_transport.curl',side_effect=fake):
                with self.assertRaises(UploadError):exchange(source,hashlib.sha256(b'original').hexdigest(),folder,'https://example.invalid')
            self.assertEqual(source.read_bytes(),b'original')
            self.assertFalse((folder/'upload-verify.partial').exists())

    def test_upload_host_is_explicit_opt_in(self):
        from datafarm.render.seedance.client import upload_host
        with patch.dict('os.environ',{'DATAFARM_SEEDANCE_UPLOAD_HOST':''}):
            with self.assertRaisesRegex(RuntimeError,'real-scene media'):upload_host()
            self.assertEqual(upload_host('https://files.example/'),'https://files.example')
        with patch.dict('os.environ',{'DATAFARM_SEEDANCE_UPLOAD_HOST':'http://plain.example'}):
            with self.assertRaises(RuntimeError):upload_host()


if __name__=='__main__':unittest.main()
