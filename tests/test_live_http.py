import json
import threading
import urllib.error
import urllib.request

import pytest
from werkzeug.serving import make_server

from app import create_app


@pytest.fixture(scope="module")
def base_url():
    server = make_server("127.0.0.1", 0, create_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def post(url, body):
    req = urllib.request.Request(url + "/api/audit", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_real_socket_round_trip(base_url):
    with urllib.request.urlopen(base_url + "/") as r:
        assert r.status == 200 and b"Fairness audit lab" in r.read()
    status, d = post(base_url, {"scenario": "loan", "seed": 43, "bias": 0.4, "target": "eo", "lam": 5})
    assert status == 200 and len(d["models"]) == 4 and len(d["curve"]) == 19
    assert d["seed"] == 43 and d["settings"]["target"] == "eo"
    status, d = post(base_url, {"target": "xx"})
    assert status == 400 and "target" in d["error"]
    with urllib.request.urlopen(base_url + "/api/adult") as r:
        assert len(json.load(r)["rows"]) == 6
