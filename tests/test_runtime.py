import logging
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def test_initialization_concurrent_idempotent_and_shutdown():
    from config import Settings
    from services.runtime import ModelRuntime

    calls = []
    entered, release = Event(), Event()

    def loader(operation, settings):
        calls.append(operation)
        if operation == "classify":
            entered.set()
            assert release.wait(5)
        return lambda **kw: None

    runtime = ModelRuntime(Settings(_env_file=None), loader)
    barrier = Barrier(4)

    def initialize():
        barrier.wait(5)
        runtime.initialize()

    with ThreadPoolExecutor(3) as pool:
        jobs = [pool.submit(initialize) for _ in range(3)]
        barrier.wait(5)
        assert entered.wait(5)
        assert not runtime.ready
        assert calls == ["classify"]  # Second model cannot start yet.
        release.set()
        for job in jobs:
            job.result(5)
    runtime.initialize()
    assert runtime.ready
    assert calls == ["classify", "ner"]
    refs = [weakref.ref(runtime.classifier), weakref.ref(runtime.ner)]
    runtime.close()
    runtime.close()
    assert not runtime.ready
    assert runtime.classifier is runtime.ner is None
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="closed"):
        runtime.initialize()


def test_startup_failure_clears_partial_state_and_logs(monkeypatch, caplog):
    from app import create_app
    from config import Settings
    from services.runtime import ModelRuntime

    refs = []

    def loader(operation, settings):
        if operation == "ner":
            raise ValueError("SECRET document credentials")
        pipe = lambda text: None  # noqa: E731
        refs.append(weakref.ref(pipe))
        return pipe

    app = create_app(Settings(_env_file=None))
    app.state.runtime = ModelRuntime(app.state.settings, loader)
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="initialization failed"):
        with TestClient(app):
            pytest.fail("Startup should fail")
    assert not app.state.runtime.ready
    assert app.state.runtime.classifier is app.state.runtime.ner is None
    assert refs[0]() is None
    assert "SECRET" not in caplog.text
    assert "runtime_initialization_failed" in caplog.text


def test_lifespan_readiness_health_and_load_counts(monkeypatch):
    import services.runtime as module
    from app import create_app
    from config import Settings

    calls, inference = [], []

    def loader(operation, settings):
        calls.append(operation)

        def pipe(*args, **kw):
            inference.append(operation)
            return [] if operation == "ner" else {"labels": ["A"], "scores": [0.9]}
        return pipe

    monkeypatch.setattr(module, "load_pipeline", loader)
    app = create_app(Settings(_env_file=None))
    assert not calls
    without_lifespan = TestClient(app)
    try:
        assert without_lifespan.get("/healthz").status_code == 200
        assert without_lifespan.get("/readyz").status_code == 503
        for endpoint in ("classify", "ner"):
            response = without_lifespan.post(f"/{endpoint}", json={"text": "x"})
            assert response.status_code == 503
        assert not calls and not inference
    finally:
        without_lifespan.close()
    with TestClient(app) as client:
        assert calls == ["classify", "ner"]
        for _ in range(2):
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 200
        assert not inference
        for _ in range(2):
            assert client.post("/classify", json={"text": "x"}).status_code == 200
            assert client.post("/ner", json={"text": "x"}).status_code == 200
        assert calls == ["classify", "ner"]
    assert not app.state.runtime.ready
    assert app.state.runtime.classifier is app.state.runtime.ner is None
    after_shutdown = TestClient(app)
    try:
        assert after_shutdown.get("/readyz").status_code == 503
        assert after_shutdown.get("/healthz").status_code == 200
        assert after_shutdown.post("/ner", json={"text": "x"}).status_code == 503
    finally:
        after_shutdown.close()
    assert calls == ["classify", "ner"]


def test_app_settings_reach_loaders_and_service(monkeypatch):
    import services.runtime as module
    from app import create_app
    from config import Settings

    loads, inputs = [], []

    def loader(operation, settings):
        loads.append((operation, settings))

        def pipe(**kwargs):
            inputs.append(kwargs)
            return {"labels": ["A"], "scores": [0.9]}
        return pipe if operation == "classify" else lambda text: []

    monkeypatch.setattr(module, "load_pipeline", loader)
    first = Settings(_env_file=None, zsc_max_len=3, zsc_multi_label=False, hf_home="/first")
    second = Settings(_env_file=None, zsc_max_len=8, zsc_multi_label=True, hf_home="/second")
    for settings in (first, second):
        with TestClient(create_app(settings)) as client:
            assert client.post("/classify", json={"text": "abcdefghij"}).status_code == 200
    assert [s for _, s in loads] == [first, first, second, second]
    assert [(x["sequences"], x["multi_label"]) for x in inputs] == [
        ("abc", False), ("abcdefgh", True),
    ]


@pytest.mark.parametrize("held,other", [("classify", "ner"), ("ner", "classify")])
@pytest.mark.parametrize("separate_apps", [False, True])
def test_real_request_concurrency(client, held, other, separate_apps):
    from app import create_app
    from config import Settings

    entered, release = Event(), Event()
    calls = []

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return [] if held == "ner" else {"labels": ["A"], "scores": [0.9]}

    def unexpected(*args, **kwargs):
        calls.append(other)
        raise AssertionError("Busy request must not reach model")

    runtime = client.app.state.runtime
    setattr(runtime, "classifier" if held == "classify" else "ner", blocking)
    with TestClient(create_app(Settings(_env_file=None))) as second:
        target = second if separate_apps else client
        attribute = "classifier" if other == "classify" else "ner"
        setattr(target.app.state.runtime, attribute, unexpected)
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(client.post, f"/{held}", json={"text": "x"})
            try:
                assert entered.wait(5)
                response = target.post(f"/{other}", json={"text": "x"})
                assert response.status_code == 503
                assert response.headers["Retry-After"] == "1"
                assert not calls
                assert target.get("/healthz").status_code == 200
                assert target.get("/readyz").status_code == 200
            finally:
                release.set()
            assert job.result(5).status_code == 200


def test_exception_releases_slot_and_safe_logs(client, caplog):
    def fail(*args, **kwargs):
        raise ValueError("SECRET CREDENTIALS PRIVATE_DOCUMENT")

    client.app.state.runtime.classifier = fail
    with caplog.at_level(logging.INFO):
        response = client.post("/classify", json={"text": "PRIVATE_DOCUMENT"})
        assert response.status_code == 500
        assert client.post("/ner", json={"text": "PRIVATE_DOCUMENT"}).status_code == 200
    assert "PRIVATE_DOCUMENT" not in caplog.text + response.text
    assert "SECRET" not in caplog.text + response.text
    assert "CREDENTIALS" not in caplog.text + response.text
    assert "inference_failed" in caplog.text
    assert "duration_seconds=" in caplog.text


def test_shutdown_waits_for_active_worker(client):
    entered, release, closing = Event(), Event(), Event()
    runtime = client.app.state.runtime

    def pipe(text):
        entered.set()
        assert release.wait(5)
        return []

    def close():
        closing.set()
        runtime.close()

    runtime.ner = pipe
    with ThreadPoolExecutor(2) as pool:
        job = pool.submit(client.post, "/ner", json={"text": "x"})
        try:
            assert entered.wait(5)
            shutdown = pool.submit(close)
            assert closing.wait(5)
            assert not shutdown.done()
            assert runtime.ner is pipe
        finally:
            release.set()
        assert job.result(5).status_code == 200
        shutdown.result(5)
    assert not runtime.ready
    assert runtime.ner is runtime.classifier is None


def test_cpu_thread_policy_once_and_conflict(monkeypatch):
    # Restore the real function hidden by the autouse fake-model fixture.
    import services.runtime as runtime

    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=calls.append))
    monkeypatch.setattr(runtime, "_CPU_THREADS", None)
    real = runtime.configure_cpu_threads.__wrapped__
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(real, [2, 2, 2]))
    assert calls == [2]
    with pytest.raises(RuntimeError, match="conflicts"):
        real(3)
    assert calls == [2]


def test_cancellation_keeps_slot_until_cpu_finishes():
    import asyncio

    from app import infer
    from config import Settings
    from services.runtime import ModelRuntime, RuntimeBusy

    entered, release = Event(), Event()
    runtime = ModelRuntime(Settings(_env_file=None), lambda *args: lambda text: [])
    runtime.initialize()

    def blocking():
        entered.set()
        assert release.wait(5)
        return []

    async def scenario():
        task = asyncio.create_task(infer(runtime, "ner", blocking))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)  # Deliver cancellation while the CPU call is running.
            assert not task.done()
            with pytest.raises(RuntimeBusy):
                with runtime.inference("classify"):
                    pytest.fail("The worker still owns the slot")
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with runtime.inference("classify"):
            pass

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        runtime.close()
