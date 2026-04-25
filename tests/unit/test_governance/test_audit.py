import io
import json

from spark_ai_functions.governance.audit import (
    CallbackAuditSink,
    CompositeAuditSink,
    InMemoryAuditSink,
    StdoutAuditSink,
)


def test_in_memory_records_and_filters():
    sink = InMemoryAuditSink()
    sink.emit({"user": "alice", "status": "success"})
    sink.emit({"user": "alice", "status": "denied"})
    sink.emit({"user": "bob", "status": "success"})
    assert len(sink.find(user="alice")) == 2
    assert sink.only(user="bob", status="success")["user"] == "bob"


def test_stdout_writes_single_line_json():
    buf = io.StringIO()
    StdoutAuditSink(stream=buf).emit({"user": "alice"})
    line = buf.getvalue().strip()
    parsed = json.loads(line)
    assert parsed == {"user": "alice"}


def test_composite_fans_out_and_swallows_errors():
    mem = InMemoryAuditSink()

    class _Broken:
        def emit(self, event):
            raise RuntimeError("oops")

    CompositeAuditSink(_Broken(), mem).emit({"x": 1})
    assert mem.events == [{"x": 1}]


def test_callback_sink():
    collected = []
    s = CallbackAuditSink(lambda e: collected.append(e))
    s.emit({"a": 1})
    assert collected == [{"a": 1}]
