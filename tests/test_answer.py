import io
import json
from urllib.error import HTTPError

import pytest

from code_assistant.answer import OpenRouterClient, answer_question, source_prompt
from code_assistant.retrieval import Evidence


class FakeClient:
    model = "test/model:free"

    def __init__(self, response):
        self.response = response
        self.prompt = None
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        self.prompt = user
        return self.response


def evidence():
    return [
        Evidence(
            1, "first", "ultralytics/engine/predictor.py", "a" * 40,
            "BasePredictor.preprocess", 10, 11,
            "def preprocess(self, im):\n    return im[0]\n",
        ),
        Evidence(
            2, "bounds", "ultralytics/engine/predictor.py", "a" * 40,
            "bounds", 12, 13, "def bounds():\n    return [0, 1]\n",
        ),
    ]


def test_empty_evidence_abstains_without_api_client():
    assert answer_question("What happens?", [], None).status == "insufficient_evidence"


def test_source_prompt_and_valid_references():
    sources = evidence()
    client = FakeClient(
        "STATUS: ANSWERED\nIt selects `im[0]`; bounds returns `[0, 1]`. [1, 2]"
    )
    result = answer_question("What do these functions return?", sources, client)
    assert result.status == "answered"
    assert "What do these functions return?" in client.prompt
    for source in sources:
        assert source.code in client.prompt
        assert f"[{source.number}] {source.citation}" in client.prompt


def test_split_branch_is_shown_in_source_order_without_filling_gaps():
    # Retrieval ranks a branch's body before its condition and final return.
    parts = [
        Evidence(1, "body", "ultralytics/data/base.py", "a" * 40, "Dataset.load", 12, 13,
                 "        im = read_image()\n        im = resize(im)\n"),
        Evidence(2, "head", "ultralytics/data/base.py", "a" * 40, "Dataset.load", 10, 11,
                 "def load(im):\n    if im is None:\n"),
        Evidence(3, "return", "ultralytics/data/base.py", "a" * 40, "Dataset.load", 14, 14,
                 "    return im\n"),
    ]
    prompt = source_prompt("Does cached input get resized?", parts)
    complete = parts[1].code + parts[0].code + parts[2].code
    assert f"```python\n{complete}\n```" in prompt
    for part in parts:
        assert f"[{part.number}] {part.citation}" in prompt
    with_gap = source_prompt("Explain load", [parts[1], parts[2]])
    assert parts[0].code not in with_gap
    assert with_gap.count("```python") == 2


@pytest.mark.parametrize("response", [
    "STATUS: ANSWERED\nNo source reference.",
    "STATUS: ANSWERED\nThis claim [1, 99] also cites [1].",
    "STATUS: INSUFFICIENT_EVIDENCE\nMissing context. [99]",
    "Unmarked response citing unknown source [99]",
    "**Status: ANSWERED**\nClaim [99]",
])
def test_missing_or_unknown_references_are_rejected(response):
    with pytest.raises(ValueError, match="citation"):
        answer_question("What happens?", evidence(), FakeClient(response))


def test_nonempty_evidence_can_return_insufficient_evidence():
    sources = evidence()
    result = answer_question(
        "What changed in the latest release?", sources,
        FakeClient("STATUS: INSUFFICIENT_EVIDENCE\nThese functions do not establish release changes."),
    )
    assert result.status == "insufficient_evidence"
    assert result.evidence == sources


@pytest.mark.parametrize("header", ["**Status: ANSWERED**", "`status: answered`", "__STATUS: ANSWERED__"])
def test_explicit_status_accepts_cosmetic_markdown_and_case(header):
    result = answer_question("What happens?", evidence(), FakeClient(header + "\nReturns the first item. [1]"))
    assert result.status == "answered"


@pytest.mark.parametrize("response", ["", "   ", "STATUS: ANSWERED\n", "ANSWERED:", "INSUFFICIENT_EVIDENCE:\n"])
def test_empty_response_or_status_without_body_is_rejected(response):
    with pytest.raises(ValueError, match="empty"):
        answer_question("What happens?", evidence(), FakeClient(response))


@pytest.mark.parametrize(("response", "status", "text"), [
    ("ANSWERED: Returns the first item. [1]", "answered", "Returns the first item. [1]"),
    ("answered:\nReturns the first item. [1]", "answered", "Returns the first item. [1]"),
    ("INSUFFICIENT_EVIDENCE: The caller is missing.", "insufficient_evidence", "The caller is missing."),
])
def test_explicit_status_alias_keeps_inline_or_multiline_body(response, status, text):
    client = FakeClient(response)
    result = answer_question("What happens?", evidence(), client)
    assert (result.status, result.text) == (status, text)
    assert client.calls == 1


@pytest.mark.parametrize("response", [
    "Unmarked answer [1]",
    "**Status: MAYBE**\nClaim [1]",
    "The method returns the first item.",
])
def test_unknown_status_preserves_raw_text_without_inferring_answered(response):
    client = FakeClient(response)
    result = answer_question("What happens?", evidence(), client)
    assert result.status == "unverified"
    assert result.text == response and result.evidence == evidence()
    assert client.calls == 1


@pytest.mark.parametrize(("payload", "error"), [
    ({"choices": [{"finish_reason": "stop", "message": {"content": "Complete answer."}}]}, None),
    ({"choices": [{"finish_reason": "length", "message": {"content": "Partial answer."}}]}, "truncated"),
    ({"choices": [{"finish_reason": "error", "message": {"content": "Partial answer."}}]}, "ended with an error"),
    ({"error": {"code": 502, "message": "provider private detail"}}, "API error"),
    ({}, "no completion choices"),
])
def test_openrouter_parses_success_and_reports_failed_responses(monkeypatch, payload, error):
    monkeypatch.setattr(
        "code_assistant.answer.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    client = OpenRouterClient("test-key", "test/model:free")
    if error is None:
        assert client.generate("system", "question") == "Complete answer."
    else:
        with pytest.raises(RuntimeError, match=error) as raised:
            client.generate("system", "question")
        assert "provider private detail" not in str(raised.value)


def test_daily_quota_is_identified_without_exposing_provider_details(monkeypatch):
    def limited(*args, **kwargs):
        body = json.dumps({"error": {"message": "free-models-per-day: provider private detail"}})
        raise HTTPError("https://example.test", 429, "Too Many Requests", {}, io.BytesIO(body.encode()))

    monkeypatch.setattr("code_assistant.answer.urlopen", limited)
    client = OpenRouterClient("test-key", "test/model:free")
    with pytest.raises(RuntimeError, match="daily request limit") as raised:
        client.generate("system", "question")
    assert "provider private detail" not in str(raised.value)
    assert client.last_response["rate_limit_scope"] == "free-models-per-day"


def test_runtime_disables_reasoning_without_changing_prompts(monkeypatch):
    sent = []

    def response(request, **kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "Complete answer."}}]}).encode())

    monkeypatch.setattr("code_assistant.answer.urlopen", response)
    OpenRouterClient("test-key", "test/model:free").generate("approved system", "same question")
    assert sent[0]["reasoning"] == {"enabled": False}
    assert sent[0]["messages"] == [{"role": "system", "content": "approved system"}, {"role": "user", "content": "same question"}]
    OpenRouterClient("test-key", "test/model:free", reasoning_enabled=None).generate("approved system", "same question")
    assert "reasoning" not in sent[1]


def stream_frames(*frames, done=True):
    body = ': keepalive\n\n' + ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames)
    if done:
        body += 'data: [DONE]\n\n'
    return io.BytesIO(body.encode())


def test_stream_measures_visible_text_and_reads_final_usage(monkeypatch):
    clock = iter([12.0, 14.0])
    monkeypatch.setattr('code_assistant.answer.time.perf_counter', lambda: next(clock))
    response = stream_frames(
        {'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]},
        {'choices': [{'delta': {'content': 'Complete '}, 'index': 0}]},
        {'choices': [{'delta': {'content': 'answer.'}, 'finish_reason': 'stop', 'index': 0}]},
        {'choices': [], 'usage': {'completion_tokens': 10, 'completion_tokens_details': {'reasoning_tokens': 0}}},
    )
    result = OpenRouterClient._read_stream(response, 10.0)
    assert result['choices'][0]['message']['content'] == 'Complete answer.'
    assert result['timing'] == {'ttft_seconds': 2.0, 'stream_seconds': 4.0,
                                'output_delivery_tokens_per_second': 5.0}


@pytest.mark.parametrize('provider_error', [False, True])
def test_stream_rejects_disconnect_and_midstream_error(monkeypatch, provider_error):
    monkeypatch.setattr('code_assistant.answer.time.perf_counter', lambda: 12.0)
    frames = [{'choices': [{'delta': {'content': 'Partial answer'}}]}]
    if provider_error:
        frames.append({'error': {'code': 502}, 'choices': []})
    result = OpenRouterClient._read_stream(stream_frames(*frames, done=provider_error), 10.0)
    assert result['error']
    assert result['partial_text'] == 'Partial answer'
    assert result['timing']['output_delivery_tokens_per_second'] is None


def test_streaming_client_preserves_request_settings_and_raw_answer(monkeypatch):
    sent = []

    def response(request, **kwargs):
        sent.append(json.loads(request.data))
        return stream_frames(
            {'choices': [{'delta': {'content': 'STATUS: ANSWERED\nAnswer [1]'}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'completion_tokens': 8, 'completion_tokens_details': {'reasoning_tokens': 0}}},
        )

    monkeypatch.setattr('code_assistant.answer.urlopen', response)
    client = OpenRouterClient('test-key', 'test/model:free', stream=True)
    assert client.generate('approved system', 'same source') == 'STATUS: ANSWERED\nAnswer [1]'
    assert sent[0]['stream'] is True
    assert sent[0]['reasoning'] == {'enabled': False}
    assert sent[0]['temperature'] == 0 and sent[0]['max_tokens'] == 2000
    assert client.last_response['usage']['reasoning_tokens'] == 0
    assert client.last_response['text'] == 'STATUS: ANSWERED\nAnswer [1]'
