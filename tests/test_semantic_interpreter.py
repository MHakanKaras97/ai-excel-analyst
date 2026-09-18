import json

from src.ai_provider import AIProviderError
from src.schema.semantic_interpreter import interpret_semantic_schema
from src.schema.semantic_prompt_builder import build_semantic_schema_prompt

CANDIDATE_COLUMNS = [
    {"name": "Discount", "dtype": "object", "inferred_type": "text", "sample_values": ["10%", "12%"],
     "candidate_role": "unknown", "candidate_semantic_type": "unknown", "candidate_confidence": 0.3},
]

VALID_RESPONSE = {"columns": [
    {"name": "Discount", "role": "measure", "semantic_type": "percentage", "unit": "%", "time_role": "none"},
]}


class FakeProvider:
    def __init__(self, response=None, error: AIProviderError | None = None):
        self._response = response
        self._error = error
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


def test_prompt_builder_includes_column_metadata_and_excludes_file_id():
    prompt = build_semantic_schema_prompt(CANDIDATE_COLUMNS)

    assert "Discount" in prompt
    assert "file_id" not in prompt.lower()


def test_valid_proposal_is_accepted():
    provider = FakeProvider(response=json.dumps(VALID_RESPONSE))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is True
    assert result["proposals"] == {
        "Discount": {"role": "measure", "semantic_type": "percentage", "unit": "%", "time_role": "none"},
    }


def test_proposal_for_unknown_column_name_is_rejected():
    response = {"columns": [
        {"name": "DoesNotExist", "role": "measure", "semantic_type": "numeric", "unit": None, "time_role": "none"},
    ]}
    provider = FakeProvider(response=json.dumps(response))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_invalid_role_enum_is_rejected():
    response = {"columns": [
        {"name": "Discount", "role": "not_a_role", "semantic_type": "percentage", "unit": "%", "time_role": "none"},
    ]}
    provider = FakeProvider(response=json.dumps(response))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_duplicate_proposal_for_same_column_is_rejected():
    response = {"columns": [
        {"name": "Discount", "role": "measure", "semantic_type": "percentage", "unit": "%", "time_role": "none"},
        {"name": "Discount", "role": "dimension", "semantic_type": "text", "unit": None, "time_role": "none"},
    ]}
    provider = FakeProvider(response=json.dumps(response))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_extra_key_in_proposal_entry_is_rejected():
    response = {"columns": [
        {"name": "Discount", "role": "measure", "semantic_type": "percentage", "unit": "%",
         "time_role": "none", "extra": 1},
    ]}
    provider = FakeProvider(response=json.dumps(response))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_extra_top_level_key_is_rejected():
    response = {"columns": VALID_RESPONSE["columns"], "notes": "extra"}
    provider = FakeProvider(response=json.dumps(response))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="not json")

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_json"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert result["is_valid"] is False and result["reason"] == "missing_api_key"


def test_partial_response_covering_fewer_columns_than_input_is_accepted():
    columns = CANDIDATE_COLUMNS + [
        {"name": "Region", "dtype": "object", "inferred_type": "text", "sample_values": ["North"],
         "candidate_role": "dimension", "candidate_semantic_type": "text", "candidate_confidence": 0.65},
    ]
    provider = FakeProvider(response=json.dumps(VALID_RESPONSE))

    result = interpret_semantic_schema(columns, provider)

    assert result["is_valid"] is True
    assert set(result["proposals"].keys()) == {"Discount"}


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=json.dumps(VALID_RESPONSE))

    interpret_semantic_schema(CANDIDATE_COLUMNS, provider)

    assert provider.call_count == 1


def test_does_not_mutate_input_candidate_columns():
    provider = FakeProvider(response=json.dumps(VALID_RESPONSE))
    columns = [dict(c) for c in CANDIDATE_COLUMNS]
    before = [dict(c) for c in columns]

    interpret_semantic_schema(columns, provider)

    assert columns == before
