import pytest

from src.schema.models import make_column_schema, make_dataset_schema


def _valid_kwargs(**overrides):
    kwargs = dict(
        name="Revenue", role="measure", semantic_type="numeric", unit=None,
        time_role="none", confidence=0.9, evidence={"header": "Revenue"},
    )
    kwargs.update(overrides)
    return kwargs


def test_make_column_schema_returns_expected_shape():
    schema = make_column_schema(**_valid_kwargs())

    assert schema == {
        "name": "Revenue", "role": "measure", "semantic_type": "numeric",
        "unit": None, "time_role": "none", "confidence": 0.9,
        "evidence": {"header": "Revenue"},
    }


def test_make_column_schema_rejects_invalid_role():
    with pytest.raises(ValueError):
        make_column_schema(**_valid_kwargs(role="not_a_role"))


def test_make_column_schema_rejects_invalid_semantic_type():
    with pytest.raises(ValueError):
        make_column_schema(**_valid_kwargs(semantic_type="not_a_type"))


def test_make_column_schema_rejects_invalid_time_role():
    with pytest.raises(ValueError):
        make_column_schema(**_valid_kwargs(time_role="not_a_time_role"))


def test_make_column_schema_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        make_column_schema(**_valid_kwargs(confidence=1.5))
    with pytest.raises(ValueError):
        make_column_schema(**_valid_kwargs(confidence=-0.1))


def test_make_column_schema_defaults_missing_evidence_to_empty_dict():
    schema = make_column_schema(**_valid_kwargs(evidence=None))

    assert schema["evidence"] == {}


def test_make_dataset_schema_wraps_columns():
    columns = [make_column_schema(**_valid_kwargs())]

    dataset_schema = make_dataset_schema(columns)

    assert dataset_schema == {"columns": columns}


def test_make_dataset_schema_does_not_mutate_input_list():
    columns = [make_column_schema(**_valid_kwargs())]
    dataset_schema = make_dataset_schema(columns)
    dataset_schema["columns"].append("extra")

    assert columns == [make_column_schema(**_valid_kwargs())]
