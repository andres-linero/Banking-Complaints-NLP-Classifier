import os

import pytest

from banking_complaints.train_bert import ComplaintTextDataset, encode_labels


class FakeTokenizer:
    def __call__(self, texts, truncation, padding, max_length):
        assert truncation is True
        assert padding == "max_length"
        return {
            "input_ids": [[index + 1] * max_length for index, _ in enumerate(texts)],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def test_encode_labels_maps_business_classes() -> None:
    labels, metadata = encode_labels(["Credit card / prepaid card", "Credit reporting", "Mortgage"])

    assert labels == [0, 1, 2]
    assert metadata["classes"] == ["Credit card / prepaid card", "Credit reporting", "Mortgage"]
    assert metadata["label2id"] == {
        "Credit card / prepaid card": 0,
        "Credit reporting": 1,
        "Mortgage": 2,
    }
    assert metadata["id2label"] == {
        "0": "Credit card / prepaid card",
        "1": "Credit reporting",
        "2": "Mortgage",
    }


@pytest.mark.skipif(
    os.getenv("RUN_BERT_TENSOR_TESTS") != "1",
    reason="Set RUN_BERT_TENSOR_TESTS=1 after running uv sync --group bert",
)
def test_complaint_text_dataset_builds_tokenized_items() -> None:
    dataset = ComplaintTextDataset(
        texts=["fee dispute", "missing payment"],
        labels=[1, 0],
        tokenizer=FakeTokenizer(),
        max_length=4,
    )

    first_item = dataset[0]

    assert len(dataset) == 2
    assert first_item["input_ids"].tolist() == [1, 1, 1, 1]
    assert first_item["attention_mask"].tolist() == [1, 1, 1, 1]
    assert first_item["labels"].item() == 1
