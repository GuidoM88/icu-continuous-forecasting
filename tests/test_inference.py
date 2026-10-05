import numpy as np

from src.data.preprocessing import Physionet2012Dataset, build_vocab
from src.data.synthetic import make_synthetic_records
from src.inference import predict_records
from src.model_bundle import save_bundle
from src.models.gru import GRUMortalityModel


def test_final_gru_bundle_roundtrip(tmp_path):
    train_records = make_synthetic_records(n=24, seed=11)
    vocab = build_vocab(train_records)

    train_ds = Physionet2012Dataset(train_records, vocab)
    normalizer = train_ds.fit_normalizer()

    model = GRUMortalityModel(
        input_dim=len(vocab),
        static_dim=len(train_ds.static_names),
        hidden=64,
    )

    bundle_path = tmp_path / "gru_bundle.pt"

    save_bundle(
        bundle_path,
        model=model,
        vocab=vocab,
        static_names=train_ds.static_names,
        normalizer=normalizer,
        threshold=0.42,
        hidden=64,
    )

    inference_records = make_synthetic_records(n=7, seed=19)
    for record in inference_records:
        record.label = None

    output = predict_records(
        inference_records,
        bundle_path,
        device="cpu",
        batch_size=4,
    )

    assert output["record_id"].shape == (7,)
    assert output["score"].shape == (7,)
    assert output["prediction"].shape == (7,)

    assert np.isfinite(output["score"]).all()
    assert ((output["score"] >= 0.0) & (output["score"] <= 1.0)).all()
    assert set(np.unique(output["prediction"])).issubset({0, 1})
