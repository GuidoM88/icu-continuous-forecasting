import numpy as np

from src.data.preprocessing import Physionet2012Dataset, build_vocab, compute_delta_t, fill_forward
from src.data.synthetic import make_synthetic_records


def test_vocab_and_shapes():
    records = make_synthetic_records(n=20, seed=1)
    vocab = build_vocab(records)
    assert len(vocab) > 0

    ds = Physionet2012Dataset(records, vocab)
    ds.fit_normalizer()
    batch = ds.collate(list(range(10)))

    B, T, D = 10, ds.n_bins, len(vocab)
    assert batch.values.shape == (B, T, D)
    assert batch.mask.shape == (B, T, D)
    assert batch.delta_t.shape == (B, T, D)
    assert batch.static.shape == (B, len(ds.static_names))
    assert batch.labels.shape == (B,)
    assert batch.times.shape == (T,)


def test_mask_matches_nan_pattern():
    records = make_synthetic_records(n=5, seed=2)
    vocab = build_vocab(records)
    ds = Physionet2012Dataset(records, vocab)
    ds.fit_normalizer()
    batch = ds.collate([0, 1, 2, 3, 4])

    is_nan = np.isnan(batch.values.numpy())
    is_unobserved = batch.mask.numpy() == 0
    assert (is_nan == is_unobserved).all()


def test_delta_t_matches_reference_recurrence():
    mask = np.array([[[1], [0], [0], [1]]], dtype=np.float32)
    delta = compute_delta_t(mask)
    # reference: delta[t] = 1 if mask[t-1] else 1 + delta[t-1]
    expected = np.zeros((1, 4, 1), dtype=np.float32)
    for t in range(1, 4):
        expected[:, t] = np.where(mask[:, t - 1] > 0, 1.0, 1.0 + expected[:, t - 1])
    assert np.allclose(delta, expected)


def test_fill_forward_carries_last_observed_value():
    values = np.array([[[5.0], [np.nan], [np.nan], [9.0]]], dtype=np.float32)
    mask = np.array([[[1], [0], [0], [1]]], dtype=np.float32)
    filled = fill_forward(values, mask)
    assert filled[0, 0, 0] == 5.0
    assert filled[0, 1, 0] == 5.0
    assert filled[0, 2, 0] == 5.0
    assert filled[0, 3, 0] == 9.0


def test_fill_forward_unobserved_prefix_is_zero():
    values = np.array([[[np.nan], [np.nan], [3.0]]], dtype=np.float32)
    mask = np.array([[[0], [0], [1]]], dtype=np.float32)
    filled = fill_forward(values, mask)
    assert filled[0, 0, 0] == 0.0
    assert filled[0, 1, 0] == 0.0
    assert filled[0, 2, 0] == 3.0


def test_vocab_filters_blank_variable_names():
    records = make_synthetic_records(n=2, seed=9)
    records[0].values[""] = [(1.0, 123.0)]
    records[0].values["   "] = [(2.0, 456.0)]
    vocab = build_vocab(records)
    assert "" not in vocab
    assert not any(v.isspace() for v in vocab)


def test_unlabeled_records_can_be_collated_for_inference():
    records = make_synthetic_records(n=4, seed=10)
    vocab = build_vocab(records)
    normalizer = Physionet2012Dataset(records, vocab).fit_normalizer()
    for r in records:
        r.label = None
    ds = Physionet2012Dataset(records, vocab, normalizer=normalizer)
    batch = ds.collate(range(4))
    assert len(ds) == 4
    assert np.isnan(batch.labels.numpy()).all()
