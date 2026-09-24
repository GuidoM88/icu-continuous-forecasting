import pytest
import torch

from src.data.preprocessing import Physionet2012Dataset, build_vocab, fill_forward
from src.data.synthetic import make_synthetic_records
from src.models.baselines import GRUD, GRUBaseline, PersistenceBaseline
from src.models.latent_ode import LatentODE, kl_standard_normal, masked_mse
from src.models.neural_cde import NeuralCDE


@pytest.fixture(scope="module")
def batch():
    records = make_synthetic_records(n=12, seed=3)
    vocab = build_vocab(records)
    ds = Physionet2012Dataset(records, vocab)
    ds.fit_normalizer()
    b = ds.collate(list(range(12)))
    return b, len(vocab), len(ds.static_names)


def _filled(b):
    return torch.from_numpy(fill_forward(b.values.numpy(), b.mask.numpy()))


def test_persistence_forward_backward(batch):
    b, D, S = batch
    model = PersistenceBaseline(D, S)
    logits = model(b)
    assert logits.shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b.labels)
    loss.backward()


def test_gru_forward_backward(batch):
    b, D, S = batch
    model = GRUBaseline(D, S)
    logits = model(b, _filled(b))
    assert logits.shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b.labels)
    loss.backward()


def test_grud_forward_backward(batch):
    b, D, S = batch
    model = GRUD(D, S)
    x_mean = torch.zeros(D)
    logits = model(b, x_mean)
    assert logits.shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b.labels)
    loss.backward()


def test_latent_ode_forward_backward(batch):
    b, D, S = batch
    model = LatentODE(D, S, latent_dim=4, hidden=16, solver="rk4")
    # rk4 is a fixed-step solver here purely to keep the *test* fast; dopri5
    # (the default) is the right choice for real training, see configs/default.yaml
    logits, recon, mean, logvar = model(b, _filled(b))
    assert logits.shape == (12,)
    assert recon.shape == b.values.shape
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, b.labels)
        + 0.1 * masked_mse(recon, b.values, b.mask).mean()
        + 1e-3 * kl_standard_normal(mean, logvar).mean()
    )
    loss.backward()


def test_neural_cde_forward_backward(batch):
    b, D, S = batch
    model = NeuralCDE(D, S, hidden_channels=8, hidden=16, solver="rk4", step_size=1.0)
    logits = model(b)
    assert logits.shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b.labels)
    loss.backward()
