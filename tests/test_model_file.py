"""The trained model shipped with the package: the file is present and holds the model of the paper."""
import torch

from oglcnacfinder.scoring import DEFAULT_MODEL


def test_shipped_model():
    final = torch.load(DEFAULT_MODEL, map_location='cpu', weights_only=True)
    assert final['plm'] == 'ESM2-150M' and final['window'] == 'pm20'
    assert final['config'] == {'hidden': 256, 'dropout': 0.25, 'lr': 0.0001}
    assert final['mean'].shape == final['std'].shape == (1280,)
    assert abs(final['platt']['a'] - 0.7919977117419091) < 1e-12
    assert abs(final['platt']['b'] + 3.3721370158396056) < 1e-12
    assert abs(final['cutoff']['probability'] - 0.08208476445548388) < 1e-12
