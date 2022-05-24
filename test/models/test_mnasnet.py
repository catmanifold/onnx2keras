import numpy as np
import pytest
from torchvision.models import mnasnet0_5, mnasnet1_0, mnasnet0_75, mnasnet1_3

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3])
def test_mnasnet(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
