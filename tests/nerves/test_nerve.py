import torch
import neurlink.nerves as nv
from neurlink.nerves.nerve import *

def test_expand():
    nndefs = list(expand([
        ((3, 1), nv.Required()),
        ((6, 1), nv.Conv2d(3)),
    ]))
    
    assert nndefs[0][0] == (3, 1)
    assert nndefs[1][0] == (6, 1)