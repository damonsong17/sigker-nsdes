"""
Parity gate: verify the pySigLib-backed kernel matches sigkernel BEFORE training.

Run from the repo root:
    python tests/parity_pysiglib.py

What to expect:
  * FORWARD values (MMD, scoring rule) should match tightly (reldiff ~1e-3 or better).
    If they are far off, the RBF `sigma` convention differs between libraries
    (sigkernel uses exp(-||x-y||^2 / sigma); pySigLib may use a different scale) --
    rescale `sigma` until the forwards agree. Same idea for `dyadic_order`.
  * GRADIENTS may differ slightly -- that is EXPECTED and is the improvement:
    pySigLib is exact, sigkernel's reversed-PDE gradient is approximate.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import sigkernel
import pysiglib
from src.gan.pysiglib_kernel import PySigKernel

torch.manual_seed(0)

DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DO, SIGMA, MB = 1, 0.5, 16
print(f"device={DEVICE} dyadic_order={DO} sigma={SIGMA}")


def make_data(requires_grad=True):
    X = torch.rand(16, 50, 3, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    Y = torch.rand(16, 50, 3, dtype=DTYPE, device=DEVICE)
    return X, Y


def reldiff(a, b):
    return abs(float(a) - float(b)) / max(abs(float(a)), 1e-12)


for ktype in ("linear", "rbf"):
    print(f"\n=== static kernel: {ktype} ===")
    if ktype == "rbf":
        sk = sigkernel.SigKernel(sigkernel.RBFKernel(sigma=SIGMA), dyadic_order=DO)
        pk = PySigKernel(pysiglib.RBFKernel(sigma=SIGMA), dyadic_order=DO)
    else:
        sk = sigkernel.SigKernel(sigkernel.LinearKernel(), dyadic_order=DO)
        pk = PySigKernel(pysiglib.LinearKernel(), dyadic_order=DO)

    # ---- MMD forward + gradient parity ----
    X, Y = make_data()
    m_sk = sk.compute_mmd(X, Y, max_batch=MB)
    g_sk = torch.autograd.grad(m_sk, X)[0]

    X2 = X.detach().clone().requires_grad_(True)
    m_pk = pk.compute_mmd(X2, Y, max_batch=MB)
    g_pk = torch.autograd.grad(m_pk, X2)[0]

    print(f"MMD   sigkernel={float(m_sk):.6e}  pysiglib={float(m_pk):.6e}  "
          f"reldiff={reldiff(m_sk, m_pk):.2e}")
    print(f"MMD   grad max|delta| = {(g_sk - g_pk).abs().max().item():.2e}")

    # ---- scoring rule forward + gradient parity ----
    # The scoring-rule discriminator calls compute_scoring_rule(X, y.unsqueeze(0)).
    X, _ = make_data()
    y = torch.rand(50, 3, dtype=DTYPE, device=DEVICE).unsqueeze(0)  # (1, stream, ch)
    try:
        s_sk = sk.compute_scoring_rule(X, y, max_batch=MB)
        gs_sk = torch.autograd.grad(s_sk, X)[0]
        X2 = X.detach().clone().requires_grad_(True)
        s_pk = pk.compute_scoring_rule(X2, y, max_batch=MB)
        gs_pk = torch.autograd.grad(s_pk, X2)[0]
        print(f"SCORE sigkernel={float(s_sk):.6e}  pysiglib={float(s_pk):.6e}  "
              f"reldiff={reldiff(s_sk, s_pk):.2e}")
        print(f"SCORE grad max|delta| = {(gs_sk - gs_pk).abs().max().item():.2e}")
    except Exception as e:
        print(f"SCORE parity could not run (check y-shape convention): {e!r}")

print("\nGuide: forward reldiff small => OK. Large => fix sigma/dyadic convention. "
      "Grad delta small-but-nonzero is expected (pySigLib exact vs sigkernel approx).")
