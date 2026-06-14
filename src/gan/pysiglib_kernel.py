"""
pySigLib-backed drop-in for the subset of ``sigkernel.SigKernel`` used in this
repo's discriminators (``compute_mmd``, ``compute_scoring_rule``, ``compute_Gram``).

Why: in sigker-nsdes the signature kernel is the *training loss*, solved forward
AND backward on every iteration. pySigLib provides exact, fast signature-kernel
gradients (vs sigkernel's approximate reversed-PDE gradient) and removes
sigkernel's hard ``max(MM, NN) < 1024`` thread cap, allowing longer paths.

The estimator formulas below intentionally mirror the repo's own gram-based
definitions in ``discriminators.py`` (MMD: lines ~185-188; scoring rule: ~337-339)
so the loss math is unchanged and only the kernel backend differs.

NOTE: ``pysiglib`` is imported lazily so that importing ``discriminators`` does not
hard-require pySigLib until a kernel is actually constructed.
"""


def _static_kernel(kernel_type, sigma):
    """Build a pySigLib static kernel matching sigker-nsdes' choices.

    Prefer the torch_api namespace if it defines the kernel classes (to stay
    consistent with the autograd-aware torch_api.sig_kernel_gram); fall back to
    the top-level pysiglib classes otherwise.
    """
    import pysiglib
    from pysiglib import torch_api
    src = torch_api if hasattr(torch_api, "RBFKernel") else pysiglib
    if kernel_type.lower() == "rbf":
        return src.RBFKernel(sigma=sigma)
    # default / "linear"
    return src.LinearKernel()


class PySigKernel:
    """Drop-in replacement exposing only the methods sigker-nsdes calls.

    Constructed via ``get_kernel`` in ``discriminators.py``. Duck-types
    ``sigkernel.SigKernel`` for ``compute_mmd`` / ``compute_scoring_rule`` /
    ``compute_Gram``; every discriminator and the training loop are unchanged.
    """

    def __init__(self, static_kernel, dyadic_order):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order

    def compute_Gram(self, X, Y, sym=False, max_batch=128):
        # IMPORTANT: call via pysiglib.torch_api so the Gram is autograd-differentiable.
        # The top-level pysiglib.sig_kernel_gram returns a DETACHED tensor (no grad_fn),
        # which breaks training (RuntimeError: ... does not require grad).
        # lead_lag=False matches sigkernel's default (time-aug, if any, is upstream).
        from pysiglib import torch_api
        return torch_api.sig_kernel_gram(
            X, Y,
            dyadic_order=self.dyadic_order,
            static_kernel=self.static_kernel,
            max_batch=max_batch,
        )

    def compute_mmd(self, X, Y, max_batch=128):
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        Kyy = self.compute_Gram(Y, Y, sym=True, max_batch=max_batch)
        Kxy = self.compute_Gram(X, Y, sym=False, max_batch=max_batch)
        m = Kxx.shape[0]
        n = Kyy.shape[0]
        mK_XX = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
        mK_YY = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
        return mK_XX + mK_YY - 2.0 * Kxy.mean()

    def compute_scoring_rule(self, X, y, max_batch=128):
        # S(P, y) = E_{X,X'}[k(X,X')] - 2 E_X[k(X,y)]   (discriminators.py:337-339)
        # NOTE: verify the y-shape convention against sigkernel via the parity test
        # before relying on this in the scoring-rule discriminator.
        Kxx = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        Kxy = self.compute_Gram(X, y, sym=False, max_batch=max_batch)
        m = Kxx.shape[0]
        mK_XX = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
        return mK_XX - 2.0 * Kxy.mean()
