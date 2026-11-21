## Follow-up: numerical review → fixes on branch

Thanks for the careful main-vs-branch analysis. We reproduced the **shared-offset** stationary failure on this Mac (expanded forms blow up at offset ≈ 1e9; direct `(x−y)` loops stay fine). Against a float64 pure oracle the blocked-Cholesky issue was milder than the mpmath table, but the recommended hybrid still matches the better path, so we landed it.

**Landed on `cip000E-numpy-blas-primitives`:**

1. **Stationary:** subtract a shared column mean before `lengthscale_grads_sqdiff` / `gradients_X_from_tmp` (translation-invariant; restores agreement with direct-difference loops at offset `1e9`).
2. **Blocked Cholesky:** when `N > block_size`, run unblocked `cholesky_vjp` on the updated diagonal block after `Dbar -= tril(Cbar.T @ C)`; keep the symbolic reverse when the whole factor fits in one block.
3. **`symmetrify`:** cache triangle indices; default block size 64.
4. **Triang maps:** cache linear tril indices + `np.take`.
5. **Tests:** offset `1e9` stationary parity; RBF + `1e-8` jitter multi-block Cholesky vs pure.
6. **`coregionalize_K`:** treating the Mac ~3× as host-specific until a Linux re-run (as you suggested).

CIP-000E Motivation updated with this follow-up. Happy to take a Linux multicore soak next if useful.
