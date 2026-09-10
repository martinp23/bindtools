from typing import Optional
import numpy as np

from bindtools.speciation.numerical import getConcs


def _solve_cubic_vectorized(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    score_fn_list,
) -> np.ndarray:
    A = b / a
    B = c / a
    C = d / a

    p = B - A**2 / 3.0
    q = C - A * B / 3.0 + 2.0 * A**3 / 27.0

    D = q**2 / 4.0 + p**3 / 27.0

    N = len(a)
    candidates = np.zeros((3, N), dtype=float)

    mask_single = D > 0
    mask_three = ~mask_single

    # Solve single root cases (D > 0)
    if np.any(mask_single):
        D_s = D[mask_single]
        q_s = q[mask_single]
        A_s = A[mask_single]

        sqrt_D = np.sqrt(D_s)
        val1 = -q_s / 2.0 + sqrt_D
        val2 = -q_s / 2.0 - sqrt_D

        u = np.sign(val1) * np.abs(val1)**(1.0 / 3.0)
        v = np.sign(val2) * np.abs(val2)**(1.0 / 3.0)

        root_s = u + v - A_s / 3.0
        candidates[0, mask_single] = root_s
        candidates[1, mask_single] = root_s
        candidates[2, mask_single] = root_s

    # Solve three root cases (D <= 0)
    if np.any(mask_three):
        p_t = p[mask_three]
        q_t = q[mask_three]
        A_t = A[mask_three]

        valid = -p_t > 1e-12
        phi = np.zeros_like(p_t)
        r = np.zeros_like(p_t)
        if np.any(valid):
            p_v = p_t[valid]
            q_v = q_t[valid]
            cos_phi = (3.0 * q_v) / (2.0 * p_v * np.sqrt(-p_v / 3.0))
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi[valid] = np.arccos(cos_phi)
            r[valid] = 2.0 * np.sqrt(-p_v / 3.0)

        y0 = r * np.cos(phi / 3.0)
        y1 = r * np.cos(phi / 3.0 + 2.0 * np.pi / 3.0)
        y2 = r * np.cos(phi / 3.0 + 4.0 * np.pi / 3.0)

        candidates[0, mask_three] = y0 - A_t / 3.0
        candidates[1, mask_three] = y1 - A_t / 3.0
        candidates[2, mask_three] = y2 - A_t / 3.0

    # Evaluate residuals using the vectorized score function
    residuals = score_fn_list(candidates)

    # Penalize out of bounds candidates
    out_of_bounds = (candidates < lower - 1e-12) | (candidates > upper + 1e-12)
    residuals[out_of_bounds] += 1e10

    best_idx = np.argmin(residuals, axis=0)
    best_root = candidates[best_idx, np.arange(N)]
    return np.clip(best_root, lower, upper)


def calc_analytical_speciation(
    comp_concs: np.ndarray,
    eq_mat: np.ndarray,
    binding_params: np.ndarray,
    topology: str,
    n_comp: int,
    complex_indices: list[int],
) -> tuple[np.ndarray, bool]:
    if n_comp != 2:
        raise ValueError("Analytical fast-exchange solver requires exactly 2 components.")

    n_rows = int(np.shape(comp_concs)[0])
    n_species = int(np.shape(eq_mat)[1])
    spec_calc = np.zeros((n_rows, n_species), dtype=float)
    error = False

    h_tot = np.maximum(comp_concs[:, 0], 0.0)
    g_tot = np.maximum(comp_concs[:, 1], 0.0)
    zero_mask = (h_tot <= 0.0) | (g_tot <= 0.0)

    beta11 = 10.0 ** float(binding_params[complex_indices[0]])

    try:
        if topology == "1:1":
            if beta11 <= 0:
                spec_calc[:, 0] = h_tot
                spec_calc[:, 1] = g_tot
            else:
                first_term = h_tot + g_tot + 1.0 / max(beta11, 1e-300)
                sqrt_val = np.sqrt(np.maximum(first_term**2 - 4.0 * h_tot * g_tot, 0.0))
                hg = 0.5 * (first_term - sqrt_val)
                hg = np.clip(hg, 0.0, np.minimum(h_tot, g_tot))
                hg[zero_mask] = 0.0

                spec_calc[:, 0] = np.maximum(h_tot - hg, 0.0)
                spec_calc[:, 1] = np.maximum(g_tot - hg, 0.0)
                spec_calc[:, complex_indices[0]] = hg

        elif topology == "1:2":
            beta12 = 10.0 ** float(binding_params[complex_indices[1]])

            # Setup coefficients for cubic equation a*g^3 + b*g^2 + c*g + d = 0
            a = np.full(n_rows, beta12, dtype=float)
            b = beta11 + 2.0 * h_tot * beta12 - g_tot * beta12
            c = 1.0 + h_tot * beta11 - g_tot * beta11
            d = -g_tot

            def score_12(g_free_candidates):
                h_t = h_tot[np.newaxis, :]
                g_t = g_tot[np.newaxis, :]
                denom = np.maximum(1.0 + beta11 * g_free_candidates + beta12 * (g_free_candidates**2), 1e-300)
                h_f = h_t / denom
                hg = beta11 * h_f * g_free_candidates
                hg2 = beta12 * h_f * (g_free_candidates**2)
                r1 = np.abs(h_t - (h_f + hg + hg2))
                r2 = np.abs(g_t - (g_free_candidates + hg + 2.0 * hg2))
                return r1 + r2

            g_free = _solve_cubic_vectorized(a, b, c, d, np.zeros(n_rows), g_tot, score_12)
            denom = np.maximum(1.0 + beta11 * g_free + beta12 * (g_free**2), 1e-300)
            h_free = h_tot / denom
            hg = beta11 * h_free * g_free
            hg2 = beta12 * h_free * (g_free**2)

            # Apply zero/negative overrides
            h_free = np.maximum(h_free, 0.0)
            g_free = np.maximum(g_free, 0.0)
            hg = np.maximum(hg, 0.0)
            hg2 = np.maximum(hg2, 0.0)

            h_free[zero_mask] = h_tot[zero_mask]
            g_free[zero_mask] = g_tot[zero_mask]
            hg[zero_mask] = 0.0
            hg2[zero_mask] = 0.0

            spec_calc[:, 0] = h_free
            spec_calc[:, 1] = g_free
            spec_calc[:, complex_indices[0]] = hg
            spec_calc[:, complex_indices[1]] = hg2

        elif topology == "2:1":
            beta21 = 10.0 ** float(binding_params[complex_indices[1]])

            # Setup coefficients for cubic equation a*h^3 + b*h^2 + c*h + d = 0
            a = np.full(n_rows, beta21, dtype=float)
            b = beta11 + 2.0 * g_tot * beta21 - h_tot * beta21
            c = 1.0 + g_tot * beta11 - h_tot * beta11
            d = -h_tot

            def score_21(h_free_candidates):
                h_t = h_tot[np.newaxis, :]
                g_t = g_tot[np.newaxis, :]
                denom = np.maximum(1.0 + beta11 * h_free_candidates + beta21 * (h_free_candidates**2), 1e-300)
                g_f = g_t / denom
                hg = beta11 * h_free_candidates * g_f
                h2g = beta21 * (h_free_candidates**2) * g_f
                r1 = np.abs(h_t - (h_free_candidates + hg + 2.0 * h2g))
                r2 = np.abs(g_t - (g_f + hg + h2g))
                return r1 + r2

            h_free = _solve_cubic_vectorized(a, b, c, d, np.zeros(n_rows), h_tot, score_21)
            denom = np.maximum(1.0 + beta11 * h_free + beta21 * (h_free**2), 1e-300)
            g_free = g_tot / denom
            hg = beta11 * h_free * g_free
            h2g = beta21 * (h_free**2) * g_free

            # Apply zero/negative overrides
            h_free = np.maximum(h_free, 0.0)
            g_free = np.maximum(g_free, 0.0)
            hg = np.maximum(hg, 0.0)
            h2g = np.maximum(h2g, 0.0)

            h_free[zero_mask] = h_tot[zero_mask]
            g_free[zero_mask] = g_tot[zero_mask]
            hg[zero_mask] = 0.0
            h2g[zero_mask] = 0.0

            spec_calc[:, 0] = h_free
            spec_calc[:, 1] = g_free
            spec_calc[:, complex_indices[0]] = hg
            spec_calc[:, complex_indices[1]] = h2g

        else:
            raise ValueError(f"Unsupported analytical topology: {topology}")
    except Exception:
        error = True
        # Fallback to numerical solver row-by-row
        for ii in range(n_rows):
            try:
                spec_calc[ii, :] = getConcs(eq_mat, comp_concs[ii], binding_params)
            except Exception:
                spec_calc[ii, :] = 0.0

    return spec_calc, error


def _infer_simple_fast_exchange_topology(eq_mat: np.ndarray, n_comp: int) -> Optional[tuple[str, list[int]]]:
    if not isinstance(eq_mat, np.ndarray) or eq_mat.ndim != 2:
        return None
    if n_comp != 2 or eq_mat.shape[0] != 2 or eq_mat.shape[1] <= n_comp:
        return None

    bound_indices = list(range(n_comp, eq_mat.shape[1]))
    sig_to_idx = {}
    for idx in bound_indices:
        a = eq_mat[0, idx]
        b = eq_mat[1, idx]
        if not np.isfinite(a) or not np.isfinite(b):
            return None
        if not np.isclose(a, round(a)) or not np.isclose(b, round(b)):
            return None
        sig = (int(round(a)), int(round(b)))
        if sig in sig_to_idx:
            return None
        sig_to_idx[sig] = idx

    sigs = set(sig_to_idx.keys())
    if len(bound_indices) == 1 and sigs == {(1, 1)}:
        return "1:1", [sig_to_idx[(1, 1)]]
    if len(bound_indices) == 2 and sigs == {(1, 1), (1, 2)}:
        return "1:2", [sig_to_idx[(1, 1)], sig_to_idx[(1, 2)]]
    if len(bound_indices) == 2 and sigs == {(1, 1), (2, 1)}:
        return "2:1", [sig_to_idx[(1, 1)], sig_to_idx[(2, 1)]]
    return None

