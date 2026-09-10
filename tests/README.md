# tests

Version 3.2

The suite needs no hardware. Run it from the checkout root:

```bash
pytest
```

The Zygo client tests run against the built-in simulation. The modal-solve
and wavefront tests drive the solvers against simulated mirrors with a known
aberration, so they take a few minutes.

## Files

| File | What it checks |
| --- | --- |
| `test_fit_profile.py` | Loop CSV reading and the profile fit, including the endpoint constraint and monotonic loading |
| `test_linearized_compensation.py` | Compensated sweeps and step plans on the `dm_d` profile: homing, anchors, clamping |
| `test_im_store.py` | Saving, loading, aligning and rejecting control matrices |
| `test_matrix_remap.py` | Re-labelling a matrix after a rewire |
| `test_adaptive_model.py` | Descriptor validation, the quantile bound and the wait rule |
| `test_adaptive_analysis.py` | Residuals from a calibration session and the descriptor files it writes |
| `test_modal.py` | 2N+1 and N+2 solves on simulated mirrors: convergence, guards, probe calibration, rails, staged hand-over |
| `test_second_moment.py` | The second moment is exactly quadratic in the aberration, and the fixed-ROI scoring |
| `test_phase_retrieval.py` | Round trips on rendered spots, the known ambiguities, robustness and conventions |
| `test_wavefront.py` | The retrieval-driven controller: unit chain, subspace, sign resolution, discarded fits |
| `test_influence_pipeline.py` | Actuator grading, surface file selection and the noise projection |
| `test_mx_client.py` | Surface export, probe points, routing and transport of the Mx client |
| `test_mx_agent_link.py` | The agent link against a fake agent: connect, measure, failures |
| `test_metrology.py` | Waves to nanometres conversion |
