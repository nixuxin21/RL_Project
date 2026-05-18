"""Reusable helpers for the MS-AirComp IRS experiments.

The package exposes module-level APIs rather than re-exporting every helper at
the package root. Prefer explicit imports such as:

    from ms_aircomp.channel_models import build_temporal_channel_trace
    from ms_aircomp.execution_policies import choose_sparse_topk_feedback_decision

See ``ms_aircomp/README.md`` for the current module boundary. The package root
does not import submodules eagerly, so importing ``ms_aircomp`` remains cheap.
"""
