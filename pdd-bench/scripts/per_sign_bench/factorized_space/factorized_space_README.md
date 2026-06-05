# factorized_space/

PGMap / paired factorized scene space. Scene materialization runs via
`per_sign_benchmark.py --materialize --backends pgmap` (or `paired`); run from
`scripts/per_sign_bench/`. Standalone tools:

### `space_definition.py` – dump the space definition (axes/sizes)
```
python factorized_space/space_definition.py --save-dir out
```

### `agent_profile_bank.py` – generate the NPC profile bank (Latin-hypercube)
```
python factorized_space/agent_profile_bank.py --n 256 --output profiles.json
```

Library modules (no CLI): `index_codec.py`, `sign_placement.py`, `ego_defaults.py`,
`benchmark_runner.py`, `materialized_sampler.py`, `paired_space.py`.
