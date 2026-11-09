# LLM run data

Output of the GitHub Actions runs between 1 and 4 November 2026 (Cerebras free tier).

| Path | What it is |
|---|---|
| `results/llm_runs/<model>/<arm>/run<N>.jsonl` | One JSON line per LLM call: tokens, latency, raw response, extracted code, fitness |
| `found/<model>_<arm>_run<N>.py` | Best program of that run |

Arm A is the (1+1) feedback loop; Arm B is the no-feedback control.

| Model | Arm A | Arm B |
|---|---|---|
| gpt-oss-120b | 5 runs, complete | 5 runs, complete |
| qwen-3.8-27b | 3 runs complete, run 4 stopped at 36 calls | none |

The runs ended when the provider began returning HTTP 402. Fake-provider plumbing tests are excluded.
