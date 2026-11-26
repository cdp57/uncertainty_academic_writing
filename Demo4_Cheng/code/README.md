# Demo 4 Qwen 4B confidence parsing fix — CPU only

The pilot shows `academic_system` has 50% confidence parse coverage (1/2 in each dataset) despite 100% answer parsing. This is a **parsing** issue only if the raw model response contains an interpretable number. The exact failed Qwen 4B raw outputs have not yet been provided, so do not assume this patch can recover every missing score.

## Install

Upload the ZIP to the **root** of `~/447_demo2/`, then:

```bash
cd ~/447_demo2
unzip -o demo4_confidence_fix.zip
python demo4_confidence_selftest.py
python demo4_selftest.py
```

`demo4_prompts.py` retains the previous Qwen `<answer>` fix, adds a conservative confidence parser for XML wrappers, plain scores, `8/10`, JSON scores, and explicit percentages. **It does not alter the experiment prompts.**

## Inspect saved Qwen 4B confidence stages first

```bash
python demo4_reparse_confidence.py --model qwen_4b
```

This prints the original confidence text for each previously unparsed result and says which scores are recoverable. Nothing is changed in dry run. Do not run while any Qwen 4B Demo 4 collection job is active.

If one or more are recoverable:

```bash
python demo4_reparse_confidence.py --model qwen_4b --apply
python demo4_export.py --allow-incomplete
```

The repair patches **only previously-null `confidence` fields**, keeping answers, raw outputs, seeds, dataset hashes, stage counts and all non-null values unchanged. A timestamped backup is created for any modified file. Your CSV export and coverage report must be regenerated afterward.

## If some are still missing

Read the printed `UNPARSED` raw responses. If there is no explicit numeric confidence value, no parser can recover one reliably. Do not fabricate a confidence score or repeatedly resubmit the unchanged job (the deterministic seed may reproduce the same response). Share these raw strings for a targeted adjustment, and only then decide whether a new targeted confidence-stage generation is necessary. Check a larger Qwen 4B pilot before the full job.
