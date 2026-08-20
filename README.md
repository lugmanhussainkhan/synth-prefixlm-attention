# Prefix LM Trainer

The training data is streamed from Hugging Face, filtered into query/answer
pairs, tokenized, and packed directly into binary files. No cleaned dataset is
materialized on disk.

## Dataset mixture

| Dataset | Selection | Cap |
| --- | --- | ---: |
| HuggingFaceH4/no_robots | All train and test rows except `Chat` and `Coding`; first user/assistant turn, with an optional system prompt prepended | All eligible rows |
| tasksource/tasksource-instruct-v0 | Reference pipeline's curated task whitelist | 10,000 documents per task |
| Open-Orca/FLAN | Reference pipeline's direct and CoT subsets | 1,500 documents per subset/task pair |
| PleIAs/SYNTH | English, `qwen-3-8b-memorization`, fewer than 800 words | 4,000,000 documents |
| MegaScience/TextbookReasoning | Biology and medicine; synthetic answer plus the reference answer for non-proof questions | All eligible rows |

The shared selection and filtering logic lives in
`scripts/dataset_mixture.py`. Both data scripts use it:

```bash
uv run python scripts/train_tokenizer.py
uv run python scripts/generate_token_bin.py
```

The bin generator writes `data/tokens.bin`, `data/document_lengths.bin`, and
`data/sequence_document_counts.bin`. Set `MAX_TOKENS` in
`scripts/generate_token_bin.py` if a token cap is preferred in addition to the
document caps above.
