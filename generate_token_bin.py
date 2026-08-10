from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import numpy as np

TOKEN_BIN    = "tokens.bin"
TRAIN_TOKENS = 5_000_000_000
SEQ_LEN      = 1024
BATCH_TEXTS  = 1000
FLUSH_EVERY  = 1_000_000

special_tokens=[
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
]

tokenizer = ByteLevelBPETokenizer(
    "./tokenizer/tokenizer-vocab.json",
    "./tokenizer/tokenizer-merges.txt",
)
tokenizer.add_special_tokens(special_tokens)

dataset = load_dataset(
    "PleIAs/SYNTH",
    split="train",
    streaming=True,
    filters=[
        ("language", "==", "en"),
        ("model", "==", "qwen-3-8b-memorization"),
        ("words", "<=", 800),
    ],
    columns=[
        "query",
        "synthetic_answer",
    ],
)

mm = np.memmap(TOKEN_BIN, dtype=np.uint16, mode="w+", shape=(TRAIN_TOKENS,))

written = 0
buffer  = []
texts   = []

pbar = tqdm(total=TRAIN_TOKENS, desc="[*] Gathering tokens", unit="tokens")

def flush():
    global written
    if not buffer:
        return False
    n = min(len(buffer), TRAIN_TOKENS - written)
    mm[written:written+n] = buffer[:n]
    written += n
    pbar.update(n)
    del buffer[:n]
    return written >= TRAIN_TOKENS

for example in dataset:
    content = (
        example["query"].strip()
        + "<|sep|>"
        + example["synthetic_answer"].strip()
    )
    texts.append(content)
    
    if len(texts) >= BATCH_TEXTS:
        encoded_batch = tokenizer.encode_batch(texts)
        texts.clear()
        for encoded in encoded_batch:
            buffer.extend(encoded.ids)
        if len(buffer) >= FLUSH_EVERY:
            if flush():
                break

if written < TRAIN_TOKENS and texts:
    encoded_batch = tokenizer.encode_batch(texts)
    for encoded in encoded_batch:
        buffer.extend(encoded.ids)

if written < TRAIN_TOKENS:
    flush()
    
pbar.close()
mm.flush()
del mm

print(f"[*] Wrote {written} tokens to {TOKEN_BIN}")