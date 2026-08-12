from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TOKEN_BIN = OUTPUT_DIR / "tokens.bin"
DOCUMENT_LENGTHS_BIN = OUTPUT_DIR / "document_lengths.bin"
SEQUENCE_DOCUMENT_COUNTS_BIN = OUTPUT_DIR / "sequence_document_counts.bin"

TRAIN_TOKENS = 5_000_000_000
SEQ_LEN = 1024
BATCH_TEXTS = 1000
PACKING_POOL_SIZE = 10_000

special_tokens = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
]

tokenizer = ByteLevelBPETokenizer(
    str(VOCAB_PATH),
    str(MERGES_PATH),
)
tokenizer.add_special_tokens(special_tokens)

dataset = load_dataset(
    "PleIAs/SYNTH",
    split="train",
    streaming=True,
    filters=[
        ("language", "==", "en"),
        ("model", "==", "qwen-3-8b-memorization"),
        ("words", "<", 800),
    ],
    columns=[
        "query",
        "synthetic_answer",
    ],
)

pad_id = tokenizer.token_to_id("<|pad|>")
bos_id = tokenizer.token_to_id("<|bos|>")
eos_id = tokenizer.token_to_id("<|eos|>")
sep_id = tokenizer.token_to_id("<|sep|>")

texts = []
packing_pool = []
document_tokens = 0
documents_written = 0
sequences_written = 0
empty_documents = 0
oversized_documents = 0
reached_token_limit = False

pbar = tqdm(total=TRAIN_TOKENS, desc="[*] Gathering document tokens", unit="tokens")


def write_packing_pool():
    global documents_written, sequences_written

    if not packing_pool:
        return

    # Best-fit-decreasing packing over a bounded pool keeps documents intact
    # without holding the full multi-billion-token dataset in memory.
    open_bins = [[] for _ in range(SEQ_LEN + 1)]
    packed_documents = []

    for document in sorted(packing_pool, key=lambda item: item[3], reverse=True):
        total_length = document[3]
        selected_bin = None
        selected_remaining = None

        for remaining in range(total_length, SEQ_LEN + 1):
            if open_bins[remaining]:
                selected_bin = open_bins[remaining].pop()
                selected_remaining = remaining
                break

        if selected_bin is None:
            selected_bin = len(packed_documents)
            selected_remaining = SEQ_LEN
            packed_documents.append([])

        packed_documents[selected_bin].append(document)
        new_remaining = selected_remaining - total_length
        if new_remaining:
            open_bins[new_remaining].append(selected_bin)

    token_rows = []
    document_lengths = []
    sequence_document_counts = []

    for documents in packed_documents:
        row = np.full(SEQ_LEN, pad_id, dtype=np.uint16)
        offset = 0

        for token_ids, query_length, answer_length, total_length in documents:
            row[offset:offset + total_length] = token_ids
            document_lengths.append((query_length, answer_length, total_length))
            offset += total_length

        token_rows.append(row)
        sequence_document_counts.append(len(documents))

    np.stack(token_rows).tofile(token_file)
    np.asarray(document_lengths, dtype=np.uint16).tofile(document_lengths_file)
    np.asarray(sequence_document_counts, dtype=np.uint16).tofile(sequence_counts_file)

    documents_written += len(document_lengths)
    sequences_written += len(token_rows)
    packing_pool.clear()


def tokenize_texts():
    global document_tokens, oversized_documents, reached_token_limit

    if not texts:
        return

    queries = [query for query, _ in texts]
    answers = [answer for _, answer in texts]
    encoded_queries = tokenizer.encode_batch(queries, add_special_tokens=False)
    encoded_answers = tokenizer.encode_batch(answers, add_special_tokens=False)
    texts.clear()

    for encoded_query, encoded_answer in zip(encoded_queries, encoded_answers):
        query_ids = encoded_query.ids
        answer_ids = encoded_answer.ids
        query_length = len(query_ids)
        answer_length = len(answer_ids)
        total_length = query_length + answer_length + 3

        if total_length > SEQ_LEN:
            oversized_documents += 1
            continue

        if document_tokens + total_length > TRAIN_TOKENS:
            reached_token_limit = True
            break

        # Every persisted document is: BOS, query, SEP, answer, EOS.
        token_ids = [bos_id, *query_ids, sep_id, *answer_ids, eos_id]
        packing_pool.append((token_ids, query_length, answer_length, total_length))
        document_tokens += total_length
        pbar.update(total_length)

        if len(packing_pool) >= PACKING_POOL_SIZE:
            write_packing_pool()


with (
    open(TOKEN_BIN, "wb") as token_file,
    open(DOCUMENT_LENGTHS_BIN, "wb") as document_lengths_file,
    open(SEQUENCE_DOCUMENT_COUNTS_BIN, "wb") as sequence_counts_file,
):
    for example in dataset:
        query = example["query"].strip() if example["query"] else ""
        answer = example["synthetic_answer"].strip() if example["synthetic_answer"] else ""

        if not query or not answer:
            empty_documents += 1
            continue

        texts.append((query, answer))

        if len(texts) >= BATCH_TEXTS:
            tokenize_texts()
            if reached_token_limit:
                break

    if not reached_token_limit:
        tokenize_texts()

    write_packing_pool()

pbar.close()

print(f"[*] Wrote {document_tokens:,} document tokens")
print(f"[*] Packed {documents_written:,} documents into {sequences_written:,} sequences")
print(f"[*] Skipped {empty_documents:,} empty and {oversized_documents:,} oversized documents")
if sequences_written:
    packing_efficiency = document_tokens / (sequences_written * SEQ_LEN)
    print(f"[*] Packing efficiency: {packing_efficiency:.2%}")
