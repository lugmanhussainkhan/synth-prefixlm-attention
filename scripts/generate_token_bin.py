from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from dataset_mixture import DATASET_NAMES, SYNTH_DOCUMENTS, iter_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TOKEN_BIN = OUTPUT_DIR / "tokens.bin"
DOCUMENT_LENGTHS_BIN = OUTPUT_DIR / "document_lengths.bin"
SEQUENCE_DOCUMENT_COUNTS_BIN = OUTPUT_DIR / "sequence_document_counts.bin"

MAX_TOKENS = None
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

pad_id = tokenizer.token_to_id("<|pad|>")
bos_id = tokenizer.token_to_id("<|bos|>")
eos_id = tokenizer.token_to_id("<|eos|>")
sep_id = tokenizer.token_to_id("<|sep|>")

texts = []
packing_pool = []
document_tokens = 0
documents_written = 0
sequences_written = 0
oversized_documents = 0
reached_token_limit = False
documents_by_source = Counter()

pbar = tqdm(total=MAX_TOKENS, desc="[*] Gathering document tokens", unit="tokens")


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

    sources = [source for source, _, _ in texts]
    queries = [query for _, query, _ in texts]
    answers = [answer for _, _, answer in texts]
    encoded_queries = tokenizer.encode_batch(queries, add_special_tokens=False)
    encoded_answers = tokenizer.encode_batch(answers, add_special_tokens=False)
    texts.clear()

    for source, encoded_query, encoded_answer in zip(sources, encoded_queries, encoded_answers):
        if source == "synth" and documents_by_source[source] >= SYNTH_DOCUMENTS:
            continue

        query_ids = encoded_query.ids
        answer_ids = encoded_answer.ids
        query_length = len(query_ids)
        answer_length = len(answer_ids)
        total_length = query_length + answer_length + 3

        if total_length > SEQ_LEN:
            oversized_documents += 1
            continue

        if MAX_TOKENS is not None and document_tokens + total_length > MAX_TOKENS:
            reached_token_limit = True
            break

        # Every persisted document is: BOS, query, SEP, answer, EOS.
        token_ids = [bos_id, *query_ids, sep_id, *answer_ids, eos_id]
        packing_pool.append((token_ids, query_length, answer_length, total_length))
        document_tokens += total_length
        documents_by_source[source] += 1
        pbar.update(total_length)

        if len(packing_pool) >= PACKING_POOL_SIZE:
            write_packing_pool()


with (
    open(TOKEN_BIN, "wb") as token_file,
    open(DOCUMENT_LENGTHS_BIN, "wb") as document_lengths_file,
    open(SEQUENCE_DOCUMENT_COUNTS_BIN, "wb") as sequence_counts_file,
):
    for source in DATASET_NAMES:
        for query, answer in iter_dataset(source, synth_documents=None):
            texts.append((source, query, answer))

            if len(texts) >= BATCH_TEXTS:
                tokenize_texts()
                if reached_token_limit:
                    break
                if source == "synth" and documents_by_source[source] >= SYNTH_DOCUMENTS:
                    break

        if not reached_token_limit:
            tokenize_texts()
        if reached_token_limit:
            break

    write_packing_pool()

pbar.close()

print(f"[*] Wrote {document_tokens:,} document tokens")
print(f"[*] Packed {documents_written:,} documents into {sequences_written:,} sequences")
for source, count in documents_by_source.items():
    print(f"[*] {source}: {count:,} documents")
print(f"[*] Skipped {oversized_documents:,} oversized documents")
if sequences_written:
    packing_efficiency = document_tokens / (sequences_written * SEQ_LEN)
    print(f"[*] Packing efficiency: {packing_efficiency:.2%}")
