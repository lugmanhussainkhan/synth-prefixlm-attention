import os

# Each worker process tokenizes on a single thread; parallelism comes from
# running many workers, so Rayon must not also fan out inside each one.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import multiprocessing as mp
import queue as queue_module
import sys
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_mixture import build_work_units, iter_unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
OUTPUT_DIR = PROJECT_ROOT / "data"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TOKEN_BIN = OUTPUT_DIR / "tokens.bin"
DOCUMENT_LENGTHS_BIN = OUTPUT_DIR / "document_lengths.bin"
SEQUENCE_DOCUMENT_COUNTS_BIN = OUTPUT_DIR / "sequence_document_counts.bin"

MAX_TOKENS = None
SEQ_LEN = 1024
PACKING_POOL_SIZE = 10_000

# Streaming is network bound, so oversubscribing cores is the point: a worker
# blocked on an HTTP range request costs nothing while its peers decode.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(24, (os.cpu_count() or 4) * 3)))
SYNTH_SHARDS = int(os.environ.get("SYNTH_SHARDS", 8))
QUEUE_DEPTH = int(os.environ.get("QUEUE_DEPTH", NUM_WORKERS * 4))

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
]


def build_tokenizer():
    tokenizer = ByteLevelBPETokenizer(str(VOCAB_PATH), str(MERGES_PATH))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


# --------------------------------------------------------------------------
# Worker: stream one unit, tokenize it, ship compact arrays to the writer
# --------------------------------------------------------------------------


def worker(units, result_queue, seq_len):
    tokenizer = build_tokenizer()
    inner = getattr(tokenizer, "_tokenizer", None)
    # encode_batch_fast skips the offset/word-index bookkeeping the packer
    # never reads; roughly 1.3x on top of encode_batch.
    raw_encode = getattr(inner, "encode_batch_fast", None) or tokenizer.encode_batch

    def encode_batch(texts):
        return raw_encode(texts, add_special_tokens=False)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    sep_id = tokenizer.token_to_id("<|sep|>")

    current_unit = None
    try:
        for unit in units:
            current_unit = unit
            source = unit[0]
            oversized = 0
            for queries, answers in iter_unit(unit):
                encoded_queries = encode_batch(queries)
                encoded_answers = encode_batch(answers)

                flat = []
                meta = []
                for encoded_query, encoded_answer in zip(encoded_queries, encoded_answers):
                    query_ids = encoded_query.ids
                    answer_ids = encoded_answer.ids
                    query_length = len(query_ids)
                    answer_length = len(answer_ids)
                    total_length = query_length + answer_length + 3

                    if total_length > seq_len:
                        oversized += 1
                        continue

                    # Every persisted document is: BOS, query, SEP, answer, EOS.
                    flat.append(bos_id)
                    flat.extend(query_ids)
                    flat.append(sep_id)
                    flat.extend(answer_ids)
                    flat.append(eos_id)
                    meta.append((query_length, answer_length, total_length))

                if meta:
                    result_queue.put(
                        (
                            source,
                            np.asarray(flat, dtype=np.uint16),
                            np.asarray(meta, dtype=np.uint16),
                            0,
                        )
                    )

            if oversized:
                result_queue.put((source, None, None, oversized))
    except Exception as error:  # surface worker failures instead of hanging
        result_queue.put(("__error__", repr(error), current_unit, 0))
    finally:
        result_queue.put(None)


# --------------------------------------------------------------------------
# Writer: best-fit-decreasing packing, unchanged semantics
# --------------------------------------------------------------------------


class Writer:
    def __init__(self, token_file, document_lengths_file, sequence_counts_file, pad_id):
        self.token_file = token_file
        self.document_lengths_file = document_lengths_file
        self.sequence_counts_file = sequence_counts_file
        self.pad_id = pad_id
        self.packing_pool = []
        self.documents_written = 0
        self.sequences_written = 0

    def add(self, flat, meta):
        offset = 0
        for query_length, answer_length, total_length in meta:
            total_length = int(total_length)
            self.packing_pool.append(
                (
                    flat[offset:offset + total_length],
                    int(query_length),
                    int(answer_length),
                    total_length,
                )
            )
            offset += total_length

    def flush(self):
        packing_pool = self.packing_pool
        if not packing_pool:
            return

        # Best-fit-decreasing packing over a bounded pool keeps documents
        # intact without holding the full multi-billion-token dataset in
        # memory.
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

        rows = np.full((len(packed_documents), SEQ_LEN), self.pad_id, dtype=np.uint16)
        document_lengths = []
        sequence_document_counts = np.empty(len(packed_documents), dtype=np.uint16)

        for index, documents in enumerate(packed_documents):
            offset = 0
            row = rows[index]
            for token_ids, query_length, answer_length, total_length in documents:
                row[offset:offset + total_length] = token_ids
                document_lengths.append((query_length, answer_length, total_length))
                offset += total_length
            sequence_document_counts[index] = len(documents)

        rows.tofile(self.token_file)
        np.asarray(document_lengths, dtype=np.uint16).tofile(self.document_lengths_file)
        sequence_document_counts.tofile(self.sequence_counts_file)

        self.documents_written += len(document_lengths)
        self.sequences_written += len(rows)
        packing_pool.clear()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    pad_id = tokenizer.token_to_id("<|pad|>")

    units = build_work_units(synth_shards=SYNTH_SHARDS)
    num_workers = max(1, min(NUM_WORKERS, len(units)))

    # Round-robin so the fourteen FLAN subsets and the SYNTH shards spread
    # across workers instead of queueing behind one another.
    assignments = [[] for _ in range(num_workers)]
    for index, unit in enumerate(units):
        assignments[index % num_workers].append(unit)

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=QUEUE_DEPTH)
    processes = [
        context.Process(target=worker, args=(assignment, result_queue, SEQ_LEN), daemon=True)
        for assignment in assignments
        if assignment
    ]

    document_tokens = 0
    oversized_documents = 0
    documents_by_source = Counter()
    tokens_by_source = Counter()
    reached_token_limit = False
    finished = 0

    progress_start = monotonic()
    pbar = tqdm(
        total=MAX_TOKENS,
        desc=f"[*] Gathering tokens ({len(processes)} workers)",
        unit="tokens",
        bar_format="{desc}: {n_fmt}{unit} [{elapsed}{postfix}]",
    )

    for process in processes:
        process.start()

    try:
        with (
            open(TOKEN_BIN, "wb") as token_file,
            open(DOCUMENT_LENGTHS_BIN, "wb") as document_lengths_file,
            open(SEQUENCE_DOCUMENT_COUNTS_BIN, "wb") as sequence_counts_file,
        ):
            writer = Writer(token_file, document_lengths_file, sequence_counts_file, pad_id)

            while finished < len(processes):
                try:
                    item = result_queue.get(timeout=60)
                except queue_module.Empty:
                    if not any(process.is_alive() for process in processes):
                        break
                    continue

                if item is None:
                    finished += 1
                    continue

                source, flat, meta, oversized = item

                if source == "__error__":
                    pbar.write(f"[!] worker failed on {meta}: {flat}")
                    continue

                if flat is None:
                    oversized_documents += oversized
                    continue

                batch_tokens = int(flat.size)

                if MAX_TOKENS is not None and document_tokens + batch_tokens > MAX_TOKENS:
                    keep = 0
                    used = 0
                    for query_length, answer_length, total_length in meta:
                        if document_tokens + used + int(total_length) > MAX_TOKENS:
                            break
                        used += int(total_length)
                        keep += 1
                    meta = meta[:keep]
                    flat = flat[:used]
                    batch_tokens = used
                    reached_token_limit = True

                if len(meta):
                    writer.add(flat, meta)
                    document_tokens += batch_tokens
                    documents_by_source[source] += len(meta)
                    tokens_by_source[source] += batch_tokens
                    pbar.update(batch_tokens)

                    if len(writer.packing_pool) >= PACKING_POOL_SIZE:
                        writer.flush()

                elapsed = max(monotonic() - progress_start, 1e-9)
                pbar.set_postfix_str(
                    f"{document_tokens / elapsed:,.0f} avg tokens/s", refresh=False
                )

                if reached_token_limit:
                    break

            writer.flush()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        pbar.close()

    print(f"[*] Wrote {document_tokens:,} document tokens")
    print(
        f"[*] Packed {writer.documents_written:,} documents "
        f"into {writer.sequences_written:,} sequences"
    )
    for source, count in documents_by_source.most_common():
        print(f"[*] {source}: {count:,} documents ({tokens_by_source[source]:,} tokens)")
    print(f"[*] Skipped {oversized_documents:,} oversized documents")
    if writer.sequences_written:
        packing_efficiency = document_tokens / (writer.sequences_written * SEQ_LEN)
        print(f"[*] Packing efficiency: {packing_efficiency:.2%}")


if __name__ == "__main__":
    main()
