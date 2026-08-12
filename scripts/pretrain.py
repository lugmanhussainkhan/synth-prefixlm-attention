import os
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("[*] Loading libraries")
import torch
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
DATA_DIR = PROJECT_ROOT.parent / "data"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TOKEN_BIN = DATA_DIR / "tokens.bin"
DOCUMENT_LENGTHS_BIN = DATA_DIR / "document_lengths.bin"
SEQUENCE_DOCUMENT_COUNTS_BIN = DATA_DIR / "sequence_document_counts.bin"
SEQ_LEN = 1024

OUTPUT_DIR = "synth-prefixlm"
RUN_NAME = "synth-prefixlm-1M-v1"
HUB_MODEL_ID = "lugman/synth-prefixlm"

special_tokens = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
]

print("[*] Loading tokenizer")
fast_tokenizer = ByteLevelBPETokenizer(
    str(VOCAB_PATH),
    str(MERGES_PATH),
)
fast_tokenizer.add_special_tokens(special_tokens)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=fast_tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    pad_token="<|pad|>",
    unk_token="<|unk|>",
    sep_token="<|sep|>"
)


class MemmapDataset(Dataset):
    def __init__(self, token_path, document_lengths_path, sequence_counts_path, seq_len=SEQ_LEN):
        self.token_path = token_path
        self.seq_len = seq_len
        token_bytes = os.path.getsize(token_path)
        bytes_per_sequence = np.dtype(np.uint16).itemsize * seq_len
        if token_bytes % bytes_per_sequence:
            raise ValueError("Token file does not contain complete packed sequences")

        self.n_chunks = token_bytes // bytes_per_sequence
        self._tokens = None

        self.sequence_document_counts = np.memmap(
            sequence_counts_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.n_chunks,),
        )

        document_length_values = os.path.getsize(document_lengths_path) // np.dtype(np.uint16).itemsize
        if document_length_values % 3:
            raise ValueError("Document length metadata must contain query, answer, and total lengths")

        self.document_lengths = np.memmap(
            document_lengths_path,
            dtype=np.uint16,
            mode="r",
            shape=(document_length_values // 3, 3),
        )
        self.document_offsets = np.zeros(self.n_chunks + 1, dtype=np.int64)
        np.cumsum(self.sequence_document_counts, out=self.document_offsets[1:])

        if self.document_offsets[-1] != len(self.document_lengths):
            raise ValueError("Sequence counts do not match the number of document length records")

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = np.memmap(
                self.token_path, dtype=np.uint16, mode="r",
                shape=(self.n_chunks * self.seq_len,)
            )
        return self._tokens

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        arr = np.asarray(self.tokens[start:start + self.seq_len], dtype=np.int64)
        ids = torch.from_numpy(arr)
        labels = torch.full_like(ids, -100)
        position_ids = torch.zeros_like(ids)
        document_ids = torch.full_like(ids, -1)
        prefix_tokens = torch.zeros(self.seq_len, dtype=torch.bool)

        first_document = self.document_offsets[idx]
        last_document = self.document_offsets[idx + 1]
        lengths = self.document_lengths[first_document:last_document]
        offset = 0

        for document_id, (query_length, answer_length, total_length) in enumerate(lengths):
            query_length = int(query_length)
            answer_length = int(answer_length)
            total_length = int(total_length)
            prefix_length = query_length + 2
            end = offset + total_length

            if total_length != query_length + answer_length + 3:
                raise ValueError("Document total length does not match its query and answer lengths")

            # BOS + query + SEP form the bidirectional prefix. Only answer +
            # EOS remain as labels; Llama shifts these labels internally.
            labels[offset + prefix_length:end] = ids[offset + prefix_length:end]
            position_ids[offset:end] = torch.arange(total_length)
            document_ids[offset:end] = document_id
            prefix_tokens[offset:offset + prefix_length] = True
            offset = end

        return {
            "input_ids": ids,
            "labels": labels,
            "position_ids": position_ids,
            "document_ids": document_ids,
            "prefix_tokens": prefix_tokens,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    position_ids = torch.stack([b["position_ids"] for b in batch])
    document_ids = torch.stack([b["document_ids"] for b in batch])
    prefix_tokens = torch.stack([b["prefix_tokens"] for b in batch])
    seq_len = input_ids.shape[1]

    valid_tokens = document_ids.ge(0)
    same_document = document_ids[:, :, None].eq(document_ids[:, None, :])
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    prefix_attention = prefix_tokens[:, :, None] & prefix_tokens[:, None, :]

    # Prefix tokens see the full prefix in their own document. Answer tokens
    # use the causal branch, and same_document makes the mask block diagonal.
    attention_mask = (
        same_document
        & valid_tokens[:, :, None]
        & valid_tokens[:, None, :]
        & (causal | prefix_attention)
    )

    padding_tokens = document_ids.eq(-1)
    padding_diagonal = padding_tokens[:, :, None] & torch.eye(seq_len, dtype=torch.bool)
    attention_mask |= padding_diagonal

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask[:, None, :, :],
    }


print("[*] Loading packed, memmap-backed dataset")
dataset = MemmapDataset(
    TOKEN_BIN,
    DOCUMENT_LENGTHS_BIN,
    SEQUENCE_DOCUMENT_COUNTS_BIN,
    seq_len=SEQ_LEN,
)
print(f"[*] Dataset ready: {len(dataset):,} packed sequences of {SEQ_LEN} tokens")

print("[*] Setting up model")
config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=144,
    intermediate_size=312,
    num_hidden_layers=6,
    num_attention_heads=4,
    head_dim=36,
    num_key_value_heads=2,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    initializer_range=0.02,
    attention_bias=False,
    mlp_bias=False,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    attn_implementation="sdpa",
)
model = LlamaForCausalLM(config)
print(f"[*] Model parameters: {model.num_parameters():,}")

print("[*] Defining training arguments")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=256,
    gradient_accumulation_steps=1,
    learning_rate=6e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    bf16=True,
    fp16=False,
    tf32=True,
    torch_compile=os.environ.get("TORCH_COMPILE", "0") == "1",
    logging_steps=50,
    logging_first_step=True,
    eval_strategy="no",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_drop_last=True,
    gradient_checkpointing=False,
    seed=42,
    remove_unused_columns=False,
    report_to="wandb",
    run_name=RUN_NAME,
    save_steps=1000,
    save_total_limit=None,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_private_repo=False,
    hub_strategy="all_checkpoints",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("[*] Starting training")
trainer.train()
