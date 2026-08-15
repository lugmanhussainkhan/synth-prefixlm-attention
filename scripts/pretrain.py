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

# Set USE_FLEX=1 to use FlexAttention block masks instead of a dense boolean
# mask. Block-sparse attention skips fully-masked 128x128 blocks entirely, so
# with ~350-token documents packed into 1024-token sequences this is roughly a
# 3x attention speedup on top of removing the host-side mask build.
USE_FLEX = os.environ.get("USE_FLEX", "0") == "1"

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
        position_ids = torch.zeros(self.seq_len, dtype=torch.long)
        # int16 rather than int64: this is transferred to the GPU every step and
        # only needs to hold a per-sequence document index.
        document_ids = torch.full((self.seq_len,), -1, dtype=torch.int16)
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
    # No mask construction here. The [B, 1, L, L] boolean mask is 268 MB per
    # batch at B=256/L=1024; building it on CPU and pinning + transferring it
    # every step was the single largest cost in the original script. We ship
    # [B, L] metadata (512 KB) instead and expand on device.
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in ("input_ids", "labels", "position_ids", "document_ids", "prefix_tokens")
    }


_STATIC_MASKS = {}


def _static_masks(seq_len, device):
    key = (seq_len, device)
    if key not in _STATIC_MASKS:
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        _STATIC_MASKS[key] = (causal, eye)
    return _STATIC_MASKS[key]


def build_attention_mask(document_ids, prefix_tokens):
    """Dense boolean prefix-LM mask, built on device. True == attend."""
    seq_len = document_ids.shape[1]
    causal, eye = _static_masks(seq_len, document_ids.device)

    valid_tokens = document_ids.ge(0)
    same_document = document_ids[:, :, None].eq(document_ids[:, None, :])
    prefix_attention = prefix_tokens[:, :, None] & prefix_tokens[:, None, :]

    # Prefix tokens see the full prefix in their own document. Answer tokens
    # use the causal branch, and same_document makes the mask block diagonal.
    attention_mask = same_document
    attention_mask &= valid_tokens[:, :, None]
    attention_mask &= valid_tokens[:, None, :]
    attention_mask &= causal | prefix_attention

    # Padding rows attend to themselves so softmax never sees an all-masked row.
    attention_mask |= (~valid_tokens)[:, :, None] & eye
    return attention_mask[:, None, :, :]


if USE_FLEX:
    from torch.nn.attention.flex_attention import create_block_mask

    _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)

    def build_attention_mask(document_ids, prefix_tokens):  # noqa: F811
        batch_size, seq_len = document_ids.shape

        def mask_mod(b, h, q_idx, kv_idx):
            valid = document_ids[b, q_idx] >= 0
            same_document = document_ids[b, q_idx] == document_ids[b, kv_idx]
            causal = q_idx >= kv_idx
            prefix = prefix_tokens[b, q_idx] & prefix_tokens[b, kv_idx]
            return torch.where(
                valid,
                same_document & (causal | prefix),
                q_idx == kv_idx,
            )

        return _compiled_create_block_mask(
            mask_mod, batch_size, None, seq_len, seq_len,
            device=document_ids.device,
        )


class PrefixLMTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        document_ids = inputs.pop("document_ids")
        prefix_tokens = inputs.pop("prefix_tokens")
        inputs["attention_mask"] = build_attention_mask(document_ids, prefix_tokens)
        return inputs


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
    # head_dim MUST be a multiple of 8 or SDPA silently falls back to the math
    # backend, which materializes the full [B, H, L, L] fp32 score matrix and
    # keeps it for backward (~4.3 GB per layer at B=256, L=1024). Llama allows
    # head_dim != hidden_size // num_attention_heads, so 32 keeps hidden_size
    # at 144 and only costs ~10k parameters.
    head_dim=32,
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
    attn_implementation="flex_attention" if USE_FLEX else "sdpa",
)
model = LlamaForCausalLM(config)
print(f"[*] Model parameters: {model.num_parameters():,}")
print(f"[*] Attention implementation: {model.config._attn_implementation}")

# ~19,000 optimizer steps at 262k tokens/step. WSD: 2% warmup, flat, 20% decay.
total_steps = len(dataset) // 256
warmup_steps = max(1, int(0.02 * total_steps))
decay_steps = int(0.20 * total_steps)
stable_steps = total_steps - warmup_steps - decay_steps

print("[*] Defining training arguments")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=256,
    gradient_accumulation_steps=1,
    learning_rate=6e-4,
    lr_scheduler_type="warmup_stable_decay",
    lr_scheduler_kwargs={
        "num_stable_steps": stable_steps,
        "num_decay_steps": decay_steps,
        "min_lr_ratio": 0.0,
    },
    warmup_steps=warmup_steps,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    bf16=True,
    fp16=False,
    tf32=True,
    torch_compile=os.environ.get("TORCH_COMPILE", "1") == "1",
    logging_steps=50,
    logging_first_step=True,
    eval_strategy="no",
    dataloader_num_workers=6,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,
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

trainer = PrefixLMTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
    # Without this, Trainer._save() skips the tokenizer branch and every
    # checkpoint is pushed without tokenizer.json / tokenizer_config.json /
    # special_tokens_map.json. On transformers < 4.46 use tokenizer=tokenizer.
    processing_class=tokenizer,
)

print("[*] Starting training")
trainer.train()

print("[*] Saving final model and tokenizer")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
trainer.push_to_hub(commit_message="End of training")