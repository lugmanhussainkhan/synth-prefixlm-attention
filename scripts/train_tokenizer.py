from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "tokenizer"
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()

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

def get_content(row):
    parts = [
        row["query"],
        row["synthetic_answer"]
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())

def get_training_corpus():
    it = iter(dataset)
    for _ in tqdm(range(50_000)):
        row = next(it)
        query = (row["query"] or "").strip()
        answer = (row["synthetic_answer"] or "").strip()
        if query:
            yield query
        if answer:
            yield answer

tokenizer.train_from_iterator(
    get_training_corpus(),
    min_frequency=2,
    vocab_size=1536,
    special_tokens=[
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|sep|>",
    ],
    show_progress=True
)

tokenizer.save_model(str(output_dir))