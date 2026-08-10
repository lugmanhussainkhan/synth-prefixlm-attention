from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

tokenizer = ByteLevelBPETokenizer()

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

def get_content(row):
    parts = [
        row["query"],
        row["synthetic_answer"]
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())

def get_training_corpus():
    it = iter(dataset)
    for example in tqdm(range(50_000), desc="Feeding examples to tokenizer"):
        yield get_content(next(it))

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

tokenizer.save_model(".", "droplet_synth_tokenizer")