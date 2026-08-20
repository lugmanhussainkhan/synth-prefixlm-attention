from itertools import islice
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from dataset_mixture import DATASET_NAMES, iter_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "tokenizer"
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
TOKENIZER_DOCUMENTS_PER_DATASET = 10_000


def get_training_corpus():
    for name in DATASET_NAMES:
        # Bin generation applies FLAN's 1,500-document per-task cap. Applying
        # that cap here can exhaust every task in a large subset before this
        # global 10,000-document sample is full, leaving the iterator scanning
        # millions of rows without yielding anything.
        documents = iter_dataset(name, flan_documents_per_type=None)
        documents = islice(documents, TOKENIZER_DOCUMENTS_PER_DATASET)
        for query, answer in tqdm(
            documents,
            total=None if name == "no_robots" else TOKENIZER_DOCUMENTS_PER_DATASET,
            desc=f"[*] Reading {name}",
        ):
            yield query
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
    show_progress=True,
)

tokenizer.save_model(str(output_dir))
