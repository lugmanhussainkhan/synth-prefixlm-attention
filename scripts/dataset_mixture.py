from collections import Counter

from datasets import load_dataset
import pyarrow as pa
import pyarrow.dataset as ds


DATASET_NAMES = (
    "no_robots",
    "tasksource",
    "flan",
    "synth",
    "textbook",
)

TASKSOURCE_DOCUMENTS_PER_TYPE = 10_000
FLAN_DOCUMENTS_PER_TYPE = 1_500
SYNTH_DOCUMENTS = 4_000_000
SHUFFLE_BUFFER_SIZE = 10_000
SEED = 42

# Rows are read in column-wise batches instead of one Python dict per row.
# Arrow hands back plain lists of strings, which is what encode_batch wants.
READ_BATCH_SIZE = 1_000

# A subset is abandoned once this many rows in a row have been scanned without
# a single document surviving the per-task cap. Without it the FLAN iterators
# stream every remaining row of a multi-hundred-GB shard set only to discard
# it, which is what made bin generation effectively unbounded.
FLAN_PATIENCE_ROWS = 250_000

# Optional hard ceiling per FLAN subset. None keeps the per-task cap as the
# only limit; set an integer to bound a subset's contribution outright.
FLAN_DOCUMENTS_PER_SUBSET = None

# The default remote Parquet range is 32 MiB. Smaller coalesced ranges with a
# deeper prefetch queue overlap the next HTTP fetch with the current decode,
# instead of alternating between one big blocking download and an idle socket.
PARQUET_SCAN_OPTIONS = ds.ParquetFragmentScanOptions(
    cache_options=pa.CacheOptions(
        prefetch_limit=4,
        range_size_limit=64 << 20,
    ),
)

FLAN_SUBSETS = (
    "dialog_fsopt_data",
    "flan_fsopt_data",
    "flan_fsnoopt_data",
    "niv2_fsopt_data",
    "t0_fsopt_data",
    "t0_fsnoopt_data",
    "dialog_zsopt_data",
    "flan_zsopt_data",
    "flan_zsnoopt_data",
    "niv2_zsopt_data",
    "t0_zsopt_data",
    "t0_zsnoopt_data",
    "cot_fsopt_data",
    "cot_zsopt_data",
)

# This is the curated task set used by the reference data pipeline. Keeping the
# selection here makes this repository self-contained without copying its
# intermediate parquet generation stage.
TASKSOURCE_TASKS = {
    "CONDAQA",
    "ConTRoL-nli",
    "FLUTE",
    "HatemojiBuild",
    "I2D2",
    "MOH",
    "MedQA-USMLE-4-options-hf",
    "PARARULE-Plus",
    "SpaceNLI",
    "Touche23-ValueEval",
    "TroFi",
    "VUAC",
    "WANLI",
    "acronym_identification",
    "add_one_rte",
    "arct",
    "autotnli",
    "avicenna",
    "balanced-copa",
    "banking77",
    "breaking_nli",
    "citation_intent",
    "cladder",
    "clcd-english",
    "cloth",
    "cnli",
    "com2sense",
    "commonsense_qa_2.0",
    "conceptrules_v2",
    "conj_nli",
    "conll2000",
    "conll2003/chunk_tags",
    "conll2003/ner_tags",
    "conll2003/pos_tags",
    "corr2cause",
    "counterfactually-augmented-imdb",
    "counterfactually-augmented-snli",
    "crowdflower/airline-sentiment",
    "crowdflower/political-media-audience",
    "crowdflower/political-media-bias",
    "crowdflower/political-media-message",
    "crowdflower/sentiment_nuclear_power",
    "crowdflower/text_emotion",
    "cycic_classification",
    "cycic_multiplechoice",
    "dadc-limit-nli",
    "defeasible-nli/atomic",
    "defeasible-nli/snli",
    "dgen",
    "dialogue_nli",
    "discosense",
    "disrpt/eng.dep.scidtb",
    "dnc",
    "dnd_style_intents",
    "dynahate",
    "dynasent/dynabench.dynasent.r1.all/r1",
    "dynasent/dynabench.dynasent.r2.all/r2",
    "e-CARE",
    "equate",
    "esci",
    "fever-evidence-related/mwong--fever-related",
    "few-nerd/supervised",
    "fig-qa",
    "folio",
    "fool-me-twice",
    "fracas",
    "gen_debiased_nli/mnli_par_z",
    "gen_debiased_nli/mnli_seq_z",
    "gen_debiased_nli/mnli_z_aug",
    "gen_debiased_nli/snli_par_z",
    "gen_debiased_nli/snli_seq_z",
    "gen_debiased_nli/snli_z_aug",
    "hate_speech18",
    "hate_speech_offensive",
    "help-nli",
    "hlgd",
    "implicatures",
    "implicit-hate-stg1",
    "jnlpba/jnlpba",
    "language-identification",
    "lex_glue/ledgar",
    "lexical_relation_classification/BLESS",
    "lexical_relation_classification/CogALexV",
    "lexical_relation_classification/EVALution",
    "lingnli",
    "logical-fallacy",
    "logiqa",
    "logiqa-2.0-nli",
    "lonli",
    "lsat_qa/all",
    "mbib-base/cognitive-bias",
    "mbib-base/gender-bias",
    "mbib-base/hate-speech",
    "mbib-base/political-bias",
    "mbib-base/racial-bias",
    "medmcqa",
    "mindgames",
    "monli",
    "monotonicity-entailment",
    "moral_stories/full",
    "mpe",
    "mutual",
    "nan-nli/joey234--nan-nli",
    "natural-language-satisfiability",
    "naturallogic",
    "ncbi_disease/ncbi_disease",
    "nli-veridicality-transitivity",
    "nli_fever",
    "onestop_qa",
    "ontonotes_english/SpeedOfMagic--ontonotes_english",
    "open_question_type",
    "parade",
    "pragmeval/emergent",
    "pragmeval/gum",
    "pragmeval/mrda",
    "pragmeval/pdtb",
    "pragmeval/sarcasm",
    "pragmeval/stac",
    "pragmeval/switchboard",
    "pragmeval/verifiability",
    "probability_words_nli/reasoning_1hop",
    "probability_words_nli/reasoning_2hop",
    "probability_words_nli/usnli",
    "propsegment/nli",
    "prost",
    "puzzte",
    "quote-repetition",
    "race-c",
    "recast/recast_factuality",
    "recast/recast_megaveridicality",
    "recast/recast_ner",
    "recast/recast_puns",
    "recast/recast_sentiment",
    "recast/recast_verbcorner",
    "recast/recast_verbnet",
    "recast_white/dpr",
    "recast_white/fnplus",
    "recast_white/sprl",
    "reclor",
    "redefine-math",
    "regset",
    "riddle_sense",
    "robustLR",
    "robust_nli/IS_CS",
    "robust_nli/LI_LI",
    "robust_nli/PI_CD",
    "robust_nli/PI_SP",
    "robust_nli/ST_LM",
    "robust_nli/ST_NE",
    "robust_nli/ST_SE",
    "robust_nli/ST_WO",
    "robust_nli_is_sd",
    "robust_nli_li_ts",
    "rotten_tomatoes",
    "ruletaker",
    "rumoureval_2019/RumourEval2019",
    "scicite",
    "scientific-exaggeration-detection",
    "scinli",
    "scone",
    "scruples",
    "sem_eval_2010_task_8",
    "sen-making/1",
    "sen-making/2",
    "sharc_modified/mod",
    "sherliic",
    "sms_spam",
    "snips_built_in_intents",
    "spartqa-mchoice",
    "spartqa-yn",
    "starcon",
    "subjectivity",
    "summarize_from_feedback/comparisons",
    "syntactic-augmentation-nli",
    "temporal-nli",
    "tomi-nli",
    "tracie",
    "tweet_eval/emoji",
    "tweet_eval/emotion",
    "tweet_eval/hate",
    "tweet_eval/irony",
    "tweet_eval/offensive",
    "tweet_eval/sentiment",
    "tweets_hate_speech_detection",
    "twentyquestions",
    "twitter-financial-news-sentiment",
    "universal_dependencies/en_ewt/deprel",
    "universal_dependencies/en_gum/deprel",
    "universal_dependencies/en_lines/deprel",
    "universal_dependencies/en_partut/deprel",
    "vitaminc/tals--vitaminc",
    "wikimedqa/medwiki",
    "winodict",
    "wnut_17/wnut_17",
}


def clean_pair(query, answer):
    query = (query or "").strip()
    answer = (answer or "").strip()
    if query and answer:
        return query, answer
    return None


def _stream(repo, *, shard=None, num_shards=None, shuffle=True, **kwargs):
    dataset = load_dataset(
        repo,
        split="train",
        streaming=True,
        fragment_scan_options=PARQUET_SCAN_OPTIONS,
        **kwargs,
    )
    # Shard before shuffling so each worker owns a disjoint slice of the
    # underlying parquet files and no byte is downloaded twice.
    if num_shards and num_shards > 1:
        dataset = dataset.shard(num_shards=num_shards, index=shard)
    if shuffle:
        dataset = dataset.shuffle(seed=SEED + (shard or 0), buffer_size=SHUFFLE_BUFFER_SIZE)
    return dataset


# --------------------------------------------------------------------------
# Work units
#
# A unit is a self-contained, independently streamable slice of one source.
# Caps are resolved per unit so workers never have to coordinate.
# --------------------------------------------------------------------------


def build_work_units(synth_shards=8, tasksource_shards=1, textbook_shards=1):
    units = [("no_robots", None, None, None)]

    for index in range(tasksource_shards):
        units.append(("tasksource", None, index, tasksource_shards))

    for subset in FLAN_SUBSETS:
        units.append(("flan", subset, None, None))

    for index in range(synth_shards):
        units.append(("synth", None, index, synth_shards))

    for index in range(textbook_shards):
        units.append(("textbook", None, index, textbook_shards))

    return units


def iter_unit(unit):
    """Yield (queries, answers) batches for one work unit."""
    source, subset, shard, num_shards = unit
    if source == "no_robots":
        return iter_no_robots()
    if source == "tasksource":
        return iter_tasksource(shard=shard, num_shards=num_shards)
    if source == "flan":
        return iter_flan_subset(subset)
    if source == "synth":
        return iter_synth(shard=shard, num_shards=num_shards)
    if source == "textbook":
        return iter_textbook(shard=shard, num_shards=num_shards)
    raise ValueError(f"Unknown dataset: {source}")


def _batched(pairs, batch_size=READ_BATCH_SIZE):
    queries = []
    answers = []
    for query, answer in pairs:
        queries.append(query)
        answers.append(answer)
        if len(queries) >= batch_size:
            yield queries, answers
            queries = []
            answers = []
    if queries:
        yield queries, answers


# --------------------------------------------------------------------------
# Per-source iterators (each yields (queries, answers) batches)
# --------------------------------------------------------------------------


def iter_no_robots():
    dataset = load_dataset("HuggingFaceH4/no_robots")

    def pairs():
        for split in dataset.values():
            for row in split:
                if (row["category"] or "").casefold() in {"chat", "coding"}:
                    continue

                messages = row["messages"]
                if len(messages) < 2:
                    continue

                system_prompt = ""
                start = 0
                if messages[0]["role"] == "system":
                    system_prompt = (messages[0]["content"] or "").strip()
                    start = 1

                if len(messages) <= start + 1:
                    continue
                if messages[start]["role"] != "user" or messages[start + 1]["role"] != "assistant":
                    continue

                query = messages[start]["content"]
                if system_prompt:
                    query = f"{system_prompt}\n\n{query}"
                pair = clean_pair(query, messages[start + 1]["content"])
                if pair:
                    yield pair

    return _batched(pairs())


def iter_tasksource(max_per_type=TASKSOURCE_DOCUMENTS_PER_TYPE, shard=None, num_shards=None):
    dataset = _stream(
        "tasksource/tasksource-instruct-v0",
        shard=shard,
        num_shards=num_shards,
        filters=[("task", "in", sorted(TASKSOURCE_TASKS))],
        columns=["inputs", "targets", "task"],
    )

    def pairs():
        counts = Counter()
        saturated = set()
        for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
            for task, inputs, targets in zip(batch["task"], batch["inputs"], batch["targets"]):
                if task not in TASKSOURCE_TASKS or counts[task] >= max_per_type:
                    continue

                pair = clean_pair(inputs, (targets or "").strip().removesuffix("."))
                if pair:
                    counts[task] += 1
                    if counts[task] >= max_per_type:
                        saturated.add(task)
                    yield pair
            # The whitelist is known up front, so this exit is exact: once
            # every task is capped there is nothing left to find.
            if len(saturated) >= len(TASKSOURCE_TASKS):
                return

    return _batched(pairs())


def iter_flan_subset(
    subset,
    max_per_type=FLAN_DOCUMENTS_PER_TYPE,
    max_documents=FLAN_DOCUMENTS_PER_SUBSET,
    patience_rows=FLAN_PATIENCE_ROWS,
):
    dataset = _stream(
        "Open-Orca/FLAN",
        data_dir=subset,
        columns=["inputs", "targets", "_task_name"],
    )

    def pairs():
        counts = Counter()
        produced = 0
        idle_rows = 0
        for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
            batch_produced = 0
            for task, inputs, targets in zip(batch["_task_name"], batch["inputs"], batch["targets"]):
                if max_per_type is not None and counts[task] >= max_per_type:
                    continue

                pair = clean_pair(inputs, targets)
                if pair:
                    counts[task] += 1
                    produced += 1
                    batch_produced += 1
                    yield pair

            if max_documents is not None and produced >= max_documents:
                return

            # The original code kept streaming (and paying for) every
            # remaining row of the subset once all tasks were capped.
            if batch_produced:
                idle_rows = 0
            else:
                idle_rows += len(batch["_task_name"])
                if patience_rows is not None and idle_rows >= patience_rows:
                    return

    return _batched(pairs())


def iter_flan(max_per_type=FLAN_DOCUMENTS_PER_TYPE):
    """Sequential fallback kept for train_tokenizer.py."""
    for subset in FLAN_SUBSETS:
        for queries, answers in iter_flan_subset(subset, max_per_type=max_per_type):
            yield from zip(queries, answers)


def iter_synth(max_documents=SYNTH_DOCUMENTS, shard=None, num_shards=None):
    dataset = _stream(
        "PleIAs/SYNTH",
        shard=shard,
        num_shards=num_shards,
        filters=[
            ("language", "==", "en"),
            ("model", "==", "qwen-3-8b-memorization"),
            ("words", "<", 800),
        ],
        columns=["query", "synthetic_answer"],
    )

    shard_budget = max_documents
    if shard_budget is not None and num_shards and num_shards > 1:
        shard_budget = -(-shard_budget // num_shards)

    def pairs():
        documents = 0
        for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
            for query, answer in zip(batch["query"], batch["synthetic_answer"]):
                pair = clean_pair(query, answer)
                if pair:
                    yield pair
                    documents += 1
                    if shard_budget is not None and documents >= shard_budget:
                        return

    return _batched(pairs())


def iter_textbook(shard=None, num_shards=None):
    dataset = _stream(
        "MegaScience/TextbookReasoning",
        shard=shard,
        num_shards=num_shards,
        columns=["question", "answer", "subject", "reference_answer"],
    )

    def pairs():
        for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
            for subject, question, answer, reference in zip(
                batch["subject"], batch["question"], batch["answer"], batch["reference_answer"]
            ):
                if (subject or "").casefold() not in {"biology", "medicine"}:
                    continue

                pair = clean_pair(question, answer)
                if pair:
                    yield pair

                lower_question = (question or "").lower()
                if "prove" not in lower_question and "show that" not in lower_question:
                    pair = clean_pair(question, reference)
                    if pair:
                        yield pair

    return _batched(pairs())


def iter_dataset(
    name,
    synth_documents=SYNTH_DOCUMENTS,
    flan_documents_per_type=FLAN_DOCUMENTS_PER_TYPE,
):
    """Row-at-a-time compatibility wrapper (used by train_tokenizer.py)."""
    if name == "no_robots":
        batches = iter_no_robots()
    elif name == "tasksource":
        batches = iter_tasksource()
    elif name == "flan":
        return iter_flan(max_per_type=flan_documents_per_type)
    elif name == "synth":
        batches = iter_synth(max_documents=synth_documents)
    elif name == "textbook":
        batches = iter_textbook()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    def rows():
        for queries, answers in batches:
            yield from zip(queries, answers)

    return rows()


def iter_mixture():
    for name in DATASET_NAMES:
        for query, answer in iter_dataset(name):
            yield name, query, answer
