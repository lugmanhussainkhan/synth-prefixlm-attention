from collections import Counter

from datasets import load_dataset


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


def iter_no_robots():
    dataset = load_dataset("HuggingFaceH4/no_robots")

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


def iter_tasksource(max_per_type=TASKSOURCE_DOCUMENTS_PER_TYPE):
    counts = Counter()
    dataset = load_dataset(
        "tasksource/tasksource-instruct-v0",
        split="train",
        streaming=True,
        columns=["inputs", "targets", "task"],
    ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    for row in dataset:
        task = row["task"]
        if task not in TASKSOURCE_TASKS or counts[task] >= max_per_type:
            continue

        pair = clean_pair(row["inputs"], (row["targets"] or "").strip().removesuffix("."))
        if pair:
            counts[task] += 1
            yield pair


def iter_flan(max_per_type=FLAN_DOCUMENTS_PER_TYPE):
    for subset in FLAN_SUBSETS:
        counts = Counter()
        dataset = load_dataset(
            "Open-Orca/FLAN",
            data_dir=subset,
            split="train",
            streaming=True,
            columns=["inputs", "targets", "_task_name"],
        ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

        for row in dataset:
            task = row["_task_name"]
            if max_per_type is not None and counts[task] >= max_per_type:
                continue

            pair = clean_pair(row["inputs"], row["targets"])
            if pair:
                counts[task] += 1
                yield pair


def iter_synth(max_documents=SYNTH_DOCUMENTS):
    dataset = load_dataset(
        "PleIAs/SYNTH",
        split="train",
        streaming=True,
        filters=[
            ("language", "==", "en"),
            ("model", "==", "qwen-3-8b-memorization"),
            ("words", "<", 800),
        ],
        columns=["query", "synthetic_answer"],
    ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    documents = 0
    for row in dataset:
        pair = clean_pair(row["query"], row["synthetic_answer"])
        if pair:
            yield pair
            documents += 1
            if max_documents is not None and documents >= max_documents:
                break


def iter_textbook():
    dataset = load_dataset(
        "MegaScience/TextbookReasoning",
        split="train",
        streaming=True,
        columns=["question", "answer", "subject", "reference_answer"],
    ).shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

    for row in dataset:
        if (row["subject"] or "").casefold() not in {"biology", "medicine"}:
            continue

        pair = clean_pair(row["question"], row["answer"])
        if pair:
            yield pair

        lower_question = (row["question"] or "").lower()
        if "prove" not in lower_question and "show that" not in lower_question:
            pair = clean_pair(row["question"], row["reference_answer"])
            if pair:
                yield pair


def iter_dataset(
    name,
    synth_documents=SYNTH_DOCUMENTS,
    flan_documents_per_type=FLAN_DOCUMENTS_PER_TYPE,
):
    if name == "no_robots":
        return iter_no_robots()
    if name == "tasksource":
        return iter_tasksource()
    if name == "flan":
        return iter_flan(max_per_type=flan_documents_per_type)
    if name == "synth":
        return iter_synth(max_documents=synth_documents)
    if name == "textbook":
        return iter_textbook()
    raise ValueError(f"Unknown dataset: {name}")


def iter_mixture():
    for name in DATASET_NAMES:
        for query, answer in iter_dataset(name):
            yield name, query, answer
