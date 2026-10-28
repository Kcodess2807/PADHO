# padho

![Padho: Learn RAG by Doing](Padho_%20Learn%20RAG%20by%20Doing.png)

Learn how RAG, embeddings and agents actually work by reading code that does
the work itself.

*padho (पढ़ो) means "read" or "study" in Hindi.*

## why I'm building this

Every time I tried to learn RAG I ended up writing something like:

```python
from some_framework import RAG
rag = RAG(docs)
rag.ask("...")
```

It worked, but I couldn't tell you what happened in between. How was the text
split? What do the embeddings look like? Why did it pick that chunk? What is
an "agent" once you take away the framework?

So I started writing everything myself in plain Python. No LangChain, no
LangGraph, no LlamaIndex. If something happens, there's a file you can open
and read.

This is a learning project, mine first. If it helps you too, great.

## what it looks like

The main idea is that you can watch the pipeline run. Add `--explain` and it
prints what happens at each stage:

```
$ python padho.py ask "what is cosine similarity" --explain

[1] query
    raw:      what is cosine similarity
    cleaned:  what is cosine similarity

[2] embed (tfidf, dim=1204)
    nonzero:  3 / 1204
    top dims: cosine=0.71  similarity=0.64  what=0.09
    took 2ms

[3] search (cosine, 142 chunks)
    scores:   max 0.58  mean 0.04  min 0.00

[4] top 3
    0.58  similarity.md #4    "Cosine similarity measures the angle betw..."
    0.41  embeddings.md #11   "Two vectors pointing the same way have a ..."
    0.19  distance.md #2      "Unlike euclidean distance, cosine doesn't..."

[5] prompt (612 tokens, use --full to print all)
    Answer using only the context below. If the answer isn't there, say so.
    ...

[6] answer (llama3.2 via ollama, 1.3s)
    Cosine similarity measures how similar two vectors are by looking at
    the angle between them, not their length [1][2].

    [1] similarity.md  [2] embeddings.md
```

Other flags:

- `--step` pauses after each stage until you hit enter
- `--full` prints everything without truncating (full vectors, full prompt, all scores)
- `--save-trace out.json` dumps the run so you can plot it later
- `--no-llm` stops after retrieval, useful if you don't have a model running

Indexing works the same way. `python padho.py index ./data --explain` shows
the loading, cleaning, chunking and embedding of every document. Chunking
happens here, not when you ask a question, which took me longer to get than
I'd like to admit.

## what's in here

Still being built, so not everything below works yet.

**RAG pipeline**
- document loading and cleaning
- chunking (fixed size, with overlap)
- cosine, euclidean and dot product, first with plain loops, then numpy
- retrieval and prompt building
- answers through ollama, or `--no-llm` for retrieval only
- `--explain` and `--step` for both indexing and asking
- a small sample dataset in `data/sample` so it runs right after cloning

**Embeddings**
- bag of words and tf-idf, written by hand
- word2vec trained from scratch in numpy on the sample data
- pretrained transformer embeddings, to compare against the ones above

**Exploring the vector space**
- clustering and plots of the embedding space
- `why-not` command: why didn't this chunk show up?
- comparing metrics and chunk sizes side by side
- examples where retrieval fails, and why it happened

**Evaluation**
- simple hit rate on a set of test questions
- RAGAS metrics written from scratch (more below)

**Graph RAG (simplified)**
- pulling entities and relations out of chunks
- building a small graph
- vector search combined with graph neighbours
- evaluated against plain RAG on the same questions

**Agents**
- an agent loop in plain Python (think, pick a tool, run it, repeat)
- a simple message format for agents to talk to each other
- `--explain` for agent runs too

## evaluation (RAGAS, written from scratch)

"The answers look fine" isn't a test, so padho has its own implementation of
the four main [RAGAS](https://arxiv.org/abs/2309.15217) metrics. I'm not
using the `ragas` library. I read the paper and the docs and wrote each
metric myself, so you can see exactly what's being measured.

- **faithfulness**: is the answer actually supported by the retrieved
  chunks? The answer gets broken into individual claims, each claim is
  checked against the context, and the score is supported claims / total
  claims. This is the one that catches hallucinations.
- **answer relevancy**: does the answer address the question? The LLM
  generates a few questions from the answer alone, and we check how close
  those are to the original question using embeddings.
- **context precision**: did the useful chunks get ranked near the top, or
  were they buried under irrelevant ones?
- **context recall**: did retrieval find everything needed? The reference
  answer is split into statements, and we check how many of them can be
  found in the retrieved chunks.

Same as everything else, `--explain` shows how each score was reached:

```
$ python padho.py eval data/sample/eval.jsonl --explain

question 3/10: what is cosine similarity

faithfulness
    claims in answer: 3
    [yes] measures similarity using the angle between vectors
    [yes] ignores the length of the vectors
    [no]  is always between 0 and 1          (not in context)
    score: 0.67

answer relevancy
    questions generated from the answer:
    0.91  how does cosine similarity compare two vectors?
    0.88  what does cosine similarity measure?
    0.79  why does cosine similarity ignore vector length?
    score: 0.86

context precision
    #1 similarity.md #4    relevant
    #2 embeddings.md #11   not relevant
    #3 distance.md #2      relevant
    score: 0.83

context recall
    statements in reference answer: 2
    found in context: 2
    score: 1.00

...

summary (10 questions)
    faithfulness        0.81
    answer relevancy    0.84
    context precision   0.72
    context recall      0.90
```

(Funny thing about that faithfulness example: cosine similarity actually
ranges from -1 to 1, so the claim that got flagged was also just wrong.
That's exactly the kind of thing this metric is for.)

A few notes:

- Faithfulness, context precision and context recall use an LLM as a judge.
  The metric logic and prompts are written here, but the judging itself is
  still done by a model, so scores can wobble a bit between runs.
- My numbers won't exactly match the official `ragas` library. Their
  prompts and implementation details differ, and they change over time. The
  point is to understand what the metrics measure, not to match their
  numbers to the decimal.
- You need a small eval set with questions and reference answers. There's
  one in `data/sample/eval.jsonl` to start with.

The main reason I wanted this: I want to run plain RAG and graph RAG on the
same questions and actually see whether the graph helps, instead of just
assuming it does.

## what isn't from scratch

Being upfront about this:

- The pretrained embedding model and the LLM are not mine. I use them, I
  don't train them. The bag of words, tf-idf and word2vec embedders are
  written here, so you can see the full path from text to vector at least
  once before using a model you can't look inside.
- numpy is used for the math after the loop versions.
- matplotlib for plots.
- The graph RAG here is a simplified version meant to show the idea. It's
  not the Microsoft GraphRAG approach.
- RAGAS metrics are my own implementation, not the `ragas` library.

## setup

```bash
git clone https://github.com/<your-username>/padho.git
cd padho
pip install -r requirements.txt
```

For answers you'll need [Ollama](https://ollama.com) running with a small
model (`ollama pull llama3.2`). Without it, use `--no-llm` and you still get
the full retrieval trace.

```bash
python padho.py index data/sample
python padho.py ask "what is cosine similarity" --explain
python padho.py eval data/sample/eval.jsonl
```

## reading order

If you want to go through the code, I'd do it in this order:

1. loading and chunking
2. embedders (bag of words → tf-idf → word2vec → pretrained)
3. similarity and search
4. the full ask pipeline
5. evaluation and RAGAS
6. graph RAG
7. agents

Run things, change numbers, break stuff. That's how most of this made sense
to me.

## contributing

Issues and PRs are welcome, especially if something in the code or the
explanations is confusing. If it confused you it's probably confusing
someone else too.

## license

MIT