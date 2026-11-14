# AI/ML Paper Q&A Assistant

A retrieval-augmented generation (RAG) app that answers questions about a collection of 21 arXiv papers on RAG, dense retrieval and LLM hallucination. Every answer cites the papers and pages it came from, and the app says so when the papers do not contain the answer. It also summarises a chosen paper and extracts its keywords.

**Live demo:** https://ai-ml-paper-q-a-assistant-1.onrender.com

The demo runs on a free Render instance, so the first visit after a quiet period can take about a minute while the app wakes up.

## Features

- **Question answering with citations.** Retrieves the most relevant passages, then an LLM answers using only those passages, citing them as [1], [2] with paper title and page.
- **Refuses out-of-scope questions.** Off-topic questions return "I could not find this in the provided papers." with no sources.
- **Paper summary and keywords.** Gemini summarises the abstract of any paper in the collection in three sentences and lists its key phrases.

## How it works

```
PDF papers (arXiv)
   -> text extraction (PyMuPDF) and cleaning
   -> 200-word chunks with 40-word overlap (bibliography chunks removed)
   -> embeddings (fastembed, ONNX runtime: all-MiniLM-L6-v2)
   -> vector store (ChromaDB, cosine similarity)

Question -> top 8 passages (max 2 per paper, similarity threshold 0.35)
        -> Gemini answers using only those passages, with citations
        -> answer + clickable sources in a Gradio app
```

**Tech stack:** Python, PyMuPDF, fastembed (ONNX, no PyTorch), ChromaDB, Google Gemini API, Gradio, Docker, Render.

The embedding model runs on ONNX instead of PyTorch so the app is small enough for a free hosting instance.

## Evaluation

Retrieval was tested on 21 hand-written questions, each with the paper that should answer it, and compared with a plain keyword (TF-IDF) baseline over the same chunks.

| Method                         | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR  |
| ------------------------------ | ----- | ----- | ----- | ----- | ---- |
| Keyword (TF-IDF)               | 81%   | 95%   | 95%   | 100%  | 0.88 |
| Semantic search (this project) | 86%   | 95%   | 95%   | 100%  | 0.91 |

- Hit@k: the correct paper appears in the top k retrieved passages. The app passes the top 8 passages to the LLM.
- The similarity threshold rejected 8 of 8 off-topic questions and 0 of 21 in-scope questions.

**Caveats:** the test set is small, and many questions share words with paper titles, which favours keyword search. The gap between the two methods is one question, so the results are comparable, not conclusive. The threshold check does not cover cases where the LLM itself declines to answer.

Reproduce with `python eval_retrieval.py` (questions are in `eval_questions.json`). The TF-IDF baseline needs scikit-learn, which the app itself does not use: run `pip install scikit-learn` first.

## Run it locally

1. Clone the repository and install the libraries (Python 3.11 recommended):

   ```
   pip install -r requirements.txt
   ```

2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and create a file named `.env` in the project folder:

   ```
   GEMINI_API_KEY=your-key-here
   ```

3. Download the papers and build the search index (takes a few minutes):

   ```
   python setup_data.py
   ```

4. Start the app and open the local address it prints:

   ```
   python app.py
   ```

Model names change over time. If you see a "model not found" error, update the `MODELS` list in `answer.py`.

### Run with Docker

```
docker build -t paper-qa .
docker run -p 10000:10000 -e GEMINI_API_KEY=your-key-here paper-qa
```

Then open http://localhost:10000. The papers are downloaded and the index is built while the image is built.

## Deploy on Render

1. Push the repository to GitHub.
2. In Render, create a **Web Service** from the repository and choose the **Docker** runtime.
3. Add an environment variable `GEMINI_API_KEY` with your key. Never commit the key to GitHub.
4. Deploy. Render builds the image, which downloads the papers and builds the index, then starts `app.py` on the port Render provides.

Every push to the connected branch triggers a new deploy.

## Project structure

| File                                       | Purpose                                                              |
| ------------------------------------------ | -------------------------------------------------------------------- |
| `setup_data.py`                            | Downloads the 21 papers by arXiv ID, extracts text, builds the index |
| `extract_text.py`                          | PDF to cleaned text, page by page                                    |
| `build_index.py`                           | Chunking, bibliography filtering, embeddings, ChromaDB               |
| `embedder.py`                              | Lightweight ONNX embeddings (all-MiniLM-L6-v2)                       |
| `search.py`                                | Semantic search with a per-paper cap                                 |
| `answer.py`                                | Prompt, LLM call with model fallback, refusal handling               |
| `nlp_tools.py`                             | Paper summaries and keywords (Gemini)                                |
| `app.py`                                   | Gradio interface                                                     |
| `Dockerfile`                               | Container build used for deployment                                  |
| `eval_retrieval.py`, `eval_questions.json` | Retrieval evaluation                                                 |

## Limitations

- The collection is small (21 papers) and covers only RAG, retrieval and hallucination.
- Broad "why is X important" questions are sometimes refused even when related passages exist.
- Questions such as "summarise all papers about X" cannot be answered fully, because only the top passages are used.
- Answers, summaries and keywords come from an LLM and can contain mistakes. Check the cited pages.
- Summaries use only the abstract area of the first page, so they can be generic.
- PDF extraction can garble tables and equations.
- The public demo shares one Gemini API key, so heavy use can hit rate limits and show a "busy" message.

## Papers

The papers are downloaded from arXiv at setup time and are not redistributed here. Each paper remains under its authors' licence.
