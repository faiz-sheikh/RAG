# Retrieval-Augmented Generation - Day 2

Classroom document-retrieval notebooks and a Streamlit retrieval interface using LangChain loaders, Hugging Face embeddings and FAISS. The goal is to find document chunks relevant to a question while preserving the original pipeline.

**Author:** [faiz-sheikh](https://github.com/faiz-sheikh)

## RAG architecture and project scope

Retrieval-Augmented Generation (RAG) combines document retrieval with language-model generation. Its typical components are document loading, text splitting, embeddings, vector indexing, retrieval, and an LLM that uses retrieved context to generate an answer.

**The notebooks and Streamlit app implement retrieval only.** They display matching source chunks. The app is titled "C++ RAG Chatbot" as a classroom example, but it does not generate answers or maintain conversation history. LLM-based generation with Ollama/Gemma is planned for a later stage; agents and advanced retrieval techniques are not implemented.

## Notebooks and documents

| Notebook | Loader | Input | Output |
|---|---|---|---|
| [RAGRP.ipynb](RAGRP.ipynb) | `PyPDFLoader` | `NIPS-2017-attention-is-all-you-need-Paper.pdf` | Relevant research-paper passages |
| [RAGTutorial.ipynb](RAGTutorial.ipynb) | UTF-8 `TextLoader` | `CPP.txt` | Relevant C++ study passages |

The supplied classroom reference documents remain beside the notebooks to preserve relative paths. Run with the repository root as the working directory. The C++ document is sample retrieval input, not a verified programming reference, and can contain inaccuracies. The research paper is [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. (2017); credit remains with its authors. No ownership of reference documents is claimed.

## Notebook workflow

1. Load the PDF into page documents or load the UTF-8 text document.
2. Split with `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)`. These are character-based settings, not token counts; separators may produce shorter chunks.
3. Generate embeddings using `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`, backed by Sentence Transformers.
4. Build an in-memory vector index with `FAISS.from_documents(finalDocuments, embeddings)`.
5. Accept a question through `input()` and call `db.similarity_search(query)` using the same embedding wrapper.
6. Print the matching chunks as numbered results. No explicit `k`, index persistence or evaluation metrics are supplied.

Embeddings map text into vectors so retrieval can reflect semantic similarity beyond exact word matches. FAISS indexes and searches these vectors; LangChain connects the loader, splitter, embedding model and vector store.

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-community` | PDF/text loaders and FAISS integration |
| `langchain-text-splitters` | Recursive character splitter |
| `langchain-huggingface` | Hugging Face embedding wrapper |
| `pypdf` | PDF parsing backend |
| `sentence-transformers` | Embedding model backend |
| `faiss-cpu` | CPU vector indexing and search |
| `ipykernel` | Notebook kernel |
| `streamlit` | Browser interface for `chatBotDemo.py` |

Streamlit is required by the new retrieval app. Ollama and the top-level `langchain` package are not used in this retrieval-only version. Pip installs transitive dependencies.

## Running in VS Code (Windows)

Install the VS Code Python and Jupyter extensions. Open this repository folder and use a Python version supported by its dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open a notebook, select **Select Kernel > Python Environments > .venv**, then run cells from top to bottom. Calling the environment's Python directly avoids PowerShell activation-policy issues. Requirements reflect actual imports and required backends, rather than an environment export. Package versions are not locked.

The first embedding run may download the model and needs internet. When the final cell requests input, try `What is self-attention?` for the paper or `What is inheritance?` for C++. No API key is needed by this pipeline.

## Expected outputs and learning outcomes

Loading prints page/document counts; splitting produces chunks; FAISS creates a vector store; the final cell prints numbered source passages. Results vary by question and package/model versions. Retrieved passages are not synthesized or guaranteed-correct answers.

The exercises demonstrate ingestion, chunk size and overlap, vector representations, semantic similarity search, and the distinction between retrieval and generation.

## Limitations and future extensions

- Small 200-character chunks can lose context; the original classroom settings are retained.
- PDF extraction quality and source accuracy affect results. OCR for scanned documents is not implemented.
- No relevance benchmark, reranking, persistence or production deployment is included.
- Model downloads and package compatibility affect setup; requirements are not a tested version lock.
- LLM-based answer generation from retrieved context is a possible future extension, not an implemented feature.

No separate Day 2 theory-notes file was present in the supplied project folder; this README documents the workflow directly.

## Saved example results

Both notebooks include outputs from successful top-to-bottom runs with the existing cached embedding model:

| Notebook | Example query | Saved result |
|---|---|---|
| `RAGRP.ipynb` | `What is self-attention?` | 11 PDF pages loaded and four matching passages printed |
| `RAGTutorial.ipynb` | `What is inheritance?` | One text document loaded and four matching passages printed |

The saved query is one example; rerunning the final cell still prompts for your own question. The classroom code, embedding model and 200/20 splitting settings are retained. Nonfatal local setup notices are excluded from the saved presentation; computed results are retained.

## Streamlit retrieval app

`chatBotDemo.py` exposes the C++ text-retrieval exercise through a small browser interface:

```text
CPP.txt → 200/20 character splitting → MiniLM embeddings → cached FAISS store
Question → similarity search (k=3) → display three matching passages
```

The app uses `st.cache_resource` so the vector store is reused across reruns instead of rebuilding it on every question. This is an in-memory cache, not a persistent database. It reads the existing `CPP.txt`; there is no document upload in this app. Restart the app or clear its cache after replacing that source file.

Run from this repository folder after installing requirements:

```powershell
.\.venv\Scripts\python.exe -m streamlit run chatBotDemo.py
```

Enter a question such as `What is inheritance?`. The app displays three retrieved context passages, without generating a combined answer. The first embedding run may require downloading the model; subsequent runs can use its local cache. No Ollama server or LLM API key is required for this stage.

The classroom filename, embedding model, variable names, chunk size 200, overlap 20 and `k=3` are retained. The loader path uses `CPP.txt` to match the supplied document exactly, including case on Linux. The two original notebooks and their saved outputs are unchanged.

The next learning stage is to pass retrieved context and the question to Ollama's `gemma2:2b` for a generated answer. That integration is not part of this commit. The separate 510-line assisted extension draft is not included.
