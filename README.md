# LangChain and Streamlit RAG

## Demo App on Community Cloud

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://st-lc-rag.streamlit.app/)

---

## What's New?

- **DeepEval Agent Evaluation:** After every answer, see automatic answer relevancy and hallucination scores in the Streamlit sidebar.
- **Supports both classic RAG and LLM-as-a-Judge metrics.**
- **Live feedback:** Quality metrics for every response, not just for batch runs.
- **Cleaner instructions for `.env` and `secrets.toml` use.**
- **Compatible with Python 3.10+ and recent LangChain versions (see notes on deprecation warnings below).**

---

## Quickstart

### Setup Python environment

Developed/tested with **Python 3.10.13**.

```bash
python -mvenv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt


If you run into issues related to hnswlib or chroma-hnswlib while installing requirements you may need to install system package for the underlying package.

For example, on Ubuntu 22.04 this was needed before pip install of hnswlib would succeed.

```bash
sudo apt install python3-hnswlib
```

### Setup .env file with API tokens needed.

```
OPENAI_API_KEY="<Put your token here>"
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"
```

### Setup Streamlit app secrets.

#### 1. Set up the .streamlit directory and secrets file.

```bash
mkdir .streamlit
touch .streamlit/secrets.toml
chmod 0600 .streamlit/secrets.toml
```

#### 2. Edit secrets.toml

**Either edit `secrets.toml` in you favorite editor.**

```toml
OPENAI_API_KEY="<Put your token here>"
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"
```

**Or, you can just reuse .env contents from above.**

```bash
cat < .env >> .streamlit/secrets.toml
```

### Verify Environment

1. Check that LangChain dependencies are working.

```bash
python basic_chain.py
```

2. Check that Streamlit and dependencies are working.

```bash
streamlit run streamlit_app.py
```

3. Run Individual Example Programs

The most of the Python source files besides `streamlit_app.py` have a main defined
so that you can execute them directly as an example or test.

For example, the main in `ensemble.py` will use context from an online version of the book [*The Problems of Philosophy* by Bertrand Russell](https://www.gutenberg.org/ebooks/5827.html.images)
to answer "What are the key problems of philosophy according to Russell?"

```bash
python ensemble.py
```

>    Split into 313 chunks
>    According to Russell, the key problems of philosophy include the uncertainty of knowledge, the limitations of metaphysical reasoning, and the inability to provide definite answers to fundamental questions. Philosophy aims to diminish the risk of error, but cannot eliminate it entirely due to human fallibility. The value of philosophy lies in its ability to challenge common sense beliefs and lead to the exploration of complex problems.


## Example Queries for Streamlit App

DeepEval Metrics Integration
This project automatically evaluates every answer using DeepEval and displays:

Answer Relevancy (0–1): How well the LLM response fits the question and context.

Hallucination (0–1): Lower is better—shows if the response strays from the provided context.

You’ll see these metrics update live in the sidebar after every response.

Example:

makefile
Copy
Edit
Relevance: 0.88
Hallucination: 0.00
DeepEval Usage (behind the scenes)
Each answer is scored with AnswerRelevancyMetric and HallucinationMetric.

The relevant code is in streamlit_app.py:

python
Copy
Edit
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
...
metrics = [AnswerRelevancyMetric(), HallucinationMetric()]
results = evaluate(test_cases=[test_case], metrics=metrics)
for metric in test_result.metrics_data:
    if metric.name.lower() == "answer relevancy":
        st.sidebar.metric("Relevance ↗", f"{metric.score:.2f}")
    elif metric.name.lower() == "hallucination":
        st.sidebar.metric("Hallucination ↘", f"{metric.score:.2f}")
Troubleshooting: If you see DeepEval errors, check that you have a working OpenAI key and that your requirements match those in requirements.txt.

Example Queries for Streamlit App
Example 1: Metabolic Rate
Question:
If I am an 195 lb male, what should my calorie intake be to lose 1 lb a week based on what you know about my basal metabolic rate?

Sidebar Scores:
Relevance: 0.88
Hallucination: 0.00

Example 2: Recipes
Note: To use this example, copy the example/us_army_recipes.txt file into your data directory (not included by default).

Question:
How do I make salmon croquettes according to the recipes I provided you?

Expected Answer:
A step-by-step recipe with all major steps and ingredients found in the recipe file.

## References


Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114). In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (SIGIR '09). Association for Computing Machinery, New York, NY, USA, 758–759. <https://doi.org/10.1145/1571941.1572114>.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Singh Chaplot, D., de las Casas, D., … & El Sayed, W. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825). arXiv e-prints, arXiv-2310. <https://doi.org/10.48550/arXiv.2310.06825>.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … & Kiela, D. (2020). [Retrieval-augmented generation for knowledge-intensive nlp tasks](https://arxiv.org/abs/2005.11401). Advances in Neural Information Processing Systems, 33, 9459–9474.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). [Lost in the middle: How language models use long contexts](https://arxiv.org/abs/2307.03172). Transactions of the Association for Computational Linguistics, 12, 157–173.

Robertson, S., & Zaragoza, H. (2009). [The probabilistic relevance framework: BM25 and beyond](https://dl.acm.org/doi/10.1561/1500000019). Foundations and Trends® in Information Retrieval, 3(4), 333–389. <https://doi.org/10.1561/1500000019>

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://dl.acm.org/doi/10.1145/3404835.3463098). In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 2288–2292. <https://doi.org/10.1145/3404835.3463098>.

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., … & Wolf, T. (2023). Zephyr: Direct Distillation of LM Alignment. arXiv e-prints, arXiv-2310. <https://doi.org/10.48550/arXiv.2310.16944>.

## Misc Notes

- There is an issue with newer langchain package versions and streamlit chat history, see https://github.com/langchain-ai/langchain/pull/18834
  - This one reason why a number of dependencies are pinned to specific values.
