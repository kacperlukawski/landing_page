---
google_colab_link: https://githubtocolab.com/kacperlukawski/qdrant-exam/blob/automate/python-environment/extractive_qa/extractive-question-answering.ipynb
reading_time_min: 5
title: Extractive Question Answering with Qdrant
---

# Extractive Question Answering with Qdrant

Welcome to a thrilling journey into the realm of AI! In this notebook, we're going to explore an exciting aspect of Natural Language Processing (NLP) - Extractive Question Answering.

Question Answering systems can respond to user queries with precise answers. 'Extractive' means our system will pull the answer directly from a given context, rather than generating new text. It's like having your own personal librarian who knows every book cover to cover and can pull the perfect quote for any question you ask!

To make our 'AI Librarian', we will be using three main components:

1. **Qdrant**: Powers our performant vector search. It's our magic bookshelf that finds the right book.
1. **Retriever Model**: It helps in embedding context passages into numerical representations (vectors) that Qdrant can store and search efficiently.
1. **Reader Model**: Once Qdrant finds the most relevant passages for a question, our reader model goes through these passages to extract the precise answer.

## Install dependencies

Let's get started by installing prerequisite packages:

```python
!pip install -qU datasets==2.12.0 qdrant-client==1.10.1 fastembed==0.3.3 sentence-transformers==2.2.2 torch==2.0.1
```

### Import libraries

```python
from datasets import load_dataset
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.auto import tqdm
from typing import List
from transformers import pipeline
```

## Load and process dataset

We'll use the [DuoRC dataset](https://huggingface.co/datasets/duorc), containing questions, plots and answers crowd-sourced from Wikipedia and IMDb movie plots.

We generate embeddings for the context passages using the retriever, index them in the Qdrant vector database, and query to retrieve the top k most relevant contexts containing potential answers to our question. We then use the reader model to extract the answers from the returned contexts.

We load the dataset into a pandas dataframe. Keep the title and plot columns, and we drop duplicates.

```python
# load the duorc dataset into a pandas dataframe
df = load_dataset(
    "duorc", "ParaphraseRC", split="train", ignore_verifications=True
).to_pandas()
df = df[["title", "plot"]]  # select only title and plot column
print(f"Before removing duplicates: {len(df)}")

df = df.drop_duplicates(
    subset="plot"
)  # drop rows containing duplicate plot passages, if any
print(f"Unique Plots: {len(df)}")
df.head()
```

```
/usr/local/lib/python3.10/dist-packages/datasets/load.py:1748: FutureWarning: 'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.
You can remove this warning by passing 'verification_mode=no_checks' instead.
  warnings.warn(
WARNING:datasets.builder:Found cached dataset parquet (/root/.cache/huggingface/datasets/parquet/ParaphraseRC-2dfadd51314ddbba/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec)
/usr/local/lib/python3.10/dist-packages/datasets/table.py:1407: FutureWarning: promote has been superseded by mode='default'.
  table = cls._concat_blocks(blocks, axis=0)


Before removing duplicates: 130245
Unique Plots: 9919
```

<div id="df-849355a9-67d9-404f-87c4-467862b6417b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>plot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ghosts of Mars</td>
      <td>Set in the second half of the 22nd century, Ma...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Noriko's Dinner Table</td>
      <td>The film starts on December 12th, 2001 with a ...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Gutterballs</td>
      <td>A brutally sadistic rape leads to a series of ...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>An Innocent Man</td>
      <td>Jimmie Rainwood (Tom Selleck) is a respected m...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>The Sorcerer's Apprentice</td>
      <td>Every hundred years, the evil Morgana (Kelly L...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

<div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-849355a9-67d9-404f-87c4-467862b6417b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

```
<script>
  const buttonEl =
    document.querySelector('#df-849355a9-67d9-404f-87c4-467862b6417b button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-849355a9-67d9-404f-87c4-467862b6417b');
    const dataTable =
      await google.colab.kernel.invokeFunction('convertToInteractive',
                                                [key], {});
    if (!dataTable) return;

    const docLinkHtml = 'Like what you see? Visit the ' +
      '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
      + ' to learn more about interactive tables.';
    element.innerHTML = '';
    dataTable['output_type'] = 'display_data';
    await google.colab.output.renderOutput(dataTable, element);
    const docLink = document.createElement('div');
    docLink.innerHTML = docLinkHtml;
    element.appendChild(docLink);
  }
</script>
```

</div>

<div id="df-86d73330-3d74-4810-b13b-8095243fc91b">
  <button class="colab-df-quickchart" onclick="quickchart('df-86d73330-3d74-4810-b13b-8095243fc91b')"
            title="Suggest charts"
            style="display:none;">

\<svg xmlns="<http://www.w3.org/2000/svg>" height="24px"viewBox="0 0 24 24"
width="24px">
<g>
<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
</g>
</svg>
</button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

<script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-86d73330-3d74-4810-b13b-8095243fc91b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>

</div>

```
</div>
```

</div>

## Initialize Qdrant client

The Qdrant collection stores vector representations of our context passages which we can retrieve using another vector (query vector)

```python
client = QdrantClient(":memory:")
```

## Create collection

Now we create a new collection called `extractive-question-answering` — we can name the collection anything we want.

We specify the metric type as "cosine" and dimension or size as 384 because the retriever we use to generate context embeddings is optimized for cosine similarity and outputs 384-dimension vectors.

```python
collection_name = "extractive-question-answering"

collections = client.get_collections()
print(collections)

# only create collection if it doesn't exist
if collection_name not in [c.name for c in collections.collections]:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )
collections = client.get_collections()
print(collections)
```

```
collections=[]
collections=[CollectionDescription(name='extractive-question-answering')]


<ipython-input-26-3a672cf6b8d1>:8: DeprecationWarning: `recreate_collection` method is deprecated and will be removed in the future. Use `collection_exists` to check collection existence and `create_collection` instead.
  client.recreate_collection(
```

## Initialize retriever

Next, we need to initialize our retriever. The retriever will mainly do two things:

- Generate embeddings for all context passages (context vectors/embeddings)
- Generate embeddings for our questions (query vector/embedding)

The retriever will generate embeddings in a way that the questions and context passages containing answers to our questions are nearby in the vector space. We can use cosine similarity to calculate the similarity between the query and context embeddings to find the context passages that contain potential answers to our question.

### Embedding model

We will use a SentenceTransformer model named `BAAI/bge-small-en-v1.5` designed for semantic search .It's also quite competitive on two embedding and retrieval benchmarks: [MTEB](https://github.com/embeddings-benchmark/mteb) and [BEIR](arxiv.org/abs/2104.08663)

```python
retriever = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
retriever
```

```
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]





<fastembed.text.text_embedding.TextEmbedding at 0x7f3dd625dc30>
```

## Generate Embeddings -> Store in Qdrant

Next, we need to generate embeddings for the context passages. We will use the `retriever.encode` for that.

When passing the documents to Qdrant, we need an:

1. id (a unique integer value),
1. context embedding, and
1. payload for each document representing context passages in the dataset. The payload is a dictionary containing data relevant to our embeddings, such as the title, plot etc.

```python
%%time

batch_size = 64  # specify batch size according to your RAM and compute, higher batch size = more RAM usage

for index in tqdm(range(0, len(df), batch_size)):
    i_end = min(index + batch_size, len(df))  # find end of batch
    batch = df.iloc[index:i_end]  # extract batch
    emb = list(retriever.embed(batch["plot"].tolist()))  # generate embeddings for batch
    emb = [e.tolist() for e in emb]
    meta = batch.to_dict(orient="records")  # get metadata
    ids = list(range(index, i_end))  # create unique IDs

    # upsert to qdrant
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(ids=ids, vectors=emb, payloads=meta),
    )

collection_vector_count = client.get_collection(
    collection_name=collection_name
).points_count
print(f"Vector count in collection: {collection_vector_count}")
assert collection_vector_count == len(df)
```

```
  0%|          | 0/155 [00:00<?, ?it/s]
```

## Initialize Reader

We use the `bert-large-uncased-whole-word-masking-finetuned-squad` model from the HuggingFace model hub as our reader model. This is finetuned on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). It is trained to extract an answer from a given context. This special mechanism is why we can use this model to extract answers from our context passages.

This is our (encoder) component which uses the contexts to extract an answer.

```python
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# load the reader model into a question-answering pipeline
reader = pipeline("question-answering", model=model_name, tokenizer=model_name)
print(reader.model, reader)
```

Now all the components we need are ready. Let's write some helper functions to execute our queries. The `get_relevant_plot` function retrieves the context embeddings containing answers to our question from the Qdrant collection, and the `extract_answer` function extracts the answers from these context passages.

## Get context

The `get_relevant_plot()` function is your librarian to the vast universe of stories stored in Qdrant.

When you have a question or need a specific story (plot), you tell this guide your question and how many top matches you want. The guide then translates your question into a language Qdrant understands, finds the best matching stories in Qdrant's massive library, and delivers you the titles and contents of these matches.

```python
def get_relevant_plot(question: str, top_k: int) -> List[str]:
    """
    Get the relevant plot for a given question

    Args:
        question (str): What do we want to know?
        top_k (int): Top K results to return

    Returns:
        context (List[str]):
    """
    try:
        encoded_query = next(
            retriever.query_embed(question)
        ).tolist()  # generate embeddings for the question

        result = client.query_points(
            collection_name=collection_name,
            query=encoded_query,
            limit=top_k,
        ).points  # search qdrant collection for context passage with the answer

        context = [
            [x.payload["title"], x.payload["plot"]] for x in result
        ]  # extract title and payload from result
        return context

    except Exception as e:
        print({e})
```

## Extracting an answer

Here is how the engine operates:

1. The central part of the function is `extract_answer`. Qdrant processes your question and retrieves all related context.

1. All related context is processed via the `reader`, which looks at each piece of context and extracts an answer that best fits your question.

1. The function sorts all answers by confidence score, with the top score at the front. Each answer has a title in order to provide context.

1. The result is a sorted list of potential answers, their confidence scores and associated titles.

That's it! All you have to do is put in a question, and wait for an ordered list of the best possible answers. The advantage of this engine is that it also tells you where the answer came from and how confident it is about the result.

```python
def extract_answer(question: str, context: List[str]):
    """
    Extract the answer from the context for a given question

    Args:
        question (str): _description_
        context (list[str]): _description_
    """
    results = []
    for c in context:
        # feed the reader the question and contexts to extract answers
        answer = reader(question=question, context=c[1])

        # add the context to answer dict for printing both together, we print only first 500 characters of plot
        answer["title"] = c[0]
        results.append(answer)

    # sort the result based on the score from reader model
    sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
    for i in range(len(sorted_result)):
        print(f"{i+1}", end=" ")
        print(
            "Answer: ",
            sorted_result[i]["answer"],
            "\n  Title: ",
            sorted_result[i]["title"],
            "\n  score: ",
            sorted_result[i]["score"],
        )


question = "In the movie 3 Idiots, what is the name of the college where the main characters Rancho, Farhan, and Raju study"
context = get_relevant_plot(question, top_k=1)
context
```

As we can see, the retriever is working fine and gets us the context passage that contains the answer to our question. Now let's use the reader to extract the exact answer from the context passage.

```python
extract_answer(question, context)
```

```
1 Answer:  Imperial College of Engineering 
  Title:  Three Idiots 
  score:  0.9049272537231445
```

The reader model predicted with 90% accuracy the correct answer as seen from the context passage. Let's run few more queries.

```python
question = "Who hates Harry Potter?"
context = get_relevant_plot(question, top_k=1)
extract_answer(question, context)
```

```
1 Answer:  . 
  Title:  Harry Potter and the Half-Blood Prince 
  score:  0.15585105121135712
```

This might look like a simple question, but it's actually a pretty tough one for our model. The answer is not explicitly mentioned in the context passage, but the model still tries to extract the answer from the context passage.

```python
question = "Who wants to kill Harry Potter?"
context = get_relevant_plot(question, top_k=1)
extract_answer(question, context)
```

```
1 Answer:  Lord Voldemort 
  Title:  Harry Potter and the Philosopher's Stone 
  score:  0.9568217992782593
```

```python
question = "In the movie The Shawshank Redemption, what was the item that Andy Dufresne used to escape from Shawshank State Penitentiary?"
context = get_relevant_plot(question, top_k=1)
extract_answer(question, context)
```

```
1 Answer:  rock hammer 
  Title:  The Shawshank Redemption 
  score:  0.8666210770606995
```

Let's run another question. This time for top 3 context passages from the retriever.

```python
question = "who killed the spy"
context = get_relevant_plot(question, top_k=3)
extract_answer(question, context)
```

```
1 Answer:  Soviet agents 
  Title:  Tinker, Tailor, Soldier, Spy 
  score:  0.7920866012573242
2 Answer:  Gila 
  Title:  Our Man Flint 
  score:  0.12037214636802673
3 Answer:  Gabriel's assassins 
  Title:  Live Free or Die Hard 
  score:  0.06259559094905853
```

### Cleaning up

We delete the collection from Qdrant and close the connection to the database. This is important to do, otherwise the collection will keep running in the background and consume resources. In a production environment, you would not want to do this. Here, we are mentioning this for completeness.

```python
client.delete_collection(collection_name=collection_name)
```

```
True
```
