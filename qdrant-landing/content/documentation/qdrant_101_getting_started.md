---
google_colab_link: https://githubtocolab.com/kacperlukawski/qdrant-exam/blob/automate/python-environment/qdrant_101_getting_started/getting_started.ipynb
reading_time_min: 9
title: Getting Started with Qdrant
---

# Getting Started with Qdrant

Vector databases shine in many applications like [semantic search](https://en.wikipedia.org/wiki/Semantic_search) and [recommendation systems](https://en.wikipedia.org/wiki/Recommender_system), and in this tutorial, you will learn how to get started building such systems with one of the most popular and fastest growing vector databases in the market, [Qdrant](qdrant.tech).

## Table of contents

1. [Learning Outcomes](##-1.-Learning-Outcomes)
1. [Installation](##-2.-Installation)
1. [Getting Started](##-3.-Getting-Started)
   - [Adding Points](###-3.1-Adding-Points)
   - [Payload](###-3.2-Payloads)
   - [Search](###-3.3-Search)
1. [Recommendations](##-4.-Recommendations)
1. [Conclusion](##-5.-Conclusion)
1. [Resources](##-6.-Resources)

## 1. Learning outcomes

By the end of this tutorial, you will be able to:

- Create, update, and query collections of vectors using Qdrant.
- Conduct semantic search based on new data.
- Develop an intuition for the mechanics behind the recommendation API of Qdrant.
- Understand and get creative with the kind of data you can add to your payload.

## 2. Installation

The open source version of Qdrant is available as a Docker image. You can download the image and run it from any machine with Docker installed on it. If you don't have Docker installed, follow the instructions [here](https://docs.docker.com/get-docker/). After you have installed Docker, Terminal and download the Qdrant image:

```sh
docker pull qdrant/qdrant
```

Next, initialize Qdrant:

```sh
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

You should see something like this:

![dockerqdrant](img/docker_qdrant_28_10_2023.png)

If you experience any issues during installation, please let us know in our [Discord channel here](https://qdrant.to/discord).

Now that you have Qdrant up and running, you need to pick a client to connect to it. We'll be using Python as it has the most mature data tools ecosystem out there. Let's start setting up our development environment and getting the libraries we'll be using.

**Note:** At the time of writing, Qdrant supports Rust, Go, Python TypeScript, Java and C#. We expect other programming languages to be added in the future.

```python
%pip install qdrant-client pandas numpy faker
```

## 3. Getting started

The two modules we are going to use are `QdrantClient` and `models`. The former lets you connect to Qdrant or to run an in-memory database by switching the parameter `location=` to `":memory:"`. The latter gives you access to most functionalities you need to interact with Qdrant.

```python
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import CollectionStatus
```

We'll start by instantiating our client using `host="localhost"` and `port=6333` (as it is the default port we used earlier with Docker). You can also follow along with the `location=":memory:"` option commented out below.

```python
client = QdrantClient(host="localhost", port=6333)
client
```

```
<qdrant_client.qdrant_client.QdrantClient at 0x1079a1710>
```

```python
# client = QdrantClient(location=":memory:")
# client
```

______________________________________________________________________

**Note:** In OLTP and OLAP databases we call specific bundles of rows and columns **Tables**. However, in vector databases, the rows are known as **Vectors**, while the columns are **Dimensions**. The combination of the two (plus some metadata) is a [**Collection**](https://qdrant.tech/documentation/concepts/collections/).

Just as we can create many tables in an OLTP or an OLAP database, we can create many collections in a vector database like Qdrant using one of its clients. The key difference to note is that when we create a collection in Qdrant, we need to specify the width of the collection (i.e. the length of the vector or amount of dimensions) beforehand with the parameter `size=...`, as well as the distance metric with the parameter `distance=...`.

The distances currently supported by Qdrant are [**Cosine Similarity**](https://en.wikipedia.org/wiki/Cosine_similarity), [**Dot Product**](https://en.wikipedia.org/wiki/Dot_product), and [**Euclidean Distance**](https://en.wikipedia.org/wiki/Euclidean_distance).

______________________________________________________________________

Let's create our first collection and have the vectors be of size 100 with a distance set to **Cosine Similarity**.

```python
my_collection = "first_collection"

first_collection = client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)
print(first_collection)
```

```
True


/var/folders/wf/617k9yhn1875htvkfrd0rmmw0000gn/T/ipykernel_70021/1220860614.py:3: DeprecationWarning: `recreate_collection` method is deprecated and will be removed in the future. Use `collection_exists` to check collection existence and `create_collection` instead.
  first_collection = client.recreate_collection(
```

We can extract information related to the health of our collection by retrieving the collection with our client. In addition, we can use this information for testing purposes, which can be very beneficial while in development mode.

```python
collection_info = client.get_collection(collection_name=my_collection)
list(collection_info)
```

```
[('status', <CollectionStatus.GREEN: 'green'>),
 ('optimizer_status', <OptimizersStatusOneOf.OK: 'ok'>),
 ('vectors_count', None),
 ('indexed_vectors_count', 0),
 ('points_count', 0),
 ('segments_count', 8),
 ('config',
  CollectionConfig(params=CollectionParams(vectors=VectorParams(size=100, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None, quantization_config=None, on_disk=None, datatype=None, multivector_config=None), shard_number=1, sharding_method=None, replication_factor=1, write_consistency_factor=1, read_fan_out_factor=None, on_disk_payload=True, sparse_vectors=None), hnsw_config=HnswConfig(m=16, ef_construct=100, full_scan_threshold=10000, max_indexing_threads=0, on_disk=False, payload_m=None), optimizer_config=OptimizersConfig(deleted_threshold=0.2, vacuum_min_vector_number=1000, default_segment_number=0, max_segment_size=None, memmap_threshold=None, indexing_threshold=20000, flush_interval_sec=5, max_optimization_threads=None), wal_config=WalConfig(wal_capacity_mb=32, wal_segments_ahead=0), quantization_config=None)),
 ('payload_schema', {})]
```

```python
assert collection_info.status == CollectionStatus.GREEN
assert collection_info.points_count == 0
```

Two important takeaways:

1. When you initiated the Docker image, you created a local directory called, `qdrant_storage`. This is where all of your collections and their metadata will be stored.

   Qdrant can use one of two options for [storage](https://qdrant.tech/documentation/concepts/storage/):

   - **in-memory** storage, which stores all vectors in RAM and has the highest speed since disk access is required only for persistence)

   - **memmap** storage, which creates a virtual address space associated with the file on disk. You can have a look at that directory in a \*nix system with `tree qdrant_storage -L 2`, and something similar to the following output should come up for you.

     ```bash
     qdrant_storage
     ├── aliases
     │   └── data.json
     ├── collections
     │   └── my_first_collection
     └── raft_state
     ```

1. You used `client.recreate_collection()`, which can be used more than once to create new collections with or without the same name. Therefore, please make sure you do not run this command multiple times and accidentally recreate a collection.

Instead, to create a brand new collection that cannot be recreated, use the `client.create_collection()` method.

The created collection will hold vectors of 100 dimensions and the distance metric has been set to Cosine Similarity.

Now that we know how to create collections, let's create a bit of dummy data and add some vectors to it.

### 3.1 Adding points

[Points](https://qdrant.tech/documentation/concepts/points/) are a central entity that Qdrant operates with. They contain records consisting of a vector, an optional `id`, and an optional `payload`.

The optional id can be represented by [unsigned integers](<https://en.wikipedia.org/wiki/Integer_(computer_science)>) or [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier)s. For this tutorial, we will use a straightforward range of numbers.

You can use [NumPy](https://numpy.org/) to create a matrix of dummy data containing 1,000 vectors and 100 dimensions. Then, you can represent the values as `float64` numbers between -1 and 1. For simplicity, imagine that each of these vectors represents one of our favorite songs. Then, each column would represent a unique characteristic of the song; e.g. the tempo, the beats, the pitch of the voice(s) of the singer(s).

```python
import numpy as np
```

<hr />

```python
data = np.random.uniform(low=-1.0, high=1.0, size=(1_000, 100))
type(data[0, 0]), data[:2, :20]
```

```
(numpy.float64,
 array([[-0.42821695, -0.0140057 , -0.48015149,  0.2772616 ,  0.49262215,
          0.92767383,  0.34668525,  0.13876603,  0.45314349,  0.04778376,
          0.54183112,  0.62803171, -0.28792279,  0.13449526, -0.74653784,
          0.35226282, -0.50233531, -0.90180375,  0.73535787,  0.56509058],
        [-0.82453141, -0.07757142, -0.40848273, -0.07197003, -0.05247721,
          0.13227664,  0.59453522, -0.7115031 ,  0.693443  ,  0.2809882 ,
          0.77522596, -0.85107386, -0.55534477,  0.80635448, -0.88174527,
         -0.92619133,  0.64621776,  0.05111648, -0.17886399,  0.84900515]]))
```

Now you can create an index for your vectors.

```python
index = list(range(len(data)))
index[-10:]
```

```
[990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
```

Once a collection has been created, you can fill it in with `client.upsert()`. You need to provide the collection's name and the appropriate uploading process from our `models` module, in this case, [`Batch`](https://qdrant.tech/documentation/points/#upload-points).

**Note:** Qdrant can only take in native Python iterables like lists and tuples. This is why you will notice the `.tolist()` method attached to our `data` matrix below.

```python
client.upsert(
    collection_name=my_collection, points=models.Batch(ids=index, vectors=data.tolist())
)
```

```
UpdateResult(operation_id=0, status=<UpdateStatus.COMPLETED: 'completed'>)
```

You can retrieve specific points based on their ID (for example, song X with ID 100) and get some additional information from that result.

```python
client.retrieve(
    collection_name=my_collection,
    ids=[100],
    with_vectors=True,  # the default is False
)
```

```
[Record(id=100, payload={}, vector=[-0.10599836, 0.11068635, 0.04444738, -0.027097967, -0.09290919, -0.08676028, 0.06970812, -0.10418644, -0.072337456, -0.14927788, -0.11333238, 0.06487068, 0.015204664, 0.0020790289, -0.059932653, -0.01583718, 0.024277804, -0.17515527, 0.008782358, 0.063005924, 0.14756365, -0.10434394, 0.083360724, -0.14625706, -0.15841727, 0.1357714, -0.0031317412, 0.11139833, -0.042295743, -0.15759756, -0.056121986, -0.14682493, -0.1764052, 0.17180207, -0.14315128, -0.0013021046, -0.01968442, 0.04458566, -0.06287631, 0.17405103, 0.14192675, -0.057540677, 0.026332472, 0.16844778, -0.047596026, -0.0472639, -0.11070978, 0.049258128, 0.03533134, 0.15809262, 0.035617046, 0.106708094, -0.13444023, 0.117557324, 0.1672869, 0.016368773, -0.0333911, 0.10456425, 0.104310416, -0.08187919, 0.070905924, 0.13841578, -0.019744948, -0.16601983, -0.019578911, 0.042030502, 0.036291942, -0.13290662, -0.14383806, 0.029476542, 0.1535214, -0.12118013, -0.1076026, -0.045408934, -0.03006504, -0.03663878, 0.056646116, -0.12882571, -0.093795806, 0.009991121, 0.03645561, -0.020317417, 0.0906876, -0.040670637, -0.0884878, -0.10439095, -0.08283463, 0.15414381, 0.08472449, 0.13265991, -0.023909355, 0.124651045, 0.16598777, 0.106182486, -0.09869526, -0.06457695, -0.16313535, 0.0076360404, -0.11167924, 0.011846884], shard_key=None, order_value=None)]
```

You can also update the collection one point at a time; e.g. as new data is coming in.

```python
def create_song():
    return np.random.uniform(low=-1.0, high=1.0, size=100).tolist()
```

<hr />

```python
client.upsert(
    collection_name=my_collection,
    points=[
        models.PointStruct(
            id=1000,
            vector=create_song(),
        )
    ],
)
```

```
UpdateResult(operation_id=1, status=<UpdateStatus.COMPLETED: 'completed'>)
```

We can also delete a point in a straightforward fashion.

```python
# this will show the amount of vectors BEFORE deleting the one we just created
client.count(
    collection_name=my_collection,
    exact=True,
)
```

```
CountResult(count=1001)
```

```python
client.delete(
    collection_name=my_collection,
    points_selector=models.PointIdsList(
        points=[1000],
    ),
)
```

```
UpdateResult(operation_id=2, status=<UpdateStatus.COMPLETED: 'completed'>)
```

```python
# this will show the amount of vectors AFTER deleting them
client.count(
    collection_name=my_collection,
    exact=True,
)
```

```
CountResult(count=1000)
```

### 3.2 Payloads

With Qdrant you can store additional information alongside vectors. This is called a [payload](https://qdrant.tech/documentation/payload/) and it is represented as JSON objects. With these payloads, not only can you retrieve information when you search the database, but you can also filter your search by the parameters in the payload, and we'll see how in a second.

Following the narrative that our dummy vectors "represent a song," in a semantic search system for this kind of data, you would want to retrieve the song file, its URL, the artist or the genre, among others.

We will take advantage of a Python package called `faker` and create a bit of fake information to add to our payload to test this functionality.

```python
from faker import Faker
```

<hr />

```python
fake_something = Faker()
fake_something.name()
```

```
'Alex Yoder'
```

For each vector, you can create list of dictionaries containing the artist's name, the song, a url to the song, the year in which it was released, and the country where it originated from.

```python
payload = []

for i in range(len(data)):
    payload.append(
        {
            "artist": fake_something.name(),
            "song": " ".join(fake_something.words()),
            "url_song": fake_something.url(),
            "year": fake_something.year(),
            "country": fake_something.country(),
        }
    )

payload[:3]
```

```
[{'artist': 'Brandy Thomas',
  'song': 'amount painting result',
  'url_song': 'https://morales.info/',
  'year': '1985',
  'country': 'Croatia'},
 {'artist': 'Dr. Lisa Flynn',
  'song': 'thing admit cultural',
  'url_song': 'http://garza.com/',
  'year': '1973',
  'country': 'Australia'},
 {'artist': 'Eric Lee',
  'song': 'yeah argue throughout',
  'url_song': 'http://roberts-cruz.net/',
  'year': '2022',
  'country': 'Gambia'}]
```

You can upsert your Points (ids, data, and payload), with the same `client.upsert()` method you used earlier.

```python
client.upsert(
    collection_name=my_collection,
    points=models.Batch(ids=index, vectors=data.tolist(), payloads=payload),
)
```

```
UpdateResult(operation_id=3, status=<UpdateStatus.COMPLETED: 'completed'>)
```

If you want to retrieve this info, use the `client.retrieve()` method.

```python
resutls = client.retrieve(
    collection_name=my_collection, ids=[10, 50, 100, 500], with_vectors=False
)

type(resutls), resutls
```

```
(list,
 [Record(id=10, payload={'artist': 'Elizabeth White', 'country': 'Madagascar', 'song': 'rock idea option', 'url_song': 'https://johnson-young.com/', 'year': '2009'}, vector=None, shard_key=None, order_value=None),
  Record(id=50, payload={'artist': 'Alex Noble', 'country': 'Monaco', 'song': 'spend result air', 'url_song': 'http://www.becker.com/', 'year': '2017'}, vector=None, shard_key=None, order_value=None),
  Record(id=100, payload={'artist': 'Jimmy Martinez', 'country': 'Qatar', 'song': 'bag they itself', 'url_song': 'https://www.bishop.com/', 'year': '2006'}, vector=None, shard_key=None, order_value=None),
  Record(id=500, payload={'artist': 'David Wilson', 'country': 'Argentina', 'song': 'same soon service', 'url_song': 'http://matthews.org/', 'year': '2002'}, vector=None, shard_key=None, order_value=None)])
```

The response is a list with records and each element inside a record can be accessed as an attribute, e.g. `.payload` or `.id`.

```python
resutls[0].payload
```

```
{'artist': 'Elizabeth White',
 'country': 'Madagascar',
 'song': 'rock idea option',
 'url_song': 'https://johnson-young.com/',
 'year': '2009'}
```

```python
resutls[0].id
```

```
10
```

Next, you will use the payload to conduct a search query.

### 3.3 Search

Now that you have your vectors with an ID and a payload, you can start searching for content when new music gets selected.

Assume that a new song (like ["living la vida loca"](https://www.youtube.com/watch?v=p47fEXGabaY&ab_channel=RickyMartinVEVO) by Ricky Martin) comes in and our model immediately transforms it into a vector. Since we don't want a large amount of values back, let's limit the search to a few points.

```python
living_la_vida_loca = create_song()
```

<hr />

```python
client.query_points(
    collection_name=my_collection, query=living_la_vida_loca, limit=3
).points
```

```
[ScoredPoint(id=874, version=3, score=0.33814278, payload={'artist': 'Yolanda Rosario', 'country': 'Puerto Rico', 'song': 'without source watch', 'url_song': 'https://li.com/', 'year': '1972'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=532, version=3, score=0.31217313, payload={'artist': 'Tamara Fowler', 'country': 'Yemen', 'song': 'security very since', 'url_song': 'https://www.jimenez-perkins.com/', 'year': '1987'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=437, version=3, score=0.3039009, payload={'artist': 'Jeffrey Wilson', 'country': 'Bangladesh', 'song': 'example bag where', 'url_song': 'https://larson.com/', 'year': '1989'}, vector=None, shard_key=None, order_value=None)]
```

Assume you only want Australian songs recommended to you. For this, you can filter the query using the information in the payload. You have to first create a filter object and then pass it to the search method as an argument to the parameter `query_filter=`.

```python
aussie_songs = models.Filter(
    must=[
        models.FieldCondition(key="country", match=models.MatchValue(value="Australia"))
    ]
)
type(aussie_songs)
```

```
qdrant_client.http.models.models.Filter
```

```python
client.query_points(
    collection_name=my_collection,
    query=living_la_vida_loca,
    query_filter=aussie_songs,
    limit=2,
).points
```

```
[ScoredPoint(id=101, version=3, score=0.09454475, payload={'artist': 'Lindsay Johnson', 'country': 'Australia', 'song': 'program television tax', 'url_song': 'http://www.wright.com/', 'year': '1985'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=1, version=3, score=0.029051725, payload={'artist': 'Dr. Lisa Flynn', 'country': 'Australia', 'song': 'thing admit cultural', 'url_song': 'http://garza.com/', 'year': '1973'}, vector=None, shard_key=None, order_value=None)]
```

Lastly, assume we want aussie songs but we don't care how new or old these songs are. Exclude the year from the payload.

```python
client.query_points(
    collection_name=my_collection,
    query=living_la_vida_loca,
    query_filter=aussie_songs,
    with_payload=models.PayloadSelectorExclude(exclude=["year"]),
    limit=5,
).points
```

```
[ScoredPoint(id=101, version=3, score=0.09454475, payload={'artist': 'Lindsay Johnson', 'country': 'Australia', 'song': 'program television tax', 'url_song': 'http://www.wright.com/'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=1, version=3, score=0.029051725, payload={'artist': 'Dr. Lisa Flynn', 'country': 'Australia', 'song': 'thing admit cultural', 'url_song': 'http://garza.com/'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=222, version=3, score=-0.06252488, payload={'artist': 'Laura Maldonado', 'country': 'Australia', 'song': 'despite sea far', 'url_song': 'https://clark.com/'}, vector=None, shard_key=None, order_value=None)]
```

As you can see, you can apply a wide-range of filtering methods to allows your users to take more control of the recommendations they are being served.

If you wanted to clear out the payload and upload a new one for the same vectors, you can use `client.clear_payload()` as below.

```python
client.clear_payload(
    collection_name=my_collection,
    points_selector=models.PointIdsList(
        points=index,
    )
)
```

## 4. Recommendation systems

A recommendation system is a technology that suggests items or content to users based on their preferences, interests, or past behavior. In its most widely-used form, recommendation systems work by analyzing data about you and other users. The system looks at your previous choices, such as movies you've watched, products you've bought, or articles you've read. It then compares this information with data from other people who have similar tastes or interests.

Such systems are used in various companies such as Netflix, Amazon, Tik-Tok, and Spotify. They aim to personalize your experience, save you time searching for things you might like, or introduce you to new and relevant content that you may not have discovered otherwise.

Qdrant's API supports such a system, letting you account for user feedback. For example, you can recommend songs based on user likes (👍) or exclude similar ones to content users have disliked (👎).

To do this, use the `client.recommend()` method and consider the following elements:

- `collection_name=` - the collection from which the vectors are selected
- `query_filter=` - optional filter to apply to your search
- `negative=` - optionally, specify the `id` of disliked songs to exclude other semantically similar songs
- `positive=` - in case of liked songs, specify their `id` to exclude similar songs (required)
- `limit=` - specifies how many songs to show to the user

Imagine there are two songs, "[Suegra](https://www.youtube.com/watch?v=p7ff5EntWsE&ab_channel=RomeoSantosVEVO)" by Romeo Santos and "[Worst Behavior](https://www.youtube.com/watch?v=U5pzmGX8Ztg&ab_channel=DrakeVEVO)" by Drake represented by the ids 17 and 120 respectively. Let's see what we would get with the former being a 👍 and the latter being a 👎.

```python
client.recommend(collection_name=my_collection, positive=[17], limit=5)
```

```
[ScoredPoint(id=46, version=3, score=0.3310853, payload={'artist': 'Felicia Yang', 'country': 'Brunei Darussalam', 'song': 'pattern help within', 'url_song': 'http://www.gonzalez.biz/', 'year': '2006'}, vector=None),
 ScoredPoint(id=840, version=3, score=0.3030827, payload={'artist': 'Charles Brown', 'country': 'Somalia', 'song': 'network national very', 'url_song': 'https://www.larson-hartman.com/', 'year': '1975'}, vector=None),
 ScoredPoint(id=771, version=3, score=0.2851892, payload={'artist': 'Lori Clark', 'country': 'Myanmar', 'song': 'why box back', 'url_song': 'https://sanchez-waters.biz/', 'year': '1992'}, vector=None),
 ScoredPoint(id=304, version=3, score=0.28324583, payload={'artist': 'Steven Fitzgerald', 'country': 'Kyrgyz Republic', 'song': 'so suddenly indicate', 'url_song': 'http://www.stevens.com/', 'year': '1994'}, vector=None),
 ScoredPoint(id=544, version=3, score=0.26716217, payload={'artist': 'Christopher Bowman', 'country': 'Yemen', 'song': 'strategy late same', 'url_song': 'http://herrera.com/', 'year': '2012'}, vector=None)]
```

```python
client.recommend(
    collection_name=my_collection,
    query_vector=living_la_vida_loca,
    positive=[17],
    negative=[120],
    limit=5,
)
```

```
[ScoredPoint(id=756, version=3, score=0.31679478, payload={'artist': 'Chad Garza', 'country': 'Namibia', 'song': 'movie find method', 'url_song': 'http://www.moore.com/', 'year': '2018'}, vector=None),
 ScoredPoint(id=46, version=3, score=0.2964203, payload={'artist': 'Felicia Yang', 'country': 'Brunei Darussalam', 'song': 'pattern help within', 'url_song': 'http://www.gonzalez.biz/', 'year': '2006'}, vector=None),
 ScoredPoint(id=233, version=3, score=0.28002173, payload={'artist': 'Julie King', 'country': 'Congo', 'song': 'since really house', 'url_song': 'https://williams.com/', 'year': '1971'}, vector=None),
 ScoredPoint(id=349, version=3, score=0.25943768, payload={'artist': 'Geoffrey Wagner', 'country': 'Zimbabwe', 'song': 'station condition candidate', 'url_song': 'https://jackson.net/', 'year': '2015'}, vector=None),
 ScoredPoint(id=304, version=3, score=0.2552938, payload={'artist': 'Steven Fitzgerald', 'country': 'Kyrgyz Republic', 'song': 'so suddenly indicate', 'url_song': 'http://www.stevens.com/', 'year': '1994'}, vector=None)]
```

Notice that, while the similarity scores are completely random for this example, it is important to we pay attention to the scores retrieved when serving recommendations in production. Even if you get 5 vectors back, it might be more useful to show random results, rather than vectors that are 0.012 similar to the query vector. With this in mind, you can actually set a threshold for our vectors with the `score_threshold=` parameter.

```python
client.query_points(
    collection_name=my_collection,
    query=models.RecommendQuery(
        recommend=models.RecommendInput(positive=[17], negative=[120, 180])
    ),
    score_threshold=0.22,
    limit=5,
).points
```

```
[ScoredPoint(id=67, version=3, score=0.32367542, payload={'artist': 'Thomas Burns', 'country': 'United Kingdom', 'song': 'at go class', 'url_song': 'https://www.wiley-morgan.com/', 'year': '1981'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=122, version=3, score=0.29786378, payload={'artist': 'Lori Meyer', 'country': 'Barbados', 'song': 'among contain strong', 'url_song': 'http://www.harris.org/', 'year': '2011'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=248, version=3, score=0.2886583, payload={'artist': 'Jacob Horn', 'country': 'Morocco', 'song': 'decide question able', 'url_song': 'http://www.miller-richmond.com/', 'year': '1994'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=956, version=3, score=0.28638673, payload={'artist': 'Amy Francis', 'country': 'Italy', 'song': 'whole beyond floor', 'url_song': 'https://www.williams.info/', 'year': '2019'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=984, version=3, score=0.2774799, payload={'artist': 'Evan Jordan', 'country': 'Nigeria', 'song': 'old raise glass', 'url_song': 'http://www.jenkins-nunez.biz/', 'year': '2005'}, vector=None, shard_key=None, order_value=None)]
```

Lastly, you can add filters in the same way as you did before. Note that these filters could be tags that your users get to pick such as, for example, genres including `reggeaton`, `bachata`, and `salsa` (sorry Drake), or the language of the song.

```python
client.query_points(
    collection_name=my_collection,
    query=models.RecommendQuery(
        recommend=models.RecommendInput(positive=[17], negative=[120, 180])
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="country", match=models.MatchValue(value="Dominican Republic")
            )
        ]
    ),
    limit=5,
).points
```

```
[ScoredPoint(id=81, version=3, score=0.037227698, payload={'artist': 'Danielle Acosta', 'country': 'Dominican Republic', 'song': 'fund stage last', 'url_song': 'http://pacheco.com/', 'year': '1974'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=833, version=3, score=0.031101033, payload={'artist': 'Vanessa Howard', 'country': 'Dominican Republic', 'song': 'report recently spring', 'url_song': 'http://rosario-romero.com/', 'year': '2019'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=787, version=3, score=-0.01613924, payload={'artist': 'John Lee', 'country': 'Dominican Republic', 'song': 'safe present so', 'url_song': 'https://church-cook.biz/', 'year': '2018'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=78, version=3, score=-0.09762458, payload={'artist': 'Mary Alvarado', 'country': 'Dominican Republic', 'song': 'great letter specific', 'url_song': 'http://cole.com/', 'year': '1986'}, vector=None, shard_key=None, order_value=None),
 ScoredPoint(id=332, version=3, score=-0.12651296, payload={'artist': 'Cheryl Garrett', 'country': 'Dominican Republic', 'song': 'drive design animal', 'url_song': 'https://kerr-davis.com/', 'year': '1971'}, vector=None, shard_key=None, order_value=None)]
```

That's it! You have now gone over a whirlwind tour of vector databases and are ready to tackle new challenges. 😎

## 5. Conclusion

To wrap up, we have explored a bit of the fascinating world of vector databases, and we learned that these databases provide efficient storage and retrieval of high-dimensional vectors, making them ideal for similarity-based search tasks and recommendation systems. Both of these use cases can be applied in a variety of industries while helping us unlock new levels of information retrieval. In particular, recommendation systems built with Qdrant provide developers with enough flexibility to add and subtract data points that users liked or disliked, respectively, and even set up a threshold for how similar a recommendation must be before our applications can serve it.

We can't wait to see what cool applications you build with Qdrant.

If you liked this introductory tutorial, make sure you keep an eye out for new ones on our website.

## 6. Resources

Here is a list with some resources that we found useful, and that helped with the development of this tutorial.

- [Fine Tuning Similar Cars Search](https://qdrant.tech/articles/cars-recognition/)
- [Q&A with Similarity Learning](https://qdrant.tech/articles/faq-question-answering/)
- [Question Answering with LangChain and Qdrant without boilerplate](https://qdrant.tech/articles/langchain-integration/)
- [Extending ChatGPT with a Qdrant-based knowledge base](https://qdrant.tech/articles/chatgpt-plugin/)
- [Word Embedding and Word2Vec, Clearly Explained!!!](https://www.youtube.com/watch?v=viZrOnJclY0&ab_channel=StatQuestwithJoshStarmer) by StatQuest with Josh Starmer
- [Word Embeddings, Bias in ML, Why You Don't Like Math, & Why AI Needs You](https://www.youtube.com/watch?v=25nC0n9ERq4&ab_channel=RachelThomas) by Rachel Thomas
