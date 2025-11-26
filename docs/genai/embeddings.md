# Embeddings

The Gemini API offers text embedding models to generate embeddings for words,
phrases, sentences, and code. These foundational embeddings power advanced NLP
tasks such as semantic search, classification, and clustering, providing more
accurate, context-aware results than keyword-based approaches.

Building Retrieval Augmented Generation (RAG) systems is a common use case for
embeddings. Embeddings play a key role in significantly enhancing model outputs
with improved factual accuracy, coherence, and contextual richness. They
efficiently retrieve relevant information from knowledge bases, represented by
embeddings, which are then passed as additional context in the input prompt to
language models, guiding it to generate more informed and accurate responses.

To learn more about the available embedding model variants, see the [Model
versions](#model-versions) section. For higher throughput serving at half the
price, try [Batch API Embedding](#batch-embedding).

## Generating embeddings

Use the `embedContent` method to generate text embeddings:

### Python

```
from google import genai

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?")

print(result.embeddings)

```

### JavaScript

```
import { GoogleGenAI } from "@google/genai";

async function main() {

    const ai = new GoogleGenAI({});

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: 'What is the meaning of life?',
    });

    console.log(response.embeddings);
}

main();

```

### Go

```
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    client, err := genai.NewClient(ctx, nil)
    if err != nil {
        log.Fatal(err)
    }

    contents := []*genai.Content{
        genai.NewContentFromText("What is the meaning of life?", genai.RoleUser),
    }
    result, err := client.Models.EmbedContent(ctx,
        "gemini-embedding-001",
        contents,
        nil,
    )
    if err != nil {
        log.Fatal(err)
    }

    embeddings, err := json.MarshalIndent(result.Embeddings, "", "  ")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(embeddings))
}

```

### REST

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/gemini-embedding-001",
     "content": {"parts":[{"text": "What is the meaning of life?"}]}
    }'

```

You can also generate embeddings for multiple chunks at once by passing them in
as a list of strings.

### Python

```
from google import genai

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ])

for embedding in result.embeddings:
    print(embedding)

```

### JavaScript

```
import { GoogleGenAI } from "@google/genai";

async function main() {

    const ai = new GoogleGenAI({});

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: [
            'What is the meaning of life?',
            'What is the purpose of existence?',
            'How do I bake a cake?'
        ],
    });

    console.log(response.embeddings);
}

main();

```

### Go

```
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    client, err := genai.NewClient(ctx, nil)
    if err != nil {
        log.Fatal(err)
    }

    contents := []*genai.Content{
        genai.NewContentFromText("What is the meaning of life?"),
        genai.NewContentFromText("How does photosynthesis work?"),
        genai.NewContentFromText("Tell me about the history of the internet."),
    }
    result, err := client.Models.EmbedContent(ctx,
        "gemini-embedding-001",
        contents,
        nil,
    )
    if err != nil {
        log.Fatal(err)
    }

    embeddings, err := json.MarshalIndent(result.Embeddings, "", "  ")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(embeddings))
}

```

### REST

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"requests": [{
    "model": "models/gemini-embedding-001",
    "content": {
    "parts":[{
        "text": "What is the meaning of life?"}]}, },
    {
    "model": "models/gemini-embedding-001",
    "content": {
    "parts":[{
        "text": "How much wood would a woodchuck chuck?"}]}, },
    {
    "model": "models/gemini-embedding-001",
    "content": {
    "parts":[{
        "text": "How does the brain work?"}]}, }, ]}' 2> /dev/null | grep -C 5 values
    ```

```

## Specify task type to improve performance

You can use embeddings for a wide range of tasks from classification to document
search. Specifying the right task type helps optimize the embeddings for the
intended relationships, maximizing accuracy and efficiency. For a complete list
of supported task types, see the [Supported task types](#supported-task-types)
table.

The following example shows how you can use
`SEMANTIC_SIMILARITY` to check how similar in meaning strings of texts are.

**Note:** Cosine similarity is a good distance metric because it focuses on
direction rather than magnitude, which more accurately reflects conceptual
closeness. Values range from -1 (opposite) to 1 (greatest similarity).

### Python

```
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?"]

result = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
]

# Calculate cosine similarity. Higher scores = greater semantic similarity.

embeddings_matrix = np.array(result)
similarity_matrix = cosine_similarity(embeddings_matrix)

for i, text1 in enumerate(texts):
    for j in range(i + 1, len(texts)):
        text2 = texts[j]
        similarity = similarity_matrix[i, j]
        print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")

```

### JavaScript

```
import { GoogleGenAI } from "@google/genai";
import * as cosineSimilarity from "compute-cosine-similarity";

async function main() {
    const ai = new GoogleGenAI({});

    const texts = [
        "What is the meaning of life?",
        "What is the purpose of existence?",
        "How do I bake a cake?",
    ];

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        contents: texts,
        taskType: 'SEMANTIC_SIMILARITY'
    });

    const embeddings = response.embeddings.map(e => e.values);

    for (let i = 0; i < texts.length; i++) {
        for (let j = i + 1; j < texts.length; j++) {
            const text1 = texts[i];
            const text2 = texts[j];
            const similarity = cosineSimilarity(embeddings[i], embeddings[j]);
            console.log(`Similarity between '${text1}' and '${text2}': ${similarity.toFixed(4)}`);
        }
    }
}

main();

```

### Go

```
package main

import (
    "context"
    "fmt"
    "log"
    "math"

    "google.golang.org/genai"
)

// cosineSimilarity calculates the similarity between two vectors.
func cosineSimilarity(a, b []float32) (float64, error) {
    if len(a) != len(b) {
        return 0, fmt.Errorf("vectors must have the same length")
    }

    var dotProduct, aMagnitude, bMagnitude float64
    for i := 0; i < len(a); i++ {
        dotProduct += float64(a[i] * b[i])
        aMagnitude += float64(a[i] * a[i])
        bMagnitude += float64(b[i] * b[i])
    }

    if aMagnitude == 0 || bMagnitude == 0 {
        return 0, nil
    }

    return dotProduct / (math.Sqrt(aMagnitude) * math.Sqrt(bMagnitude)), nil
}

func main() {
    ctx := context.Background()
    client, _ := genai.NewClient(ctx, nil)
    defer client.Close()

    texts := []string{
        "What is the meaning of life?",
        "What is the purpose of existence?",
        "How do I bake a cake?",
    }

    var contents []*genai.Content
    for _, text := range texts {
        contents = append(contents, genai.NewContentFromText(text, genai.RoleUser))
    }

    result, _ := client.Models.EmbedContent(ctx,
        "gemini-embedding-001",
        contents,
        &genai.EmbedContentRequest{TaskType: genai.TaskTypeSemanticSimilarity},
    )

    embeddings := result.Embeddings

    for i := 0; i < len(texts); i++ {
        for j := i + 1; j < len(texts); j++ {
            similarity, _ := cosineSimilarity(embeddings[i].Values, embeddings[j].Values)
            fmt.Printf("Similarity between '%s' and '%s': %.4f\n", texts[i], texts[j], similarity)
        }
    }
}

```

### REST

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"task_type": "SEMANTIC_SIMILARITY",
    "content": {
    "parts":[{
    "text": "What is the meaning of life?"}, {"text": "How much wood would a woodchuck chuck?"}, {"text": "How does the brain work?"}]}
    }'

```

The following shows an example output from this code snippet:

```
Similarity between 'What is the meaning of life?' and 'What is the purpose of existence?': 0.9481

Similarity between 'What is the meaning of life?' and 'How do I bake a cake?': 0.7471

Similarity between 'What is the purpose of existence?' and 'How do I bake a cake?': 0.7371

```

### Supported task types

Task type | Description | Examples || **SEMANTIC\_SIMILARITY** | Embeddings optimized to assess text similarity. | Recommendation systems, duplicate detection |
| **CLASSIFICATION** | Embeddings optimized to classify texts according to preset labels. | Sentiment analysis, spam detection |
| **CLUSTERING** | Embeddings optimized to cluster texts based on their similarities. | Document organization, market research, anomaly detection |
| **RETRIEVAL\_DOCUMENT** | Embeddings optimized for document search. | Indexing articles, books, or web pages for search. |
| **RETRIEVAL\_QUERY** | Embeddings optimized for general search queries. Use `RETRIEVAL_QUERY` for queries; `RETRIEVAL_DOCUMENT` for documents to be retrieved. | Custom search |
| **CODE\_RETRIEVAL\_QUERY** | Embeddings optimized for retrieval of code blocks based on natural language queries. Use `CODE_RETRIEVAL_QUERY` for queries; `RETRIEVAL_DOCUMENT` for code blocks to be retrieved. | Code suggestions and search |
| **QUESTION\_ANSWERING** | Embeddings for questions in a question-answering system, optimized for finding documents that answer the question. Use `QUESTION_ANSWERING` for questions; `RETRIEVAL_DOCUMENT` for documents to be retrieved. | Chatbox |
| **FACT\_VERIFICATION** | Embeddings for statements that need to be verified, optimized for retrieving documents that contain evidence supporting or refuting the statement. Use `FACT_VERIFICATION` for the target text; `RETRIEVAL_DOCUMENT` for documents to be retrieved | Automated fact-checking systems |

## Controlling embedding size

The Gemini embedding model, `gemini-embedding-001`, is trained using the
Matryoshka Representation Learning (MRL) technique which teaches a model to
learn high-dimensional embeddings that have initial segments (or prefixes) which
are also useful, simpler versions of the same data.

Use the `output_dimensionality` parameter to control the size of
the output embedding vector. Selecting a smaller output dimensionality can save
storage space and increase computational efficiency for downstream applications,
while sacrificing little in terms of quality. By default, it outputs a
3072-dimensional embedding, but you can truncate it to a smaller size without
losing quality to save storage space. We recommend using 768, 1536, or 3072
output dimensions.

### Python

```
from google import genai
from google.genai import types

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(output_dimensionality=768)
)

[embedding_obj] = result.embeddings
embedding_length = len(embedding_obj.values)

print(f"Length of embedding: {embedding_length}")

```

### JavaScript

```
import { GoogleGenAI } from "@google/genai";

async function main() {
    const ai = new GoogleGenAI({});

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-001',
        content: 'What is the meaning of life?',
        outputDimensionality: 768,
    });

    const embeddingLength = response.embedding.values.length;
    console.log(`Length of embedding: ${embeddingLength}`);
}

main();

```

### Go

```
package main

import (
    "context"
    "fmt"
    "log"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    // The client uses Application Default Credentials.
    // Authenticate with 'gcloud auth application-default login'.
    client, err := genai.NewClient(ctx, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    contents := []*genai.Content{
        genai.NewContentFromText("What is the meaning of life?", genai.RoleUser),
    }

    result, err := client.Models.EmbedContent(ctx,
        "gemini-embedding-001",
        contents,
        &genai.EmbedContentRequest{OutputDimensionality: 768},
    )
    if err != nil {
        log.Fatal(err)
    }

    embedding := result.Embeddings[0]
    embeddingLength := len(embedding.Values)
    fmt.Printf("Length of embedding: %d\n", embeddingLength)
}

```

### REST

```
curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
        "content": {"parts":[{ "text": "What is the meaning of life?"}]},
        "output_dimensionality": 768
    }'

```

Example output from the code snippet:

```
Length of embedding: 768

```

## Ensuring quality for smaller dimensions

The 3072 dimension embedding is normalized. Normalized embeddings produce more
accurate semantic similarity by comparing vector direction, not magnitude. For
other dimensions, including 768 and 1536, you need to normalize the embeddings
as follows:

### Python

```
import numpy as np
from numpy.linalg import norm

embedding_values_np = np.array(embedding_obj.values)
normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

print(f"Normed embedding length: {len(normed_embedding)}")
print(f"Norm of normed embedding: {np.linalg.norm(normed_embedding):.6f}") # Should be very close to 1

```

Example output from this code snippet:

```
Normed embedding length: 768
Norm of normed embedding: 1.000000

```

The following table shows the MTEB scores, a commonly used benchmark for
embeddings, for different dimensions. Notably, the result shows that performance
is not strictly tied to the size of the embedding dimension, with lower
dimensions achieving scores comparable to their higher dimension counterparts.

MRL Dimension | MTEB Score || 2048 | 68.16 |
| 1536 | 68.17 |
| 768 | 67.99 |
| 512 | 67.55 |
| 256 | 66.19 |
| 128 | 63.31 |

## Use cases

Text embeddings are crucial for a variety of common AI use cases, such as:

* **Retrieval-Augmented Generation (RAG):** Embeddings enhance the quality
  of generated text by retrieving and incorporating relevant information into
  the context of a model.
* **Information retrieval:** Search for the most semantically similar text or
  documents given a piece of input text.

  [Document search tutorialtask](https://github.com/google-gemini/cookbook/blob/main/examples/Talk_to_documents_with_embeddings.ipynb)
* **Search reranking**: Prioritize the most relevant items by semantically
  scoring initial results against the query.

  [Search reranking tutorialtask](https://github.com/google-gemini/cookbook/blob/main/examples/Search_reranking_using_embeddings.ipynb)
* **Anomaly detection:** Comparing groups of embeddings can help identify
  hidden trends or outliers.

  [Anomaly detection tutorialbubble\_chart](https://github.com/google-gemini/cookbook/blob/main/examples/Anomaly_detection_with_embeddings.ipynb)
* **Classification:** Automatically categorize text based on its content, such
  as sentiment analysis or spam detection

  [Classification tutorialtoken](https://github.com/google-gemini/cookbook/blob/main/examples/Classify_text_with_embeddings.ipynb)
* **Clustering:** Effectively grasp complex relationships by creating clusters
  and visualizations of your embeddings.

  [Clustering visualization tutorialbubble\_chart](https://github.com/google-gemini/cookbook/blob/main/examples/clustering_with_embeddings.ipynb)

## Storing embeddings

As you take embeddings to production, it is common to
use **vector databases** to efficiently store, index, and retrieve
high-dimensional embeddings. Google Cloud offers managed data services that
can be used for this purpose including
[BigQuery](https://cloud.google.com/bigquery/docs/introduction),
[AlloyDB](https://cloud.google.com/alloydb/docs/overview), and
[Cloud SQL](https://cloud.google.com/sql/docs/postgres/introduction).

The following tutorials show how to use other third party vector databases
with Gemini Embedding.

* [ChromaDB tutorialsbolt](https://github.com/google-gemini/cookbook/tree/main/examples/chromadb)
* [QDrant tutorialsbolt](https://github.com/google-gemini/cookbook/tree/main/examples/qdrant)
* [Weaviate tutorialsbolt](https://github.com/google-gemini/cookbook/tree/main/examples/weaviate)
* [Pinecone tutorialsbolt](https://github.com/google-gemini/cookbook/blob/main/examples/langchain/Gemini_LangChain_QA_Pinecone_WebLoad.ipynb)

## Model versions

| Property | Description |
| --- | --- |
| id\_cardModel code | **Gemini API**  `gemini-embedding-001` |
| saveSupported data types | **Input**  Text  **Output**  Text embeddings |
| token\_autoToken limits[[\*]](/gemini-api/docs/tokens) | **Input token limit**  2,048  **Output dimension size**  Flexible, supports: 128 - 3072, Recommended: 768, 1536, 3072 |
| 123Versions | Read the [model version patterns](/gemini-api/docs/models/gemini#model-versions) for more details.  * Stable: `gemini-embedding-001` * Experimental: `gemini-embedding-exp-03-07` (deprecating in Oct of 2025) |
| calendar\_monthLatest update | June 2025 |

## Batch embeddings

If latency is not a concern, try using the Gemini Embeddings model with
[Batch API](/gemini-api/docs/batch-api#batch-embedding). This
allows for much higher throughput at 50% of interactive Embedding pricing.
Find examples on how to get started in the [Batch API cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Batch_mode.ipynb).

## Responsible use notice

Unlike generative AI models that create new content, the Gemini Embedding model
is only intended to transform the format of your input data into a numerical
representation. While Google is responsible for providing an embedding model
that transforms the format of your input data to the numerical-format requested,
users retain full responsibility for the data they input and the resulting
embeddings. By using the Gemini Embedding model you confirm that you have the
necessary rights to any content that you upload. Do not generate content that
infringes on others' intellectual property or privacy rights. Your use of this
service is subject to our [Prohibited Use
Policy](https://policies.google.com/terms/generative-ai/use-policy) and
[Google's Terms of Service](/gemini-api/terms).

## Start building with embeddings

Check out the [embeddings quickstart
notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb)
to explore the model capabilities and learn how to customize and visualize your
embeddings.

## Deprecation notice for legacy models

The following models will be deprecated in October, 2025:
- `embedding-001`
- `embedding-gecko-001`
- `gemini-embedding-exp-03-07` (`gemini-embedding-exp`)

# Embeddings

Embeddings are a numerical representation of text input that open up a number of unique use cases, such as clustering, similarity measurement and information retrieval. For an introduction, check out the [Embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings).

Unlike generative AI models that create new content, the Gemini Embedding model is only intended to transform the format of your input data into a numerical representation. While Google is responsible for providing an embedding model that transforms the format of your input data to the numerical-format requested, users retain full responsibility for the data they input and the resulting embeddings. By using the Gemini Embedding model you confirm that you have the necessary rights to any content that you upload. Do not generate content that infringes on others' intellectual property or privacy rights. Your use of this service is subject to our [Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy) and [Google's Terms of Service](https://ai.google.dev/gemini-api/terms).

## Method: models.embedContent

* [Endpoint](#body.HTTP_TEMPLATE)
* [Path parameters](#body.PATH_PARAMETERS)
* [Request body](#body.request_body)
  + [JSON representation](#body.request_body.SCHEMA_REPRESENTATION)
* [Response body](#body.response_body)
* [Authorization scopes](#body.aspect)
* [Example request](#body.codeSnippets)
  + [Basic](#body.codeSnippets.group)

Generates a text embedding vector from the input `Content` using the specified [Gemini Embedding model](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding).

### Endpoint

post

`https://generativelanguage.googleapis.com/v1beta/{model=models/*}:embedContent`

### Path parameters

`model` |

`string`

Required. The model's resource name. This serves as an ID for the Model to use.

This name should match a model name returned by the `models.list` method.

Format: `models/{model}` It takes the form `models/{model}`.

### Request body

The request body contains data with the following structure:

| Fields | |
| --- | --- |

`content` |

`object ([Content](/api/caching#Content))`

Required. The content to embed. Only the `parts.text` fields will be counted.

`taskType` |

`enum ([TaskType](/api/embeddings#v1beta.TaskType))`

Optional. Optional task type for which the embeddings will be used. Not supported on earlier models (`models/embedding-001`).

`title` |

`string`

Optional. An optional title for the text. Only applicable when TaskType is `RETRIEVAL_DOCUMENT`.

Note: Specifying a `title` for `RETRIEVAL_DOCUMENT` provides better quality embeddings for retrieval.

`outputDimensionality` |

`integer`

Optional. Optional reduced dimension for the output embedding. If set, excessive values in the output embedding are truncated from the end. Supported by newer models since 2024 only. You cannot set this value if using the earlier model (`models/embedding-001`).

### Example request

### Python

```
from google import genai
from google.genai import types

client = genai.Client()
text = "Hello World!"
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=types.EmbedContentConfig(output_dimensionality=10),
)
print(result.embeddings)

embed.py

```

### Node.js

```
// Make sure to include the following import:
// import {GoogleGenAI} from '@google/genai';
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const text = "Hello World!";
const result = await ai.models.embedContent({
  model: "gemini-embedding-001",
  contents: text,
  config: { outputDimensionality: 10 },
});
console.log(result.embeddings);

embed.js

```

### Go

```
ctx := context.Background()
client, err := genai.NewClient(ctx, &genai.ClientConfig{
	APIKey:  os.Getenv("GEMINI_API_KEY"),
	Backend: genai.BackendGeminiAPI,
})
if err != nil {
	log.Fatal(err)
}

text := "Hello World!"
outputDim := int32(10)
contents := []*genai.Content{
	genai.NewContentFromText(text, genai.RoleUser),
}
result, err := client.Models.EmbedContent(ctx, "gemini-embedding-001",
	contents, &genai.EmbedContentConfig{
		OutputDimensionality: &outputDim,
})
if err != nil {
	log.Fatal(err)
}

embeddings, err := json.MarshalIndent(result.Embeddings, "", "  ")
if err != nil {
	log.Fatal(err)
}
fmt.Println(string(embeddings))

embed.go

```

### Shell

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/gemini-embedding-001",
     "content": {
     "parts":[{
     "text": "What is the meaning of life?"}]}
    }'

embed.sh

```

### Response body

If successful, the response body contains an instance of `[EmbedContentResponse](/api/embeddings#v1beta.EmbedContentResponse)`.

## Method: models.batchEmbedContents

* [Endpoint](#body.HTTP_TEMPLATE)
* [Path parameters](#body.PATH_PARAMETERS)
* [Request body](#body.request_body)
  + [JSON representation](#body.request_body.SCHEMA_REPRESENTATION)
* [Response body](#body.response_body)
  + [JSON representation](#body.BatchEmbedContentsResponse.SCHEMA_REPRESENTATION)
* [Authorization scopes](#body.aspect)
* [Example request](#body.codeSnippets)
  + [Basic](#body.codeSnippets.group)

Generates multiple embedding vectors from the input `Content` which consists of a batch of strings represented as `EmbedContentRequest` objects.

### Endpoint

post

`https://generativelanguage.googleapis.com/v1beta/{model=models/*}:batchEmbedContents`

### Path parameters

`model` |

`string`

Required. The model's resource name. This serves as an ID for the Model to use.

This name should match a model name returned by the `models.list` method.

Format: `models/{model}` It takes the form `models/{model}`.

### Request body

The request body contains data with the following structure:

| Fields | |
| --- | --- |

`requests[]` |

`object ([EmbedContentRequest](/api/batch-api#EmbedContentRequest))`

Required. Embed requests for the batch. The model in each of these requests must match the model specified `BatchEmbedContentsRequest.model`.

### Example request

### Python

```
from google import genai
from google.genai import types

client = genai.Client()
texts = [
    "What is the meaning of life?",
    "How much wood would a woodchuck chuck?",
    "How does the brain work?",
]
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(output_dimensionality=10),
)
print(result.embeddings)

embed.py

```

### Node.js

```
// Make sure to include the following import:
// import {GoogleGenAI} from '@google/genai';
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const texts = [
  "What is the meaning of life?",
  "How much wood would a woodchuck chuck?",
  "How does the brain work?",
];
const result = await ai.models.embedContent({
  model: "gemini-embedding-001",
  contents: texts,
  config: { outputDimensionality: 10 },
});
console.log(result.embeddings);

embed.js

```

### Go

```
ctx := context.Background()
client, err := genai.NewClient(ctx, &genai.ClientConfig{
	APIKey:  os.Getenv("GEMINI_API_KEY"),
	Backend: genai.BackendGeminiAPI,
})
if err != nil {
	log.Fatal(err)
}

contents := []*genai.Content{
	genai.NewContentFromText("What is the meaning of life?", genai.RoleUser),
	genai.NewContentFromText("How much wood would a woodchuck chuck?", genai.RoleUser),
	genai.NewContentFromText("How does the brain work?", genai.RoleUser),
}

outputDim := int32(10)
result, err := client.Models.EmbedContent(ctx, "gemini-embedding-001", contents, &genai.EmbedContentConfig{
	OutputDimensionality: &outputDim,
})
if err != nil {
	log.Fatal(err)
}

embeddings, err := json.MarshalIndent(result.Embeddings, "", "  ")
if err != nil {
	log.Fatal(err)
}
fmt.Println(string(embeddings))

embed.go

```

### Shell

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"requests": [{
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "What is the meaning of life?"}]}, },
      {
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "How much wood would a woodchuck chuck?"}]}, },
      {
      "model": "models/gemini-embedding-001",
      "content": {
      "parts":[{
        "text": "How does the brain work?"}]}, }, ]}' 2> /dev/null | grep -C 5 values

embed.sh

```

### Response body

The response to a `BatchEmbedContentsRequest`.

If successful, the response body contains data with the following structure:

| Fields | |
| --- | --- |

`embeddings[]` |

`object ([ContentEmbedding](/api/embeddings#v1beta.ContentEmbedding))`

Output only. The embeddings for each request, in the same order as provided in the batch request.

| JSON representation |
| --- |
| ``` {   "embeddings": [     {       object (ContentEmbedding)     }   ] } ``` |

## Method: models.asyncBatchEmbedContent

* [Endpoint](#body.HTTP_TEMPLATE)
* [Path parameters](#body.PATH_PARAMETERS)
* [Request body](#body.request_body)
  + [JSON representation](#body.request_body.SCHEMA_REPRESENTATION)
    - [JSON representation](#body.request_body.SCHEMA_REPRESENTATION.batch.SCHEMA_REPRESENTATION)
    - [JSON representation](#body.request_body.SCHEMA_REPRESENTATION.batch.SCHEMA_REPRESENTATION_1)
    - [JSON representation](#body.request_body.SCHEMA_REPRESENTATION.batch.SCHEMA_REPRESENTATION_2)
* [Response body](#body.response_body)
* [Authorization scopes](#body.aspect)

Enqueues a batch of `models.embedContent` requests for batch processing. We have a `models.batchEmbedContents` handler in `GenerativeService`, but it was synchronized. So we name this one to be `Async` to avoid confusion.

### Endpoint

post

`https://generativelanguage.googleapis.com/v1beta/{batch.model=models/*}:asyncBatchEmbedContent`

### Path parameters

`batch.model` |

`string`

Required. The name of the `Model` to use for generating the completion.

Format: `models/{model}`. It takes the form `models/{model}`.

### Request body

The request body contains data with the following structure:

| Fields | |
| --- | --- |

`batch.name` |

`string`

Output only. Identifier. Resource name of the batch.

Format: `batches/{batchId}`.

`batch.displayName` |

`string`

Required. The user-defined name of this batch.

`batch.inputConfig` |

`object ([InputEmbedContentConfig](/api/embeddings#InputEmbedContentConfig))`

Required. Input configuration of the instances on which batch processing are performed.

`batch.output` |

`object ([EmbedContentBatchOutput](/api/embeddings#EmbedContentBatchOutput))`

Output only. The output of the batch request.

`batch.createTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch was created.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`batch.endTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch processing completed.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`batch.updateTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch was last updated.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`batch.batchStats` |

`object ([EmbedContentBatchStats](/api/embeddings#EmbedContentBatchStats))`

Output only. Stats about the batch.

`batch.state` |

`enum ([BatchState](/api/batch-api#v1beta.BatchState))`

Output only. The state of the batch.

`batch.priority` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Optional. The priority of the batch. Batches with a higher priority value will be processed before batches with a lower priority value. Negative values are allowed. Default is 0.

### Response body

If successful, the response body contains an instance of `[Operation](/api/batch-api#Operation)`.

## EmbedContentResponse

* [JSON representation](#SCHEMA_REPRESENTATION)

The response to an `EmbedContentRequest`.

| Fields | |
| --- | --- |

`embedding` |

`object ([ContentEmbedding](/api/embeddings#v1beta.ContentEmbedding))`

Output only. The embedding generated from the input content.

| JSON representation |
| --- |
| ``` {   "embedding": {     object (ContentEmbedding)   } } ``` |

## ContentEmbedding

* [JSON representation](#SCHEMA_REPRESENTATION)

A list of floats representing an embedding.

| Fields | |
| --- | --- |

`values[]` |

`number`

The embedding values.

| JSON representation |
| --- |
| ``` {   "values": [     number   ] } ``` |

## TaskType

Type of task for which the embedding will be used.

| Enums | |
| --- | --- |
| `TASK_TYPE_UNSPECIFIED` | Unset value, which will default to one of the other enum values. |
| `RETRIEVAL_QUERY` | Specifies the given text is a query in a search/retrieval setting. |
| `RETRIEVAL_DOCUMENT` | Specifies the given text is a document from the corpus being searched. |
| `SEMANTIC_SIMILARITY` | Specifies the given text will be used for STS. |
| `CLASSIFICATION` | Specifies that the given text will be classified. |
| `CLUSTERING` | Specifies that the embeddings will be used for clustering. |
| `QUESTION_ANSWERING` | Specifies that the given text will be used for question answering. |
| `FACT_VERIFICATION` | Specifies that the given text will be used for fact verification. |
| `CODE_RETRIEVAL_QUERY` | Specifies that the given text will be used for code retrieval. |

## EmbedContentBatch

* [JSON representation](#SCHEMA_REPRESENTATION)
* [InputEmbedContentConfig](#InputEmbedContentConfig)
  + [JSON representation](#InputEmbedContentConfig.SCHEMA_REPRESENTATION)
* [InlinedEmbedContentRequests](#InlinedEmbedContentRequests)
  + [JSON representation](#InlinedEmbedContentRequests.SCHEMA_REPRESENTATION)
* [InlinedEmbedContentRequest](#InlinedEmbedContentRequest)
  + [JSON representation](#InlinedEmbedContentRequest.SCHEMA_REPRESENTATION)
* [EmbedContentBatchOutput](#EmbedContentBatchOutput)
  + [JSON representation](#EmbedContentBatchOutput.SCHEMA_REPRESENTATION)
* [InlinedEmbedContentResponses](#InlinedEmbedContentResponses)
  + [JSON representation](#InlinedEmbedContentResponses.SCHEMA_REPRESENTATION)
* [InlinedEmbedContentResponse](#InlinedEmbedContentResponse)
  + [JSON representation](#InlinedEmbedContentResponse.SCHEMA_REPRESENTATION)
* [EmbedContentBatchStats](#EmbedContentBatchStats)
  + [JSON representation](#EmbedContentBatchStats.SCHEMA_REPRESENTATION)

A resource representing a batch of `EmbedContent` requests.

| Fields | |
| --- | --- |

`model` |

`string`

Required. The name of the `Model` to use for generating the completion.

Format: `models/{model}`.

`name` |

`string`

Output only. Identifier. Resource name of the batch.

Format: `batches/{batchId}`.

`displayName` |

`string`

Required. The user-defined name of this batch.

`inputConfig` |

`object ([InputEmbedContentConfig](/api/embeddings#InputEmbedContentConfig))`

Required. Input configuration of the instances on which batch processing are performed.

`output` |

`object ([EmbedContentBatchOutput](/api/embeddings#EmbedContentBatchOutput))`

Output only. The output of the batch request.

`createTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch was created.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`endTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch processing completed.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`updateTime` |

`string ([Timestamp](https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp) format)`

Output only. The time at which the batch was last updated.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.

`batchStats` |

`object ([EmbedContentBatchStats](/api/embeddings#EmbedContentBatchStats))`

Output only. Stats about the batch.

`state` |

`enum ([BatchState](/api/batch-api#v1beta.BatchState))`

Output only. The state of the batch.

`priority` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Optional. The priority of the batch. Batches with a higher priority value will be processed before batches with a lower priority value. Negative values are allowed. Default is 0.

| JSON representation |
| --- |
| ``` {   "model": string,   "name": string,   "displayName": string,   "inputConfig": {     object (InputEmbedContentConfig)   },   "output": {     object (EmbedContentBatchOutput)   },   "createTime": string,   "endTime": string,   "updateTime": string,   "batchStats": {     object (EmbedContentBatchStats)   },   "state": enum (BatchState),   "priority": string } ``` |

## InputEmbedContentConfig

Configures the input to the batch request.

| Fields | |
| --- | --- |

|  |
| --- |
| `source` |

`Union type`

Required. The source of the input. `source` can be only one of the following:

`fileName` |

`string`

The name of the `File` containing the input requests.

`requests` |

`object ([InlinedEmbedContentRequests](/api/embeddings#InlinedEmbedContentRequests))`

The requests to be processed in the batch.

| JSON representation |
| --- |
| ``` {    // source   "fileName": string,   "requests": {     object (InlinedEmbedContentRequests)   }   // Union type } ``` |

## InlinedEmbedContentRequests

The requests to be processed in the batch if provided as part of the batch creation request.

| Fields | |
| --- | --- |

`requests[]` |

`object ([InlinedEmbedContentRequest](/api/embeddings#InlinedEmbedContentRequest))`

Required. The requests to be processed in the batch.

| JSON representation |
| --- |
| ``` {   "requests": [     {       object (InlinedEmbedContentRequest)     }   ] } ``` |

## InlinedEmbedContentRequest

The request to be processed in the batch.

| Fields | |
| --- | --- |

`request` |

`object ([EmbedContentRequest](/api/batch-api#EmbedContentRequest))`

Required. The request to be processed in the batch.

`metadata` |

`object ([Struct](https://protobuf.dev/reference/protobuf/google.protobuf/#struct) format)`

Optional. The metadata to be associated with the request.

| JSON representation |
| --- |
| ``` {   "request": {     object (EmbedContentRequest)   },   "metadata": {     object   } } ``` |

## EmbedContentBatchOutput

The output of a batch request. This is returned in the `AsyncBatchEmbedContentResponse` or the `EmbedContentBatch.output` field.

| Fields | |
| --- | --- |

|  |
| --- |
| `output` |

`Union type`

The output of the batch request. `output` can be only one of the following:

`responsesFile` |

`string`

Output only. The file ID of the file containing the responses. The file will be a JSONL file with a single response per line. The responses will be `EmbedContentResponse` messages formatted as JSON. The responses will be written in the same order as the input requests.

`inlinedResponses` |

`object ([InlinedEmbedContentResponses](/api/embeddings#InlinedEmbedContentResponses))`

Output only. The responses to the requests in the batch. Returned when the batch was built using inlined requests. The responses will be in the same order as the input requests.

| JSON representation |
| --- |
| ``` {    // output   "responsesFile": string,   "inlinedResponses": {     object (InlinedEmbedContentResponses)   }   // Union type } ``` |

## InlinedEmbedContentResponses

The responses to the requests in the batch.

| Fields | |
| --- | --- |

`inlinedResponses[]` |

`object ([InlinedEmbedContentResponse](/api/embeddings#InlinedEmbedContentResponse))`

Output only. The responses to the requests in the batch.

| JSON representation |
| --- |
| ``` {   "inlinedResponses": [     {       object (InlinedEmbedContentResponse)     }   ] } ``` |

## InlinedEmbedContentResponse

The response to a single request in the batch.

| Fields | |
| --- | --- |

`metadata` |

`object ([Struct](https://protobuf.dev/reference/protobuf/google.protobuf/#struct) format)`

Output only. The metadata associated with the request.

|  |
| --- |
| `output` |

`Union type`

The output of the request. `output` can be only one of the following:

`error` |

`object ([Status](/api/files#v1beta.Status))`

Output only. The error encountered while processing the request.

`response` |

`object ([EmbedContentResponse](/api/embeddings#v1beta.EmbedContentResponse))`

Output only. The response to the request.

| JSON representation |
| --- |
| ``` {   "metadata": {     object   },    // output   "error": {     object (Status)   },   "response": {     object (EmbedContentResponse)   }   // Union type } ``` |

## EmbedContentBatchStats

Stats about the batch.

| Fields | |
| --- | --- |

`requestCount` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Output only. The number of requests in the batch.

`successfulRequestCount` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Output only. The number of requests that were successfully processed.

`failedRequestCount` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Output only. The number of requests that failed to be processed.

`pendingRequestCount` |

`string ([int64](https://developers.google.com/discovery/v1/type-format) format)`

Output only. The number of requests that are still pending processing.

| JSON representation |
| --- |
| ``` {   "requestCount": string,   "successfulRequestCount": string,   "failedRequestCount": string,   "pendingRequestCount": string } ``` |


Last updated 2025-10-24 UTC.