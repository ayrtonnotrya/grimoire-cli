# Entender e contar tokens

O Gemini e outros modelos de IA generativa processam entradas e saídas em uma granularidade chamada *token*.

**Para modelos do Gemini, um token equivale a cerca de quatro caracteres.
100 tokens equivalem a cerca de 60 a 80 palavras em inglês.**

## Sobre tokens

Os tokens podem ser caracteres únicos, como `z`, ou palavras inteiras, como `cat`. Palavras longas são divididas em vários tokens. O conjunto de todos os tokens usados pelo modelo é chamado de vocabulário, e o processo de dividir o texto em tokens é chamado de *tokenização*.

Quando o faturamento está ativado, o [custo de uma chamada para a API Gemini](https://ai.google.dev/pricing?hl=pt-br) é determinado em parte pelo número de tokens de entrada e saída. Por isso, saber como contar tokens pode ser útil.

Python
JavaScript
Go

## Teste a contagem de tokens em um Colab

Você pode testar a contagem de tokens usando um Colab.

|  |  |  |
| --- | --- | --- |
| [![](https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png?hl=pt-br)](https://ai.google.dev/gemini-api/docs/tokens?hl=pt-br) | [![](https://www.tensorflow.org/images/colab_logo_32px.png?hl=pt-br)](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Counting_Tokens.ipynb?hl=pt-br) | [![](https://www.tensorflow.org/images/GitHub-Mark-32px.png?hl=pt-br)](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Counting_Tokens.ipynb?hl=pt-br) |

## Janelas de contexto

Os modelos disponíveis na API Gemini têm janelas de contexto medidas em tokens. A janela de contexto define a quantidade de entrada que você pode fornecer e a quantidade de saída que o modelo pode gerar. Para determinar o tamanho da janela de contexto, chame o [endpoint getModels](https://ai.google.dev/api/rest/v1/models/get?hl=pt-br) ou consulte a [documentação de modelos](https://ai.google.dev/gemini-api/docs/models/gemini?hl=pt-br).

No exemplo a seguir, o modelo `gemini-1.5-flash` tem um limite de entrada de cerca de 1 milhão de tokens e um limite de saída de cerca de 8 mil tokens, o que significa que uma janela de contexto é de 1 milhão de tokens.

```
from google import genai

client = genai.Client()
model_info = client.models.get(model="gemini-2.0-flash")
print(f"{model_info.input_token_limit=}")
print(f"{model_info.output_token_limit=}")
# ( e.g., input_token_limit=30720, output_token_limit=2048 )

count_tokens.py

```

## Contar tokens

Todas as entradas e saídas da API Gemini são tokenizadas, incluindo texto, arquivos de imagem e outras modalidades não textuais.

É possível contar tokens das seguintes maneiras:

* **Chame [`count_tokens`](https://ai.google.dev/api/rest/v1/models/countTokens?hl=pt-br) com a entrada da solicitação.**
  : retorna o número total de tokens *apenas na entrada*. Você pode fazer essa chamada antes de enviar a entrada ao modelo para verificar o tamanho das solicitações.
* **Use o atributo `usage_metadata` no objeto `response` depois de chamar
  `generate_content`.**
   Isso retorna o número total de tokens *tanto na entrada quanto na saída*: `total_token_count`.
   Ele também retorna as contagens de tokens da entrada e da saída separadamente: `prompt_token_count` (tokens de entrada) e `candidates_token_count` (tokens de saída).

  Se você estiver usando um [modelo de pensamento](https://ai.google.dev/gemini-api/docs/thinking?hl=pt-br), como os 2.5, os tokens usados durante o processo de pensamento serão retornados em `thoughts_token_count`. Se você estiver usando o [armazenamento em cache de contexto](https://ai.google.dev/gemini-api/docs/caching?hl=pt-br), a contagem de tokens em cache estará em `cached_content_token_count`.

### Contar tokens de texto

Se você chamar `count_tokens` com uma entrada somente de texto, ele vai retornar a contagem de tokens do texto *apenas na entrada* (`total_tokens`). É possível fazer essa chamada antes de chamar `generate_content` para verificar o tamanho das solicitações.

Outra opção é chamar `generate_content` e usar o atributo `usage_metadata`
no objeto `response` para receber o seguinte:

* As contagens de tokens separadas da entrada (`prompt_token_count`), do conteúdo em cache (`cached_content_token_count`) e da saída (`candidates_token_count`).
* A contagem de tokens para o processo de pensamento (`thoughts_token_count`)
* O número total de tokens na *entrada e na saída*
  (`total_token_count`)

```
from google import genai

client = genai.Client()
prompt = "The quick brown fox jumps over the lazy dog."

# Count tokens using the new client method.
total_tokens = client.models.count_tokens(
    model="gemini-2.0-flash", contents=prompt
)
print("total_tokens: ", total_tokens)
# ( e.g., total_tokens: 10 )

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
)

# The usage_metadata provides detailed token counts.
print(response.usage_metadata)
# ( e.g., prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )

count_tokens.py

```

### Contar tokens multiturno (chat)

Se você chamar `count_tokens` com o histórico de chat, ele vai retornar a contagem total de tokens do texto de cada função no chat (`total_tokens`).

Outra opção é chamar `send_message` e usar o atributo `usage_metadata`
no objeto `response` para receber o seguinte:

* As contagens de tokens separadas da entrada (`prompt_token_count`), do conteúdo em cache (`cached_content_token_count`) e da saída (`candidates_token_count`).
* A contagem de tokens para o processo de pensamento (`thoughts_token_count`)
* O número total de tokens na *entrada e na saída*
  (`total_token_count`)

Para entender o tamanho da sua próxima conversa, adicione-a ao histórico ao chamar `count_tokens`.

```
from google import genai
from google.genai import types

client = genai.Client()

chat = client.chats.create(
    model="gemini-2.0-flash",
    history=[
        types.Content(
            role="user", parts=[types.Part(text="Hi my name is Bob")]
        ),
        types.Content(role="model", parts=[types.Part(text="Hi Bob!")]),
    ],
)
# Count tokens for the chat history.
print(
    client.models.count_tokens(
        model="gemini-2.0-flash", contents=chat.get_history()
    )
)
# ( e.g., total_tokens: 10 )

response = chat.send_message(
    message="In one sentence, explain how a computer works to a young child."
)
print(response.usage_metadata)
# ( e.g., prompt_token_count: 25, candidates_token_count: 21, total_token_count: 46 )

# You can count tokens for the combined history and a new message.
extra = types.UserContent(
    parts=[
        types.Part(
            text="What is the meaning of life?",
        )
    ]
)
history = chat.get_history()
history.append(extra)
print(client.models.count_tokens(model="gemini-2.0-flash", contents=history))
# ( e.g., total_tokens: 56 )

count_tokens.py

```

### Contar tokens multimodais

Todas as entradas da API Gemini são tokenizadas, incluindo texto, arquivos de imagem e outras modalidades não textuais. Confira os principais pontos sobre a tokenização de entradas multimodais durante o processamento pela API Gemini:

* Com o Gemini 2.0, as entradas de imagem com as duas dimensões <=384 pixels são contadas como
  258 tokens. Imagens maiores em uma ou ambas as dimensões são cortadas e dimensionadas conforme
  necessário em blocos de 768 x 768 pixels, cada um contado como 258 tokens. Antes do Gemini 2.0, as imagens usavam 258 tokens fixos.
* Os arquivos de vídeo e áudio são convertidos em tokens nas seguintes taxas fixas: vídeo a 263 tokens por segundo e áudio a 32 tokens por segundo.

#### Resoluções de mídia

O pré-lançamento do Gemini 3 Pro apresenta controle granular sobre o processamento de visão multimodal com o parâmetro
`media_resolution`. O parâmetro `media_resolution` determina o **número máximo de tokens alocados por imagem de entrada ou frame de vídeo**.
Resoluções mais altas melhoram a capacidade do modelo de ler textos pequenos ou identificar detalhes pequenos, mas aumentam o uso de tokens e a latência.

Para mais detalhes sobre o parâmetro e como ele pode afetar os cálculos de token, consulte o guia de [resolução de mídia](https://ai.google.dev/gemini-api/docs/media-resolution?hl=pt-br).

#### Arquivos de imagem

Se você chamar `count_tokens` com uma entrada de texto e imagem, ele vai retornar a contagem combinada de tokens do texto e da imagem *apenas na entrada* (`total_tokens`). Você pode fazer essa chamada antes de chamar `generate_content` para verificar o tamanho das suas solicitações. Você também pode chamar `count_tokens` no texto e no arquivo separadamente.

Outra opção é chamar `generate_content` e usar o atributo `usage_metadata`
no objeto `response` para receber o seguinte:

* As contagens de tokens separadas da entrada (`prompt_token_count`), do conteúdo em cache (`cached_content_token_count`) e da saída (`candidates_token_count`).
* A contagem de tokens para o processo de pensamento (`thoughts_token_count`)
* O número total de tokens na *entrada e na saída*
  (`total_token_count`)

**Observação**: você vai receber a mesma contagem de tokens se usar um arquivo enviado por upload com a
API File ou se fornecer o arquivo como dados inline.

Exemplo que usa uma imagem enviada da API File:

```
from google import genai

client = genai.Client()
prompt = "Tell me about this image"
your_image_file = client.files.upload(file=media / "organ.jpg")

print(
    client.models.count_tokens(
        model="gemini-2.0-flash", contents=[prompt, your_image_file]
    )
)
# ( e.g., total_tokens: 263 )

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[prompt, your_image_file]
)
print(response.usage_metadata)
# ( e.g., prompt_token_count: 264, candidates_token_count: 80, total_token_count: 345 )

count_tokens.py

```

Exemplo que fornece a imagem como dados inline:

```
from google import genai
import PIL.Image

client = genai.Client()
prompt = "Tell me about this image"
your_image_file = PIL.Image.open(media / "organ.jpg")

# Count tokens for combined text and inline image.
print(
    client.models.count_tokens(
        model="gemini-2.0-flash", contents=[prompt, your_image_file]
    )
)
# ( e.g., total_tokens: 263 )

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[prompt, your_image_file]
)
print(response.usage_metadata)
# ( e.g., prompt_token_count: 264, candidates_token_count: 80, total_token_count: 345 )

count_tokens.py

```

#### Arquivos de vídeo ou áudio

O áudio e o vídeo são convertidos em tokens nas seguintes taxas fixas:

* Vídeo: 263 tokens por segundo
* Áudio: 32 tokens por segundo

Se você chamar `count_tokens` com uma entrada de texto e vídeo/áudio, ela vai retornar a contagem combinada de tokens do texto e do arquivo de vídeo/áudio *somente na entrada* (`total_tokens`). É possível fazer essa chamada antes de chamar `generate_content` para verificar o tamanho das suas solicitações. Também é possível chamar `count_tokens` no texto e no arquivo separadamente.

Outra opção é chamar `generate_content` e usar o atributo `usage_metadata`
no objeto `response` para receber o seguinte:

* As contagens de tokens separadas da entrada (`prompt_token_count`), do conteúdo em cache (`cached_content_token_count`) e da saída (`candidates_token_count`).
* A contagem de tokens para o processo de pensamento (`thoughts_token_count`)
* O número total de tokens na *entrada e na saída*
  (`total_token_count`)

**Observação**: você vai receber a mesma contagem de tokens se usar um arquivo enviado com a API File ou se fornecer o arquivo como dados inline.

```
from google import genai
import time

client = genai.Client()
prompt = "Tell me about this video"
your_file = client.files.upload(file=media / "Big_Buck_Bunny.mp4")

# Poll until the video file is completely processed (state becomes ACTIVE).
while not your_file.state or your_file.state.name != "ACTIVE":
    print("Processing video...")
    print("File state:", your_file.state)
    time.sleep(5)
    your_file = client.files.get(name=your_file.name)

print(
    client.models.count_tokens(
        model="gemini-2.0-flash", contents=[prompt, your_file]
    )
)
# ( e.g., total_tokens: 300 )

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[prompt, your_file]
)
print(response.usage_metadata)
# ( e.g., prompt_token_count: 301, candidates_token_count: 60, total_token_count: 361 )

count_tokens.py

```

### Instruções e ferramentas do sistema

As instruções e ferramentas do sistema também são contabilizadas no total de tokens da entrada.

Se você usar instruções do sistema, a contagem de `total_tokens` vai aumentar para refletir a adição de `system_instruction`.

Se você usar a chamada de função, a contagem de `total_tokens` vai aumentar para refletir a adição de `tools`.

Envie comentários

Exceto em caso de indicação contrária, o conteúdo desta página é licenciado de acordo com a [Licença de atribuição 4.0 do Creative Commons](https://creativecommons.org/licenses/by/4.0/), e as amostras de código são licenciadas de acordo com a [Licença Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Para mais detalhes, consulte as [políticas do site do Google Developers](https://developers.google.com/site-policies?hl=pt-br). Java é uma marca registrada da Oracle e/ou afiliadas.

Última atualização 2025-11-20 UTC.