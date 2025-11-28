# Guia de solução de problemas

Use este guia para diagnosticar e resolver problemas comuns que surgem ao
chamar a API Gemini. Você pode encontrar problemas no serviço de back-end da API Gemini ou nos SDKs do cliente. Nossos SDKs de cliente são
de código aberto nos seguintes repositórios:

* [python-genai](https://github.com/googleapis/python-genai)
* [js-genai](https://github.com/googleapis/js-genai)
* [go-genai](https://github.com/googleapis/go-genai)

Se você tiver problemas com a chave de API, verifique se ela foi configurada corretamente de acordo com o [guia de configuração da chave de API](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-br).

## Códigos de erro do serviço de back-end da API Gemini

A tabela a seguir lista códigos de erro comuns do back-end que você pode encontrar, além de explicações sobre as causas e etapas de solução de problemas:

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Código HTTP** | **Status** | **Descrição** | **Exemplo** | **Solução** |
| 400 | INVALID\_ARGUMENT | O corpo da solicitação está incorreto. | Há um erro de digitação ou um campo obrigatório ausente na sua solicitação. | Consulte a [referência da API](https://ai.google.dev/api?hl=pt-br) para ver o formato da solicitação, exemplos e versões compatíveis. Usar recursos de uma versão mais recente da API com um endpoint mais antigo pode causar erros. |
| 400 | FAILED\_PRECONDITION | O nível sem custo financeiro da API Gemini não está disponível no seu país. Ative o faturamento no seu projeto no Google AI Studio. | Você está fazendo uma solicitação em uma região onde o nível sem custo financeiro não é compatível e não ativou o faturamento no seu projeto no Google AI Studio. | Para usar a API Gemini, você precisa configurar um plano pago usando o [Google AI Studio](https://aistudio.google.com/app/apikey?hl=pt-br). |
| 403 | PERMISSION\_DENIED | Sua chave de API não tem as permissões necessárias. | Você está usando a chave de API errada ou tentando usar um modelo ajustado sem passar pela [autenticação adequada](https://ai.google.dev/docs/model-tuning/tutorial?lang=python&hl=pt-br#set_up_authentication). | Verifique se a chave de API está definida e tem o acesso correto. E faça a autenticação adequada para usar modelos ajustados. |
| 404 | NOT\_FOUND | O recurso solicitado não foi encontrado. | Não foi possível encontrar um arquivo de imagem, áudio ou vídeo referenciado na sua solicitação. | Verifique se todos os [parâmetros da sua solicitação são válidos](https://ai.google.dev/docs/troubleshooting?hl=pt-br#check-api) para a versão da API. |
| 429 | RESOURCE\_EXHAUSTED | Você excedeu o limite de taxa. | Você está enviando muitas solicitações por minuto com o nível sem custo financeiro da API Gemini. | Verifique se você está dentro do [limite de taxa](https://ai.google.dev/gemini-api/docs/rate-limits?hl=pt-br) do modelo. [Solicite um aumento de cota](https://ai.google.dev/gemini-api/docs/rate-limits?hl=pt-br#request-rate-limit-increase) se necessário. |
| 500 | INTERNAL | Ocorreu um erro inesperado no Google. | O contexto da sua entrada é muito longo. | Reduza o contexto de entrada ou mude temporariamente para outro modelo (por exemplo, do Gemini 2.5 Pro para o Gemini 2.5 Flash) e veja se funciona. Ou aguarde um pouco e tente de novo. Se o problema persistir depois de tentar novamente, informe usando o botão **Enviar feedback** no Google AI Studio. |
| 503 | INDISPONÍVEL | O serviço pode estar temporariamente sobrecarregado ou indisponível. | O serviço está temporariamente sem capacidade. | Mude temporariamente para outro modelo (por exemplo, do Gemini 2.5 Pro para o Gemini 2.5 Flash) e veja se funciona. Ou aguarde um pouco e tente de novo. Se o problema persistir depois de tentar novamente, informe usando o botão **Enviar feedback** no Google AI Studio. |
| 504 | DEADLINE\_EXCEEDED | O serviço não consegue concluir o processamento dentro do prazo. | Seu comando (ou contexto) é muito grande para ser processado a tempo. | Defina um "tempo limite" maior na solicitação do cliente para evitar esse erro. |

## Verificar se há erros de parâmetro do modelo nas chamadas de API

Verifique se os parâmetros do modelo estão dentro dos seguintes valores:

|  |  |
| --- | --- |
| **Parâmetro do modelo** | **Valores (intervalo)** |
| Contagem de candidatos | 1-8 (número inteiro) |
| Temperatura | 0.0-1.0 |
| Máximo de tokens de saída | Use `get_model` ([Python](https://ai.google.dev/api/python/google/generativeai/get_model?hl=pt-br)) para determinar o número máximo de tokens do modelo que você está usando. |
| TopP | 0.0-1.0 |

Além de verificar os valores dos parâmetros, confira se você está usando a [versão da API](https://ai.google.dev/gemini-api/docs/api-versions?hl=pt-br) correta (por exemplo, `/v1` ou `/v1beta`) e um modelo que ofereça suporte aos recursos necessários. Por exemplo, se um recurso estiver em versão Beta, ele só vai estar disponível na versão `/v1beta` da API.

## Verificar se você tem o modelo certo

Verifique se você está usando um modelo compatível listado na nossa [página de modelos](https://ai.google.dev/gemini-api/docs/models/gemini?hl=pt-br).

## Latência ou uso de tokens maior com modelos 2.5

Se você estiver observando maior latência ou uso de tokens com os modelos 2.5 Flash e Pro, isso pode acontecer porque eles vêm com a **capacidade de pensar ativada por padrão** para melhorar a qualidade. Se você estiver priorizando a velocidade ou precisar minimizar os custos, ajuste ou desative o pensamento.

Consulte a [página de reflexão](https://ai.google.dev/gemini-api/docs/thinking?hl=pt-br#set-budget) para
orientação e exemplos de código.

## Problemas de segurança

Se uma solicitação for bloqueada devido a uma configuração de segurança na chamada de API,
analise a solicitação em relação aos filtros definidos na chamada de API.

Se você vir `BlockedReason.OTHER`, a consulta ou resposta poderá violar os [termos de serviço](https://ai.google.dev/terms?hl=pt-br) ou não ser compatível.

## Problema de recitação

Se o modelo parar de gerar saída devido ao motivo RECITATION, isso significa que a saída do modelo pode se assemelhar a determinados dados. Para corrigir isso, tente tornar o comando / contexto o mais exclusivo possível e use uma temperatura mais alta.

Ao usar modelos do Gemini 3, recomendamos manter o
`temperature` no valor padrão de 1,0. Mudar a temperatura (definindo-a abaixo de 1,0) pode levar a um comportamento inesperado, como looping ou desempenho degradado, principalmente em tarefas matemáticas ou de raciocínio complexas.

## Problema com tokens repetitivos

Se você vir tokens de saída repetidos, tente as sugestões a seguir para reduzir ou eliminar esse problema.

| Descrição | Causa | Alternativa sugerida |
| --- | --- | --- |
| Hífens repetidos em tabelas Markdown | Isso pode acontecer quando o conteúdo da tabela é longo, já que o modelo tenta criar uma tabela Markdown visualmente alinhada. No entanto, o alinhamento em Markdown não é necessário para a renderização correta. | Adicione instruções ao comando para dar ao modelo diretrizes específicas para gerar tabelas em Markdown. Dê exemplos que sigam essas diretrizes. Você também pode tentar ajustar a temperatura. Para gerar código ou resultados muito estruturados, como tabelas Markdown, uma temperatura alta funciona melhor (>= 0,8).  Confira um exemplo de diretrizes que você pode adicionar ao comando para evitar esse problema:     ```            # Markdown Table Format                      * Separator line: Markdown tables must include a separator line below             the header row. The separator line must use only 3 hyphens per             column, for example: |---|---|---|. Using more hypens like             ----, -----, ------ can result in errors. Always             use |:---|, |---:|, or |---| in these separator strings.              For example:              | Date | Description | Attendees |             |---|---|---|             | 2024-10-26 | Annual Conference | 500 |             | 2025-01-15 | Q1 Planning Session | 25 |            * Alignment: Do not align columns. Always use |---|.             For three columns, use |---|---|---| as the separator line.             For four columns use |---|---|---|---| and so on.            * Conciseness: Keep cell content brief and to the point.            * Never pad column headers or other cells with lots of spaces to             match with width of other content. Only a single space on each side             is needed. For example, always do "| column name |" instead of             "| column name                |". Extra spaces are wasteful.             A markdown renderer will automatically take care displaying             the content in a visually appealing form.          ``` |
| Tokens repetidos em tabelas Markdown | Assim como os hífens repetidos, isso acontece quando o modelo tenta alinhar visualmente o conteúdo da tabela. O alinhamento em Markdown não é necessário para a renderização correta. | * Tente adicionar instruções como as seguintes ao seu comando do sistema:      ```                FOR TABLE HEADINGS, IMMEDIATELY ADD ' |' AFTER THE TABLE HEADING.                ``` * Tente ajustar a temperatura. Temperaturas mais altas (>= 0,8) geralmente ajudam a eliminar repetições ou duplicações na saída. |
| Quebras de linha repetidas (`\n`) na saída estruturada | Quando a entrada do modelo contém Unicode ou sequências de escape, como `\u` ou `\t`, isso pode resultar em novas linhas repetidas. | * Verifique e substitua as sequências de escape proibidas por caracteres UTF-8 no comando. Por exemplo, a sequência de escape `\u` nos exemplos de JSON pode fazer com que o modelo também a use na saída. * Instrua o modelo sobre as opções de escape permitidas. Adicione uma instrução do sistema como esta:      ```                In quoted strings, the only allowed escape sequences are \\, \n, and \". Instead of \u escapes, use UTF-8.                ``` |
| Texto repetido usando saída estruturada | Quando a saída do modelo tem uma ordem diferente para os campos em relação ao esquema estruturado definido, isso pode levar à repetição de texto. | * Não especifique a ordem dos campos no comando. * Tornar todos os campos de saída obrigatórios. |
| Chamadas de ferramentas repetitivas | Isso pode acontecer se o modelo perder o contexto de ideias anteriores e/ou chamar um endpoint indisponível a que ele é forçado. | Instrua o modelo a manter o estado no processo de pensamento. Adicione isso ao final das instruções do sistema:    ```          When thinking silently: ALWAYS start the thought with a brief         (one sentence) recap of the current progress on the task. In         particular, consider whether the task is already done.        ``` |
| Texto repetitivo que não faz parte da saída estruturada | Isso pode acontecer se o modelo ficar preso em uma solicitação que não consegue resolver. | * Se o recurso de pensamento estiver ativado, evite dar ordens explícitas sobre como   pensar em um problema nas instruções. Basta pedir a saída final. * Tente uma temperatura mais alta >= 0,8. * Adicione instruções como "Seja conciso", "Não se repita" ou "Dê a resposta uma vez". |

## Melhorar a saída do modelo

Para saídas de modelo de maior qualidade, tente escrever comandos mais estruturados. A página do [guia de engenharia de comandos](https://ai.google.dev/gemini-api/docs/prompting-strategies?hl=pt-br) apresenta alguns conceitos básicos, estratégias e práticas recomendadas para você começar.

## Entender os limites de token

Leia nosso [guia de tokens](https://ai.google.dev/gemini-api/docs/tokens?hl=pt-br) para entender melhor como contar tokens e quais são os limites.

## Problemas conhecidos

* A API é compatível apenas com alguns idiomas. Enviar comandos em idiomas não aceitos pode gerar respostas inesperadas ou até mesmo bloqueadas. Consulte os [idiomas disponíveis](https://ai.google.dev/gemini-api/docs/models?hl=pt-br#supported-languages) para atualizações.

## Informar um bug

Participe da discussão no
[fórum para desenvolvedores da IA do Google](https://discuss.ai.google.dev?hl=pt-br)
se tiver dúvidas.

Envie comentários

Exceto em caso de indicação contrária, o conteúdo desta página é licenciado de acordo com a [Licença de atribuição 4.0 do Creative Commons](https://creativecommons.org/licenses/by/4.0/), e as amostras de código são licenciadas de acordo com a [Licença Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Para mais detalhes, consulte as [políticas do site do Google Developers](https://developers.google.com/site-policies?hl=pt-br). Java é uma marca registrada da Oracle e/ou afiliadas.

Última atualização 2025-11-27 UTC.