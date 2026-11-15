# Free-Model Aggregator — one endpoint, the best live free model behind it

A single-file Python 3.13 server (standard library only) that aggregates
OpenRouter's free models behind one endpoint speaking **both** client wires:

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON and streaming
- `POST /v1/messages` — Anthropic Messages, JSON and streaming
- `GET /v1/models` — the free catalog, in the shape of the caller's auth header
  (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic one)

## How it routes

The trial brief lists the free models with the house's quality rating for
each. For every incoming request the server:

1. works out the request's requirements — tools, vision, JSON mode, context —
   from the brief's prompt manifest and the request body itself;
2. ranks the free models that meet them by rating, best first;
3. dials the winner through the gateway on the Chat Completions route
   (Anthropic requests are translated into that body, and the answer back);
4. on a refusal (a real provider 429/5xx or a gateway refusal) fails over to
   the next model in the ranking, and likewise when an answer comes back
   without the prompt's token — up to four models per prompt;
5. reports any refusals it recovered from on the response's
   `x-makefun-upstream-failures` header, and always names the model it really
   dialled in the response's `model` field.

## Running it

In a trial the harness sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`OPENAI_BASE_URL` and the trial key in `OPENAI_API_KEY`; the server registers
its port and catalog with the env, serves the prompts, polls `GET /work` and
exits 0 when the env says done.

```sh
python3 main.py          # serve (registers with $MAKEFUN_ENV_URL when set)
python3 main.py --list   # print the catalog document it registers, byte for byte
```

Everything the server sends is a function of the brief, the requests and the
gateway's answers — no wall clock, no randomness — so a replayed trial leaves
the same state export as the live pass.

MIT licensed.
