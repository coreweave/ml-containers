# vLLM over a tailnet PoC

For the full setup, a captured end-to-end LLM request/response, and the
productionization plan, see
[`SETUP_AND_PRODUCTIONIZATION.md`](SETUP_AND_PRODUCTIONIZATION.md).

This proof of concept runs vLLM locally and exposes it only through two embedded
`tailscale-py` nodes:

```text
curl/OpenAI client -> 127.0.0.1:18080 -> tailscale-py client
  -> tailnet -> tailscale-py server -> 127.0.0.1:8000 -> vLLM
```

The Python bridge forwards raw TCP bytes, so HTTP keep-alive, streaming
responses, and WebSockets pass through without protocol-specific handling.

## Install

Python 3.12 or newer is required.

```shell
uv sync --python 3.12
```

## Start a small public model

This command uses the official ARM64 CPU image and a public 135M-parameter
instruct model. It does not require a GPU or Hugging Face token.

```shell
docker run --rm --name vllm-tsnet-poc \
  --platform linux/arm64 \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --shm-size 4g \
  -p 127.0.0.1:8000:8000 \
  -e VLLM_CPU_KVCACHE_SPACE=1 \
  vllm/vllm-openai-cpu:latest-arm64 \
  HuggingFaceTB/SmolLM2-135M-Instruct \
  --dtype float \
  --max-model-len 1024
```

Wait until `curl http://127.0.0.1:8000/health` succeeds.

## Start the embedded server node

```shell
uv run python proxy.py server
```

On its first run, `tailscale-py` prints an `auth_url`. Open it in a browser and
approve the node. The process continues automatically after authorization and
stores its identity under `.state/server.json` with owner-only permissions.

The library can refresh the URL while it waits. An already-open authorization
page remains associated with the same node; finish approving that page rather
than opening every URL printed by the retry loop.

## Start the embedded client node

In a second terminal:

```shell
uv run python proxy.py client --target-name vllm-poc-server
```

Open and approve the second `auth_url`. Its identity is stored separately under
`.state/client.json`.

## Exercise vLLM across the tailnet

```shell
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 16
  }'
```

To test streaming, add `"stream": true` to the request body and use `curl -N`.

For non-interactive runs, set `TS_AUTH_KEY` before launching either bridge.
