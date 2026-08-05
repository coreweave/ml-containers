# vLLM over a Tailnet with `tailscale-py`

Status: working proof of concept, verified on 2026-08-05.

This document describes how to expose vLLM's OpenAI-compatible HTTP server over
a tailnet using embedded Python Tailscale nodes on both ends. It also records
the tested request path and lays out the work needed to turn the PoC into a
production service.

## Result

The following path has been exercised end to end:

```text
HTTP/OpenAI client
  -> 127.0.0.1:18080
  -> tailscale-py client node
  -> encrypted tailnet connection
  -> tailscale-py server node at 100.89.249.24:8080
  -> 127.0.0.1:8000
  -> vLLM 0.26.0
  -> HuggingFaceTB/SmolLM2-135M-Instruct
```

Verified behavior:

- Interactive browser authentication for both embedded nodes.
- Reuse of persisted node identities without another login.
- vLLM `/health` through the complete path.
- Non-streaming OpenAI chat completions.
- Streaming chat completions using server-sent events (SSE).
- Multiple HTTP requests over the same keep-alive TCP connection.

## Package naming

There are several similarly named packages:

- `tailscale.com/tsnet` is Tailscale's Go package for embedding a supported
  Tailscale node.
- `tailscale-py` is the Python package used by this PoC. It is imported as
  `tailscale` and is built from the experimental `tailscale-rs` project.
- The PyPI package named `tsnet` is unrelated to Tailscale.
- The PyPI package named `tailscale` is an administration API client, not an
  embedded tailnet node.

The PoC pins `tailscale-py==0.4.0` and requires Python 3.12 or newer.

References:

- [`tailscale-py` on PyPI](https://pypi.org/project/tailscale-py/)
- [`tailscale-rs` and its current caveats](https://github.com/tailscale/tailscale-rs#caveats)
- [Official Go `tsnet` API](https://pkg.go.dev/tailscale.com/tsnet)

## Why this uses TCP forwarding

`tailscale-py` exposes asynchronous `tcp_listen`, `tcp_connect`, `recv`, and
`send` operations. Its `TcpStream` is not an operating-system socket and does
not have a file descriptor. It therefore cannot be passed directly to Uvicorn,
asyncio's normal socket integration, `requests`, `httpx`, or the OpenAI Python
client.

The bridge solves that mismatch by copying raw bytes in both directions:

```text
asyncio StreamReader/StreamWriter <-> tailscale-py TcpStream
```

Because it forwards bytes rather than parsing HTTP, the same code supports:

- HTTP/1.1 keep-alive.
- Chunked request and response bodies.
- SSE token streaming.
- WebSocket upgrade traffic.
- Any other TCP protocol that does not require source-address preservation.

The bridge handles partial tailnet sends, local-stream backpressure, concurrent
connections, and cancellation when either side closes.

## Repository contents

```text
vllm-tensorizer/tsnet-poc/
├── .gitignore
├── pyproject.toml
├── proxy.py
├── README.md
├── SETUP_AND_PRODUCTIONIZATION.md
└── uv.lock
```

Runtime node state is stored under `.state/`. The directory is ignored by Git,
and `proxy.py` restricts each state file to mode `0600` because it contains the
node's private identity material.

## Prerequisites

- Python 3.12 or newer.
- `uv` for the commands below, or another environment manager capable of
  installing `tailscale-py==0.4.0`.
- A tailnet account that can register devices.
- A running vLLM OpenAI-compatible server reachable on the server bridge's
  loopback interface.
- Tailnet policy that allows the client identity to reach the server identity
  on TCP port 8080.

This local test used Docker Desktop on ARM64 and the official vLLM CPU image.
The repository's `vllm-tensorizer` CUDA image can be substituted on a GPU host.

## Set up the Python environment

From this directory:

```shell
uv sync --python 3.12
```

This creates a local environment from the checked-in lock file.

## Start vLLM

### Small CPU model used by the PoC

The following starts vLLM 0.26.0 with a public 135M-parameter model and binds
the host port only to loopback:

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

Wait for the backend to become healthy:

```shell
curl --fail http://127.0.0.1:8000/health
```

The official vLLM CPU installation documentation describes the available CPU
images and platform-specific tags:
[vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/).

### GPU deployment using this repository's image

On a CUDA-capable host, run the built `vllm-tensorizer` image with the same
network boundary:

```shell
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY"
```

The current image builds vLLM and exposes port 8080 as image metadata, but it
does not define an entrypoint. The deployment must start vLLM and the bridge
explicitly. Port 8000 is the loopback backend in this design; port 8080 is the
tailnet-facing listener.

## Start the server-side embedded node

Run:

```shell
uv run python proxy.py server
```

On first use, no auth key is passed. `tailscale-py` emits a line containing an
`auth_url`, for example:

```text
please authorize this machine or pass an auth key auth_url=https://login.tailscale.com/a/...
```

Open the URL and approve the node. The library may print refreshed URLs while
it waits; completing an already-open authorization page is sufficient. Once
approved, the process continues automatically and prints output similar to:

```text
Tailnet listener ready at 100.89.249.24:8080; forwarding to 127.0.0.1:8000
```

The node identity is persisted in `.state/server.json`. Later runs use that
state and do not require interactive authentication.

Server options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--state-file` | `.state/server.json` | Persistent node identity |
| `--hostname` | `vllm-poc-server` | Requested tailnet hostname |
| `--tailnet-port` | `8080` | Tailnet TCP listener |
| `--backend-host` | `127.0.0.1` | Local vLLM address |
| `--backend-port` | `8000` | Local vLLM port |
| `--tag` | none | Repeatable requested node tag |

## Start the client-side embedded node

In another terminal, run:

```shell
uv run python proxy.py client --target-name vllm-poc-server
```

Approve the second interactive login URL. Once connected, the process resolves
the server's tailnet IPv4 address from the peer map and prints:

```text
Local listener ready at 127.0.0.1:18080; forwarding to 100.89.249.24:8080 over the tailnet
```

The client identity is persisted independently in `.state/client.json`.

Client options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--state-file` | `.state/client.json` | Persistent client identity |
| `--hostname` | `vllm-poc-client` | Requested tailnet hostname |
| `--local-host` | `127.0.0.1` | Local HTTP-facing listener |
| `--local-port` | `18080` | Local HTTP-facing port |
| `--target-name` | required unless IP given | Tailnet server hostname |
| `--target-ip` | required unless name given | Explicit server tailnet IPv4 |
| `--target-port` | `8080` | Server's tailnet port |
| `--tag` | none | Repeatable requested node tag |

## Call vLLM through the tailnet

Applications use a normal localhost URL. No HTTP client integration with
`tailscale-py` is necessary.

### Exact verified request

```http
POST http://127.0.0.1:18080/v1/chat/completions
Content-Type: application/json

{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France? Reply in one short sentence."
    }
  ],
  "max_tokens": 32,
  "temperature": 0
}
```

Equivalent `curl` command:

```shell
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France? Reply in one short sentence."
      }
    ],
    "max_tokens": 32,
    "temperature": 0
  }'
```

### Exact verified response

```json
{
  "id": "chatcmpl-bcff3ed13804db1b",
  "object": "chat.completion",
  "created": 1785952979,
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null,
      "routed_experts": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": "vllm-0.26.0-dbf9b11d",
  "usage": {
    "prompt_tokens": 44,
    "total_tokens": 52,
    "completion_tokens": 8,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "prompt_text": null,
  "kv_transfer_params": null,
  "ec_transfer_params": null,
  "metrics": null
}
```

The important transport evidence is that the request was made only to the
client's loopback port, while vLLM received it through the two embedded nodes.
The response content itself is model behavior, independent of the transport.

### OpenAI Python client

Any OpenAI-compatible client can use the local forwarder:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18080/v1",
    api_key="the-vllm-api-key",
)

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    messages=[{"role": "user", "content": "Hello over the tailnet"}],
)
print(response.choices[0].message.content)
```

If vLLM was started without `--api-key` for a local PoC, the client still needs
a placeholder value because the OpenAI SDK requires one.

### Streaming

Set `stream` and disable `curl` output buffering:

```shell
curl --no-buffer http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Tell me a short story."}],
    "max_tokens": 64,
    "stream": true
  }'
```

The verified response arrived as multiple `data:` events followed by:

```text
data: [DONE]
```

## Authentication behavior

### Interactive PoC authentication

Interactive login works by calling:

```python
await tailscale.connect(state_path, None, hostname=hostname)
```

The call remains pending until the browser authorization finishes. Once it
completes, the node state is written to the configured path and reused on
restart.

Interactive login is convenient for development, but the node remains present
in the tailnet until it is removed through Tailscale administration. Treat the
state files as credentials and never commit or copy them into container images.

### Non-interactive PoC authentication

The bridge optionally reads `TS_AUTH_KEY`:

```shell
TS_AUTH_KEY='tskey-auth-...' uv run python proxy.py server
```

Do not put auth keys directly in shell history or Kubernetes manifests. Inject
them from a secret manager.

## Stop and restart

Stop each Python bridge with `Ctrl-C`. Stop the local model container with:

```shell
docker stop vllm-tsnet-poc
```

Restarting the bridge commands reuses `.state/server.json` and
`.state/client.json`. Removing those files creates new embedded identities and
requires new authentication; it does not automatically delete the old devices
from the tailnet.

## Current PoC boundaries

The current implementation intentionally favors a small, inspectable test:

- `tailscale-py`/`tailscale-rs` is alpha software and currently warns against
  relying on it for production security.
- The current Rust implementation uses DERP relays rather than direct
  peer-to-peer connections, which can affect latency and bandwidth.
- `TcpStream` does not expose explicit close or half-close methods. The PoC
  closes a tailnet connection by ending tasks and dropping stream references.
- vLLM sees the bridge as a loopback client. The original tailnet source IP or
  user identity is not forwarded in HTTP headers.
- The bridge currently has no connection limit, idle timeout, metrics endpoint,
  structured logging, or retry policy.
- Peer lookup resolves the requested hostname to a numeric tailnet IPv4 address
  before connecting.
- The PoC was tested on one host with two embedded nodes. Multi-host behavior
  should be verified before any broader rollout.

## Recommended production architecture

The production decision is less about vLLM and more about which embedded
Tailscale implementation owns the network boundary.

### Recommended: a small Go `tsnet` proxy

Use the official `tailscale.com/tsnet` package for the network bridge while
leaving vLLM and its clients unchanged:

```text
client application -> localhost Go tsnet proxy
  -> tailnet -> server Go tsnet proxy -> localhost vLLM
```

The same small binary can have two modes:

- Server mode uses `tsnet.Server.Listen("tcp", ":8080")` and forwards accepted
  `net.Conn` connections to `127.0.0.1:8000`.
- Client mode listens on `127.0.0.1:18080` and uses
  `tsnet.Server.Dial(ctx, "tcp", "vllm-server:8080")`.

Advantages over the current Python binding include standard Go `net.Conn`
lifecycle semantics, direct peer connectivity, explicit shutdown, mature
Tailscale authentication options, LocalAPI/WhoIs support, and the ability to
advertise a stable Tailscale Service.

### Alternative: continue with `tailscale-py`

This may become reasonable once `tailscale-rs` completes its security audit,
adds direct connections, stabilizes its API, and exposes complete TCP lifecycle
operations. Until then, use it only where its published caveats are acceptable.

## Kubernetes deployment shape

A sidecar design avoids combining vLLM process supervision with the network
bridge:

```text
Pod: vLLM server
├── vLLM container
│   └── listens on 127.0.0.1:8000
└── tsnet proxy container
    └── listens on the tailnet at :8080
```

Containers in a Pod share a network namespace, so the proxy can reach vLLM on
loopback without exposing vLLM through a Service or host port. Use an exec probe
or a small proxy health endpoint for readiness because kubelet HTTP probes
normally target the Pod IP rather than its loopback address.

Client workloads can run an equivalent sidecar that exposes only
`127.0.0.1:18080`. The application continues to use an ordinary OpenAI base
URL.

## Identity and access control

Production workloads should not depend on a human opening login pages.

Recommended identity model:

- Give server proxies a `tag:vllm-server` identity.
- Give approved client proxies a `tag:vllm-client` identity.
- Use tagged, pre-approved credentials delivered by a secret manager or use a
  supported workload-identity flow.
- Give every replica its own node identity. Do not copy a persisted node-state
  file into an image or share one writable state file among replicas.
- Use ephemeral client identities for short-lived jobs when appropriate.
- Use persistent server identities only when stable per-node names are needed;
  otherwise prefer Tailscale Services for service discovery and replicas.

An example least-privilege grant is:

```json
{
  "tagOwners": {
    "tag:vllm-server": ["autogroup:admin"],
    "tag:vllm-client": ["autogroup:admin"]
  },
  "grants": [
    {
      "src": ["tag:vllm-client"],
      "dst": ["tag:vllm-server"],
      "ip": ["tcp:8080"]
    }
  ]
}
```

Tailscale recommends Grants for new policy:
[Grants documentation](https://tailscale.com/docs/features/access-control/grants).

## Application security

Tailnet access should not be the only authorization boundary for a production
inference service.

- Start vLLM with `--api-key` and distribute separate application credentials
  to callers.
- Add rate limits, request-size limits, model allowlists, token limits, and
  audit logging at an HTTP gateway if different clients need different policy.
- Do not log prompts, responses, bearer tokens, or node private state by
  default.
- Keep vLLM and the local side of the proxy bound to loopback.
- Do not publish the vLLM backend through a Kubernetes Service, host port, or
  public load balancer unless separately required and protected.
- Consider application-layer TLS if organizational policy requires encryption
  independently of the tailnet tunnel.

## Lifecycle and reliability work

Before production rollout, the proxy should add:

- Signal-aware graceful shutdown and connection draining.
- Explicit connect, idle, and total request timeouts appropriate for long LLM
  streams.
- Bounded concurrent connections and per-client admission control.
- Retry with jittered backoff for control-plane and peer reconnection.
- A readiness endpoint that requires both tailnet registration and a healthy
  vLLM backend.
- A liveness endpoint that does not fail merely because the model is busy.
- Fail-closed startup: do not bind an externally accessible kernel socket when
  tailnet registration fails.
- Clear behavior during node-key expiry, credential rotation, peer restart,
  and tailnet policy changes.
- Tests for half-close, abrupt reset, client cancellation, slow readers, and
  server shutdown during SSE generation.

## Observability

Add metrics and structured logs without recording model data:

- Active and accepted TCP connections.
- Connect failures by stage: local backend, tailnet registration, peer lookup,
  and peer dial.
- Bytes transferred in both directions.
- Connection lifetime and idle duration.
- Time to first response byte through the bridge.
- Tailnet path type and DERP region where the selected implementation exposes
  that information.
- vLLM health and model readiness.
- Correlation IDs propagated from an HTTP-aware gateway, if one is introduced.

The raw bridge cannot inject identity headers safely because it does not parse
HTTP. If downstream vLLM or a gateway must know the Tailscale caller identity,
use the Go `tsnet` LocalClient/WhoIs API and an HTTP-aware reverse proxy with a
strict header-sanitization boundary.

## Performance validation

Benchmark the proxy against a direct localhost baseline using the expected
model, prompts, and concurrency. Capture at least:

- Time to first token.
- Inter-token latency.
- End-to-end request latency.
- Requests per second and output tokens per second.
- CPU and memory consumed by each bridge.
- Large prompt, multimodal, embeddings, and audio payload throughput if those
  APIs are in scope.
- Direct peer versus DERP-relayed performance for the production networking
  implementation.
- Behavior under hundreds or thousands of simultaneous streaming connections.

The alpha Rust implementation's DERP-only limitation makes this comparison
especially important; inference latency may hide the overhead for small text
requests while large request bodies and high concurrency expose it.

## Packaging and supply chain

- Pin the Tailscale dependency and record its source revision.
- Pin downloaded wheels or Go modules with checksums.
- Build both `linux/amd64` and `linux/arm64` artifacts used by CoreWeave
  workloads.
- Produce an SBOM and run normal image vulnerability scanning.
- Run the proxy as a non-root user with a read-only root filesystem and a
  narrowly scoped writable state volume.
- Avoid baking auth keys, state files, Hugging Face tokens, or vLLM API keys
  into images or build layers.
- Keep the bridge image separate from the large CUDA image when possible so
  network-layer updates do not trigger vLLM rebuilds.

## Test plan

Automate the following before rollout:

1. Unit-test partial sends and both copy directions with fake streams.
2. Integration-test a local HTTP server through two embedded nodes in an
   isolated test tailnet.
3. Exercise vLLM health, models, chat completions, regular completions, SSE,
   cancellation, and keep-alive.
4. Verify the tailnet policy allows the approved client tag and denies an
   unapproved identity.
5. Restart each side independently during an active stream.
6. Rotate authentication credentials and expire node keys.
7. Run load and soak tests with expected production concurrency.
8. Confirm no request is reachable through the Pod IP, host IP, or public
   network when only tailnet access is intended.

## Prioritized next steps

1. Implement a small Go `tsnet` proxy with matching `server` and `client`
   modes, preserving this PoC's CLI and port layout.
2. Add unit tests for the byte pumps and an automated HTTP/SSE integration
   test.
3. Build a minimal multi-architecture proxy image and run it as a sidecar next
   to the existing `vllm-tensorizer` image.
4. Define tagged workload identities and the `tcp:8080` tailnet Grant in a
   non-production tailnet.
5. Add readiness, metrics, graceful shutdown, connection bounds, and timeouts.
6. Benchmark direct vLLM access, `tailscale-py`, and Go `tsnet` under realistic
   prompt sizes and concurrency.
7. Add vLLM API authentication and an HTTP-aware gateway only if application
   authorization, rate limiting, or caller identity is required.
8. Run a multi-host GPU validation on CoreWeave before choosing the final
   deployment model.

## PoC acceptance criteria met

- A real open-source model ran under vLLM rather than a mock server.
- Both ends embedded a Tailscale implementation through the Python package.
- Both nodes used interactive browser authentication.
- The client made ordinary HTTP calls to localhost.
- The server was reachable over the tailnet and forwarded to loopback vLLM.
- Non-streaming, streaming, and keep-alive traffic worked.
- Setup is reproducible from files contained in this directory.
