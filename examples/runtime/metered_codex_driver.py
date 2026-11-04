#!/usr/bin/env python3
"""Example driver: run the Codex CLI behind a local metered Responses gateway.  Not supported tooling.

It shows how a deployment can meter and cap model spend for one agent session.  It starts `codex exec` with
--dangerously-bypass-approvals-and-sandbox, so the agent runs with no sandbox and no approval prompts: use it only on a
trusted task in a disposable environment.  The gateway meters requests that pass through it; it is not a security
boundary against code the agent runs.
"""

import argparse, json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from datafarm.agent.metered_provider import Meter, MeteredGateway, load_pricing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--compact-handoff", action="store_true")
    parser.add_argument(
        "--pricing",
        default=os.environ.get("DATAFARM_METERED_PRICING"),
        help="JSON file naming the metered model and its USD per million tokens (default: env DATAFARM_METERED_PRICING)",
    )
    args = parser.parse_args()
    if not args.pricing:
        parser.error("--pricing or DATAFARM_METERED_PRICING is required (metered model id and price table; no default)")
    pricing = load_pricing(args.pricing)
    sys.path.insert(0, str(Path(args.source) / "src"))
    from datafarm.agent.credentials import KeysFileCredentials

    creds = KeysFileCredentials(args.credentials)()
    receipt = Path(os.environ["DATAFARM_COST_RECEIPT"])
    receipt.parent.mkdir(parents=True, exist_ok=True)
    meter = Meter(receipt.with_name("provider-usage.jsonl"), os.environ["DATAFARM_SESSION_MAX_USD"], pricing=pricing)
    gateway = MeteredGateway(meter, creds.endpoint.rstrip("/") + "/v1/responses", creds.api_key)
    server = gateway.server
    url = f"http://127.0.0.1:{server.server_port}/v1"
    argv = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-m",
        pricing.model,
        "-c",
        'model_provider="datafarm_metered"',
        "-c",
        'model_providers.datafarm_metered.name="DataFarm metered"',
        "-c",
        f'model_providers.datafarm_metered.base_url="{url}"',
        "-c",
        'model_providers.datafarm_metered.wire_api="responses"',
        "-c",
        "model_providers.datafarm_metered.requires_openai_auth=false",
        "-c",
        "model_providers.datafarm_metered.stream_idle_timeout_ms=960000",
        "-c",
        "model_providers.datafarm_metered.request_max_retries=0",
        "-c",
        "model_providers.datafarm_metered.stream_max_retries=0",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'service_tier="default"',
        "-c",
        "features.responses_websockets=false",
        "-c",
        "features.responses_websockets_v2=false",
        "-",
    ]
    if args.resume:
        argv[2:2] = ["resume", args.resume]
    prompt = sys.stdin.read()
    if not args.smoke:
        messages = json.loads(prompt)
        if args.resume or args.compact_handoff:
            messages = messages[-1:]
        prompt = "\n\n".join(str(x.get("content", "")) for x in messages)
    prompt += (
        "\nYou ARE the sole task executor inside the active CLI, not an external observer. The metered driver and CLI processes in this task workspace are YOUR OWN parents/session; never wait on them or poll your own stdout, stderr, execution or cost logs. No other agent will create task products for you. Only wait on simulator jobs YOU actually submit. For white-only scope, provide independently verified white-stage artifacts and a white-delivery report; do not fabricate render submissions to satisfy the full rendered-delivery schema. Continue implementation now.\nAll model calls must use the inherited datafarm_metered provider. Do not invoke external model CLIs/APIs, read credentials, or alter cost ledgers. Independent reviewer subagents must inherit this provider and use "
        + pricing.model
        + ". No hosted web tools; local shell and image tools only. Stop cleanly on cost reservation errors. Maintain evidence of every independent review. No rendering or training."
    )
    code = 1
    try:
        child = subprocess.run(argv, input=prompt, text=True, timeout=21600)
        code = child.returncode
    finally:
        gateway.close()
        result = {
            "schema": "datafarm.session-cost/1",
            "reservation_id": os.environ["DATAFARM_COST_RESERVATION_ID"],
            "includes_descendants": True,
            "complete": not meter.unknown,
            "cost_usd": float(meter.spent),
            "cost_basis": "api-equivalent",
            "evidence": str(meter.path),
            "provider_calls": meter.calls,
            "pricing_source": pricing.source,
            "exit_code": code,
        }
        result["stop_reason"] = (
            "unknown-liability" if meter.unknown else ("request-bound" if meter.denied else "executor-exit")
        )
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
