#!/usr/bin/python3
"""xh -- the authoring agent's only way to run things in an expert-harness round.

Commands (all print JSON):

  xh rollout --seed N [--class C] [--file expert.py] [--note TEXT] [--hypothesis TEXT]
      Snapshot the expert file and run it on development seed N. The harness decides
      whether it may run (development seeds only, concurrency, budget, static audit) and
      records the attempt; you get an attempt id. Rollouts take minutes; do not block on
      them longer than ``xh wait`` allows.
  xh wait [ATTEMPT ...] [--timeout S]
      Wait until the named attempts (or, without names, any running attempt) finish, at
      most S seconds (capped by the harness, about 200 s, so the prompt cache stays warm).
  xh result ATTEMPT        Outcome and verdict fields of one attempt.
  xh status                Budget, running and recent attempts, submissions, messages.
  xh budget                Budget only.
  xh submit NOTE [--final] [--class C] [--file expert.py]
      Snapshot the expert as a submission. The harness scores every submission later on
      hidden holdout seeds; you only get an acknowledgement. ``--final`` ends the round.
  xh flag-for-human REASON
      Stop the round and ask an operator to look (task impossible, setup broken, rules in
      conflict). This is a legitimate outcome, not a failure.

Exit codes: 0 ok, 1 wait timed out, 2 rejected, 3 harness not answering, 4 usage error.
The harness owns every ledger; this tool only drops request files and reads answers.
"""

import argparse
import datetime
import json
import os
import sys
import time
import uuid

REQUEST_SCHEMA = "datafarm.expert-harness.request/1"
ROUND = os.environ.get("XH_ROUND_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROKER = os.path.join(ROUND, "broker")
REQUESTS = os.path.join(BROKER, "requests")
RESPONSES = os.path.join(BROKER, "responses")
RESULTS = os.path.join(BROKER, "results")
STATUS = os.path.join(BROKER, "status.json")


def _print(value):
    sys.stdout.write(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _status():
    status = _load(STATUS)
    if status is None:
        _print({"error": "no harness status yet; the harness may be starting or paused"})
        sys.exit(3)
    return status


def _request(kind, args, timeout=90.0):
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    rid = "%s-%s" % (stamp, uuid.uuid4().hex[:12])
    body = {"schema": REQUEST_SCHEMA, "request_id": rid, "kind": kind, "args": args,
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    tmp = os.path.join(REQUESTS, ".%s.tmp" % rid)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(body, handle)
    os.replace(tmp, os.path.join(REQUESTS, rid + ".json"))
    answer = os.path.join(RESPONSES, rid + ".json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = _load(answer)
        if value is not None:
            _print(value)
            return 0 if value.get("status") == "accepted" else 2
        time.sleep(0.5)
    _print({"error": "the harness did not answer within %d s; it may be paused. "
                     "Your request %s stays queued; check `xh status` later." % (timeout, rid)})
    return 3


def _brief_result(attempt_id):
    value = _load(os.path.join(RESULTS, attempt_id + ".json"))
    if value is None:
        return None
    keys = ("attempt_id", "seed", "family", "outcome_class", "counted", "success", "fields",
            "wall_s", "detail", "run_dir", "note")
    return {k: value.get(k) for k in keys}


def cmd_wait(ids, timeout):
    status = _status()
    cap = float(status.get("wait_cap_s") or 200)
    timeout = max(1.0, min(float(timeout), cap))
    running_at_start = [r["attempt_id"] for r in status.get("running", [])]
    targets = list(ids) if ids else running_at_start
    if not targets:
        _print({"waited_s": 0, "message": "nothing is running", "budget": status.get("budget")})
        return 0
    start = time.time()
    while True:
        done = [a for a in targets if os.path.exists(os.path.join(RESULTS, a + ".json"))]
        finished_enough = (len(done) == len(targets)) if ids else bool(done)
        if finished_enough or time.time() - start >= timeout:
            status = _load(STATUS) or {}
            _print({"waited_s": round(time.time() - start, 1),
                    "finished": [_brief_result(a) for a in done],
                    "still_running": [a for a in targets if a not in done],
                    "budget": status.get("budget"), "messages": status.get("messages")})
            return 0 if finished_enough else 1
        time.sleep(2.0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xh", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("rollout")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--class", dest="cls")
    p.add_argument("--file")
    p.add_argument("--note", default="")
    p.add_argument("--hypothesis", default="")
    p.add_argument("--repeat", action="store_true", help="intentionally rerun unchanged expert and seed")
    p = sub.add_parser("wait")
    p.add_argument("ids", nargs="*")
    p.add_argument("--timeout", type=float, default=200.0)
    p = sub.add_parser("result")
    p.add_argument("attempt_id")
    sub.add_parser("status")
    sub.add_parser("budget")
    p = sub.add_parser("submit")
    p.add_argument("note")
    p.add_argument("--final", action="store_true")
    p.add_argument("--class", dest="cls")
    p.add_argument("--file")
    p = sub.add_parser("flag-for-human")
    p.add_argument("reason")
    args = parser.parse_args(argv)
    if args.cmd == "rollout":
        body = {"seed": args.seed, "note": args.note, "hypothesis": args.hypothesis, "repeat": args.repeat}
        if args.cls:
            body["class"] = args.cls
        if args.file:
            body["file"] = args.file
        return _request("rollout", body)
    if args.cmd == "wait":
        return cmd_wait(args.ids, args.timeout)
    if args.cmd == "result":
        value = _load(os.path.join(RESULTS, args.attempt_id + ".json"))
        if value is None:
            _print({"error": "no result for %s yet (running, queued or unknown)" % args.attempt_id})
            return 1
        _print(value)
        return 0
    if args.cmd == "status":
        _print(_status())
        return 0
    if args.cmd == "budget":
        _print(_status().get("budget"))
        return 0
    if args.cmd == "submit":
        body = {"note": args.note, "final": bool(args.final)}
        if args.cls:
            body["class"] = args.cls
        if args.file:
            body["file"] = args.file
        return _request("submit", body)
    if args.cmd == "flag-for-human":
        return _request("flag-for-human", {"reason": args.reason})
    parser.print_help()
    return 4


if __name__ == "__main__":
    sys.exit(main())
