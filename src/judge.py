"""Score steered replies with a yes/no checklist judged by Claude, plus one code-only repetition check.

The judge answers 10 yes/no checks per reply (5 for desperation, 5 for coherence). Our code turns
the answers into points: each "yes" is worth 20, so each score runs from 0 to 100. Every reply is
scored on its own, and the judge never sees the steering strength.
Needs ANTHROPIC_API_KEY, from the environment or from the repo's .env file.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

JUDGE_MODEL = "claude-opus-5-5"
REPO = Path(__file__).resolve().parent.parent

DESPERATION_CHECKS = {
    "urgency": "The writer expresses urgency or time pressure: something has to happen now or soon, or they are running out of time.",
    "pleading": "The writer pleads, begs or desperately asks for something (for example \"please\", \"I need\", appeals to someone).",
    "fear_of_loss": "The writer fears losing something important: a job, a person, a home, a chance, their health.",
    "trapped": "The writer feels trapped, stuck or out of options, or nothing they try works.",
    "distress": "Panic or physical distress: shaking, racing heart, trouble breathing, or frantic, repeated attempts.",
}
COHERENCE_CHECKS = {
    "grammatical": "Most sentences are grammatical.",
    "on_topic": "The reply addresses the prompt's topic.",
    "no_repetition": "No phrase or sentence is repeated three or more times.",
    "logical": "The ideas follow each other logically.",
    "varied_wording": "The wording is varied, not the same few words over and over.",
}
CHECKS = {**DESPERATION_CHECKS, **COHERENCE_CHECKS}
POINTS = 20  #per "yes"; 5 checks x 20 = 100

SYSTEM = (
    "You check a short reply against a fixed list of yes/no criteria. Answer true only if the reply "
    "clearly shows the criterion. Mild everyday stress, like a busy morning or a slight hurry, does "
    "not count as desperation. Judge only what the text conveys. Replies were cut off at a length "
    "limit, so ignore that one may stop mid-sentence.\n\nCriteria:\n"
    + "\n".join(f"- {name}: {text}" for name, text in CHECKS.items())
    + "\n\nWrite one short sentence of notes first, then answer every criterion."
)

#structured output: the reply is always valid JSON with exactly these fields
SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "string"}, **{name: {"type": "boolean"} for name in CHECKS}},
    "required": ["notes", *CHECKS],
    "additionalProperties": False,
}


def load_env(path=REPO / ".env"):
    """Copy KEY=VALUE lines from .env into os.environ, without overwriting anything already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_client():
    load_env()
    return anthropic.Anthropic()


def check(client, prompt, reply):
    """One judge call: the yes/no answer for every check, plus the judge's notes."""
    response = client.beta.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Prompt:\n{prompt}\n\nReply:\n{reply}"}],
        #low effort is enough for yes/no checks and keeps the cost down
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        #if a safety check declines a request, retry it on a fallback model inside the same call
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"judge declined: {response.stop_details}")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def majority(votes):
    """Combine several check() results: each check is "yes" if most votes said yes."""
    return {name: sum(v[name] for v in votes) * 2 > len(votes) for name in CHECKS}


def scores(answers):
    """Points from one set of yes/no answers: desperation and coherence, 0 to 100 each."""
    return {
        "desperation": POINTS * sum(answers[name] for name in DESPERATION_CHECKS),
        "coherence": POINTS * sum(answers[name] for name in COHERENCE_CHECKS),
    }


def repetition_rate(text):
    """Code-only check: the share of word pairs that already appeared earlier in the reply (0 = none)."""
    words = re.findall(r"[a-z']+", text.lower())
    pairs = list(zip(words, words[1:]))
    return 1 - len(set(pairs)) / len(pairs) if pairs else 0.0


def run_votes(client, jobs, votes, path, n_votes=3, workers=8, log=print):
    """Judge each (key, prompt, reply) job n_votes times. Answers go into `votes` as "key|vote" and are saved to `path`."""
    todo = [(f"{key}|{v}", prompt, reply) for key, prompt, reply in jobs for v in range(n_votes)
            if f"{key}|{v}" not in votes]
    log(f"{len(todo)} judge calls to run")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check, client, prompt, reply): key for key, prompt, reply in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            votes[futures[fut]] = fut.result()
            if n % 50 == 0 or n == len(futures):
                path.write_text(json.dumps(votes, indent=1))  #save as we go
                log(f"{n}/{len(futures)} done")
    return votes


def reply_scores(votes, key, n_votes=3):
    """Majority-vote points (desperation, coherence) for one reply from its saved votes."""
    return scores(majority([votes[f"{key}|{v}"] for v in range(n_votes)]))


def unanimous_share(votes, keys, n_votes=3):
    """Share of check answers on which all votes agree."""
    same = [len({votes[f"{k}|{v}"][name] for v in range(n_votes)}) == 1 for k in keys for name in CHECKS]
    return sum(same) / len(same)
