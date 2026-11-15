"""RQ2 - Query an LLM for line-level defect localization (Section 4.4.2, Listing 1).

Sends the prompt of Listing 1 with the diff of each true-positive commit of CodeT5+ and stores
the raw answers, in commit order, in <out-dir>/responses-<model>_exec-<run>.pkl (or
responses-<model>.pkl for a single run). The file is saved after every answer, so an interrupted
run resumes where it stopped. scripts/rq2_metrics.py computes the metrics from these files.

The answers used in the paper are already in jit-dl-llms/; this script is only needed to query
the models again (LLM answers are not deterministic, so new runs will differ slightly).

Models and calls are the ones used in the paper:
  gpt-5   OpenAI Responses API, model "gpt-5" (default reasoning effort); OPENAI_API_KEY.
          --openrouter uses OpenRouter's "openai/gpt-5" instead; OPENROUTER_API_KEY.
  claude  Anthropic Messages API, model "claude-sonnet-4-5-20250929", max_tokens 20000; prompts
          longer than 20000 tokens are halved until they fit; ANTHROPIC_API_KEY.
          Claude Sonnet 4.5 is deprecated and will be retired by Anthropic on 2026-11-30.
  gemini  Google GenAI API, model "gemini-2.5-pro"; prompts longer than 1M tokens are halved
          until they fit; GEMINI_API_KEY (or GOOGLE_API_KEY).

Examples:
    python scripts/rq2_llm.py --model gpt-5 --runs 5
    python scripts/rq2_llm.py --model claude --limit 3 --out-dir results/rq2/responses
"""
import argparse
import os
import pickle

from jitdl_lib import DATA_DIR, RESULTS_DIR, build_prompt, load_true_positives

CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_MAX_TOKENS = 20000
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_MAX_TOKENS = 1000000


def gpt5_client(openrouter):
    from openai import OpenAI
    if openrouter:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        model = "openai/gpt-5"
    else:
        client, model = OpenAI(), "gpt-5"
    return lambda prompt: client.responses.create(model=model, input=prompt).output_text


def claude_client():
    import anthropic
    client = anthropic.Anthropic()

    def ask(prompt):
        while client.messages.count_tokens(model=CLAUDE_MODEL, messages=[{"role": "user", "content": prompt}]
                                           ).input_tokens > CLAUDE_MAX_TOKENS:
            prompt = prompt[:len(prompt) // 2]
        message = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        return message.content[0].text

    return ask


def gemini_client():
    from google import genai
    client = genai.Client()

    def ask(prompt):
        while client.models.count_tokens(model=GEMINI_MODEL, contents=prompt).total_tokens > GEMINI_MAX_TOKENS:
            prompt = prompt[:len(prompt) // 2]
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text

    return ask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["gpt-5", "claude", "gemini"], required=True)
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (the paper used 5 for GPT-5).")
    parser.add_argument("--openrouter", action="store_true", help="Call GPT-5 through OpenRouter.")
    parser.add_argument("--limit", type=int, help="Only the first N commits (quick check).")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "responses"))
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and the files to be written.")
    args = parser.parse_args()

    tp = load_true_positives(args.data_dir)
    if args.limit:
        tp = tp.head(args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    names = [f"responses-{args.model}.pkl"] if args.runs == 1 else \
        [f"responses-{args.model}_exec-{r}.pkl" for r in range(args.runs)]

    if args.dry_run:
        print(build_prompt(tp.loc[0, "diff"])[:3000])
        print(f"\n{len(tp)} commits -> " + ", ".join(os.path.join(args.out_dir, n) for n in names))
        return

    ask = {"gpt-5": lambda: gpt5_client(args.openrouter), "claude": claude_client, "gemini": gemini_client}[args.model]()

    for name in names:
        path = os.path.join(args.out_dir, name)
        responses = pickle.load(open(path, "rb")) if os.path.exists(path) else []
        if responses:
            print(f"{path}: resuming after {len(responses)} answers")
        for i in range(len(responses), len(tp)):
            responses.append(ask(build_prompt(tp.loc[i, "diff"])))
            with open(path, "wb") as f:
                pickle.dump(responses, f)
            print(f"{name}: {i + 1}/{len(tp)}", flush=True)
        print(f"{path}: {len(responses)} answers")


if __name__ == "__main__":
    main()
