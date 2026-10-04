"""
Team PR helper - a thin wrapper around the GitHub CLI (`gh`).

Usage (from the project root):

    python -m cli.prs list                 # open PRs
    python -m cli.prs list --state all     # open + closed + merged
    python -m cli.prs mine                 # PRs you opened
    python -m cli.prs review               # PRs waiting for your review
    python -m cli.prs view 12              # summary of PR #12
    python -m cli.prs checks 12            # CI status of PR #12
    python -m cli.prs diff 12              # diff of PR #12
    python -m cli.prs files 12             # files changed in PR #12
    python -m cli.prs checkout 12          # check the PR branch out locally
    python -m cli.prs open 12              # open PR #12 in the browser

Requirements: Git and the GitHub CLI (https://cli.github.com), logged in
with `gh auth login`. The repository is detected from the `origin` remote;
use --repo OWNER/NAME to override.
"""

import argparse
import shutil
import subprocess
import sys


def run_gh(args, repo=None):
    """Run `gh <args>`, streaming its output. Returns the exit code."""
    cmd = ["gh"] + args
    if repo:
        cmd += ["--repo", repo]
    return subprocess.call(cmd)


def check_gh():
    if shutil.which("gh") is None:
        sys.exit(
            "GitHub CLI (gh) not found. Install it from "
            "https://cli.github.com and run `gh auth login`."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m cli.prs",
        description="Check and review team pull requests.",
    )
    parser.add_argument(
        "--repo",
        help="OWNER/NAME (default: detected from the git remote)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list pull requests")
    p.add_argument(
        "--state",
        choices=["open", "closed", "merged", "all"],
        default="open",
    )
    p.add_argument("--limit", type=int, default=30)

    sub.add_parser("mine", help="PRs you opened")
    sub.add_parser("review", help="PRs requesting your review")

    for name, text in [
        ("view", "show PR summary and description"),
        ("checks", "show CI check status"),
        ("diff", "show the PR diff"),
        ("files", "list files changed"),
        ("checkout", "check out the PR branch locally"),
        ("open", "open the PR in the browser"),
    ]:
        p = sub.add_parser(name, help=text)
        p.add_argument("number", type=int, help="PR number")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_gh()
    repo = args.repo
    cmd = args.command

    if cmd == "list":
        return run_gh(
            ["pr", "list", "--state", args.state, "--limit", str(args.limit)],
            repo,
        )
    if cmd == "mine":
        return run_gh(["pr", "list", "--author", "@me"], repo)
    if cmd == "review":
        return run_gh(["pr", "list", "--search", "review-requested:@me"], repo)
    if cmd == "view":
        return run_gh(["pr", "view", str(args.number), "--comments"], repo)
    if cmd == "checks":
        return run_gh(["pr", "checks", str(args.number)], repo)
    if cmd == "diff":
        return run_gh(["pr", "diff", str(args.number)], repo)
    if cmd == "files":
        return run_gh(["pr", "diff", str(args.number), "--name-only"], repo)
    if cmd == "checkout":
        return run_gh(["pr", "checkout", str(args.number)], repo)
    if cmd == "open":
        return run_gh(["pr", "view", str(args.number), "--web"], repo)

    return 1


if __name__ == "__main__":
    sys.exit(main())
