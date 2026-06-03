cat > ./setup-bx-codex-tsk-framework.sh <<'SETUP_EOF'
#!/usr/bin/env bash
set -euo pipefail

BX_BIN_DIR="${HOME}/.local/bin"

BPUB_ROOT="${BPUB_ROOT:-${HOME}/.work/openai_codex/.bpub}"
TASK_LOCAL_ROOT="${TASK_LOCAL_ROOT:-${BPUB_ROOT}/tsk}"
TASK_REMOTE_ROOT="${TASK_REMOTE_ROOT:-https://github.com/blusjune/.bpub/tsk}"

CODEX_DIR="${HOME}/.codex"
CODEX_CONFIG="${CODEX_DIR}/config.toml"
CODEX_SKILL_ROOT="${CODEX_DIR}/skills/bx-codex-task-manager"
CODEX_SKILL_FILE="${CODEX_SKILL_ROOT}/SKILL.md"

MGMT_LIB="${BX_BIN_DIR}/bx.codex.tsk_mgmt_lib.py"
FRONTEND="${BX_BIN_DIR}/bx.codex.tsk.sh"

mkdir -p "${BX_BIN_DIR}" "${TASK_LOCAL_ROOT}" "${CODEX_SKILL_ROOT}" "${CODEX_DIR}"

export BX_BIN_DIR BPUB_ROOT TASK_LOCAL_ROOT TASK_REMOTE_ROOT CODEX_CONFIG CODEX_SKILL_FILE MGMT_LIB FRONTEND

python3 - <<'PYGEN'
from pathlib import Path
import os

mgmt = Path(os.environ["MGMT_LIB"])
frontend = Path(os.environ["FRONTEND"])
skill_file = Path(os.environ["CODEX_SKILL_FILE"])
codex_config = Path(os.environ["CODEX_CONFIG"])

bpub_root = os.environ["BPUB_ROOT"]
task_local_root = os.environ["TASK_LOCAL_ROOT"]
task_remote_root = os.environ["TASK_REMOTE_ROOT"]

mgmt.parent.mkdir(parents=True, exist_ok=True)
frontend.parent.mkdir(parents=True, exist_ok=True)
skill_file.parent.mkdir(parents=True, exist_ok=True)
codex_config.parent.mkdir(parents=True, exist_ok=True)

mgmt.write_text(r'''#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: python3 -m pip install --user pyyaml", file=sys.stderr)
    sys.exit(2)

TASK_LOCAL_ROOT = Path(os.environ.get("TASK_LOCAL_ROOT", str(Path.home() / ".work" / "openai_codex" / ".bpub" / "tsk")))
TASK_REMOTE_ROOT = os.environ.get("TASK_REMOTE_ROOT", "https://github.com/blusjune/.bpub/tsk")
TASK_DEF_FILE = os.environ.get("TASK_DEF_FILE", ".task_def.yaml")
TASK_META_FILE = os.environ.get("TASK_META_FILE", ".task_meta.yaml")
TASK_LOG_FILE = os.environ.get("TASK_LOG_FILE", ".task_log.txt")
TASK_MSG_FILE = os.environ.get("TASK_MSG_FILE", ".task_msg.txt")
MAX_GIT_FILE_BYTES = int(os.environ.get("BX_CODEX_MAX_GIT_FILE_BYTES", str(50 * 1024 * 1024)))

STATUS_TBS = "100_TBS"
STATUS_STP = "150_STP"
STATUS_WIP = "200_WIP"
STATUS_DID = "300_DID"
STATUS_ERR = "800_ERR"

def now_iso():
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()

def git_root():
    return TASK_LOCAL_ROOT.parent

def year_dir():
    return TASK_LOCAL_ROOT / f"Y{dt.datetime.now().year}"

def slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9가-힣._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "untitled"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data

def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def append_line(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def log(task_root, action, message=""):
    append_line(task_root / TASK_LOG_FILE, f"[{now_iso()}] {action} {message}".rstrip())

def msg(task_root, action, message=""):
    append_line(task_root / TASK_MSG_FILE, f"[{now_iso()}] {action} {message}".rstrip())

def validate_task_def(data, path):
    summary = data.get("summary")
    subtasks = data.get("subtasks")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"{path} must contain mandatory non-empty string field: summary")
    if not isinstance(subtasks, list) or len(subtasks) == 0:
        raise ValueError(f"{path} must contain mandatory non-empty list field: subtasks")

def find_task_root(task_id):
    matches = list(TASK_LOCAL_ROOT.glob(f"Y*/{task_id}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous task_id: {task_id}")
    return matches[0]

def read_meta(task_root):
    return load_yaml(task_root / TASK_META_FILE)

def write_meta(task_root, meta):
    meta["updated_at"] = now_iso()
    save_yaml(task_root / TASK_META_FILE, meta)

def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)

def relative_to_git(path):
    return str(path.resolve().relative_to(git_root().resolve()))

def large_files_under(root):
    skipped = []
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_size > MAX_GIT_FILE_BYTES:
            skipped.append((path, path.stat().st_size))
    return skipped

def sync_task(task_root, reason):
    gr = git_root()
    if not (gr / ".git").exists():
        msg(task_root, "SYNC_SKIPPED", f"git_root={gr} reason=no_git_repository")
        return

    skipped = large_files_under(task_root)
    skipped_set = {p.resolve() for p, _ in skipped}

    for path, size in skipped:
        msg(task_root, "LARGE_FILE_SKIPPED", f"path={path} size={size} threshold={MAX_GIT_FILE_BYTES}")

    files_to_add = [p for p in task_root.rglob("*") if p.is_file() and p.resolve() not in skipped_set]
    if not files_to_add:
        msg(task_root, "SYNC_SKIPPED", "reason=no_files_to_add")
        return

    for path in files_to_add:
        add = run_cmd(["git", "-C", str(gr), "add", relative_to_git(path)])
        if add.returncode != 0:
            msg(task_root, "GIT_ADD_FAILED", f"path={path} error={(add.stdout + add.stderr).strip()[:1000]}")

    commit = run_cmd(["git", "-C", str(gr), "commit", "-m", f"Update task {task_root.name}: {reason}"])
    combined_commit = (commit.stdout + "\n" + commit.stderr).strip()
    if commit.returncode != 0:
        if "nothing to commit" in combined_commit.lower():
            msg(task_root, "SYNC_NO_CHANGE", f"reason={reason}")
            return
        msg(task_root, "GIT_COMMIT_FAILED", combined_commit[:1000])
        return

    push = run_cmd(["git", "-C", str(gr), "push"])
    combined_push = (push.stdout + "\n" + push.stderr).strip()
    if push.returncode != 0:
        msg(task_root, "GIT_PUSH_FAILED", combined_push[:1000])
    else:
        msg(task_root, "SYNC_DONE", f"reason={reason}")

def ensure_output_dirs(task_root):
    for subdir in ["output", "output/artifacts", "output/reports", "output/patches", "output/logs"]:
        (task_root / subdir).mkdir(parents=True, exist_ok=True)

def cmd_enq(args):
    src = Path(args.task_def_file).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: task definition file not found: {src}", file=sys.stderr)
        return 1

    data = load_yaml(src)
    validate_task_def(data, src)

    summary = data["summary"].strip()
    task_id = f"{dt.datetime.now().strftime('%y%m%d_%H%M%S')}__{slugify(summary)}"
    root = year_dir() / task_id
    root.mkdir(parents=True, exist_ok=False)
    ensure_output_dirs(root)
    shutil.copy2(src, root / TASK_DEF_FILE)

    meta = {
        "task_id": task_id,
        "title": summary,
        "status": STATUS_TBS,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "task_local_root": str(TASK_LOCAL_ROOT),
        "task_remote_root": TASK_REMOTE_ROOT,
        "per_task_root": str(root),
        "task_def": str(root / TASK_DEF_FILE),
        "task_log": str(root / TASK_LOG_FILE),
        "task_msg": str(root / TASK_MSG_FILE),
        "output_root": str(root / "output"),
    }
    save_yaml(root / TASK_META_FILE, meta)
    log(root, "ENQ", f"status={STATUS_TBS} task_id={task_id}")
    msg(root, "TASK_CREATED", f"task_id={task_id}")
    sync_task(root, "enqueue")
    print(task_id)
    return 0

def transition(task_id, target_status, allowed_from, action):
    root = find_task_root(task_id)
    if not root:
        print(f"ERROR: task not found: {task_id}", file=sys.stderr)
        return 1
    meta = read_meta(root)
    current = meta.get("status")
    if current not in allowed_from:
        print(f"ERROR: invalid transition {current} -> {target_status}", file=sys.stderr)
        return 1
    meta["status"] = target_status
    write_meta(root, meta)
    log(root, action, f"status={target_status}")
    sync_task(root, action.lower())
    print(f"{task_id} {current} -> {target_status}")
    return 0

def cmd_start(args):
    return transition(args.task_id, STATUS_WIP, {STATUS_TBS, STATUS_STP, STATUS_ERR}, "START")

def cmd_stop(args):
    return transition(args.task_id, STATUS_STP, {STATUS_WIP}, "STOP")

def cmd_mark_done(args):
    return transition(args.task_id, STATUS_DID, {STATUS_WIP}, "MARK_DONE")

def cmd_mark_error(args):
    return transition(args.task_id, STATUS_ERR, {STATUS_WIP}, "MARK_ERROR")

def cmd_deq(args):
    root = find_task_root(args.task_id)
    if not root:
        print(f"ERROR: task not found: {args.task_id}", file=sys.stderr)
        return 1
    meta = read_meta(root)
    current = meta.get("status")
    if current != STATUS_TBS:
        print(f"ERROR: deq allowed only when status={STATUS_TBS}; current={current}", file=sys.stderr)
        return 1

    log(root, "DEQ", "removing task")
    gr = git_root()
    rel = relative_to_git(root) if (gr / ".git").exists() else None
    shutil.rmtree(root)

    if rel and (gr / ".git").exists():
        run_cmd(["git", "-C", str(gr), "rm", "-r", "--ignore-unmatch", rel])
        commit = run_cmd(["git", "-C", str(gr), "commit", "-m", f"Remove task {args.task_id}"])
        if commit.returncode == 0:
            run_cmd(["git", "-C", str(gr), "push"])

    print(f"removed {args.task_id}")
    return 0

def all_task_roots():
    if not TASK_LOCAL_ROOT.exists():
        return []
    return sorted([p for p in TASK_LOCAL_ROOT.glob("Y*/*") if p.is_dir()])

def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = " | ".join("{:<" + str(width) + "}" for width in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt.format(*[str(cell) for cell in row]))

def cmd_status(args):
    roots = [find_task_root(args.task_id)] if args.task_id else all_task_roots()
    roots = [root for root in roots if root]
    rows = []
    for root in roots:
        try:
            meta = read_meta(root)
            rows.append((meta.get("task_id", root.name), meta.get("title", ""), meta.get("status", ""), meta.get("updated_at", "")))
        except Exception as exc:
            rows.append((root.name, f"ERROR: {exc}", "", ""))
    print_table(["task_id", "title", "status", "updated_at"], rows)
    return 0

def cmd_grep(args):
    regex = re.compile(args.regular_expression)
    rows = []
    for root in all_task_roots():
        meta = read_meta(root)
        haystacks = [
            ("task_id", meta.get("task_id", "")),
            ("title", meta.get("title", "")),
            ("status", meta.get("status", "")),
        ]
        task_def = root / TASK_DEF_FILE
        if task_def.exists():
            haystacks.append(("task_def", task_def.read_text(encoding="utf-8", errors="replace")))
        first_match = ""
        for field, value in haystacks:
            match = regex.search(str(value))
            if match:
                first_match = f"{field}: {match.group(0)}"
                break
        if first_match:
            rows.append((meta.get("task_id", root.name), meta.get("title", ""), meta.get("status", ""), first_match))
    print_table(["task_id", "title", "status", "first_match"], rows)
    return 0

def cmd_sync(args):
    root = find_task_root(args.task_id)
    if not root:
        print(f"ERROR: task not found: {args.task_id}", file=sys.stderr)
        return 1
    sync_task(root, "manual_sync")
    return 0

def cmd_paths(args):
    root = find_task_root(args.task_id)
    if not root:
        print(f"ERROR: task not found: {args.task_id}", file=sys.stderr)
        return 1
    meta = read_meta(root)
    for key in ["task_id", "status", "per_task_root", "task_def", "task_log", "task_msg", "output_root"]:
        print(f"{key}: {meta.get(key, '')}")
    return 0

def build_parser():
    parser = argparse.ArgumentParser(prog="bx.codex.tsk_mgmt_lib.py")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("enq")
    p.add_argument("task_def_file")
    p.set_defaults(func=cmd_enq)

    p = sub.add_parser("start")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("deq")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_deq)

    p = sub.add_parser("status")
    p.add_argument("task_id", nargs="?")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("grep")
    p.add_argument("regular_expression")
    p.set_defaults(func=cmd_grep)

    p = sub.add_parser("mark_done")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_mark_done)

    p = sub.add_parser("mark_error")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_mark_error)

    p = sub.add_parser("sync")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("paths")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_paths)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
''', encoding="utf-8")

frontend.write_text(r'''#!/usr/bin/env bash
set -euo pipefail

MGMT_LIB="${HOME}/.local/bin/bx.codex.tsk_mgmt_lib.py"
BPUB_ROOT="${BPUB_ROOT:-${HOME}/.work/openai_codex/.bpub}"
TASK_ROOT="${TASK_LOCAL_ROOT:-${BPUB_ROOT}/tsk}"

usage() {
  cat <<USAGE
Usage:
  bx.codex.tsk.sh enq <task_def_file>
  bx.codex.tsk.sh start <task_id>
  bx.codex.tsk.sh stop <task_id>
  bx.codex.tsk.sh deq <task_id>
  bx.codex.tsk.sh status [task_id]
  bx.codex.tsk.sh grep <regular_expression>
  bx.codex.tsk.sh mark_done <task_id>
  bx.codex.tsk.sh mark_error <task_id>
  bx.codex.tsk.sh sync <task_id>
  bx.codex.tsk.sh paths <task_id>

Environment:
  BPUB_ROOT=${BPUB_ROOT}
  TASK_LOCAL_ROOT=${TASK_ROOT}
USAGE
}

require_mgmt_lib() {
  if [ ! -x "${MGMT_LIB}" ]; then
    echo "ERROR: ${MGMT_LIB} not found or not executable." >&2
    exit 1
  fi
}

cmd="${1:-}"
if [ -z "${cmd}" ]; then
  usage
  exit 1
fi

export BPUB_ROOT
export TASK_LOCAL_ROOT="${TASK_ROOT}"

case "${cmd}" in
  enq|stop|deq|status|grep|mark_done|mark_error|sync|paths)
    require_mgmt_lib
    shift
    exec "${MGMT_LIB}" "${cmd}" "$@"
    ;;

  start)
    require_mgmt_lib
    task_id="${2:-}"
    if [ -z "${task_id}" ]; then
      echo "ERROR: task_id is required." >&2
      usage
      exit 1
    fi

    exec codex \
      -C "${BPUB_ROOT}" \
      --add-dir "${BPUB_ROOT}" \
      exec \
      "Use the \$bx-codex-task-manager skill. Start and execute task ${task_id}.

Required workflow:
1. Run '${MGMT_LIB} status ${task_id}'.
2. Run '${MGMT_LIB} start ${task_id}' if the task can be started.
3. Run '${MGMT_LIB} paths ${task_id}' and identify PER_TASK_ROOT, TASK_DEF, TASK_LOG, TASK_MSG, and output_root.
4. Read the task definition file.
5. Execute all subtasks under PER_TASK_ROOT.
6. Save all task results under PER_TASK_ROOT/output/.
7. Use output/README.md for final summary, output/reports/ for reports, output/artifacts/ for artifacts, output/patches/ for diffs or patches, and output/logs/ for command logs.
8. Append meaningful progress to .task_log.txt.
9. Append operational warnings, skipped large files, blocked issues, and important messages to .task_msg.txt.
10. Run '${MGMT_LIB} sync ${task_id}' after each meaningful step.
11. If all acceptance criteria are satisfied, run '${MGMT_LIB} mark_done ${task_id}'.
12. If blocked or failed, run '${MGMT_LIB} mark_error ${task_id}' and record the reason in .task_msg.txt."
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    echo "ERROR: Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
''', encoding="utf-8")

skill_file.write_text(f'''---
name: bx-codex-task-manager
description: Manage and execute Brian's bx.codex task workflow using bx.codex.tsk_mgmt_lib.py and bx.codex.tsk.sh. Use this for enqueue, start, stop, dequeue, status, grep, mark_done, mark_error, output organization, task logs, task messages, and git-backed task synchronization.
---

# bx.codex Task Manager

## Purpose

Use Brian's Git-backed task workflow.

Command files:

- `bx.codex.tsk_mgmt_lib.py`: deterministic Python task management backend
- `bx.codex.tsk.sh`: human-facing Bash frontend and Codex wrapper

## Paths

- `BPUB_ROOT`: `{bpub_root}`
- `TASK_LOCAL_ROOT`: `{task_local_root}`
- `TASK_REMOTE_ROOT`: `{task_remote_root}`
- Task definition file: `.task_def.yaml`
- Task metadata file: `.task_meta.yaml`
- Task log file: `.task_log.txt`
- Task message file: `.task_msg.txt`
- Task output root: `<PER_TASK_ROOT>/output/`

## Statuses

- `100_TBS`: to be started
- `150_STP`: stopped
- `200_WIP`: work in progress
- `300_DID`: did it
- `800_ERR`: error

## Task definition schema

Mandatory fields:

    summary: overall task description
    subtasks:
      - subtask 1
      - subtask 2

Optional fields:

    special_requests:
      - request 1
    tags:
      - tag 1
    acceptance_criteria:
      - criterion 1

## Task execution workflow

When asked to start a task:

1. Run `bx.codex.tsk_mgmt_lib.py status <task_id>`.
2. Run `bx.codex.tsk_mgmt_lib.py start <task_id>` if the task can be started.
3. Run `bx.codex.tsk_mgmt_lib.py paths <task_id>` to identify paths.
4. Read `<PER_TASK_ROOT>/.task_def.yaml`.
5. Execute all subtasks.
6. Save all results only under `<PER_TASK_ROOT>/output/`.
7. Use:
   - `output/README.md` for final summary
   - `output/reports/` for reports
   - `output/artifacts/` for generated artifacts
   - `output/patches/` for patches or diffs
   - `output/logs/` for command logs
8. Append meaningful progress notes to `.task_log.txt`.
9. Append warnings, blocked issues, skipped large files, and operational messages to `.task_msg.txt`.
10. Run `bx.codex.tsk_mgmt_lib.py sync <task_id>` after each meaningful step.
11. If all acceptance criteria are satisfied, run `bx.codex.tsk_mgmt_lib.py mark_done <task_id>`.
12. If blocked or failed, run `bx.codex.tsk_mgmt_lib.py mark_error <task_id>` and record the reason in `.task_msg.txt`.

## Safety rules

- Do not delete task directories manually.
- Use `deq` only for `100_TBS`.
- Do not log secrets.
- Do not write task outputs outside `<PER_TASK_ROOT>/output/` unless explicitly required.
- Keep changes minimal and auditable.
- Do not mark `300_DID` unless acceptance criteria are satisfied.
''', encoding="utf-8")

codex_config.write_text(f'''approval_policy = "on-request"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
writable_roots = [
  "{bpub_root}"
]
''', encoding="utf-8")

mgmt.chmod(0o755)
frontend.chmod(0o755)
PYGEN

python3 - <<'PYCHK' >/dev/null 2>&1 || python3 -m pip install --user pyyaml
import yaml
PYCHK

chmod 700 "${CODEX_DIR}"
chmod 600 "${CODEX_CONFIG}"
bash -n "${FRONTEND}"
python3 -m py_compile "${MGMT_LIB}"

echo "Installed:"
echo "  ${MGMT_LIB}"
echo "  ${FRONTEND}"
echo "  ${CODEX_SKILL_FILE}"
echo "  ${CODEX_CONFIG}"
echo
echo "BPUB_ROOT=${BPUB_ROOT}"
echo "TASK_LOCAL_ROOT=${TASK_LOCAL_ROOT}"
echo
echo "Try:"
echo "  bx.codex.tsk.sh status"
SETUP_EOF

chmod +x ./setup-bx-codex-tsk-framework.sh
./setup-bx-codex-tsk-framework.sh
