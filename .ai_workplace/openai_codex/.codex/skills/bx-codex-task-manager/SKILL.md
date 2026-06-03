---
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

- `BPUB_ROOT`: `/home/blusjune/.work/openai_codex/.bpub`
- `TASK_LOCAL_ROOT`: `/home/blusjune/.work/openai_codex/.bpub/tsk`
- `TASK_REMOTE_ROOT`: `https://github.com/blusjune/.bpub/tsk`
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

    title: brief task description
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
