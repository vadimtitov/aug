"""Per-tool display formatters for the Telegram rolling tool-status draft.

Each registered tool has a (label, args_formatter) pair.
format_tool() returns (label, args_preview) suitable for single-line display.
"""

from urllib.parse import urlparse

_ARG_TRUNCATE = 40


def _first_arg(args: dict) -> str:
    if not args:
        return ""
    value = next(iter(args.values()))
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value)
    return text[:_ARG_TRUNCATE] + "…" if len(text) > _ARG_TRUNCATE else text


def _fetch_page_fmt(args: dict) -> str:
    urls = args.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    hosts = [urlparse(u).netloc or u for u in urls[:2]]
    text = ", ".join(hosts)
    return text[:_ARG_TRUNCATE] + "…" if len(text) > _ARG_TRUNCATE else text


def _ssh_fmt(args: dict) -> str:
    target = str(args.get("target", "?"))
    cmd = str(args.get("command") or args.get("remote_path") or args.get("local_path") or "")
    inner = f"{target}: {cmd}" if cmd else target
    return inner[:_ARG_TRUNCATE] + "…" if len(inner) > _ARG_TRUNCATE else inner


def _prompt_fmt(args: dict) -> str:
    """Format the first arg of run_subagent (prompt), truncated for display."""
    prompt = str(args.get("prompt", ""))
    return prompt[:_ARG_TRUNCATE] + "…" if len(prompt) > _ARG_TRUNCATE else prompt


_REGISTRY: dict[str, tuple[str, object]] = {
    "brave_search": ("🔍 Search", _first_arg),
    "fetch_page": ("🌐 Fetch", _fetch_page_fmt),
    "run_bash": ("💻 Bash", _first_arg),
    "browser": ("🖥️ Browser", _first_arg),
    "note": ("📝 Note", _first_arg),
    "gmail_search": ("📧 Gmail", _first_arg),
    "gmail_read_thread": ("📧 Gmail", _first_arg),
    "gmail_send": ("📤 Send email", _first_arg),
    "gmail_draft": ("📝 Draft email", _first_arg),
    "respond_with_file": ("📎 Send file", lambda a: str(a.get("filename", ""))),
    "generate_image": ("🖼️ Image", _first_arg),
    "edit_image": ("✏️ Edit image", _first_arg),
    "portainer_list_containers": ("🐳 Portainer", _first_arg),
    "portainer_container_logs": ("🐳 Logs", _first_arg),
    "portainer_container_action": ("🐳 Action", _first_arg),
    "portainer_list_stacks": ("🐳 Stacks", _first_arg),
    "portainer_deploy_stack": ("🐳 Deploy", _first_arg),
    "portainer_stack_action": ("🐳 Stack", _first_arg),
    "run_ssh": ("🔌 SSH", _ssh_fmt),
    "list_ssh_targets": ("🔌 SSH targets", lambda _: ""),
    "download_ssh_file": ("🔌 SSH download", _ssh_fmt),
    "upload_ssh_file": ("🔌 SSH upload", _ssh_fmt),
    "create_task": ("📋 Create task", _first_arg),
    "list_tasks": ("📋 List tasks", lambda _: ""),
    "update_task": ("📋 Update task", _first_arg),
    "delete_task": ("📋 Delete task", _first_arg),
    "get_skill": ("⚡ Get skill", _first_arg),
    "save_skill": ("⚡ Save skill", _first_arg),
    "write_skill_file": ("⚡ Write skill", _first_arg),
    "delete_skill": ("⚡ Delete skill", _first_arg),
    "run_subagent": ("🤖 Agent", _prompt_fmt),
}


def format_tool(tool_name: str, args: dict) -> tuple[str, str]:
    """Return (label, args_preview) for a tool call.

    Falls back to (tool_name, first_arg_preview) for unregistered tools.
    """
    entry = _REGISTRY.get(tool_name)
    if entry:
        label, formatter = entry
        return label, formatter(args)  # type: ignore[operator]
    return tool_name, _first_arg(args)
