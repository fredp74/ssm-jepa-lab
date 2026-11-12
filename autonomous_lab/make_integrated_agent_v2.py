from pathlib import Path

source_path = Path("autonomous_lab/integrated_agent.py")
destination = Path("autonomous_lab/integrated_agent_v2.py")
if destination.exists():
    raise SystemExit("Version v2 deja presente : aucun remplacement.")

source = source_path.read_text(encoding="utf-8-sig")

patches = [
    (
        '            "epoch": 0,',
        '            "epoch": 0,\n'
        '            "audit_cursor": 0,',
    ),
    (
        '            return known[(step // 4) % len(known)], "memory_audit"',
        '            cursor = self.state.get("audit_cursor", 0)\n'
        '            index = known[cursor % len(known)]\n'
        '            self.state["audit_cursor"] = cursor + 1\n'
        '            return index, "memory_audit"',
    ),
    (
        '            self.state["epoch"] += 1',
        '            self.state["epoch"] += 1\n'
        '            self.state["audit_cursor"] = 0',
    ),
    (
        'Path("autonomous_lab/runs/integrated_agent_01")',
        'Path("autonomous_lab/runs/integrated_agent_02")',
    ),
    (
        '    second = tuple(1-value for value in first)',
        '    second = tuple(\n'
        '        value ^ int(index == 17)\n'
        '        for index, value in enumerate(first)\n'
        '    )',
    ),
    (
        '                "environment_switch_after": 48,',
        '                "environment_switch_after": 48,\n'
        '                "change_type": "single_output_flip",\n'
        '                "changed_input_index": 17,\n'
        '                "audit_policy": "persistent_round_robin_cursor",',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Version creee : {destination}")
