import subprocess
from app.agent import policies as policy
from app.agent.policies import PolicyEngine

TOOLS = ["filesystem.read", "filesystem.write", "calculator.evaluate",
         "shell.execute", "browser.snapshot", "my.custom_tool"]


def show(title):
    engine = PolicyEngine()
    print(f"\n=== {title} (FULL_ACCESS={policy.FULL_ACCESS}) ===")
    for name in TOOLS:
        t = engine.tool_for(name)
        print(f"{name:22} risk={t.risk.value:7} needs_approval={engine.requires_approval(name)}")
    return engine


policy.FULL_ACCESS = True
engine = show("Full access")
assert not any(engine.requires_approval(n) for n in TOOLS)

policy.FULL_ACCESS = False
engine = show("Approvals on")
assert engine.requires_approval("shell.execute")
assert not engine.requires_approval("filesystem.read")

policy.FULL_ACCESS = True
engine = PolicyEngine()
if not engine.requires_approval("shell.execute"):
    r = subprocess.run("echo hello from AEXN", shell=True, capture_output=True, text=True)
    print("\nshell.execute ran without asking:", r.stdout.strip())

print("\nAll checks passed.")
