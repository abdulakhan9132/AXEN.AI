import tempfile
import unittest
from pathlib import Path

from app.agent.executor import Executor
from app.agent.memory import MemoryStore
from app.agent.observer import Observer
from app.agent.planner import Planner
from app.agent.recovery import RecoveryManager
from app.agent.state import TaskState, TaskStore
from app.agent.tools import ToolRegistry
from app.agent.verifier import Verifier
from app.ai.base_model import AIModel
from app.ai.context import ContextManager
from app.security.permissions import SecurityPolicy
from app.tools.registry import build_registry


class MockModel(AIModel):
    def generate(self, prompt, *, system=None, context=None, response_format="text"):
        return '{"goal": "demo task", "steps": [{"id": 1, "objective": "Run the tool", "tool": "calculator.evaluate", "requires_approval": false, "arguments": {"expression": "2 + 2"}}]}'

    def stream(self, prompt, *, system=None, context=None, response_format="text"):
        yield "step"

    def structured_output(self, prompt, *, schema=None, system=None, context=None):
        return {
            "goal": "demo task",
            "steps": [{
                "id": 1,
                "objective": "Run the tool",
                "tool": "calculator.evaluate",
                "requires_approval": False,
                "arguments": {"expression": "2 + 2"},
            }],
        }

    def health_check(self):
        return True


class MockModelManager:
    def __init__(self):
        self.model = MockModel()

    def health_check(self):
        return True

    def get_model(self):
        return self.model


class TestPlanner(unittest.TestCase):
    def test_planner_creates_steps(self):
        planner = Planner(model_manager=MockModelManager())
        plan = planner.plan("Find all PDF files in the project.")
        self.assertIn("steps", plan)
        self.assertTrue(len(plan["steps"]) >= 1)


class TestExecutor(unittest.TestCase):
    def test_executor_runs_calculator(self):
        registry = build_registry()
        executor = Executor(registry, SecurityPolicy())
        result = executor.execute_step({"tool": "calculator.evaluate", "arguments": {"expression": "2 + 2"}})
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["result"], 4)


class TestObserver(unittest.TestCase):
    def test_observer_separates_facts_and_assumptions(self):
        observer = Observer()
        observation = observer.observe({"success": True, "result": {"files": ["a.pdf", "b.pdf"]}})
        self.assertTrue(any("success" in fact.lower() for fact in observation.facts))
        self.assertTrue(any("Found 2 matching" in fact for fact in observation.facts))


class TestVerifier(unittest.TestCase):
    def test_verifier_checks_file_creation(self):
        verifier = Verifier()
        path = Path("test_verifier_output.txt")
        path.write_text("demo", encoding="utf-8")
        result = verifier.verify({"tool": "filesystem.write", "arguments": {"path": str(path)}}, {"success": True, "result": {"result": "ok"}})
        self.assertTrue(result.passed_)
        path.unlink(missing_ok=True)


class TestMemory(unittest.TestCase):
    def test_memory_store_persists_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir) / "memory.db")
            entry_id = store.add_memory(content="Remember the task", source="test", category="working_memory")
            self.assertTrue(entry_id)
            self.assertGreater(len(store.search("Remember")), 0)


class TestToolRegistry(unittest.TestCase):
    def test_registry_contains_expected_tools(self):
        registry = build_registry()
        tools = {tool.name for tool in registry.list()}
        self.assertIn("filesystem.search", tools)
        self.assertIn("calculator.evaluate", tools)
        self.assertIn("shell.execute", tools)
        self.assertIn("memory.read", tools)


class TestPermissions(unittest.TestCase):
    def test_dangerous_shell_commands_are_blocked(self):
        policy = SecurityPolicy()
        with self.assertRaises(PermissionError):
            policy.validate_tool_call("shell.execute", {"command": "rm -rf /"})


class TestContextManagement(unittest.TestCase):
    def test_context_builds_bundle(self):
        manager = ContextManager()
        bundle = manager.build_context(
            task="Find PDFs",
            plan={"goal": "Find PDFs", "steps": [{"id": 1, "objective": "Search"}]},
            memory=["Important fact"],
            tool_results=["Listed files"],
            constraints=["Keep it local"],
        )
        self.assertEqual(bundle.current_task, "Find PDFs")
        self.assertIn("Important fact", bundle.relevant_memory)


class TestRecovery(unittest.TestCase):
    def test_recovery_manager_flags_retryable_failure(self):
        manager = RecoveryManager(max_retries=3)
        result = manager.recover(error="Temporary network issue", step={"tool": "browser.snapshot"}, attempts=1)
        self.assertEqual(result["status"], "replan")
        self.assertTrue(result["retryable"])


class TestStructuredOutputs(unittest.TestCase):
    def test_planner_outputs_json_with_steps(self):
        planner = Planner(model_manager=MockModelManager())
        plan = planner.plan("Create a calculator")
        self.assertIsInstance(plan.get("steps"), list)
        self.assertTrue(all("objective" in step for step in plan["steps"]))


class TestTaskState(unittest.TestCase):
    def test_task_store_tracks_states(self):
        store = TaskStore()
        task = store.create_task("Test task")
        store.update_task(task.task_id, state=TaskState.EXECUTING)
        self.assertEqual(store.get_task(task.task_id).state, TaskState.EXECUTING)


if __name__ == "__main__":
    unittest.main()
