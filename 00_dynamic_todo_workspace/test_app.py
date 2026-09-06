import unittest

from app import Task, TaskInput, create_task, summarize, TASKS


class TodoCoreTests(unittest.TestCase):
    def setUp(self):
        TASKS.clear()

    def test_create_and_summarize(self):
        first = create_task(TaskInput(title="Audit model", priority="high", estimate_minutes=30))
        first.completed = True
        create_task(TaskInput(title="Record demo", estimate_minutes=60))
        report = summarize(list(TASKS.values()))
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["completion_rate"], 0.5)
        self.assertEqual(report["open_minutes"], 60)

    def test_empty_summary(self):
        self.assertEqual(summarize([])["completion_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
