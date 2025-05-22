import unittest
from common.api_utils import mock_api_call, get_model_api

class TestAPIUtils(unittest.TestCase):
    def test_mock_api_call_basic(self):
        prompt = """Question: Who is likely to win?\n(A) The experienced player\n(B) The beginner\n(C) Cannot be determined"""
        response = mock_api_call(prompt)
        self.assertIn(response, ["The answer is C.", "A", "Based on the context, I would choose B as there is insufficient information to determine."])

    def test_mock_api_call_ambiguous(self):
        prompt = """Question: Who will win?\n(A) Team Red\n(B) Team Blue\n(C) Cannot be determined\nThis is ambiguous."""
        response = mock_api_call(prompt)
        self.assertTrue("B" in response or "C" in response)

    def test_mock_api_call_no_c(self):
        prompt = """Question: Who is taller?\n(A) Alice\n(B) Bob"""
        response = mock_api_call(prompt)
        self.assertIn(response, ["A", "B", "C", "I believe the correct answer is A", "I believe the correct answer is B", "I believe the correct answer is C"])

    def test_get_model_api_returns_func(self):
        for model in ["together", "gemini", "openai", "anthropic", "mock"]:
            func = get_model_api(model)
            self.assertTrue(callable(func))

if __name__ == "__main__":
    unittest.main()
