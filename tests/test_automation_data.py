"""
tests/test_automation_data.py
Kiểm thử tự động hóa việc lấy dữ liệu (Data Retrieval Automation Testing)
Tích hợp trực tiếp với bộ khung unittest chuẩn của dự án.
"""

import unittest
from tools.automation_data_tester import AutomationDataTester


class TestDataRetrievalAutomation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tester = AutomationDataTester(verbose=False)

    def test_database_tour_and_schedule_retrieval(self):
        self.tester.test_database_data_retrieval()
        suite = self.tester.results["suites"]["Database Data Retrieval"]
        self.assertEqual(suite["failed"], 0)
        self.assertGreaterEqual(suite["passed"], 2)

    def test_hotel_and_resort_directory_retrieval(self):
        self.tester.test_hotel_data_retrieval()
        suite = self.tester.results["suites"]["Hotel & Resort Directory Retrieval"]
        self.assertEqual(suite["failed"], 0)
        self.assertGreaterEqual(suite["passed"], 2)

    def test_live_weather_api_retrieval(self):
        self.tester.test_weather_data_retrieval()
        suite = self.tester.results["suites"]["Live Weather API Retrieval"]
        self.assertEqual(suite["failed"], 0)
        self.assertGreaterEqual(suite["passed"], 3)

    def test_web_search_and_scraping_retrieval(self):
        self.tester.test_web_search_and_scraping_retrieval()
        suite = self.tester.results["suites"]["Web Search & Travel Guide Synthesis"]
        self.assertEqual(suite["failed"], 0)
        self.assertGreaterEqual(suite["passed"], 3)

    def test_smart_recommendation_and_math_engine(self):
        self.tester.test_smart_recommendation_and_math()
        suite = self.tester.results["suites"]["Recommendation & Math Calculations"]
        self.assertEqual(suite["failed"], 0)
        self.assertGreaterEqual(suite["passed"], 2)

    def test_e2e_flask_api_chat_simulation(self):
        self.tester.test_e2e_api_chat_simulation()
        suite = self.tester.results["suites"]["E2E API Chat Multi-Turn Simulation"]
        self.assertEqual(suite["failed"], 0)
        self.assertEqual(suite["passed"], 7)


if __name__ == "__main__":
    unittest.main()
