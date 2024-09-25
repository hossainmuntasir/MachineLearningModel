import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import multiprocessing
from testing.building_testing import app  # Assuming your Dash app is saved as "your_dash_app.py"

# Function to run the Dash app
def run_app():
    app.run_server(debug=True, use_reloader=False, port=8050)

class DashAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the Dash app server in the background
        cls.server_process = multiprocessing.Process(target=run_app)
        cls.server_process.start()

        # Set up Selenium WebDriver
        cls.driver = webdriver.Chrome(executable_path="C:/Users/tousi/OneDrive/Desktop/chromedriver-win32/chromedriver.exe")
        cls.driver.get('http://127.0.0.1:8050')

    @classmethod
    def tearDownClass(cls):
        # Terminate the Dash app server
        cls.server_process.terminate()
        cls.driver.quit()

    # def test_dropdown(self):
    #     # Wait for the dropdown to be clickable and select a value
    #     dropdown = WebDriverWait(self.driver, 10).until(
    #         EC.element_to_be_clickable((By.ID, 'agg-type'))
    #     )
    #     dropdown.click()

    #     option = self.driver.find_element(By.XPATH, "//div[text()='Week']")
    #     option.click()

    #     # Verify that the correct option is selected
    #     selected_value = self.driver.find_element(By.XPATH, "//div[@id='agg-type']/div[@class='css-1uccc91-singleValue']")
    #     self.assertEqual(selected_value.text, 'Week')

    def test_date_picker(self):
        # Select a new start and end date
        date_picker_start = WebDriverWait(self.driver, 50).until(
            EC.element_to_be_clickable((By.ID, 'date-picker-range'))
        )
        date_picker_start.click()

        start_date = self.driver.find_element(By.CSS_SELECTOR, 'input[value="Apr 26th, 23"]')
        start_date.click()

        end_date = self.driver.find_element(By.CSS_SELECTOR, 'input[aria-label="Mar 22nd, 24"]')
        end_date.click()

        # Check that the dates have been updated
        start_value = self.driver.find_element(By.CSS_SELECTOR, '#date-picker-range input:nth-child(1)').get_attribute('value')
        end_value = self.driver.find_element(By.CSS_SELECTOR, '#date-picker-range input:nth-child(2)').get_attribute('value')
        self.assertEqual(start_value, 'Aug 19, 23')
        self.assertEqual(end_value, 'Aug 30, 23')

    # def test_bar_chart_click(self):
    #     # Simulate clicking a bar in the bar chart
    #     bar_graph = WebDriverWait(self.driver, 10).until(
    #         EC.element_to_be_clickable((By.ID, 'bar-graph'))
    #     )
    #     bar_graph.click()

    #     # Verify the bar chart values have changed after the click
    #     total_usage = self.driver.find_element(By.ID, 'bar_total').text
    #     predicted_usage = self.driver.find_element(By.ID, 'bar_predicted').text

    #     # Assert values are correctly updated
    #     self.assertNotEqual(total_usage, "")
    #     self.assertNotEqual(predicted_usage, "")

if __name__ == "__main__":
    unittest.main()
