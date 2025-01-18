from selenium import webdriver
from selenium.webdriver.common.by import By
import random
import time

# Replace with the URL of your Google Form
google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSdBbIOWKFix8ANWVNvli3E6SeiDa5GizlkaRlbAU6Lzr1o36g/viewform"

# Initialize the WebDriver (Make sure to replace the path with your actual chromedriver path)
driver = webdriver.Chrome()

def fill_form():
    try:
        # Open the Google Form
        driver.get(google_form_url)
        time.sleep(2)  # Wait for the page to load

        # Fill out the first question (replace with the actual field XPath or CSS selector)
        # Select "Male" or "Female" radio button randomly
        gender_options = [
            '//*[@aria-label="Male"]',   # XPath for Male radio button
            '//*[@aria-label="Female"]' # XPath for Female radio button
        ]
        selected_option = random.choice(gender_options)
        gender_button = driver.find_element(By.XPATH, selected_option)
        gender_button.click()

        random_age = str(random.randint(18, 24))

        age_field = driver.find_element(By.XPATH, '//input[@aria-labelledby="i18 i21"]')
        age_field.send_keys(random_age)

        # Options and their corresponding weights
        options = ["Albanian", "Macedonian", "Turkish", "macedonian", "albanian", "turkish"]
        weights = [4, 1, 4, 1, 4, 4]  # Higher weights for "Albanian" and "Turkish"
        
        # Select a random option based on weights
        random_language = random.choices(options, weights=weights, k=1)[0]

        # Locate the text field and input the selected option
        language_field = driver.find_element(By.XPATH, '//input[@aria-labelledby="i23 i26"]')
        language_field.send_keys(random_language)

        uni_options = ["Balkan University", "Kiril and Metodij", "SEEU University", "Tetovo University", "Mother Theresa"]
        uni_weights = [3, 1, 3, 3, 3]  # Higher weights for "Albanian" and "Turkish"
        
        # Select a random option based on weights
        random_uni = random.choices(uni_options, weights=uni_weights, k=1)[0]

        # Locate the text field and input the selected option
        fac_field = driver.find_element(By.XPATH, '//input[@aria-labelledby="i28 i31"]')
        fac_field.send_keys(random_uni)

        fac_options = ["Business Administration", "Electrical Engineering", "Architecture", "Law", "Pedagogy", "Computer Science", "Economics", "Marketing"]
        fac_weights = [4, 1, 1, 4, 4, 4, 4, 3]  # Higher weights for "Albanian" and "Turkish"
        
        # Select a random option based on weights
        random_uni = random.choices(fac_options, weights=fac_weights, k=1)[0]

        # Locate the text field and input the selected option
        fac_field = driver.find_element(By.XPATH, '//input[@aria-labelledby="i33 i36"]')
        fac_field.send_keys(random_uni)

        # -----------------------------

        # Define options with weights (higher weight = more likely to be selected)
        options = [
            {"label": "First year", "xpath": '//div[@aria-label="First year"]'},
            {"label": "Second year", "xpath": '//div[@aria-label="Second year"]'},
            {"label": "Third year", "xpath": '//div[@aria-label="Third year"]'},
            {"label": "Fourth year", "xpath": '//div[@aria-label="Fourth year"]'},
            {"label": "Other", "xpath": '//div[@id="i55" and @data-value="__other_option__" and @role="radio"]'}
        ]
        weights = [3, 3, 3, 3, 1]  # Rarely select "Other"

        # Randomly select an option based on weights
        selected_option = random.choices(options, weights=weights, k=1)[0]

        # Click on the selected option
        driver.find_element(By.XPATH, selected_option["xpath"]).click()
        print(f"Selected option: {selected_option['label']}")

        # If "Other" is selected, input "Master's"
        if selected_option["label"] == "Other":
            other_input = driver.find_element(By.XPATH, '//input[@aria-label="Other response"]')
            other_input.send_keys("Master's")
            print("Entered 'Master's' in the Other response.")

        # ----------------------------------------------------

        # Define the XPath for the radio buttons (1 to 5)
        question_elements = driver.find_elements(By.CSS_SELECTOR, "div.Qr7Oae")

        # Loop through each question element (one at a time)
        for question_element in question_elements:
            # Get the radio buttons for the current question (within that specific div.Qr7Oae)
            radio_buttons = question_element.find_elements(By.CSS_SELECTOR, "label.T5pZmf div[role='radio']")
            
            # Print the number of radio buttons found for debugging
            print(f"Number of radio buttons for this question: {len(radio_buttons)}")
            
            # Ensure that there are radio buttons available for this question
            if len(radio_buttons) > 0:
                # Randomly select a radio button from the available ones
                random_index = random.randint(0, len(radio_buttons) - 1)  # Pick a valid random index
                
                # Click the randomly selected radio button for this question
                radio_buttons[random_index].click()
            else:
                print("No radio buttons found for this question.")


        # Loop through each question div
        for question_element in question_elements:
            # Find all the radio buttons for the current question div (be more specific)
            radio_buttons = question_element.find_elements(By.CSS_SELECTOR, "div[role='radiogroup'] div[role='radio']")
            
            # Print the number of radio buttons for debugging
            print(f"Number of radio buttons for this question: {len(radio_buttons)}")
            
            # If radio buttons are found, select one
            if len(radio_buttons) > 0:
                # Pick a random radio button index
                random_index = random.randint(0, len(radio_buttons) - 1)
                
                # Click the selected radio button
                radio_buttons[random_index].click()
            else:
                print("No radio buttons found for this question.")


        # Print which option was selected
        print(f"Selected option: {random_index + 1}")


        # Submit the form (replace with the actual submit button XPath or CSS selector)
        submit_button = driver.find_element(By.XPATH, '//*[@role="button"][@data-disable-on-invalid="true"]')
        submit_button.click()

        print("Form submitted successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the form submission 5 times
for _ in range(78):
    fill_form()

# Close the browser after 5 submissions
driver.quit()
