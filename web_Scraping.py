from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re
import numpy as np
def scrape_flipkart_reviews(product_name, product_url):
    # Initialize the WebDriver
    browser = webdriver.Chrome()  # Ensure that you have the correct path to your ChromeDriver

    # Open the provided URL
    browser.get(product_url)

    # Lists to store all data
    all_data = []

    # Loop through each page
    current_page = 1
    while True:
        try:
            # Click all "READ MORE" links to expand the reviews
            read_more_links = browser.find_elements(By.XPATH, "//span[contains(text(),'READ MORE')]")
            for link in read_more_links:
                try:
                    browser.execute_script("arguments[0].click();", link)
                    time.sleep(1)  # Add a small delay to ensure the content loads
                except Exception as e:
                    print(f"Error clicking 'READ MORE': {e}")

            # Extract all rating elements (class 'XQDdHH')
            rating_elements = browser.find_elements(By.XPATH, "//div[contains(@class, 'XQDdHH Ga3i8K')]")

            # Extract all review elements (class 'ZmyHeo')
            review_elements = browser.find_elements(By.XPATH, "//div[contains(@class, 'ZmyHeo')]")

            # Extract all dislikes
            dislikes_elements = browser.find_elements(By.XPATH,
                                                      "//div[contains(@class, '_6kK6mk') and contains(@class, 'aQymJL')]//span[contains(@class, 'tl9VpF')]")

            # Extract all likes
            likes_elements = browser.find_elements(By.XPATH,
                                                   "//div[contains(@class, '_6kK6mk') and not(contains(@class, 'aQymJL'))]//span[contains(@class, 'tl9VpF')]")


            # Loop through each review, rating, likes, and dislikes element on the current page
            for review, rating, likes, dislikes in zip(review_elements, rating_elements, likes_elements, dislikes_elements):
                # Convert likes and dislikes to integers
                try:
                    # Handle possible missing or non-integer values for likes and dislikes
                    likes_text = likes.text.replace(',', '')
                    dislikes_text = dislikes.text.replace(',', '')
                    likes_count = int(likes_text) if likes_text.isdigit() else 0
                    dislikes_count = int(dislikes_text) if dislikes_text.isdigit() else 0
                    all_data.append({
                        'Product Name': product_name,
                        'Rating': rating.text,
                        'Review': review.text,
                        'Likes': likes_count,
                        'Dislikes': dislikes_count
                    })
                except Exception as e:
                    pass

            # Wait for the "Next" button to become clickable
            next_button = WebDriverWait(browser, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'Next')]"))
            )

            # Check if the next_button element is not None and if it's not disabled
            if next_button is not None and 'disabled' not in next_button.get_attribute('class'):
                # Use JavaScript to click the "Next" button to avoid the interception issue
                browser.execute_script("arguments[0].click();", next_button)
                current_page += 1
                time.sleep(2)
            else:
                # Break out of the loop if the "Next" button is disabled or not found
                break
        except Exception as e:
            # print(f"An error occurred: {e}")
            break

    # Close the browser
    browser.quit()

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)

    # Data Cleaning
    # Clearing the Special Character
    pattern = re.compile(r'[^\x00-\x7F]+')
    df["Review"] = df["Review"].replace({pattern: ' '}, regex=True)
    # Trim spaces from the values in the Review column
    df['Review'] = df['Review'].str.strip()
    # Replace empty strings with NaN
    df['Review'] = df['Review'].replace({'': np.nan})
    # Drop rows where Review is NaN
    df = df.dropna(subset=['Review'])
    # Optionally, reset the index if you want a clean DataFrame without gaps in the index
    df = df.reset_index(drop=True)
    csv_filename = f"{product_name}_reviews.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved reviews to {csv_filename}")

url_list={
    "SAMSUNG Galaxy A34 5G" : "https://www.flipkart.com/samsung-galaxy-a34-5g-awesome-violet-256-gb/product-reviews/itm7e3c2507eac6b?pid=MOBGNE4SAZEPPZNN&lid=LSTMOBGNE4SAZEPPZNNDQZJKA&marketplace=FLIPKART&page=1",
    "REDMI Note 13 Pro+ 5G" : "https://www.flipkart.com/redmi-note-13-pro-5g-fusion-purple-512-gb/product-reviews/itm23c59d243f177?pid=MOBGWFHFCRGFQGDK&lid=LSTMOBGWFHFCRGFQGDKSW03TT&marketplace=FLIPKART&page=1",
    "OPPO Reno11 5G" : "https://www.flipkart.com/oppo-reno11-5g-rock-grey-256-gb/product-reviews/itm5665de8ad650f?pid=MOBGWU4CQDZHMSEY&lid=LSTMOBGWU4CQDZHMSEY9VWVUM&marketplace=FLIPKART&page=1",
    "vivo V25 Pro 5G" : "https://www.flipkart.com/vivo-v25-pro-5g-pure-black-128-gb/product-reviews/itmeccdbf93bbfc6?pid=MOBGGRB79Z4ANQ9G&lid=LSTMOBGGRB79Z4ANQ9GXEIMMZ&marketplace=FLIPKART&page=1",
    "POCO F4 5G" : "https://www.flipkart.com/poco-f4-5g-nebula-green-128-gb/product-reviews/itmae3dc60dfbc3a?pid=MOBGE853R2HGTSZN&lid=LSTMOBGE853R2HGTSZNYJGBAM&marketplace=FLIPKART&page=1",
    "realme 12+ 5G" : "https://www.flipkart.com/realme-12-5g-navigator-beige-128-gb/product-reviews/itmaf084e49c78e2?pid=MOBGYAFKS5P3GAHY&lid=LSTMOBGYAFKS5P3GAHYLSL84Z&marketplace=FLIPKART&page=1"
}

for product_name, product_url in url_list.items():
    scrape_flipkart_reviews(product_name, product_url)


