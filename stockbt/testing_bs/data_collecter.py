from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def scrape_yahoo_finance_selenium(ticker, prev_price=None):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)
    try:
        driver.get(url)
    except Exception:
        driver.quit()
        print("Failed to load page or extract data.")
        return None

    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-testid='qsp-price']")))
    except Exception:
        driver.quit()
        print("Failed to load page or extract data.")
        return None

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Extract the main price
    price_elem = soup.find("span", attrs={"data-testid": "qsp-price"})
    price_value = float(price_elem.text.replace(',', '').strip()) if price_elem and price_elem.text.strip().replace('.', '', 1).replace(',', '').isdigit() else None

    # Calculate price change
    price_change = None
    if price_value is not None and prev_price is not None:
        price_change = price_value - prev_price
    elif price_value is not None:
        price_change = 0.0

    # Extract Bid/Ask and Volumes
    value_elems = soup.find_all(class_="value yf-1jj98ts")
    bid_price = ask_price = buy_vol = sell_vol = None
    bid_found = ask_found = False
    for elem in value_elems:
        text = elem.text.strip()
        if 'x' in text:
            price, volume = text.split('x', 1)
            price = price.strip().replace(',', '')
            volume = volume.strip().replace(',', '')
            try:
                price_f = float(price)
                volume_i = int(volume)
            except ValueError:
                continue
            if not bid_found:
                bid_price = price_f
                buy_vol = volume_i
                bid_found = True
            elif not ask_found:
                ask_price = price_f
                sell_vol = volume_i
                ask_found = True
            if bid_found and ask_found:
                break

    driver.quit()

    vector = [price_value, price_change, bid_price, ask_price, buy_vol, sell_vol]
    print(vector)
    return vector

if __name__ == "__main__":
    scrape_yahoo_finance_selenium("NVDA")
