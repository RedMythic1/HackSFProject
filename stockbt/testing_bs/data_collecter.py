from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

def scrape_yahoo_finance_selenium(ticker, prev_price=None, prev_total_vol=None):
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
    except Exception as e:
        driver.quit()
        print(f"Failed to load page or extract data. Exception: {e}")
        return None, None

    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-testid='qsp-price']")))
    except Exception as e:
        driver.quit()
        print(f"Failed to load page or extract data. Exception: {e}")
        return None, None

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

    # Extract Bid/Ask
    value_elems = soup.find_all(class_="value yf-1jj98ts")
    bid_price = ask_price = None
    bid_found = ask_found = False
    for elem in value_elems:
        text = elem.text.strip()
        if 'x' in text:
            price, _ = text.split('x', 1)
            price = price.strip().replace(',', '')
            try:
                price_f = float(price)
            except ValueError:
                continue
            if not bid_found:
                bid_price = price_f
                bid_found = True
            elif not ask_found:
                ask_price = price_f
                ask_found = True
            if bid_found and ask_found:
                break

    # Extract total volume
    total_vol_elem = soup.find("fin-streamer", attrs={"data-field": "regularMarketVolume"})
    total_vol = None
    if total_vol_elem and total_vol_elem.get("data-value"):
        total_vol_str = total_vol_elem["data-value"].replace(',', '')
        try:
            total_vol = int(total_vol_str)
        except ValueError:
            total_vol = None

    # Calculate change in total volume
    total_vol_change = None
    if total_vol is not None and prev_total_vol is not None:
        total_vol_change = total_vol - prev_total_vol
    elif total_vol is not None:
        total_vol_change = 0

    driver.quit()

    vector = [price_value, price_change, bid_price, ask_price, total_vol_change]
    print(vector)
    return price_value, total_vol, vector

if __name__ == "__main__":
    ticker = "NVDA"
    prev_price = None
    prev_total_vol = None
    while True:
        price, total_vol, vector = scrape_yahoo_finance_selenium(ticker, prev_price, prev_total_vol)
        if price is not None:
            prev_price = price
        if total_vol is not None:
            prev_total_vol = total_vol
        time.sleep(10)  # Wait 10 seconds between scrapes
