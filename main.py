import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone

# Selenium scraping
data = []
youtube_video_url = "https://www.youtube.com/watch?v=NYc-I1bntmc"
chrome_driver_path = r"C:\Program Files\Google\Chrome\Application\chromedriver.exe"
service = Service(chrome_driver_path)

driver = webdriver.Chrome(service=service)
wait = WebDriverWait(driver, 15)
driver.get(youtube_video_url)

last_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
    time.sleep(5)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Implement infinite scrolling
while True:
    # Scroll down the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # Check for "Load more comments" button or other loading indicator
    # If found, click it
    # ...

    # Check if all comments are loaded (e.g., count the number of comments)
    # If all comments are loaded, break the loop
    # ...

comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content #content-text")))
for comment in comments:
    data.append(comment.text)

driver.quit()

df = pd.DataFrame(data, columns=['comment'])
print(df)

# Embedding comments
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.numpy()

# Initialize Pinecone
pinecone.init(api_key='cb574695-dac0-45ca-806b-3fa18bdd5ef9', environment='us-west1-gcp')
index_name = 'youtube-comments'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)

index = pinecone.Index(index_name)

# Upsert comments to Pinecone
for i, comment in enumerate(df['comment']):
    vector = embed_text(comment)[0]
    index.upsert([(f'comment-{i}', vector)])

print("Comments stored in Pinecone vector database.")

# Query Pinecone
query_vector = embed_text("Sample query text")[0]
response = index.query(
    vector=query_vector,
    top_k=10,
    include_values=True
)

print("Query response:", response)
