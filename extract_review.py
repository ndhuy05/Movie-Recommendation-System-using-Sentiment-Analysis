import pandas as pd
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import time
import os
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Install the google-genai SDK if not already installed
try:
    import google.genai as genai
except ImportError:
    raise ImportError("Please install the google-genai SDK: pip install google-genai")

# Function to load API keys from file
def load_api_keys(file_path='api_keys.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Đọc tất cả dòng và loại bỏ khoảng trắng/newline
        keys = [line.strip() for line in f.readlines() if line.strip()]
        
    return keys

# Load API keys từ file
gemini_api_keys = load_api_keys()

# Track current API key index for each thread
key_manager_lock = threading.Lock()
current_key_index = 0

def get_client():
    """Get a client with the current API key"""
    with key_manager_lock:
        return genai.Client(api_key=gemini_api_keys[current_key_index])

def switch_to_next_api_key():
    global current_key_index
    with key_manager_lock:
        current_key_index = (current_key_index + 1) % len(gemini_api_keys)
        print(f"Switched to API key #{current_key_index + 1}")

# Read the CSV and extract only 'id' and 'reviewText' columns
df = pd.read_csv('data/rotten_tomatoes_movie_reviews.csv', usecols=['id', 'reviewText'])

# Define the categories
categories = ['actor', 'director', 'script', 'image', 'sound', 'plot', 'pacing',
              'character', 'editing', 'music', 'visual_effects', 'costume',
              'set_design', 'cinematography', 'dialogue', 'atmosphere', 'themes',
              'lighting', 'makeup', 'stunts', 'performances', 'originality',
              'adaptation', 'humor', 'emotional_impact']

# Prepare the output DataFrame
output = []

# Optimized prompt template - extract only single adjectives
prompt_template = (
    """Extract only single adjectives (one word) that describe sentiment for these movie aspects from the review. If there are none for a category, return 'N/A'. Return JSON only:
    {{"actor": "good|bad|great", "director": "talented|poor", "script": "brilliant|weak", "image": "stunning|ugly", "sound": "clear|loud", "plot": "engaging|boring", "pacing": "fast|slow", "character": "complex|flat", "editing": "smooth|choppy", "music": "beautiful|annoying", "visual_effects": "amazing|terrible", "costume": "elegant|cheap", "set_design": "detailed|simple", "cinematography": "gorgeous|poor", "dialogue": "witty|awkward", "atmosphere": "dark|bright", "themes": "deep|shallow", "lighting": "dramatic|harsh", "makeup": "realistic|fake", "stunts": "exciting|dangerous", "performances": "outstanding|weak", "originality": "fresh|stale", "adaptation": "faithful|loose", "humor": "funny|unfunny", "emotional_impact": "powerful|weak"}}
    
    Review: "{review}"
    """
)
def extract_review(row, prompt):
    global current_key_index
    max_retries = len(gemini_api_keys)
    retries = 0
    
    while retries < max_retries:
        try:
            client = get_client()  # Get fresh client for each request
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'temperature': 0.1,  # Lower temperature for more consistent results
                }
            )
            
            # Handle None response
            if response is None or response.text is None:
                print("Received None response, retrying...")
                time.sleep(1)
                continue
                            
            result = response.text.strip()
            
            # Check for empty response
            if not result:
                print("Empty response received, retrying...")
                time.sleep(1)
                continue
            
            # Debug: print first few characters of response
            if len(result) < 50:
                print(f"Short response received: '{result}'")
            
            try:
                result_json = json.loads(result)
                row_result = {
                    'id': row['id'],
                    'review': row['reviewText'],
                }
                for cat in categories:
                    row_result[cat] = result_json.get(cat, 'N/A')
                return row_result
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Raw response length: {len(result)}")
                print(f"Raw response: '{result[:200]}...'")
                
                # Try to fix truncated JSON by adding missing closing braces
                if result.count('{') > result.count('}'):
                    fixed_result = result + '}'
                    try:
                        result_json = json.loads(fixed_result)
                        print("Fixed truncated JSON successfully!")
                        row_result = {
                            'id': row['id'],
                            'review': row['reviewText'],
                        }
                        for cat in categories:
                            row_result[cat] = result_json.get(cat, 'N/A')
                        return row_result
                    except:
                        pass
                
                # If still can't parse, return N/A
                row_result = {
                    'id': row['id'],
                    'review': row['reviewText'],
                }
                for cat in categories:
                    row_result[cat] = 'N/A'
                return row_result
                
        except Exception as e:
            print(f"Error with API key #{current_key_index + 1}: {str(e)[:100]}...")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"API key #{current_key_index + 1} quota exceeded, switching to next key...")
                switch_to_next_api_key()
                retries += 1
                time.sleep(2)  # Shorter delay
                continue
            elif "RATE_LIMIT_EXCEEDED" in str(e):
                time.sleep(1)  # Short wait for rate limits
                continue
            else:
                # For other errors, wait briefly and retry
                time.sleep(5)
                continue
    
    # If all keys are exhausted, stop the entire program
    print("All API keys exhausted! Stopping execution.")
    exit(1)

def process_batch(batch_data):
    """Process a batch of reviews"""
    results = []
    for idx, row in batch_data.iterrows():
        review = str(row['reviewText'])
        
        prompt = prompt_template.format(review=review)
        row_result = extract_review(row, prompt)
        results.append(row_result)
        
        # Small delay to respect rate limits
        time.sleep(0.1)
    
    return results

def save_results_batch(results, output_file):
    """Save results to CSV file in batch"""
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'review'] + categories)
        for result in results:
            writer.writerow(result)

# Optimized processing with batching
output_file = 'data/extracted_sentiments.csv'

# Check if file exists and get last processed index
start_idx = 0
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file)
        start_idx = len(existing_df)
        print(f"Resuming from index {start_idx}")
    except:
        start_idx = 0

# Create file with header if it doesn't exist
if not os.path.exists(output_file) or start_idx == 0:
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'review'] + categories)
        writer.writeheader()

# Process in batches for better performance
batch_size = 50  # Process 50 reviews at a time
df_subset = df.iloc[start_idx:]  # Start from where we left off

print(f"Processing {len(df_subset)} reviews starting from index {start_idx}")

for i in range(0, len(df_subset), batch_size):
    batch_end = min(i + batch_size, len(df_subset))
    batch = df_subset.iloc[i:batch_end]
    
    print(f"Processing batch {i//batch_size + 1}: rows {start_idx + i} to {start_idx + batch_end - 1}")
    
    try:
        # Process batch
        batch_results = process_batch(batch)
        
        # Save results immediately
        save_results_batch(batch_results, output_file)
        
        print(f"Completed batch {i//batch_size + 1}/{(len(df_subset) + batch_size - 1)//batch_size}")
        
        # Brief pause between batches
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue

print("Extraction completed!")

def convert_sentiment_to_score(sentiment, sentiment_score):
    if sentiment == "" or sentiment == "N/A" or pd.isna(sentiment):
        return 0

    splited_sentiment = sentiment.split("|")
    result = 0
    for word in splited_sentiment:
        if not sentiment_score[sentiment_score['CONCEPT'] == word]['SENTIMENT_SCORE'].empty:
            score = sentiment_score[sentiment_score['CONCEPT'] == word]['SENTIMENT_SCORE'].iloc[0]
        else:
            score = 0
        result += score

    return result / len(splited_sentiment)

sentiment_score = pd.read_csv('data/senticnet.csv')
extracted_sentiment = pd.read_csv('data/extracted_sentiments.csv')

def apply_sentiment_score(sentiment):
    return convert_sentiment_to_score(sentiment, sentiment_score)

for category in categories:
    extracted_sentiment[category] = extracted_sentiment[category].apply(apply_sentiment_score)

extracted_sentiment = extracted_sentiment.groupby('id').mean(numeric_only=True)

extracted_sentiment.to_csv("data/sentiment_score.csv")

# 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '250'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '15s'}]}}