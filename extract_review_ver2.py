import pandas as pd
import time
import os
import csv
import threading
import json

# Install the google-genai SDK if not already installed
try:
    import google.genai as genai
except ImportError:
    raise ImportError("Please install the google-genai SDK: pip install google-genai")

# Function to load API keys from file
def load_api_keys(file_path='api_keys.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        keys = [line.strip() for line in f.readlines() if line.strip()]
    return keys

# Load API keys and split them for extraction and validation
all_api_keys = load_api_keys()
extraction_keys = all_api_keys[:len(all_api_keys)//2]  # First half for extraction
validation_keys = all_api_keys[len(all_api_keys)//2:]  # Second half for validation

print(f"Loaded {len(extraction_keys)} keys for extraction, {len(validation_keys)} keys for validation")

# Track current API key index for each service
extraction_key_lock = threading.Lock()
validation_key_lock = threading.Lock()
current_extraction_key_index = 0
current_validation_key_index = 0

def get_extraction_client():
    """Get a client with the current extraction API key"""
    with extraction_key_lock:
        return genai.Client(api_key=extraction_keys[current_extraction_key_index])

def get_validation_client():
    """Get a client with the current validation API key"""
    with validation_key_lock:
        return genai.Client(api_key=validation_keys[current_validation_key_index])

def switch_extraction_key():
    global current_extraction_key_index
    with extraction_key_lock:
        current_extraction_key_index = (current_extraction_key_index + 1) % len(extraction_keys)
        print(f"Switched to extraction API key #{current_extraction_key_index + 1}")

def switch_validation_key():
    global current_validation_key_index
    with validation_key_lock:
        current_validation_key_index = (current_validation_key_index + 1) % len(validation_keys)
        print(f"Switched to validation API key #{current_validation_key_index + 1}")

# Read the CSV and extract only 'id' and 'reviewText' columns
df = pd.read_csv('data/rotten_tomatoes_movie_reviews.csv', usecols=['id', 'reviewText'])

# Define the categories - same as original code
categories = ['actor', 'director', 'script', 'image', 'sound', 'plot', 'pacing',
              'character', 'editing', 'music', 'visual_effects', 'costume',
              'set_design', 'cinematography', 'dialogue', 'atmosphere', 'themes',
              'lighting', 'makeup', 'stunts', 'performances', 'originality',
              'adaptation', 'humor', 'emotional_impact']

# Define the extraction prompt
EXTRACTION_PROMPT = """ROLE: You are a professional sentiment and aspect-based opinion mining assistant specializing in analyzing movie reviews.
ACTION: Analyze the provided movie review to extract sentiment expressions and classify them into appropriate film-related aspects.
TASK:
Extract sentiment expressions, defined as:
- A single sentiment word, optionally preceded by a modifier:
- Negations (e.g., not, never)
- Intensifiers (e.g., very, extremely, too, so)
- Do not extract phrases longer than 2 words.
- Keep the modifier together with the word (e.g., not good, very bad), and treat it as one sentiment expression.
- Aspects to classify into:
	+) actor: Sentiment expressions about acting performances, cast, actors
	+) director: Sentiment expressions about directing, direction, filmmaker's vision
	+) script: Sentiment expressions about screenplay, writing, storyline
	+) image: Sentiment expressions about visual imagery, cinematography, visuals
	+) sound: Sentiment expressions about audio, sound effects, sound design
	+) plot: Sentiment expressions about story, narrative, plot structure
	+) pacing: Sentiment expressions about rhythm, tempo, flow of the film
	+) character: Sentiment expressions about character development, characterization
	+) editing: Sentiment expressions about film editing, cuts, transitions
	+) music: Sentiment expressions about soundtrack, score, musical elements
	+) visual_effects: Sentiment expressions about VFX, CGI, special effects
	+) costume: Sentiment expressions about costumes, wardrobe, clothing
	+) set_design: Sentiment expressions about sets, production design, locations
	+) cinematography: Sentiment expressions about camera work, shots, filming
	+) dialogue: Sentiment expressions about conversations, lines, speech
	+) atmosphere: Sentiment expressions about mood, ambiance, feeling
	+) themes: Sentiment expressions about underlying messages, themes
	+) lighting: Sentiment expressions about illumination, lighting effects
	+) makeup: Sentiment expressions about makeup, prosthetics, appearance
	+) stunts: Sentiment expressions about action sequences, stunt work
	+) performances: Sentiment expressions about overall acting quality
	+) originality: Sentiment expressions about creativity, uniqueness, innovation
	+) adaptation: Sentiment expressions about adaptation from source material
	+) humor: Sentiment expressions about comedy, jokes, funny elements
	+) emotional_impact: Sentiment expressions about emotional effect, feelings evoked

INSTRUCTIONS
- Only extract sentiment expressions (single-word or modifier + word).
- Do not include neutral or factual terms without sentiment.
- Do not infer or generate new expressions — only extract from the actual review.
- Classify each sentiment expression under exactly one aspect.

EXAMPLES:
1.	Review #1: The actors were not convincing, and the dialogue felt forced.
Extract:
- actor: not convincing
- dialogue: forced
2. Review #2: A visually stunning movie with breathtaking shots and very immersive sound design.
Extract:
- image: stunning, breathtaking
- sound: very immersive
3.	Review #3: The director's choices were bold but sometimes confusing.
Extract:
- director: bold, confusing
4.	Review #4: Although the film had a slow start, the lead actor delivered an exceptionally powerful performance.
Extract:
- pacing: slow
- actor: exceptionally powerful
5.	Review: Not bad, but not great either. Just average.
Extract:
- emotional_impact: not bad, not great, average

OUTPUT FORMAT:
{{
  "actor": ["..."],
  "director": ["..."],
  "script": ["..."],
  "image": ["..."],
  "sound": ["..."],
  "plot": ["..."],
  "pacing": ["..."],
  "character": ["..."],
  "editing": ["..."],
  "music": ["..."],
  "visual_effects": ["..."],
  "costume": ["..."],
  "set_design": ["..."],
  "cinematography": ["..."],
  "dialogue": ["..."],
  "atmosphere": ["..."],
  "themes": ["..."],
  "lighting": ["..."],
  "makeup": ["..."],
  "stunts": ["..."],
  "performances": ["..."],
  "originality": ["..."],
  "adaptation": ["..."],
  "humor": ["..."],
  "emotional_impact": ["..."]
}}

Review: "{review}"
"""

# Define the validation prompt
VALIDATION_PROMPT = """ROLE: You are a validation assistant for aspect-based sentiment extraction in movie reviews.

TASK: Your job is to verify whether all sentiment expressions extracted from a movie review have been correctly classified into their appropriate aspects.
You will be given:
	- The original movie review
	- A JSON object containing extracted sentiment expressions, grouped under 25 film aspects
Your validation criteria:
	1.	Each sentiment expression must appear in the review exactly as written.
	2.	Each expression must be assigned to the correct aspect, based on the meaning of the sentence in which it appears.
	3.	If even one sentiment expression is misclassified, the output is "No".
	4.	If all expressions are correctly assigned, the output is "Yes".

The 25 aspects are:
- actor, director, script, image, sound, plot, pacing, character, editing, music
- visual_effects, costume, set_design, cinematography, dialogue, atmosphere, themes
- lighting, makeup, stunts, performances, originality, adaptation, humor, emotional_impact

OUTPUT FORMAT:
	- Yes: if all expressions are correctly classified
	- No: if any expression is misclassified

EXAMPLE:
Review #1:
The director's choices were bold but sometimes confusing. The actors delivered very powerful performances. The visuals were stunning.
Extracted JSON:
{{
  "director": ["bold", "confusing"],
  "actor": ["very powerful"],
  "image": ["stunning"],
  "script": [],
  "sound": [],
  "plot": [],
  "pacing": [],
  "character": [],
  "editing": [],
  "music": [],
  "visual_effects": [],
  "costume": [],
  "set_design": [],
  "cinematography": [],
  "dialogue": [],
  "atmosphere": [],
  "themes": [],
  "lighting": [],
  "makeup": [],
  "stunts": [],
  "performances": [],
  "originality": [],
  "adaptation": [],
  "humor": [],
  "emotional_impact": []
}}
Output: Yes

Another Example:
Review #2:
The film looked amazing but the main actor felt flat.
Extracted JSON:
{{
  "director": [],
  "actor": ["amazing"],
  "image": ["flat"],
  "script": [],
  "sound": [],
  "plot": [],
  "pacing": [],
  "character": [],
  "editing": [],
  "music": [],
  "visual_effects": [],
  "costume": [],
  "set_design": [],
  "cinematography": [],
  "dialogue": [],
  "atmosphere": [],
  "themes": [],
  "lighting": [],
  "makeup": [],
  "stunts": [],
  "performances": [],
  "originality": [],
  "adaptation": [],
  "humor": [],
  "emotional_impact": []
}}
Output: No

Review: "{review}"
Extracted JSON: {extracted_json}
"""

def extract_sentiments(review_text, review_id):
    """Extract sentiment expressions from a review using LLM"""
    max_retries = len(extraction_keys)
    retries = 0
    
    while retries < max_retries:
        try:
            client = get_extraction_client()
            prompt = EXTRACTION_PROMPT.format(review=review_text)
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'temperature': 0.1,
                }
            )
            
            if response is None or response.text is None:
                print(f"Review {review_id}: Received None response, retrying...")
                time.sleep(1)
                continue
                
            result = response.text.strip()
            
            if not result:
                print(f"Review {review_id}: Empty response received, retrying...")
                time.sleep(1)
                continue
            
            try:
                result_json = json.loads(result)
                return result_json
                
            except json.JSONDecodeError as e:
                print(f"Review {review_id}: JSON parsing failed: {e}")
                print(f"Raw response: '{result[:200]}...'")
                
                # Try to fix truncated JSON
                if result.count('{') > result.count('}'):
                    fixed_result = result + '}'
                    try:
                        result_json = json.loads(fixed_result)
                        print(f"Review {review_id}: Fixed truncated JSON successfully!")
                        return result_json
                    except:
                        pass
                
                # Return empty structure if can't parse
                empty_result = {cat: [] for cat in categories}
                return empty_result
                
        except Exception as e:
            print(f"Review {review_id}: Error with extraction API key #{current_extraction_key_index + 1}: {str(e)[:100]}...")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"Extraction API key #{current_extraction_key_index + 1} quota exceeded, switching...")
                switch_extraction_key()
                retries += 1
                time.sleep(2)
                continue
            elif "RATE_LIMIT_EXCEEDED" in str(e):
                time.sleep(1)
                continue
            else:
                time.sleep(5)
                continue
    
    print("All extraction API keys exhausted!")
    return {cat: [] for cat in categories}

def validate_extraction(review_text, extracted_json, review_id):
    """Validate the extraction using a second LLM"""
    max_retries = len(validation_keys)
    retries = 0
    
    while retries < max_retries:
        try:
            client = get_validation_client()
            prompt = VALIDATION_PROMPT.format(
                review=review_text, 
                extracted_json=json.dumps(extracted_json)
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'temperature': 0.1,
                }
            )
            
            if response is None or response.text is None:
                print(f"Review {review_id}: Validation received None response, retrying...")
                time.sleep(1)
                continue
                
            result = response.text.strip().lower()
            
            if "yes" in result:
                return "Yes"
            elif "no" in result:
                return "No"
            else:
                print(f"Review {review_id}: Unclear validation response: {result}")
                return "Unclear"
                
        except Exception as e:
            print(f"Review {review_id}: Error with validation API key #{current_validation_key_index + 1}: {str(e)[:100]}...")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"Validation API key #{current_validation_key_index + 1} quota exceeded, switching...")
                switch_validation_key()
                retries += 1
                time.sleep(2)
                continue
            elif "RATE_LIMIT_EXCEEDED" in str(e):
                time.sleep(1)
                continue
            else:
                time.sleep(5)
                continue
    
    print("All validation API keys exhausted!")
    return "Error"

def save_extraction_results(results, output_file):
    """Save extraction results to CSV"""
    fieldnames = ['id', 'review'] + categories + ['validation']
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for result in results:
            # Convert lists to pipe-separated strings
            row = {
                'id': result['id'],
                'review': result['review'],
                'validation': result['validation']
            }
            # Add all categories
            for cat in categories:
                row[cat] = '|'.join(result[cat])
            writer.writerow(row)

def process_batch(batch_data):
    """Process a batch of reviews"""
    extraction_results = []
    
    # Step 1: Extract sentiments for all reviews in batch
    print(f"Step 1: Extracting sentiments for batch of {len(batch_data)} reviews...")
    for idx, (_, row) in enumerate(batch_data.iterrows()):
        review_id = row['id']
        review_text = str(row['reviewText'])
        
        print(f"Processing review {review_id}...")
        
        # Extract sentiments
        extracted_json = extract_sentiments(review_text, review_id)
        
        # Store extraction result
        extraction_result = {
            'id': review_id,
            'review': review_text,
        }
        # Add all categories
        for cat in categories:
            extraction_result[cat] = extracted_json[cat]
        
        # Store both the result and the original extracted_json for validation
        extraction_result['_extracted_json'] = extracted_json
        extraction_results.append(extraction_result)
        
        # Small delay to respect rate limits
        time.sleep(0.2)
    
    # Step 2: Validate the entire batch
    print(f"Step 2: Validating entire batch of {len(extraction_results)} reviews...")
    for i, result in enumerate(extraction_results):
        review_id = result['id']
        review_text = result['review']
        
        # Use the stored extracted_json directly
        extracted_json = result['_extracted_json']
        
        print(f"Validating review {review_id}...")
        validation_result = validate_extraction(review_text, extracted_json, review_id)
        
        # Add validation result and remove temporary field
        extraction_results[i]['validation'] = validation_result
        del extraction_results[i]['_extracted_json']
        
        # Small delay for validation API calls
        time.sleep(0.1)
    
    return extraction_results

# File paths
extraction_output_file = 'data/extracted_sentiments_v2.csv'

# Check if files exist and get last processed index
start_idx = 0
if os.path.exists(extraction_output_file):
    try:
        existing_df = pd.read_csv(extraction_output_file)
        start_idx = len(existing_df)
        print(f"Resuming from index {start_idx}")
    except:
        start_idx = 0

# Create files with headers if they don't exist
if not os.path.exists(extraction_output_file) or start_idx == 0:
    with open(extraction_output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'review'] + categories + ['validation'])
        writer.writeheader()

# Process in batches
batch_size = 20  # Smaller batches for better control
df_subset = df.iloc[start_idx:]

print(f"Processing {len(df_subset)} reviews starting from index {start_idx}")
print(f"Each batch will be extracted and validated")

for i in range(0, len(df_subset), batch_size):
    batch_end = min(i + batch_size, len(df_subset))
    batch = df_subset.iloc[i:batch_end]
    
    print(f"Processing batch {i//batch_size + 1}: rows {start_idx + i} to {start_idx + batch_end - 1}")
    
    try:
        # Process batch - extract all reviews first, then validate all
        extraction_results = process_batch(batch)
        
        # Save results immediately
        save_extraction_results(extraction_results, extraction_output_file)
        
        print(f"Completed batch {i//batch_size + 1}/{(len(df_subset) + batch_size - 1)//batch_size}")
        print(f"Extracted and validated {len(extraction_results)} reviews")
        
        # Brief pause between batches
        time.sleep(2)
        
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue

print("Extraction and validation completed!")

# Step 5: Read results, group by film, convert to scores and calculate averages
print("Processing sentiment scores...")

def convert_sentiment_to_score(sentiment_expressions, sentiment_score_df):
    """Convert sentiment expressions to numerical scores"""
    if not sentiment_expressions or sentiment_expressions == "":
        return 0
    
    expressions = sentiment_expressions.split('|') if isinstance(sentiment_expressions, str) else sentiment_expressions
    if not expressions or expressions == ['']:
        return 0
    
    total_score = 0
    valid_expressions = 0
    
    for expr in expressions:
        expr = expr.strip()
        if not expr:
            continue
            
        # Look for the expression in sentiment score database
        matches = sentiment_score_df[sentiment_score_df['CONCEPT'].str.lower() == expr.lower()]
        if not matches.empty:
            score = matches['SENTIMENT_SCORE'].iloc[0]
            total_score += score
            valid_expressions += 1
        else:
            # If not found, assign neutral score
            total_score += 0
            valid_expressions += 1
    
    return total_score / valid_expressions if valid_expressions > 0 else 0

# Load sentiment score database
sentiment_score_df = pd.read_csv('data/senticnet.csv')

# Load extracted sentiments
extracted_sentiments_df = pd.read_csv(extraction_output_file)

# Convert sentiment expressions to scores
for category in categories:
    extracted_sentiments_df[f'{category}_score'] = extracted_sentiments_df[category].apply(
        lambda x: convert_sentiment_to_score(x, sentiment_score_df)
    )

# Group by film ID and calculate averages
score_columns = [f'{cat}_score' for cat in categories]
film_scores = extracted_sentiments_df.groupby('id')[score_columns].mean().reset_index()

# Save final sentiment scores
film_scores.to_csv("data/sentiment_scores_v2.csv", index=False)

print("Sentiment analysis completed!")
print(f"Final sentiment scores saved to data/sentiment_scores_v2.csv")
print(f"Extraction and validation results saved to {extraction_output_file}")
