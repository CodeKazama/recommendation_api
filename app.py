import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from supabase import create_client, Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk

# This line loads your secret keys from a special .env file
load_dotenv()

# --- Initialize Flask App (The core of our API) ---
app = Flask(__name__)

# --- Supabase Connection ---
# !!! IMPORTANT FIX !!!
# The os.environ.get() function needs the NAME of the variable, not the value itself.
# This was the cause of your "Invalid URL" error. I have corrected it.
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# Check if the keys were found
if not url or not key:
    print("🔴 FATAL ERROR: Supabase URL or Key not found.")
    print("🔴 Make sure you have set them in your .env file (for local) or in Render's Environment Variables (for deployment).")
    exit() # Stop the app if it can't connect to the database

# Create the connection to your Supabase database
supabase: Client = create_client(url, key)
print("✅ Successfully configured Supabase client.")

# Download a small helper file for NLTK (our text processing library)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' for NLTK...")
    nltk.download('punkt')
    print("Download complete.")

# --- The Recommendation Logic ---
def build_model_and_recommend(all_users_data, target_user_id):
    """
    This function does all the heavy lifting.
    It takes all user profiles and the ID of the user asking for a recommendation.
    """
    print("Building recommendation model in memory...")
    
    # 1. Convert the list of users from Supabase into a pandas DataFrame
    df = pd.DataFrame(all_users_data)
    
    # 2. Data Cleaning and Preparation
    required_columns = ['id', 'name', 'sex', 'age', 'goal', 'deadline', 'academic_qualification']
    for col in required_columns:
        if col not in df.columns:
            df[col] = '' 
    
    df.fillna('', inplace=True)
    df['age'] = df['age'].astype(str)

    # 3. Create the "tags" 
    df['tags'] = (
        df['sex'].str.lower() + ' ' +
        df['age'] + ' ' +
        df['goal'].str.lower() + ' ' +
        df['deadline'].str.replace(" ", "").str.lower() + ' ' +
        df['academic_qualification'].str.replace(" ", "").str.lower()
    )
    print("Tags created for all users.")

    # 4. Text Processing (Stemming)
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])
    df['tags'] = df['tags'].apply(stem)

    # 5. Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # 6. Similarity Calculation
    similarity = cosine_similarity(vectors)
    
    # 7. Find the Best Match
    try:
        user_index = df[df['id'] == target_user_id].index[0]
    except IndexError:
        print(f"Warning: User {target_user_id} not found in the fetched data.")
        return None 

    distances = similarity[user_index]
    similar_students_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    
    recommended_user_data = None
    for i, score in similar_students_list:
        if df.iloc[i]['id'] != target_user_id:
            recommended_user_data = df.iloc[i].to_dict()
            print(f"Found best match: {recommended_user_data['name']} with score {score}")
            break
            
    return recommended_user_data

# --- API Endpoint Definition ---
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    # ... (The rest of this function is unchanged)
    print("\n--- New Request Received at /recommend ---")
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'Request is missing the user_id field.'}), 400
    
    user_id = data['user_id']
    print(f"Request for user: {user_id}")

    try:
        response = supabase.table('profiles').select('*').execute()
        if not response.data or len(response.data) < 2:
            return jsonify({'message': 'Not enough users in the database to make a recommendation.'}), 404

        all_users = response.data
        print(f"Fetched {len(all_users)} users from Supabase.")
        
        recommendation = build_model_and_recommend(all_users, user_id)
        
        if recommendation:
            print(f"Returning recommendation: {recommendation.get('name')}")
            return jsonify(recommendation)
        else:
            return jsonify({'message': 'No suitable recommendation found.'}), 404

    except Exception as e:
        print(f"🚨 An unexpected error occurred: {e}")
        return jsonify({'error': f'An internal server error occurred on the server.'}), 500

# A simple "health check" to see if the server is running
@app.route('/', methods=['GET'])
def health_check():
    return "Checkmate Recommendation API is running!"

# --- Main execution block ---
if __name__ == '__main__':
    # Get the port number from the environment variable provided by Render.
    # If it's not found (i.e., we're running locally), it defaults to 4000.
    port = int(os.environ.get('PORT', 4000))
    
    # The host '0.0.0.0' makes the server publicly accessible.
    app.run(host='0.0.0.0', port=port)
    print(f"Example app listening on port {port}")


