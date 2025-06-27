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
# These lines read your secret Supabase URL and Key from the environment.
# This is a secure way to handle secrets, never write them directly in the code.
url = os.environ.get("https://wsxnzuiinwzsmccsrovn.supabase.co")
key = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndzeG56dWlpbnd6c21jY3Nyb3ZuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MDUyNzMyNCwiZXhwIjoyMDY2MTAzMzI0fQ.4eQcHCMptP1YPl5OIGjte1mIDimrRZ-XR-w3Tc6b00s")

# Check if the keys were found
if not url or not key:
    print("🔴 FATAL ERROR: Supabase URL or Key not found.")
    print("🔴 Make sure you have a .env file with SUPABASE_URL and SUPABASE_KEY.")
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
    # Ensure required columns exist to prevent errors. Fill any missing values with an empty string.
    required_columns = ['id', 'name', 'sex', 'age', 'goal', 'deadline', 'academic_qualification']
    for col in required_columns:
        if col not in df.columns:
            df[col] = '' 
    
    df.fillna('', inplace=True)
    df['age'] = df['age'].astype(str)

    # 3. Create the "tags" - This is the core of our model
    # We combine all the important profile features into a single string for each user.
    df['tags'] = (
        df['sex'].str.lower() + ' ' +
        df['age'] + ' ' +
        df['goal'].str.lower() + ' ' +
        df['deadline'].str.replace(" ", "").str.lower() + ' ' +
        df['academic_qualification'].str.replace(" ", "").str.lower()
    )
    print("Tags created for all users.")

    # 4. Text Processing (Stemming)
    # This simplifies words to their root (e.g., "coding" -> "code").
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])
    df['tags'] = df['tags'].apply(stem)

    # 5. Vectorization
    # We convert the text tags into numerical vectors that a machine can understand.
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # 6. Similarity Calculation
    # We calculate how similar each user is to every other user.
    similarity = cosine_similarity(vectors)
    
    # 7. Find the Best Match
    try:
        # Get the row number (index) of our target user
        user_index = df[df['id'] == target_user_id].index[0]
    except IndexError:
        print(f"Warning: User {target_user_id} not found in the fetched data.")
        return None 

    # Get the similarity scores for our user against everyone else
    distances = similarity[user_index]
    
    # Sort the users by similarity, from highest to lowest.
    # We take the item at index 1 because index 0 will be the user themselves (with a perfect score of 1.0).
    similar_students_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    
    # Find the first user in the sorted list that is not the user themselves.
    recommended_user_data = None
    for i, score in similar_students_list:
        if df.iloc[i]['id'] != target_user_id:
            recommended_user_data = df.iloc[i].to_dict()
            print(f"Found best match: {recommended_user_data['name']} with score {score}")
            break
            
    return recommended_user_data

# --- API Endpoint Definition ---
# This is the public "door" to our API. The Flutter app will knock on this door.
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    print("\n--- New Request Received at /recommend ---")
    # 1. Get the user's ID from the message sent by Flutter
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'Request is missing the user_id field.'}), 400
    
    user_id = data['user_id']
    print(f"Request for user: {user_id}")

    try:
        # 2. Fetch ALL user profiles from the Supabase 'profiles' table
        response = supabase.table('profiles').select('*').execute()
        
        if not response.data or len(response.data) < 2:
            return jsonify({'message': 'Not enough users in the database to make a recommendation.'}), 404

        all_users = response.data
        print(f"Fetched {len(all_users)} users from Supabase.")
        
        # 3. Call our main function to do the work
        recommendation = build_model_and_recommend(all_users, user_id)
        
        if recommendation:
            print(f"Returning recommendation: {recommendation.get('name')}")
            # Send the recommended user's full profile back to the Flutter app
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

if __name__ == '__main__':
    # This makes the API accessible on your network
    app.run(host='0.0.0.0', port=5000)
