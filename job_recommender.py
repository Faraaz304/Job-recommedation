from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import re
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os 
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["Placement_Management_System"]
collection = db["jobs"]

def parse_resume(resume_file):
    """
    Parse the resume to extract skills, experience, and location.

    :param resume_file: The uploaded PDF resume file.
    :return: Dictionary with extracted details: 'skills', 'experience', and 'location'.
    """
    # Extract text from the resume PDF using PyPDF2
    try:
        reader = PdfReader(resume_file)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Replace or remove problematic characters
                text = text.encode('ascii', 'ignore').decode('ascii')
                resume_text += text
    except Exception as e:
        raise ValueError(f"Error reading the resume file: {e}")

    # Initialize result dictionary
    resume_details = {
        'skills': [],
        'experience': 0,
        'location': ''
    }

    # Extended list of predefined skills
    predefined_skills = [
        'python', 'java', 'machine learning', 'data analysis', 'project management',
        'sql', 'cloud computing', 'communication', 'teamwork', 'problem-solving',
        'c++', 'javascript', 'html', 'css', 'deep learning', 'artificial intelligence',
        'data science', 'devops', 'aws', 'azure', 'docker', 'kubernetes', 'node.js', 'r',
        'tensorflow', 'pytorch', 'big data', 'biotechnology', 'digital marketing',
        'leadership', 'agile', 'scrum', 'git', 'linux', 'flutter', 'react', 'vue.js', 'firebase'
    ]
    found_skills = [skill for skill in predefined_skills if skill.lower() in resume_text.lower()]
    resume_details['skills'] = found_skills

    # Extract years of experience
    experience_match = re.search(r'(\d+)\s+years?\s+of\s+experience', resume_text.lower())
    if experience_match:
        resume_details['experience'] = int(experience_match.group(1))

    # Extract location (refined pattern to avoid irrelevant matches)
    location_match = re.search(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s+[A-Z]{2})?)\b', 
        resume_text
    )
    # Check against an extended predefined list of valid location names
    valid_locations = [
        "New York", "San Francisco", "Chicago", "Bangalore", "London", "Toronto",
        "Los Angeles", "Paris", "Berlin", "Sydney", "Tokyo", "Seattle", "Austin",
        "Vancouver", "Boston", "Dallas", "Mumbai", "Delhi", "Singapore", "Hong Kong", "Cape Town"
    ]
    if location_match and location_match.group(1) in valid_locations:
        resume_details['location'] = location_match.group(1)
    else:
        resume_details['location'] = "Unknown"  # Default location if no valid match found

    # Handle missing fields gracefully
    if not resume_details['skills']:
        resume_details['skills'] = ['N/A']  # Placeholder if no skills are found
    if not resume_details['experience']:
        resume_details['experience'] = 0  # Default experience to 0 if not mentioned

    return resume_details

def train_similarity_model():
    """
    Train a simple neural network model to combine weighted similarity scores.
    Returns the trained model.
    """
    model = Sequential([
        Dense(8, input_dim=3, activation='relu'),  # 3 inputs: skills, location, experience
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')  # Output a single score between 0-1
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Example training data for weights tuning (skills_similarity, location_match, experience_similarity)
    X_train = np.array([
        [0.9, 1, 0.8],  # High match across all dimensions
        [0.7, 1, 0.6],  # Good skills and location, decent experience
        [0.8, 0, 0.7],  # Good skills and experience, location mismatch
        [0.4, 0, 0.5],  # Medium-low match
        [0.1, 1, 0.2],  # Location match but poor skills/experience
        [0.2, 0, 0.1],  # Low match across all dimensions
    ])
    # Expected scores (higher weight on skills, then experience, then location)
    y_train = np.array([0.95, 0.80, 0.70, 0.35, 0.30, 0.15])
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

# At the beginning of the file, add:
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Then modify the recommend_jobs_from_resume function's return section:
def recommend_jobs_from_resume(resume_file, top_n=5):
    """
    Recommend jobs based on a user's uploaded resume.
    
    :param resume_file: Path to the uploaded resume PDF file.
    :param top_n: Number of top recommendations to return.
    :return: List of recommended jobs with scores.
    """
    # Parse the resume
    parsed_resume = parse_resume(resume_file)
    resume_skills = " ".join(parsed_resume['skills'])
    user_location = parsed_resume['location']
    user_experience = parsed_resume['experience']
    
    print(f"Extracted from resume - Skills: {parsed_resume['skills']}, Location: {user_location}, Experience: {user_experience} years")

    # Fetch all jobs from MongoDB
    try:
        jobs = list(collection.find())
        if not jobs:
            raise ValueError("No jobs available in the database.")
        print(f"Found {len(jobs)} jobs in the database")
        
        # Debug: Print first job to see its structure
        if jobs:
            print(f"Sample job structure: {list(jobs[0].keys())}")
    except Exception as e:
        raise ConnectionError(f"Error connecting to MongoDB: {e}")

    # Extract job information
    job_skills = []
    job_locations = []
    job_experience = []
    
    for job in jobs:
        # Debug job structure if needed
        # print(f"Processing job: {job.get('title', 'Unknown')} - Keys: {list(job.keys())}")
        
        # Handle skills - check both 'required_skills' and 'skills' fields
        if 'required_skills' in job and job['required_skills']:
            if isinstance(job['required_skills'], list):
                job_skills.append(" ".join(job['required_skills']))
            else:
                job_skills.append(str(job['required_skills']))
        elif 'skills' in job and job['skills']:
            if isinstance(job['skills'], list):
                job_skills.append(" ".join(job['skills']))
            else:
                job_skills.append(str(job['skills']))
        else:
            job_skills.append("")
            
        # Handle locations
        job_locations.append(job.get('location', ""))
        
        # Default all jobs to 0 experience requirement since this field might not exist
        job_experience.append(0)

    # Skills Similarity (TF-IDF + Cosine Similarity)
    if resume_skills.strip():
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            # Add all skills to the corpus to ensure better matching
            all_skills = [resume_skills] + job_skills
            tfidf_matrix = vectorizer.fit_transform(all_skills)
            skills_similarity = np.array(tfidf_matrix[0:1].dot(tfidf_matrix[1:].T).toarray()).flatten()
            
            # Debug: Print some similarity scores
            print(f"Skills similarity range: {min(skills_similarity) if len(skills_similarity) > 0 else 'N/A'} to {max(skills_similarity) if len(skills_similarity) > 0 else 'N/A'}")
        except Exception as e:
            print(f"Error in skills similarity calculation: {e}")
            # Fallback if vectorization fails
            skills_similarity = np.zeros(len(jobs))
    else:
        skills_similarity = np.zeros(len(jobs))

    # Location Similarity (Binary Match with fallback)
    location_similarity = np.array([
        1 if (user_location.lower() in job_location.lower() or job_location.lower() in user_location.lower()) else 0 
        for job_location in job_locations
    ])

    # Use a constant experience similarity since we don't have reliable experience data
    experience_similarity = np.ones(len(jobs))

    # Combine scores using the neural network model
    similarity_model = train_similarity_model()
    
    # Ensure all arrays have the same length
    min_length = min(len(skills_similarity), len(location_similarity), len(experience_similarity), len(jobs))
    skills_similarity = skills_similarity[:min_length]
    location_similarity = location_similarity[:min_length]
    experience_similarity = experience_similarity[:min_length]
    jobs_subset = jobs[:min_length]
    
    # Stack features and predict scores
    input_features = np.stack([skills_similarity, location_similarity, experience_similarity], axis=1)
    scores = similarity_model.predict(input_features).flatten()

    # Rank Jobs by Score
    ranked_jobs = sorted(
        zip(jobs_subset, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return Top N Recommendations with more details
    recommendations = []
    for job, score in ranked_jobs[:top_n]:
        try:
            # Clean and encode strings before adding to recommendations
            job_title = str(job.get("title", "Unknown Title")).encode('utf-8', 'ignore').decode('utf-8')
            company_name = str(job.get("company_name", "Unknown Company")).encode('utf-8', 'ignore').decode('utf-8')
            job_location = str(job.get("location", "Unknown Location")).encode('utf-8', 'ignore').decode('utf-8')
            job_skills = job.get("required_skills", [])
            
            # Clean skills list
            if isinstance(job_skills, list):
                job_skills = [str(skill).encode('utf-8', 'ignore').decode('utf-8') for skill in job_skills]
            else:
                job_skills = []
            
            recommendations.append({
                "job_id": str(job.get("_id", "")),
                "title": job_title,
                "company": company_name,
                "location": job_location,
                "skills": job_skills,
                "score": round(float(score), 2)
            })
        except Exception as e:
            print(f"Warning: Skipping job due to encoding error: {str(e)}")
            continue

    return recommendations

# Modify the main section:
if __name__ == "__main__":
    resume_path = "D:\\Faraaz\\Flask projects\\job reccomedation\\faraaz.pdf"
    
    try:
        recommendations = recommend_jobs_from_resume(resume_path, top_n=10)
        print("\nTop Job Recommendations:")
        for i, job in enumerate(recommendations, 1):
            print(f"\n{i}. Job Details:")
            print(f"   Title: {job['title']}")
            print(f"   Company: {job['company']}")
            print(f"   Location: {job['location']}")
            print(f"   Skills Required: {', '.join(job['skills'])}")
            print(f"   Match Score: {job['score']}")
            print("   " + "-"*50)
    except Exception as e:
        print(f"Error in job recommendation: {str(e)}")
        import traceback
        print(traceback.format_exc())