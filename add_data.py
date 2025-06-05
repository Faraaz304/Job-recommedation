from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection string from environment variables
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")

# Connect to MongoDB
client = MongoClient(MONGODB_CONNECTION_STRING)

# Specify the database and collection
db = client["Placement_Management_System"]
collection = db["jobs"]

# Load job postings from the JSON file
try:
    with open("job_postings.json", "r") as file:
        job_postings = json.load(file)
    
    # Clear existing data in the collection
    collection.delete_many({})
    print("Existing data cleared from the collection.")
    
    # Transform the data to match your MongoDB schema if needed
    transformed_jobs = []
    for job in job_postings:
        transformed_job = {
            "_id": job["job_id"],
            "title": job["job_title"],
            "required_skills": job["requirements"].split(", "),
            "location": job["location"],
            "description": job["description"],
            "salary_range": job["salary_range"].replace("$", ""),
            "company_name": job["company_name"],
            "date_posted": job["posted_date"],
            "employment_type": job["job_type"]
        }
        transformed_jobs.append(transformed_job)
    
    # Insert the new data into the collection
    collection.insert_many(transformed_jobs)
    print(f"Successfully imported {len(transformed_jobs)} jobs from job_postings.json to MongoDB.")
    
except FileNotFoundError:
    print("Error: job_postings.json file not found.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format in job_postings.json.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the MongoDB connection
    client.close()



