from faker import Faker
import random
import json

# Initialize Faker
fake = Faker()

# List of industries and skills for randomization
industries = ["Tech", "Healthcare", "Finance", "Education", "Retail"]
skills = ["Python", "Java", "SQL", "Machine Learning", "React", "AWS", "Project Management"]

# Function to generate a fake job posting
def generate_job_posting():
    return {
        "job_id": fake.uuid4(),
        "job_title": fake.job(),
        "company_name": fake.company(),
        "location": fake.city(),
        "industry": random.choice(industries),
        "description": fake.paragraph(nb_sentences=5),
        "requirements": ", ".join(random.sample(skills, k=random.randint(3, 5))),
        "salary_range": f"${random.randint(50000, 150000)} - ${random.randint(150000, 250000)}",
        "job_type": random.choice(["Full-time", "Part-time", "Contract", "Internship"]),
        "posted_date": fake.date_this_year().strftime("%Y-%m-%d")  # Convert datetime to string
    }

# Function to generate a fake user profile
def generate_user_profile():
    return {
        "user_id": fake.uuid4(),
        "name": fake.name(),
        "email": fake.email(),
        "location": fake.city(),
        "education": {
            "degree": random.choice(["B.Tech", "MBA", "MS", "PhD"]),
            "field": random.choice(["Computer Science", "Business", "Data Science", "Engineering"]),
            "institution": fake.company()
        },
        "work_experience": [
            {
                "job_title": fake.job(),
                "company": fake.company(),
                "years": random.randint(1, 10)
            }
            for _ in range(random.randint(1, 3))  # 1-3 past jobs
        ],
        "skills": ", ".join(random.sample(skills, k=random.randint(3, 6))),
        "preferred_job_type": random.choice(["Full-time", "Part-time", "Remote"]),
        "preferred_location": fake.city()
    }

# Function to generate a fake skills dataset
def generate_skills_dataset():
    return {
        "skill_id": fake.uuid4(),
        "skill_name": random.choice(skills),
        "related_job_titles": [fake.job() for _ in range(random.randint(3, 5))]
    }

# Generate fake data
job_postings = [generate_job_posting() for _ in range(100)]  # 100 job postings
user_profiles = [generate_user_profile() for _ in range(50)]  # 50 user profiles
skills_dataset = [generate_skills_dataset() for _ in range(20)]  # 20 skills

# Save data to JSON files
with open("job_postings.json", "w") as f:
    json.dump(job_postings, f, indent=4)

with open("user_profiles.json", "w") as f:
    json.dump(user_profiles, f, indent=4)

with open("skills_dataset.json", "w") as f:
    json.dump(skills_dataset, f, indent=4)

print("Fake data generated and saved to JSON files!")
# Function to generate a fake job posting
def generate_job_posting():
    return {
        "job_id": fake.uuid4(),
        "job_title": fake.job(),
        "company_name": fake.company(),
        "location": fake.city(),
        "industry": random.choice(industries),
        "description": fake.paragraph(nb_sentences=5),
        "requirements": ", ".join(random.sample(skills, k=random.randint(3, 5))),
        "salary_range": f"${random.randint(50000, 150000)} - ${random.randint(150000, 250000)}",
        "job_type": random.choice(["Full-time", "Part-time", "Contract", "Internship"]),
        "posted_date": fake.date_this_year().strftime("%Y-%m-%d")  # Convert datetime to string
    }