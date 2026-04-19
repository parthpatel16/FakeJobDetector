import pandas as pd
import random
import os

# Define the columns (matching the original CSV)
columns = [
    "job_id", "title", "location", "department", "salary_range", "company_profile",
    "description", "requirements", "benefits", "telecommuting", "has_company_logo",
    "has_questions", "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent"
]

# Indian Cities
cities = ["Bangalore, IN", "Mumbai, IN", "Delhi, IN", "Hyderabad, IN", "Pune, IN", "Chennai, IN", "Ahmedabad, IN", "Gurgaon, IN"]

# Real Indian Job Templates
real_job_templates = [
    {
        "title": "Software Engineer",
        "company": "TCS (Tata Consultancy Services)",
        "desc": "Looking for a Software Engineer with experience in Java, Spring Boot, and Microservices. Join our digital transformation team in India.",
        "req": "Bachelor's degree in CS/IT, 2+ years of experience in backend development.",
        "department": "IT Development",
        "industry": "IT Services"
    },
    {
        "title": "Front-End Developer",
        "company": "Infosys Limited",
        "desc": "Join our UI/UX team to build responsive web applications using React.js and modern JavaScript frameworks.",
        "req": "Proficiency in HTML5, CSS3, React.js, and Redux. Good communication skills.",
        "department": "Engineering",
        "industry": "Engineering"
    },
    {
        "title": "Business Analyst",
        "company": "Zomato",
        "desc": "Help us analyze market trends and optimize our food delivery operations across India.",
        "req": "MBA or equivalent, strong analytical skills, experience with SQL and Excel.",
        "department": "Business Ops",
        "industry": "Food Tech"
    },
    {
        "title": "Full Stack Developer",
        "company": "Wipro",
        "desc": "We are hiring Full Stack Developers with expertise in Node.js and Angular for our cloud-native projects.",
        "req": "Graduate in Engineering, 3-5 years of experience in JavaScript frameworks.",
        "department": "Digital",
        "industry": "Consulting"
    }
]

# Fake Indian Job Templates
fake_job_templates = [
    {
        "title": "Work From Home Data Entry",
        "company": "Fast Cash Solutions",
        "desc": "Earn 5000-10000 INR daily just by typing simple data. No experience needed. Immediate joining. Work from mobile or laptop.",
        "req": "Basic typing skills, smartphone. Interested candidates contact on WhatsApp +91-XXXXX-XXXXX.",
        "department": "Back Office",
        "industry": "Data Entry"
    },
    {
        "title": "Part-Time Captcha Entry",
        "company": "Digital Earning Pvt Ltd",
        "desc": "We are looking for part-time workers for captcha entry. 2000 INR per day guaranteed. Registration fee of 500 INR is required for training.",
        "req": "Internet connection, computer/mobile. Pay security deposit to start working today.",
        "department": "Operations",
        "industry": "Freelance"
    },
    {
        "title": "Online SMS Sending Job",
        "company": "Global SMS Marketing",
        "desc": "Send 100 SMS daily and earn 30000 INR monthly. Genuine work from home. No interview. Apply now on Telegram @job_india.",
        "req": "Smartphone, knowledge of Hindi/English. Must share bank details for salary deposit.",
        "department": "Marketing",
        "industry": "Marketing"
    },
    {
        "title": "Urgent Requirement - Amazon India (Work from Home)",
        "company": "Amazon Recruitment Services (Fake)",
        "desc": "Amazon is hiring for remote customer support. Salary 40000 INR. To apply, pay background verification charges of 1500 INR.",
        "req": "12th pass or Graduate. Good voice quality. Must have PAN card and Aadhaar card.",
        "department": "Customer Service",
        "industry": "Retail"
    }
]

def generate_data(n=200):
    rows = []
    base_id = 20000  # Starting ID far from typical range
    
    for i in range(n):
        is_fraud = random.choice([0, 1])
        template = random.choice(fake_job_templates if is_fraud else real_job_templates)
        
        row = {
            "job_id": base_id + i,
            "title": template["title"],
            "location": random.choice(cities),
            "department": template["department"],
            "salary_range": "300000-800000" if not is_fraud else "50000-100000",
            "company_profile": f"Welcome to {template['company']}. We are leaders in our field.",
            "description": template["desc"],
            "requirements": template["req"],
            "benefits": "Standard industry benefits." if not is_fraud else "High bonuses and daily pay.",
            "telecommuting": 1 if "Home" in template["title"] or "remote" in template["desc"].lower() else 0,
            "has_company_logo": 1 if not is_fraud else 0,
            "has_questions": 1 if not is_fraud else 0,
            "employment_type": "Full-time" if not is_fraud else "Part-time",
            "required_experience": "Entry level" if is_fraud else "Mid-Senior level",
            "required_education": "Bachelor's Degree" if not is_fraud else "Unspecified",
            "industry": template["industry"],
            "function": template["department"],
            "fraudulent": is_fraud
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Generating Indian job data...")
    new_data = generate_data(300)
    
    csv_file = "fake_job_postings.csv"
    if os.path.exists(csv_file):
        print(f"Reading existing data from {csv_file}...")
        # Note: Original CSV might have quote issues, but pandas usually handles it.
        # However, to be safe and avoid memory issues on huge files, we just append.
        existing_df = pd.read_csv(csv_file)
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        combined_df.to_csv("fake_job_postings_augmented.csv", index=False)
        print(f"Success! Augmented data saved to 'fake_job_postings_augmented.csv'")
        print(f"New total rows: {len(combined_df)}")
    else:
        print(f"Error: {csv_file} not found. Saving only new data to 'indian_jobs.csv'")
        new_data.to_csv("indian_jobs.csv", index=False)
