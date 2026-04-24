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
cities = ["Bangalore, IN", "Mumbai, IN", "Delhi, IN", "Hyderabad, IN", "Pune, IN", "Chennai, IN", "Ahmedabad, IN", "Gurgaon, IN", "Remote, IN"]

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

# NEW PROCEDURAL FAKE GENERATOR based on specific Red Flags
fake_keywords = {
    "money": ["earn $500–$5000 per week from home", "no investment required", "small registration fee", "work from home and earn daily", "get paid to your PayPal/crypto wallet", "weekly cash payments", "earn while you sleep", "passive income opportunity", "financial freedom", "unlimited earning potential", "be your own boss", "6-figure income", "money-back guarantee", "Upto 10,000/Month Please hit like or comment Interested so", "earn ₹50,000/month seamlessly"],
    "urgency": ["limited seats available", "apply before friday or miss out", "urgent hiring", "immediate joiners only", "only 5 spots left", "don't miss this opportunity", "act now", "hiring closes tonight", "first come first served"],
    "vague": ["data entry work", "online work from home", "part-time easy work", "simple tasks", "no experience required", "freshers welcome – earn ₹50,000/month", "work just 2 hours a day", "miscellaneous tasks", "ad posting job", "typing job", "reselling job", "multiple positions in weintern it services"],
    "contact": ["contact us on WhatsApp only", "DM for details", "reach us on Telegram", "interview on Google Hangouts/Telegram", "no formal interview required", "you're selected, just pay the fee", "HR will contact you on WhatsApp"],
    "fees": ["registration fee", "training fee", "security deposit", "refundable deposit", "kit fee", "starter kit", "verification fee", "ID card charges", "uniform fee", "background check fee", "joining fee"]
}

fake_companies = ["XYZ Pvt Ltd", "ABC Solutions", "Google Recruitment Cell", "We IT Services", "Amazon Work From Home Team", "Global Data Solutions", "Digital Earning Pvt Ltd", "Fast Cash Solutions"]

def generate_procedural_fake_job():
    desc_points = [
        random.choice(fake_keywords["vague"]),
        random.choice(fake_keywords["money"]),
        random.choice(fake_keywords["urgency"]),
        random.choice(fake_keywords["fees"]) + " asked upfront to secure spot." if random.random() > 0.4 else "",
        random.choice(fake_keywords["contact"])
    ]
    random.shuffle(desc_points)
    desc = " ".join([d for d in desc_points if d])
    
    title = random.choice([
        "Data Entry Operator", "Online Typist Needed", "Work From Home Exec",
        "Part Time Earner", "Ad Poster", "Remote Consultant", "Copy Paste Job",
        "Virtual Assistant", "Multiple Positions in IT Services"
    ])
    
    return {
        "title": title,
        "company": random.choice(fake_companies),
        "desc": desc,
        "req": random.choice(["Must have smartphone and internet.", "No experience required.", "Undergraduates welcome.", "Can work from mobile."]),
        "department": "Operations",
        "industry": "Freelance"
    }

def generate_data(n=2000):
    rows = []
    base_id = 90000  # Starting ID far from typical range
    
    for i in range(n):
        # We will heavily oversample FAKE jobs to penalize these words
        is_fraud = 1 if random.random() > 0.2 else 0 
        
        if is_fraud:
            template = generate_procedural_fake_job()
        else:
            template = random.choice(real_job_templates)
            
        row = {
            "job_id": base_id + i,
            "title": template["title"],
            "location": random.choice(cities),
            "department": template["department"],
            "salary_range": "300000-800000" if not is_fraud else "50000-100000",
            "company_profile": f"Welcome to {template['company']}." if not is_fraud else f"Company: {template['company']}.",
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
    print("Generating heavily targeted simulated scam data...")
    new_data = generate_data(3000)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_file = os.path.join(BASE_DIR, "data", "fake_job_postings.csv")
    if os.path.exists(csv_file):
        print(f"Reading existing data from {csv_file}...")
        existing_df = pd.read_csv(csv_file)
        
        # Optionally, remove previously generated synthetic data to prevent ID overflow 
        # but Pandas 'concat' is very reliable and memory-efficient compared to just ID conflicts.
        # We'll just concatenate.
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        combined_df.to_csv(csv_file, index=False)
        print(f"Success! Augmented data SAVED to '{csv_file}' for ML consumption.")
        print(f"New total rows: {len(combined_df)}")
    else:
        print(f"Error: {csv_file} not found. Saving new data to 'backend/data/fake_job_postings.csv'")
        new_data.to_csv(csv_file, index=False)
