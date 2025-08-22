import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy.stats import beta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of students
N = 100

# ------- 1. Basic Student Demographics & A/L Metrics -------
districts = [
    'Colombo', 'Gampaha', 'Kurunegala', 'Kandy', 'Kalutara', 'Ratnapura', 'Galle',
    'Anuradhapura', 'Kegalle', 'Badulla', 'Matara', 'Puttalam', 'Nuwara Eliya',
    'Ampara', 'Hambantota', 'Jaffna', 'Batticaloa', 'Matale', 'Monaragala',
    'Polonnaruwa', 'Trincomalee', 'Vavuniya', 'Kilinochchi', 'Mannar', 'Mullaitivu'
]
populations = [
    2460, 2421, 1727, 1482, 1279, 1188, 1139, 950, 892, 896, 869, 845, 781, 749,
    680, 628, 582, 525, 509, 445, 443, 196, 136, 116, 99
]

district_to_category = {
    'Colombo': 'Close', 'Gampaha': 'Close', 'Kalutara': 'Close',
    'Galle': 'Medium', 'Matara': 'Medium', 'Hambantota': 'Medium',
    'Kandy': 'Medium', 'Matale': 'Medium', 'Nuwara Eliya': 'Medium',
    'Batticaloa': 'Far', 'Ampara': 'Far', 'Trincomalee': 'Far',
    'Jaffna': 'Far', 'Kilinochchi': 'Far', 'Mannar': 'Far',
    'Vavuniya': 'Far', 'Mullaitivu': 'Far',
    'Kurunegala': 'Medium', 'Puttalam': 'Medium',
    'Anuradhapura': 'Far', 'Polonnaruwa': 'Far',
    'Ratnapura': 'Medium', 'Kegalle': 'Medium',
    'Badulla': 'Far', 'Monaragala': 'Far'
}

category_to_factor = {'Close': 1.0, 'Medium': 0.5, 'Far': 0.2}

adjusted_populations = [populations[i] * category_to_factor[district_to_category[districts[i]]] for i in range(len(districts))]
total_adj_pop = sum(adjusted_populations)
district_weights = [adj_pop / total_adj_pop for adj_pop in adjusted_populations]

sampled_districts = np.random.choice(districts, size=N, p=district_weights)

district_to_province = {
    'Colombo': 'Western', 'Gampaha': 'Western', 'Kalutara': 'Western',
    'Galle': 'Southern', 'Matara': 'Southern', 'Hambantota': 'Southern',
    'Kandy': 'Central', 'Matale': 'Central', 'Nuwara Eliya': 'Central',
    'Batticaloa': 'Eastern', 'Ampara': 'Eastern', 'Trincomalee': 'Eastern',
    'Jaffna': 'Northern', 'Vavuniya': 'Northern', 'Kilinochchi': 'Northern',
    'Mannar': 'Northern', 'Mullaitivu': 'Northern',
    'Kurunegala': 'North Western', 'Puttalam': 'North Western',
    'Anuradhapura': 'North Central', 'Polonnaruwa': 'North Central',
    'Ratnapura': 'Sabaragamuwa', 'Kegalle': 'Sabaragamuwa',
    'Badulla': 'Uva', 'Monaragala': 'Uva'
}
sampled_provinces = [district_to_province[dist] for dist in sampled_districts]

genders = np.random.choice(['Male', 'Female'], N, p=[0.65, 0.35])
ages = np.random.choice([19, 20, 21, 22], size=N, p=[0.4, 0.4, 0.15, 0.05])
z_scores = np.round(1.45 + 0.55 * beta.rvs(a=2, b=9, size=N), 2)
intake_years = np.random.choice([2020, 2021, 2022, 2023, 2024], size=N)
student_ids = np.arange(1, N + 1)

students_df = pd.DataFrame({
    'student_id': student_ids,
    'gender': genders,
    'district': sampled_districts,
    'province': sampled_provinces,
    'age_at_enrollment': ages,
    'z_score_AL': z_scores,
    'intake_year': intake_years
})

current_year = 2025
students_df['years_since_intake'] = current_year - students_df['intake_year']
students_df['is_graduated'] = students_df['years_since_intake'] >= 4
students_df['current_year_of_study'] = np.where(students_df['is_graduated'], np.nan, students_df['years_since_intake'] + 1)

pathways = ["Cyber Security", "Data Science", "Artificial Intelligence", "Scientific Computing", "Standard"]
students_df['pathway'] = np.random.choice(pathways, size=N)

pathway_to_prefix = {
    "Cyber Security": "CSEC",
    "Data Science": "DSCI",
    "Artificial Intelligence": "AINT",
    "Scientific Computing": "SCOM",
    "Standard": "CSCI"
}

# ------- 2. Curriculum with Compulsory/Optional Flags -------
curriculum = {
    "Year1_Sem1": [
        ("CSCI 11014", True), ("CSCI 11023", True), ("CSCI 11032", True),
        ("CSCI 11042", True), ("CSCI 11052", True), ("CSCI 11062", True),
        ("CSCI 11072", False), ("DELT 13302", True)
    ],
    "Year1_Sem2": [
        ("CSCI 12013", True), ("CSCI 12022", True), ("CSCI 12033", True),
        ("CSCI 12042", True), ("CSCI 12052", True), ("CSCI 12063", True)
    ],
    "Year2_Sem1": [
        ("CSCI 21013", True), ("CSCI 21023", True), ("CSCI 21033", True),
        ("CSCI 21042", True), ("CSCI 21052", True), ("CSCI 21062", True),
        ("CSCI 23072", True), ("DELT 21212", True), ("MGMT 21012", True)
    ],
    "Year2_Sem2": [
        ("CSCI 22012", True), ("CSCI 22022", True), ("CSCI 22032", True),
        ("CSCI 22042", True), ("CSCI 22052", True), ("CSCI 22062", True),
        ("CSCI 22072", False), ("CSCI 22082", True), ("MGMT 22012", True)
    ],
    "Year3_Sem1": [
        ("CSCI 31014", True), ("DELT 33212", True), ("MGMT 31012", True),
        ("CSCI 31022", False), ("CSCI 31032", False), ("CSCI 31042", False),
        ("CSCI 31052", False), ("CSCI 31062", False), ("CSCI 31072", False),
        ("CSCI 31082", True), ("CSEC 31012", True), ("CSEC 31022", True),
        ("AINT 31012", False), ("AINT 31022", False),
        ("SCOM 31013", False), ("SCOM 31022", False), ("SCOM 31032", False)
    ],
    "Year3_Sem2": [
        ("CSCI 32012", True), ("CSCI 32022", True), ("CSCI 32032", True),
        ("CSCI 32042", True), ("DELT 33212", True), ("MGMT 31012", True),
        ("CSCI 32052", False), ("CSCI 32062", False), ("CSCI 32073", False),
        ("CSCI 32083", False), ("CSCI 32092", False), ("CSEC 32012", True),
        ("CSEC 32022", True), ("CSEC 32032", True), ("DSCI 32012", False),
        ("AINT 32012", False), ("AINT 32022", False), ("SCOM 32012", False)
    ],
    "Year4": [
        ("CSCI 43018", True), ("CSCI 44026", True),
        ("CSCI 44032", False), ("CSCI 44042", False), ("CSCI 44052", False),
        ("CSCI 44062", False), ("CSCI 44072", False), ("CSCI 44082", False),
        ("CSCI 44092", False), ("CSCI 44103", False), ("CSCI 44112", False),
        ("CSEC 44012", True), ("CSEC 44022", True), ("CSEC 44032", True),
        ("CSEC 44042", True), ("CSEC 44052", False), ("CSEC 44062", True),
        ("CSEC 44072", True), ("CSEC 44082", False), ("CSEC 44092", False),
        ("CSEC 44102", False), ("DSCI 44012", False), ("DSCI 44022", True),
        ("DSCI 44033", True), ("DSCI 44042", False), ("DSCI 44052", True),
        ("DSCI 44062", False), ("DSCI 44072", False), ("AINT 44012", False),
        ("AINT 44022", False), ("AINT 44032", False), ("AINT 44042", True),
        ("AINT 44052", False), ("AINT 44062", False), ("AINT 44072", False),
        ("SCOM 44012", False), ("SCOM 44022", True), ("SCOM 44033", False),
        ("SCOM 44043", False), ("SCOM 44052", False)
    ]
}

# ------- 3. Build the Kelaniya Course Enrollments DataFrame -------
course_data = []
grade_points_list = [4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0]

for _, row in students_df.iterrows():
    sid = row['student_id']
    start_year = row['intake_year']
    max_sem = min(8, 2 * (current_year - start_year))
    semester_labels = [f'Fall {start_year + i//2}' if i % 2 == 0 else f'Spring {start_year + i//2 + 1}' for i in range(8)]
    ability = (row['z_score_AL'] - 1.45) / (2.0 - 1.45)
    for sem_offset in range(max_sem):
        sem_label = semester_labels[sem_offset]
        if sem_offset < 2:
            sem_key = f"Year1_Sem{sem_offset + 1}"
        elif sem_offset < 4:
            sem_key = f"Year2_Sem{(sem_offset - 2) + 1}"
        elif sem_offset < 6:
            sem_key = f"Year3_Sem{(sem_offset - 4) + 1}"
        else:
            sem_key = "Year4"
        for course, compulsory in curriculum[sem_key]:
            if compulsory:
                credits = 3
                mean_index = 6 - 6 * ability
                index = np.clip(np.random.normal(mean_index, 1.5), 0, 6)
                grade_index = int(np.round(index))
                grade = grade_points_list[grade_index]
                course_data.append({
                    'student_id': sid,
                    'semester_number': sem_offset + 1,
                    'term_year': sem_label,
                    'course_code': course,
                    'credits': credits,
                    'grade_points': grade
                })
        if sem_key in ["Year3_Sem1", "Year3_Sem2", "Year4"]:
            prefix = pathway_to_prefix[row['pathway']]
            optional_courses = [c for c, comp in curriculum[sem_key] if not comp and c.startswith(prefix)]
        else:
            optional_courses = [c for c, comp in curriculum[sem_key] if not comp]
        chosen_opt = np.random.choice(optional_courses, size=min(2, len(optional_courses)), replace=False)
        for course in chosen_opt:
            credits = np.random.choice([2, 3])
            mean_index = 6 - 6 * ability
            index = np.clip(np.random.normal(mean_index, 1.5), 0, 6)
            grade_index = int(np.round(index))
            grade = grade_points_list[grade_index]
            course_data.append({
                'student_id': sid,
                'semester_number': sem_offset + 1,
                'term_year': sem_label,
                'course_code': course,
                'credits': credits,
                'grade_points': grade
            })

courses_df = pd.DataFrame(course_data)

# ------- 4. GPA Trajectories -------
gpa_data = []
for sid in student_ids:
    sc = courses_df[courses_df['student_id'] == sid].copy()
    sc.sort_values('semester_number', inplace=True)
    total_credits = 0
    total_points = 0
    max_sem = sc['semester_number'].max()
    for sem_idx in range(1, int(max_sem) + 1):
        sem_courses = sc[sc['semester_number'] == sem_idx]
        sem_credits = sem_courses['credits'].sum()
        sem_points = (sem_courses['credits'] * sem_courses['grade_points']).sum()
        total_credits += sem_credits
        total_points += sem_points
        cumulative_gpa = round(total_points / total_credits, 2) if total_credits > 0 else np.nan
        gpa_data.append({
            'student_id': sid,
            'semester_number': sem_idx,
            'cumulative_gpa': cumulative_gpa
        })

gpa_df = pd.DataFrame(gpa_data)

# ------- 5. Internships -------
companies = ['Virtusa', 'MillenniumIT', 'WSO2', 'IFS', 'hSenid', 'Dialog Axiata', '99x',
             'Sysco LABS', 'Virtuosoft', 'Calcey Technologies', 'Microimage',
             'Codegen', 'Xiteb', 'Johnkeellsit', 'Epictechnology', 'Arimac',
             'Bhasha', 'Insfra', 'Builtapps']
pathway_to_intern_roles = {
    "Artificial Intelligence": ["AI Intern", "Machine Learning Intern", "Data Science Intern"],
    "Cyber Security": ["Security Intern", "Network Security Intern"],
    "Data Science": ["Data Analyst Intern", "Data Science Intern"],
    "Scientific Computing": ["Computational Science Intern", "Research Intern"],
    "Standard": ["Software Intern", "Web Development Intern"]
}
internship_data = []

for _, row in students_df.iterrows():
    sid = row['student_id']
    if (row['current_year_of_study'] == 4 or row['is_graduated']) and np.random.rand() < 0.8:
        employer = np.random.choice(companies)
        role = np.random.choice(pathway_to_intern_roles[row['pathway']])
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        end_date = start_date + timedelta(days=180)
        performance_rating = round(np.random.uniform(3.0, 5.0), 2)
        internship_data.append({
            'student_id': sid,
            'employer': employer,
            'role_title': role,
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'performance_rating': performance_rating
        })

internships_df = pd.DataFrame(internship_data)

# ------- 6. Capstone Projects -------
pathway_to_domain = {
    "Cyber Security": "Network and Information Security",
    "Data Science": "Data Mining and Warehousing",
    "Artificial Intelligence": "Machine Learning",
    "Scientific Computing": "Scientific Computing",
    "Standard": "Software Engineering"
}
pathway_to_tech = {
    "Cyber Security": "Python, Wireshark, Kali Linux",
    "Data Science": "Python, Pandas, Scikit-Learn, R",
    "Artificial Intelligence": "Python, TensorFlow, PyTorch",
    "Scientific Computing": "MATLAB, Python",
    "Standard": "Java, Spring, React, SQL"
}
capstone_data = []

for sid in student_ids:
    pathway = students_df.loc[students_df['student_id'] == sid, 'pathway'].values[0]
    domain = pathway_to_domain[pathway]
    tech = pathway_to_tech[pathway]
    outcome = np.random.choice(["Prototype", "Published Paper", "Industry Collaboration", "University Showcase"])
    capstone_data.append({
        'student_id': sid,
        'pathway': pathway,
        'domain': domain,
        'technologies_used': tech,
        'outcome': outcome
    })

capstone_df = pd.DataFrame(capstone_data)

# ------- 7. Extended Project Templates by Subcategory -------
project_subcategories = {
    "Artificial Intelligence - Computer Vision & Image Processing": {
        "Medical Image Analysis System": ["Python", "TensorFlow", "OpenCV", "Keras", "DICOM"],
        "Facial Recognition Attendance System": ["Python", "PyTorch", "OpenCV", "Flask", "SQLite"],
        "Object Detection for Autonomous Vehicles": ["Python", "YOLO", "TensorFlow", "Computer Vision"],
        "Handwritten Digit Recognition": ["Python", "CNN", "TensorFlow", "Jupyter Notebook"],
        "Traffic Sign Recognition System": ["Python", "TensorFlow", "OpenCV", "Keras"],
        "Augmented Reality Application": ["Python", "OpenCV", "Unity", "ARCore"]
    },
    "Artificial Intelligence - Natural Language Processing": {
        "Sentiment Analysis for Social Media": ["Python", "NLTK", "TensorFlow", "Twitter API", "Pandas"],
        "Chatbot with Intent Recognition": ["Python", "PyTorch", "Transformers", "Flask", "NLP"],
        "Automatic Text Summarization": ["Python", "BERT", "Transformers", "Streamlit"],
        "Language Translation System": ["Python", "Seq2Seq", "TensorFlow", "Google Translate API"],
        "Voice Assistant for Local Languages": ["Python", "SpeechRecognition", "NLTK", "TensorFlow"],
        "Text Classification for News Articles": ["Python", "Scikit-Learn", "Pandas", "NLP"]
    },
    "Artificial Intelligence - Predictive Analytics": {
        "Stock Price Prediction Model": ["Python", "LSTM", "TensorFlow", "Yahoo Finance API", "Pandas"],
        "Customer Churn Prediction": ["Python", "Random Forest", "PyTorch", "Scikit-learn"],
        "Disease Prediction System": ["Python", "Neural Networks", "TensorFlow", "Medical Dataset"],
        "Recommendation Engine": ["Python", "Collaborative Filtering", "PyTorch", "Matrix Factorization"],
        "Credit Risk Assessment Model": ["Python", "Scikit-Learn", "Pandas", "XGBoost"],
        "Energy Consumption Forecasting": ["Python", "Prophet", "Pandas", "Time Series Analysis"]
    },
    "Cyber Security - Network Security": {
        "Intrusion Detection System": ["Python", "Wireshark", "Scapy", "Machine Learning", "Kali Linux"],
        "Network Traffic Analyzer": ["Python", "Wireshark", "Packet Sniffing", "TCP/IP", "Network Protocols"],
        "Firewall Configuration Tool": ["Python", "iptables", "Network Security", "Linux"],
        "DDoS Attack Detection": ["Python", "Wireshark", "Network Analysis", "Real-time Monitoring"],
        "Secure File Transfer Protocol": ["Python", "Cryptography", "Sockets", "SSL/TLS"],
        "Network Intrusion Prevention": ["Python", "Snort", "Machine Learning", "Network Security"]
    },
    "Cyber Security - Security Assessment": {
        "Vulnerability Scanner": ["Python", "Nmap", "Kali Linux", "Port Scanning", "Security Testing"],
        "Password Strength Analyzer": ["Python", "Cryptography", "Hash Functions", "Security Algorithms"],
        "Web Application Security Scanner": ["Python", "OWASP", "SQL Injection Detection", "XSS Prevention"],
        "Penetration Testing Framework": ["Python", "Metasploit", "Kali Linux", "Ethical Hacking"],
        "Automated Penetration Testing Tool": ["Python", "Metasploit", "Nmap", "Burp Suite"],
        "Security Audit Framework": ["Python", "OWASP", "Security Standards", "Compliance"]
    },
    "Cyber Security - Digital Forensics": {
        "Digital Evidence Analyzer": ["Python", "File System Analysis", "Metadata Extraction", "Forensics Tools"],
        "Network Forensics Tool": ["Python", "Wireshark", "Network Packet Analysis", "Evidence Collection"],
        "Malware Analysis Platform": ["Python", "Reverse Engineering", "Static Analysis", "Sandbox"],
        "Memory Forensics Analyzer": ["Python", "Volatility", "Memory Analysis", "Malware Detection"],
        "Mobile Device Forensics Tool": ["Python", "Android Debug Bridge", "iOS Forensics", "Data Extraction"]
    },
    "Data Science - Data Analytics & Visualization": {
        "Sales Performance Dashboard": ["Python", "Pandas", "Plotly", "Dash", "Business Intelligence"],
        "Customer Segmentation Analysis": ["Python", "K-Means", "Scikit-Learn", "Pandas", "Seaborn"],
        "Web Analytics Platform": ["Python", "Google Analytics API", "Pandas", "Matplotlib"],
        "Market Basket Analysis": ["Python", "Association Rules", "Pandas", "R", "Retail Analytics"],
        "Fraud Detection System": ["Python", "Pandas", "Scikit-Learn", "Matplotlib", "Anomaly Detection"],
        "Social Media Analytics Dashboard": ["Python", "Pandas", "Plotly", "Twitter API", "Sentiment Analysis"]
    },
    "Data Science - Data Warehousing": {
        "ETL Pipeline for E-commerce": ["Python", "Apache Airflow", "Pandas", "PostgreSQL", "Data Integration"],
        "Real-time Data Streaming Platform": ["Python", "Apache Kafka", "Pandas", "Stream Processing"],
        "Data Quality Assessment Tool": ["Python", "Pandas", "Data Profiling", "Quality Metrics"],
        "Multi-source Data Integration": ["Python", "Pandas", "API Integration", "Database Connectors"],
        "Data Lake Architecture": ["Python", "Apache Spark", "Hadoop", "AWS S3", "Data Ingestion"],
        "Real-time Analytics Pipeline": ["Python", "Apache Kafka", "Spark Streaming", "Elasticsearch"]
    },
    "Data Science - Statistical Analysis": {
        "A/B Testing Framework": ["Python", "Statistical Testing", "Pandas", "R", "Hypothesis Testing"],
        "Time Series Forecasting": ["Python", "ARIMA", "Pandas", "Scikit-Learn", "R"],
        "Survey Data Analysis Tool": ["Python", "Statistical Analysis", "Pandas", "R", "Survey Analytics"],
        "Bayesian Inference System": ["Python", "PyMC3", "Pandas", "Statistical Modeling"],
        "Survival Analysis Tool": ["Python", "Lifelines", "Pandas", "R"]
    },
    "Standard - Web Applications": {
        "Task Management System": ["Java", "Spring Boot", "React", "PostgreSQL", "REST API"],
        "Online Banking System": ["Java", "Spring Security", "React", "MySQL", "JWT Authentication"],
        "E-learning Platform": ["Java", "Spring", "React", "MongoDB", "Video Streaming"],
        "Hospital Management System": ["Java", "Spring MVC", "React", "SQL Server", "Medical Records"],
        "Online Marketplace Platform": ["Java", "Spring Boot", "React", "MongoDB", "Payment Gateway"],
        "Content Management System": ["Java", "Spring MVC", "Angular", "MySQL", "User Authentication"]
    },
    "Standard - Enterprise Applications": {
        "Employee Management System": ["Java", "Spring Framework", "React", "Oracle DB", "HR Management"],
        "Inventory Control System": ["Java", "Spring Boot", "React", "MySQL", "Barcode Integration"],
        "Customer Relationship Management": ["Java", "Spring", "React", "PostgreSQL", "CRM Features"],
        "Supply Chain Management": ["Java", "Microservices", "React", "Database Integration"],
        "Human Resource Management": ["Java", "Spring Framework", "Vue.js", "PostgreSQL", "Employee Management"],
        "ERP Module": ["Java", "Spring Boot", "React", "Oracle DB", "Business Processes"]
    },
    "Standard - API & Backend Systems": {
        "RESTful API Gateway": ["Java", "Spring Cloud", "Microservices", "SQL Database", "API Management"],
        "Document Management System": ["Java", "Spring Boot", "React", "File Upload", "Version Control"],
        "Payment Processing System": ["Java", "Spring Security", "React", "Payment Gateway", "Transaction Management"],
        "Microservices E-commerce Backend": ["Java", "Spring Cloud", "Docker", "Kubernetes", "API Gateway"],
        "Real-time Chat Backend": ["Java", "Spring WebSocket", "Redis", "Message Queue"]
    },
    "Scientific Computing - Mathematical Modeling": {
        "Climate Change Simulation": ["MATLAB", "Python", "NumPy", "SciPy", "Data Modeling"],
        "Population Dynamics Model": ["MATLAB", "Python", "Mathematical Modeling", "Differential Equations"],
        "Fluid Dynamics Simulator": ["MATLAB", "Python", "Numerical Methods", "Computational Physics"],
        "Optimization Algorithm Comparison": ["MATLAB", "Python", "Optimization", "Algorithm Analysis"],
        "Epidemic Spread Simulation": ["MATLAB", "Python", "Differential Equations", "Agent-based Modeling"],
        "Structural Engineering Analysis": ["MATLAB", "Finite Element Method", "Numerical Analysis", "Python"]
    },
    "Scientific Computing - Data Analysis & Simulation": {
        "Monte Carlo Simulation": ["MATLAB", "Python", "Statistical Simulation", "Random Sampling"],
        "Signal Processing System": ["MATLAB", "Python", "Digital Signal Processing", "Fourier Transform"],
        "Image Processing Pipeline": ["MATLAB", "Python", "Computer Vision", "Image Enhancement"],
        "Financial Risk Modeling": ["MATLAB", "Python", "Risk Analysis", "Financial Mathematics"],
        "Quantum Computing Simulation": ["Python", "Qiskit", "Quantum Algorithms", "Simulation"],
        "Bioinformatics Data Analysis": ["Python", "Biopython", "Pandas", "Genomics", "R"]
    }
}

pathway_to_subcategories = {
    "Artificial Intelligence": [
        "Artificial Intelligence - Computer Vision & Image Processing",
        "Artificial Intelligence - Natural Language Processing",
        "Artificial Intelligence - Predictive Analytics"
    ],
    "Cyber Security": [
        "Cyber Security - Network Security",
        "Cyber Security - Security Assessment",
        "Cyber Security - Digital Forensics"
    ],
    "Data Science": [
        "Data Science - Data Analytics & Visualization",
        "Data Science - Data Warehousing",
        "Data Science - Statistical Analysis"
    ],
    "Standard": [
        "Standard - Web Applications",
        "Standard - Enterprise Applications",
        "Standard - API & Backend Systems"
    ],
    "Scientific Computing": [
        "Scientific Computing - Mathematical Modeling",
        "Scientific Computing - Data Analysis & Simulation"
    ]
}

# ------- 8. Student Projects: average 3 from pathway subcategories + 1 from other areas -------
all_projects = {}
for subcat in project_subcategories:
    for title, techs in project_subcategories[subcat].items():
        all_projects[title] = techs

pathway_to_related_projects = {}
pathway_to_other_projects = {}
all_subcats = list(project_subcategories.keys())
for P in pathway_to_subcategories:
    related_subcats = set(pathway_to_subcategories[P])
    other_subcats = [subcat for subcat in all_subcats if subcat not in related_subcats]
    related_projects = [title for subcat in pathway_to_subcategories[P] for title in project_subcategories[subcat].keys()]
    other_projects = [title for subcat in other_subcats for title in project_subcategories[subcat].keys()]
    pathway_to_related_projects[P] = related_projects
    pathway_to_other_projects[P] = other_projects

projects_data = []
project_id = 1

for sid in student_ids:
    P = students_df.loc[students_df['student_id'] == sid, 'pathway'].values[0]
    related_list = pathway_to_related_projects[P]
    other_list = pathway_to_other_projects[P]
    r = np.random.poisson(3)
    selected_r = min(r, len(related_list))
    chosen_related = np.random.choice(related_list, size=selected_r, replace=False) if selected_r > 0 else []
    o = np.random.poisson(1)
    selected_o = min(o, len(other_list))
    chosen_other = np.random.choice(other_list, size=selected_o, replace=False) if selected_o > 0 else []
    all_chosen_titles = list(chosen_related) + list(chosen_other)
    for title in all_chosen_titles:
        techs = all_projects[title]
        start_date = datetime(2023, np.random.randint(1, 13), np.random.randint(1, 28))
        end_date = start_date + timedelta(days=np.random.randint(30, 180))
        outcome = np.random.choice(["Completed", "Ongoing", "Abandoned", "Deployed"])
        projects_data.append({
            'project_id': project_id,
            'student_id': sid,
            'project_title': title,
            'technologies_used': ", ".join(techs),
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'outcome': outcome
        })
        project_id += 1

projects_df = pd.DataFrame(projects_data)

# ------- 9. Certifications & Workshops -------
pathway_specific_certifications = {
    "Artificial Intelligence": [
        "Google Cloud Certified - Machine Learning Engineer",
        "IBM Watson AI",
        "Microsoft Certified: Azure AI Engineer Associate",
        "Coursera IBM AI Engineering",
        "edX MicroMasters in Artificial Intelligence"
    ],
    "Cyber Security": [
        "CompTIA Security+",
        "Certified Ethical Hacker (CEH)",
        "Certified Information Systems Security Professional (CISSP)",
        "Offensive Security Certified Professional (OSCP)",
        "GIAC Security Essentials (GSEC)"
    ],
    "Data Science": [
        "Google Data Analytics",
        "Microsoft Certified: Azure Data Scientist Associate",
        "SAS Certified Data Scientist",
        "Cloudera Certified Data Analyst",
        "IBM Data Science Professional Certificate"
    ],
    "Scientific Computing": [
        "MATLAB Certified Associate",
        "Python for Scientific Computing Certification",
        "R for Data Science Certification"
    ],
    "Standard": [
        "Oracle Certified Java Programmer",
        "Cisco CCNA",
        "AWS Certified Cloud Practitioner",
        "Microsoft Certified: Azure Administrator Associate",
        "Google Cloud Certified - Professional Cloud Developer"
    ]
}

general_certifications = [
    "Project Management Professional (PMP)",
    "Scrum Master Certification",
    "ITIL Foundation",
    "TOGAF 9 Certified",
    "Six Sigma Green Belt"
]

certification_data = []

for sid in student_ids:
    P = students_df.loc[students_df['student_id'] == sid, 'pathway'].values[0]
    pathway_certs = pathway_specific_certifications[P]
    num_pathway_certs = min(np.random.poisson(3), len(pathway_certs))
    if num_pathway_certs > 0:
        chosen_pathway_certs = np.random.choice(pathway_certs, size=num_pathway_certs, replace=False)
        for cert in chosen_pathway_certs:
            cert_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            certification_data.append({
                'student_id': sid,
                'certificate_name': cert,
                'date_earned': cert_date.date(),
                'type': 'pathway-specific'
            })
    num_general_certs = min(np.random.poisson(1), len(general_certifications))
    if num_general_certs > 0:
        chosen_general_certs = np.random.choice(general_certifications, size=num_general_certs, replace=False)
        for cert in chosen_general_certs:
            cert_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            certification_data.append({
                'student_id': sid,
                'certificate_name': cert,
                'date_earned': cert_date.date(),
                'type': 'general'
            })

certifications_df = pd.DataFrame(certification_data)

# ------- 10. Placement Outcomes (for all students) -------
pathway_to_job_roles = {
    "Artificial Intelligence": ["AI Engineer", "Machine Learning Engineer", "Data Scientist"],
    "Cyber Security": ["Cybersecurity Specialist", "Network Security Engineer"],
    "Data Science": ["Data Scientist", "Data Analyst"],
    "Scientific Computing": ["Scientific Programmer", "Computational Scientist"],
    "Standard": ["Software Engineer", "Web Developer"]
}
placement_data = []

latest_gpa_df = gpa_df.loc[gpa_df.groupby('student_id')['semester_number'].idxmax()]

for _, row in students_df.iterrows():
    sid = row['student_id']
    graduation_date = datetime(row['intake_year'] + 4, 5, 15)
    latest_gpa = latest_gpa_df[latest_gpa_df['student_id'] == sid]['cumulative_gpa'].values[0]
    placed = np.random.choice([True, False], p=[0.85, 0.15])
    placement_data.append({'student_id': sid, 'placed': placed})
    if placed:
        company = np.random.choice(companies)
        role = np.random.choice(pathway_to_job_roles[row['pathway']])
        location = np.random.choice(['Colombo', 'Kandy', 'Galle', 'Jaffna'])
        job_start_date = graduation_date + timedelta(days=np.random.randint(30, 180))
        starting_salary = 50000 + (latest_gpa - 2.0) / 2.0 * 70000
        starting_salary = round(starting_salary, -3)
        bonus = np.random.choice([0, 5000, 10000])
        time_to_job = (job_start_date - graduation_date).days
        employed_one_year = np.random.choice([True, False], p=[0.7, 0.3])
        promotion = np.random.choice([True, False], p=[0.2, 0.8])
        placement_data[-1].update({
            'company_name': company,
            'role_title': role,
            'location': location,
            'job_start_date': job_start_date.date(),
            'starting_salary_lkr': starting_salary,
            'bonus_lkr': bonus,
            'time_to_job_days': time_to_job,
            'employed_one_year': employed_one_year,
            'promotion_within_year': promotion
        })

placement_df = pd.DataFrame(placement_data)

# ------- 11. External Industry Indicators (IT sector) -------
industry_df = pd.DataFrame([{
    'sector': 'Information Technology',
    'avg_hiring_rate': 0.10,
    'emerging_job_titles': 'AI Engineer, Cloud Engineer',
    'salary_benchmark_monthly_lkr': 206938
}])

# ------- 12. Industry Trends Table -------
industry_trends = pd.DataFrame([
    {'role_title': 'AI & Machine Learning Specialist', 'demand_level': 'High'},
    {'role_title': 'Data Engineer / Data Scientist', 'demand_level': 'High'},
    {'role_title': 'Senior Software Engineer', 'demand_level': 'High'},
    {'role_title': 'Software Developer', 'demand_level': 'High'},
    {'role_title': 'Cloud / DevOps Engineer', 'demand_level': 'High'},
    {'role_title': 'Cybersecurity Analyst / Security Engineer', 'demand_level': 'High'},
    {'role_title': 'Network Engineer / Cloud Architect', 'demand_level': 'Medium'},
    {'role_title': 'Business Analyst (IT)', 'demand_level': 'Medium'},
    {'role_title': 'Mobile App Developer', 'demand_level': 'Medium'},
    {'role_title': 'UX/UI Designer', 'demand_level': 'Medium'}
])

# ------- Save All DataFrames to CSV Files -------
students_df.to_csv('kelaniya_students.csv', index=False)
courses_df.to_csv('kelaniya_courses.csv', index=False)
gpa_df.to_csv('kelaniya_gpa.csv', index=False)
internships_df.to_csv('kelaniya_internships.csv', index=False)
capstone_df.to_csv('kelaniya_capstone.csv', index=False)
projects_df.to_csv('kelaniya_projects.csv', index=False)
certifications_df.to_csv('kelaniya_certifications.csv', index=False)
placement_df.to_csv('kelaniya_placement.csv', index=False)
industry_df.to_csv('kelaniya_industry.csv', index=False)
industry_trends.to_csv('kelaniya_industry_trends.csv', index=False)

print("CSV files generated successfully:")
print(" - kelaniya_students.csv")
print(" - kelaniya_courses.csv")
print(" - kelaniya_gpa.csv")
print(" - kelaniya_internships.csv")
print(" - kelaniya_capstone.csv")
print(" - kelaniya_projects.csv")
print(" - kelaniya_certifications.csv")
print(" - kelaniya_placement.csv")
print(" - kelaniya_industry.csv")
print(" - kelaniya_industry_trends.csv")