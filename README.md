# 💼 AI Job Recommendation System

An AI-powered job recommendation system that helps users discover relevant job opportunities based on their skills, experience, and preferences.  
This project leverages **machine learning** and **NLP** to provide personalized job suggestions.

---

## 📑 Table of Contents
1. [Demo](#-demo)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Installation & Setup](#-installation--setup)
5. [Usage](#-usage)
6. [Dataset](#-dataset)
7. [Model & Approach](#-model--approach)
8. [Project Structure](#-project-structure)
9. [Contributing](#-contributing)
10. [Future Improvements](#-future-improvements)
11. [Acknowledgements](#-acknowledgements)
12. [License](#-license)

---

## 🎥 Demo
https://github.com/Krishdshah/AI-job-recommendation-system/blob/main/demo.mp4

---

## ✨ Features
- Personalized job recommendations  
- Filtering by role, skills, location, and experience  
- Resume parsing and matching  
- Machine learning model for intelligent ranking  
- Easy-to-use interface  

---

## 🛠 Tech Stack
- **Backend & Interface**: Python, **Streamlit**  
- **PDF Processing**: **pdfplumber**, **PyPDF2**  
- **Machine Learning / NLP**: **sentence-transformers**, **torch**, **transformers**, **scikit-learn**  
- **Data Handling**: **numpy**, **pandas**  

---

## ⚙️ Installation & Setup
```bash
# Clone repo
git clone https://github.com/Krishdshah/AI-job-recommendation-system.git
cd AI-job-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
