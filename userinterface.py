#!/usr/bin/env python
# coding: utf-8

'''Required libraries for data handling, visualization,
network analysis, evaluation metrics, and the Streamlit interface.'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re
import io
import networkx as nx
from sklearn.metrics import silhouette_score, r2_score
from sklearn.preprocessing import LabelEncoder

# Libraries used to create the interactive interface in Streamlit.
import streamlit as st
import os

# ────୨ৎ────────୨ৎ────────୨ৎ──────
#  STREAMLIT CONFIGURATION
# ────୨ৎ────────୨ৎ────────୨ৎ──────
# Set page config at the very start to avoid errors
try:
    st.set_page_config(page_title="Student Anxiety Decision Support System", layout="wide")
except:
    pass  # Ignore if running in standard python mode


#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────
# DATA LOADING & PRE-PROCESSING
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────
'''This section loads the survey data and prepares it for analysis by cleaning and organizing the variables.'''

# Cache the data so it is not reloaded every time the app refreshes.

@st.cache_data
def load_data():
    raw_text_data = """
    "I Agree to participate in the research project under the conditions described above","The name of your institution","The name of your program of study","Your current class level is","Your gender","Your current age","Living with family?","Are you happy with your academic Condition?","Your CGPA","Are you addicted to any drugs?","Are you in a relationship?","Did you have a recent breakup?","How often do you conflict with your friend?","Do you have financial problem in your family?","Violence in family?","Have you ever been bullied","Have you ever been sexually harassed or abused?","How many hours do you spend on social media?","1. I found it hard to wind down","2. I tended to over-react to situations","3 .I felt that I was using a lot of nervous energy","4. I found myself getting agitated","5. I found it difficult to relax","6. I was intolerant of anything that kept me from getting on with what I was doing","7. I felt that I was rather touchy","1. _____ Extraverted, enthusiastic.","2. _____ Critical, quarrelsome.","3. _____ Dependable, self-disciplined.","4. _____ Anxious, easily upset.","5. _____ Open to new experiences, complex.","6. _____ Reserved, quiet.","7. _____ Sympathetic, warm.","8. _____ Disorganized, careless.","9. _____ Calm, emotionally stable","10. _____ Conventional, uncreative."
    "Yes, I agree to participate,","EDU","CSE","Freshman (Undergrad)","Female","> 25","Yes","Yes","3.01-3.50","No","Yes","Yes","Sometimes","Yes","Most of the time","No","Maybe","< 1 Hours","0","1","2","3","1","2","3","1","5","4","7","7","7","7","5","1","1"
    "Yes, I agree to participate,","EDU","CSE","Senior (Undergrad)","Male","21-25","Yes","Yes","3.01-3.50","No","Yes","No","Sometimes","Yes","Never","No","Yes","> 3 Hours","1","3","2","1","3","3","2","6","5","2","7","5","4","7","5","1","5"
    "Yes, I agree to participate,","East Delta University","Computer Science and Engineering","Senior (Undergrad)","Male","> 25","Yes","Yes","> 3.50","No","No","No","Never occurs","No","Never","No","No","> 3 Hours","0","0","0","0","0","0","0","2","5","7","2","5","6","5","5","3","2"
    "Yes, I agree to participate,","East Delta University ","CSE ","Senior (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","No","No","Never occurs","Yes","Never","No","No","> 3 Hours","1","1","0","1","2","1","1","1","1","3","4","6","4","6","5","6","1"
    "Yes, I agree to participate,","East Delta University","BSC in CSE","Senior (Undergrad)","Male","21-25","Yes","Yes","> 3.50","No","No","No","Sometimes","No","Never","No","Maybe","> 3 Hours","2","2","2","3","0","0","0","2","5","6","6","5","5","5","5","3","3"
    "Yes, I agree to participate,","East Delta University ","CSE","Junior (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","Yes","No","Sometimes","Yes","Never","Yes","No","1-3 Hours","3","2","2","1","2","0","1","2","1","6","1","7","7","7","5","6","7"
    "Yes, I agree to participate,","East Delta University","CSE","Senior (Undergrad)","Male","21-25","Yes","Yes","3.01-3.50","No","Yes","No","Never occurs","Yes","Never","No","No","> 3 Hours","3","2","1","2","0","2","3","2","5","6","5","3","2","6","5","5","5"
    "Yes, I agree to participate,","University of Chittagong ","MBA","Graduate student","Male","21-25","Yes","No","3.01-3.50","Yes","Yes","No","Never occurs","No","Never","Yes","Maybe","> 3 Hours","1","0","2","0","0","1","0","7","6","4","5","3","2","6","5","5","2"
    "Yes, I agree to participate,","East Delta University ","BSc","Senior (Undergrad)","Male","21-25","Yes","Yes","3.01-3.50","No","Yes","No","Never occurs","No","Rarely","Yes","No","< 1 Hours","0","1","1","2","1","1","1","1","4","5","7","5","2","1","5","1","4"
    "Yes, I agree to participate,","East Delta University ","CSE","Senior (Undergrad)","Male","21-25","Yes","Yes","> 3.50","No","No","No","Never occurs","No","Never","No","No","< 1 Hours","0","1","0","1","0","0","1","1","7","5","7","5","7","4","5","1","7"
    "Yes, I agree to participate,","East Delta University ","Computer Science and Engineering","Senior (Undergrad)","Male","21-25","Yes","Yes","3.01-3.50","No","No","No","Never occurs","No","Never","No","No","> 3 Hours","0","0","0","0","0","0","0","5","3","6","6","3","7","5","5","5","7"
    "Yes, I agree to participate,","International Islamic University Chittagong ","Masters in English Language Teaching","Graduate student","Male","> 25","Yes","No","< 3.00","No","No","Yes","Never occurs","Yes","Never","Yes","No","> 3 Hours","1","0","2","0","0","0","0","6","5","6","6","6","4","7","5","3","3"
    "Yes, I agree to participate,","CVASU ","Food Science and Technology ","Senior (Undergrad)","Male","21-25","No","No","< 3.00","No","No","No","Sometimes","No","Rarely","Yes","No","> 3 Hours","1","0","1","1","1","2","1","5","1","4","6","5","2","7","5","2","1"
    "Yes, I agree to participate,","International Islamic University Chittagong ","English Language and Literature ","Graduate student","Male","21-25","Yes","Yes","3.01-3.50","No","No","No","Sometimes","Yes","Rarely","No","No","> 3 Hours","2","0","1","2","1","2","1","5","2","5","5","2","5","3","5","6","2"
    "Yes, I agree to participate,","East Delta University ","BSc. in Computer Science  & Engineering ","Senior (Undergrad)","Male","21-25","Yes","No","3.01-3.50","No","No","No","Sometimes","No","Never","Yes","No","> 3 Hours","1","3","3","3","3","3","0","3","5","6","1","6","6","4","5","7","2"
    "Yes, I agree to participate,","Government Commerce College ","BBS","Freshman (Undergrad)","Male","21-25","Yes","No","3.01-3.50","No","No","Yes","Sometimes","Yes","Never","No","No","> 3 Hours","1","3","1","1","3","3","1","3","5","4","5","6","5","5","5","4","4"
    "Yes, I agree to participate,","East Delta University ","CSE","Freshman (Undergrad)","Male","21-25","Yes","Yes","< 3.00","No","No","Yes","Sometimes","Yes","Never","No","Maybe","> 3 Hours","0","2","0","1","1","1","2","1","4","5","7","6","5","4","5","1","3"
    "Yes, I agree to participate,","East Delta University ","EEE","Graduate student","Male","21-25","Yes","Yes","< 3.00","No","No","No","Sometimes","Yes","Never","No","No","> 3 Hours","0","2","1","1","1","1","1","1","7","4","0","1","6","4","5","3","1"
    "Yes, I agree to participate,","International Islamic University Chittagong ","B.A in ELL","Graduate student","Male","21-25","Yes","No","3.01-3.50","No","No","No","Never occurs","No","Often","Yes","Yes","> 3 Hours","2","1","3","2","2","1","1","2","7","7","1","7","5","4","5","6","2"
    "Yes, I agree to participate,","East Delta University ","English ","Senior (Undergrad)","Female","21-25","Yes","No","3.01-3.50","No","No","No","Sometimes","No","Never","No","No","> 3 Hours","0","1","1","1","1","1","1","2","2","5","4","6","5","7","5","3","6"
    "Yes, I agree to participate,","East Delta University ","EEE","Senior (Undergrad)","Male","21-25","No","No","< 3.00","No","Yes","No","Never occurs","No","Never","No","No","1-3 Hours","0","0","1","2","1","1","2","7","1","3","6","6","7","2","5","1","1"
    "Yes, I agree to participate,","East Delta University ","CSE","Senior (Undergrad)","Female","> 25","Yes","No","3.01-3.50","No","No","No","Sometimes","No","Rarely","Yes","Maybe","> 3 Hours","3","1","0","3","3","3","1","1","6","5","7","6","1","5","5","2","7"
    "Yes, I agree to participate,","University of Chittagong ","Law","Graduate student","Male","21-25","No","No","3.01-3.50","No","Yes","No","Sometimes","Yes","Rarely","Yes","No","1-3 Hours","2","1","1","2","0","0","1","3","3","5","5","2","6","5","5","1","1"
    "Yes, I agree to participate,","East Delta University ","CSE","Senior (Undergrad)","Female","> 25","Yes","No","3.01-3.50","No","Yes","No","Never occurs","No","Never","Yes","Yes","> 3 Hours","1","1","2","2","1","1","3","1","5","5","6","5","5","5","5","3","5"
    "Yes, I agree to participate,","BGC Trust University Bangladesh","Computer Science and Engineering ","Sophomore (Undergrad)","Male","> 25","Yes","No","> 3.50","No","No","No","Sometimes","Yes","Most of the time","No","Yes","1-3 Hours","1","1","2","1","2","1","1","2","4","3","5","3","6","1","5","3","4"
    "Yes, I agree to participate,","Chittagong Independent University   ","ELT","Graduate student","Male","21-25","No","No","> 3.50","No","No","No","Sometimes","No","Never","No","No","> 3 Hours","1","1","1","3","1","3","1","2","6","5","7","5","6","6","5","1","2"
    "Yes, I agree to participate,","East Delta University ","Bsc in CSE","Senior (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","Yes","No","Sometimes","No","Never","Yes","No","1-3 Hours","3","1","3","3","3","3","2","1","1","7","7","7","7","5","5","1","1"
    "Yes, I agree to participate,","University of Chittagong ","Education and Research ","Freshman (Undergrad)","Male","21-25","No","No","> 3.50","No","No","No","Never occurs","No","Rarely","Yes","No","> 3 Hours","1","0","1","2","2","3","1","4","3","1","7","7","2","1","5","3","1"
    "Yes, I agree to participate,","East Delta University ","CSE","Senior (Undergrad)","Male","21-25","No","Yes","3.01-3.50","No","No","No","Sometimes","No","Rarely","No","No","> 3 Hours","0","1","0","1","1","3","1","7","1","6","6","7","1","5","5","4","7"
    "Yes, I agree to participate,","Islamic University Kushtia ","LLB","Senior (Undergrad)","Male","21-25","Yes","No","3.01-3.50","No","Yes","No","Sometimes","No","Rarely","Yes","Yes","> 3 Hours","1","1","0","1","3","3","0","0","0","1","1","0","2","2","5","2","0"
    "Yes, I agree to participate,","East Delta University ","CSE","Sophomore (Undergrad)","Female","21-25","Yes","Yes","3.01-3.50","No","No","No","Most of the time","No","Never","Yes","No","> 3 Hours","1","3","3","3","3","1","3","5","0","6","0","7","7","7","5","7","3"
    "Yes, I agree to participate,","CUET","Electrical and Electronic Engineering ","Senior (Undergrad)","Male","21-25","Yes","Yes","> 3.50","No","No","No","Never occurs","No","Rarely","No","No","1-3 Hours","2","1","0","2","2","2","2","4","1","5","7","7","7","3","5","3","3"
    "Yes, I agree to participate,","Islamic university ","English ","Sophomore (Undergrad)","Female","21-25","No","No","> 3.50","No","Yes","No","Never occurs","Yes","Rarely","No","Yes","1-3 Hours","1","1","1","2","2","3","1","3","1","4","1","6","6","4","5","6","2"
    "Yes, I agree to participate,","Chittagong University of Engineering and Technology ","Petroleum and Mining Engineering ","Sophomore (Undergrad)","Male","> 25","No","No","3.01-3.50","Yes","Yes","No","Sometimes","No","Rarely","No","No","1-3 Hours","2","1","2","1","1","1","0","2","2","5","5","5","5","2","5","5","5"
    "Yes, I agree to participate,","University of Dhaka","BSC- Physics ","Senior (Undergrad)","Male","21-25","Yes","No","> 3.50","No","No","No","Never occurs","Yes","Rarely","No","Maybe","> 3 Hours","2","2","2","2","1","1","0","2","1","6","6","5","7","7","5","5","2"
    "Yes, I agree to participate,","East Delta university ","MBA","Graduate student","Male","> 25","Yes","No","> 3.50","No","No","Yes","Sometimes","Yes","Never","No","No","> 3 Hours","2","3","3","3","3","3","3","1","3","4","5","6","7","5","5","3","5"
    "Yes, I agree to participate,","Mymensingh Medical College ","MBBS","Junior (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","No","No","Sometimes","No","Never","Yes","No","> 3 Hours","3","3","3","3","3","3","3","6","5","3","7","7","3","6","5","3","5"
    "Yes, I agree to participate,","AIUB ","Bsc in CSE (Graduate) ","Graduate student","Male","21-25","Yes","Yes","< 3.00","No","Yes","Yes","Sometimes","No","Never","Yes","No","> 3 Hours","2","1","2","3","3","2","2","5","5","3","4","6","4","6","5","6","2"
    "Yes, I agree to participate,","Eden Mohila College  ","B A honours ","Sophomore (Undergrad)","Female","21-25","Yes","No","3.01-3.50","No","No","No","Never occurs","No","Rarely","Yes","Yes","> 3 Hours","1","1","2","3","1","3","0","2","5","3","4","2","4","2","5","3","3"
    "Yes, I agree to participate,","Butex","B.Sc in Textile Engineering","Senior (Undergrad)","Male","21-25","Yes","No","3.01-3.50","No","No","No","Sometimes","Yes","Rarely","Yes","No","> 3 Hours","0","0","1","2","1","1","1","5","4","3","2","7","2","5","5","2","1"
    "Yes, I agree to participate,","Bangladesh Agricultural University ","B.Sc. Fisheries( Hons.) ","Sophomore (Undergrad)","Female","21-25","Yes","No","> 3.50","No","No","No","Sometimes","No","Never","Yes","Yes","> 3 Hours","1","1","3","2","2","2","2","4","5","5","5","7","4","3","5","4","1"
    "Yes, I agree to participate,","CU","B.A (Hon's), M.A in History ","Graduate student","Male","21-25","Yes","Yes","3.01-3.50","Yes","No","Yes","Sometimes","Yes","Rarely","No","No","1-3 Hours","0","0","0","1","0","1","0","3","5","3","5","7","5","6","5","4","2"
    "Yes, I agree to participate,","Bangladesh Agricultural University ","Fisheries ","Senior (Undergrad)","Female","21-25","Yes","No","> 3.50","No","No","No","Never occurs","Yes","Most of the time","Yes","Yes","1-3 Hours","2","1","2","2","3","2","2","6","2","7","2","7","3","5","5","6","7"
    "Yes, I agree to participate,","East Delta University ","BSc in CSE","Senior (Undergrad)","Male","21-25","No","Yes","< 3.00","Yes","Yes","No","Sometimes","No","Never","Yes","Yes","> 3 Hours","0","3","2","3","3","3","3","6","4","6","5","7","3","7","5","5","2"
    "Yes, I agree to participate,","Bangladesh Agricultural University ","Fisheries ","Junior (Undergrad)","Female","21-25","Yes","Yes","< 3.00","No","No","No","Sometimes","Yes","Rarely","No","No","> 3 Hours","1","1","1","1","2","2","0","2","7","2","7","1","1","1","5","1","7"
    "Yes, I agree to participate,","University of Chittagong ","Law","Junior (Undergrad)","Female","21-25","Yes","No","3.01-3.50","No","No","No","Sometimes","Yes","Most of the time","Yes","No","< 1 Hours","3","3","3","3","3","3","3","2","3","5","4","4","2","6","5","1","7"
    "Yes, I agree to participate,","Chittagong University ","Law","Freshman (Undergrad)","Female","> 25","No","No","3.01-3.50","No","No","Yes","Never occurs","Yes","Never","Yes","Yes","1-3 Hours","2","3","3","3","2","2","2","1","6","3","4","7","2","6","5","1","5"
    "Yes, I agree to participate,","University of Dhaka ","BBA","Junior (Undergrad)","Female","21-25","Yes","No","> 3.50","No","Yes","No","Never occurs","Yes","Never","No","Maybe","1-3 Hours","1","2","0","1","1","1","2","2","6","5","7","6","5","7","5","7","1"
    "Yes, I agree to participate,","UNIVERSITY OF CHITTAGONG ","L.L.B","Freshman (Undergrad)","Male","21-25","Yes","No","> 3.50","No","Yes","No","Sometimes","Yes","Rarely","Yes","Yes","> 3 Hours","2","3","2","3","2","2","3","2","6","7","2","6","5","3","5","1","7"
    "Yes, I agree to participate,","Chittagong University ","LL.B.","Senior (Undergrad)","Female","21-25","Yes","No","> 3.50","No","No","No","Sometimes","No","Never","Yes","No","> 3 Hours","3","1","2","2","2","3","3","2","4","7","5","7","5","3","5","5","5"
    "Yes, I agree to participate,","International Islamic University Chittagong","ELL","Graduate student","Male","21-25","No","Yes","> 3.50","No","No","No","Sometimes","Yes","Never","No","No","1-3 Hours","2","1","3","3","1","1","3","7","5","7","5","3","1","7","5","4","5"
    "Yes, I agree to participate,","Abdul Malek Ukil Medical College,Noakhali","MBBS","Junior (Undergrad)","Female","21-25","Yes","No","3.01-3.50","No","Yes","No","Sometimes","Yes","Never","No","No","> 3 Hours","1","3","3","3","3","1","3","5","5","0","5","3","6","6","5","2","1"
    "Yes, I agree to participate,","Port City International University","B.sc in EEE","Senior (Undergrad)","Male","21-25","Yes","No","3.01-3.50","No","No","Yes","Never occurs","Yes","Never","Yes","Maybe","1-3 Hours","2","0","0","2","1","3","3","2","6","7","1","7","5","5","5","1","4"
    "Yes, I agree to participate,","Abdul malek ukil medical college","Medicine ","Senior (Undergrad)","Female","21-25","Yes","Yes","3.01-3.50","No","No","Yes","Never occurs","No","Rarely","No","Yes","> 3 Hours","1","3","3","3","1","3","3","5","6","7","6","6","1","7","5","1","3"
    "Yes, I agree to participate,","হাজী সোনামিয়া ডিগ্রী কলেজ","Pornograpy and Culture ","Freshman (Undergrad)","Male","21-25","No","Yes","> 3.50","Yes","No","Yes","Most of the time","Yes","Most of the time","Yes","No","> 3 Hours","0","0","0","0","0","0","0","4","4","6","5","5","5","7","5","7","2"
    "Yes, I agree to participate,","Dhaka University ","BA honours ","Freshman (Undergrad)","Male","21-25","No","Yes","3.01-3.50","No","No","No","Never occurs","No","Never","Yes","No","1-3 Hours","0","0","2","0","0","0","0","3","5","5","1","7","5","5","5","4","3"
    "Yes, I agree to participate,","Dhaka University ","B.A Honours","Junior (Undergrad)","Female","21-25","No","No","< 3.00","No","Yes","No","Never occurs","No","Never","Yes","Yes","< 1 Hours","2","1","2","3","3","3","3","5","5","5","5","7","4","6","5","6","2"
    "Yes, I agree to participate,","National University ","MSS","Graduate student","Male","21-25","Yes","No","< 3.00","Yes","No","No","Sometimes","Yes","Never","No","No","> 3 Hours","1","0","2","2","1","2","0","2","5","6","2","0","4","2","5","3","3"
    "Yes, I agree to participate,","Kapasgola city corporation mohila college","Management ","Graduate student","Female","21-25","Yes","No","3.01-3.50","No","Yes","No","Sometimes","No","Rarely","No","No","1-3 Hours","0","0","2","2","2","2","2","5","4","3","2","7","2","5","5","2","1"
    "Yes, I agree to participate,","University of Dhaka","English","Graduate student","Female","21-25","Yes","No","3.01-3.50","No","No","No","Never occurs","No","Rarely","No","No","> 3 Hours","3","2","2","3","3","3","3","4","5","5","5","7","4","3","5","4","1"
    "Yes, I agree to participate,","Port city international University ","BBA ","Senior (Undergrad)","Male","21-25","Yes","Yes","3.01-3.50","No","Yes","No","Never occurs","Yes","Never","Yes","No","> 3 Hours","0","2","3","3","0","3","1","3","5","3","5","7","5","6","5","4","2"
    "Yes, I agree to participate,","abdul malek ukil medical college","mbbs","Graduate student","Female","21-25","Yes","Yes","< 3.00","No","No","No","Sometimes","Yes","Rarely","Yes","Yes","> 3 Hours","0","1","0","1","0","1","1","6","2","7","2","7","3","5","5","6","7"
    "Yes, I agree to participate,","East Delta university ","EEE","Senior (Undergrad)","Female","21-25","Yes","No","< 3.00","No","Yes","No","Sometimes","Yes","Rarely","Yes","Yes","> 3 Hours","2","1","1","2","2","3","2","6","4","6","5","7","3","7","5","5","2"
    "Yes, I agree to participate,","Port City International University ","B.A. in English ","Graduate student","Female","< 20","No","Yes","3.01-3.50","No","No","No","Sometimes","No","Never","No","No","1-3 Hours","1","0","0","1","0","0","0","3","4","1","7","5","5","3","5","1","6"
    "Yes, I agree to participate,","Amumc","Mbbs","Senior (Undergrad)","Female","21-25","No","No","3.01-3.50","No","No","No","Sometimes","Yes","Never","Yes","No","> 3 Hours","3","3","3","3","3","3","3","0","0","5","0","0","0","4","5","6","0"
    "Yes, I agree to participate,","BGC Trust University Bangladesh ","CSE ","Senior (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","No","No","Never occurs","Yes","Most of the time","Yes","No","> 3 Hours","0","3","0","0","0","3","0","7","0","7","5","7","0","7","5","7","0"
    "Yes, I agree to participate,","Port City International University ","MA in English ","Graduate student","Female","21-25","No","No","< 3.00","No","No","No","Sometimes","Yes","Most of the time","Yes","Yes","> 3 Hours","3","3","3","3","3","3","3","7","0","7","7","0","7","7","5","0","6"
    "Yes, I agree to participate,","Port City International University ","B.A. in English ","Graduate student","Male","> 25","Yes","No","3.01-3.50","No","No","Yes","Never occurs","Yes","Rarely","No","No","> 3 Hours","2","1","2","3","2","3","2","2","2","3","4","3","2","3","5","2","2"
    "Yes, I agree to participate,","Bandarban gov.College ","Bss honors 2nd year(Economics) ","Graduate student","Male","< 20","No","No","3.01-3.50","No","No","No","Never occurs","Yes","Never","No","No","> 3 Hours","0","0","0","1","0","0","1","6","1","6","1","6","5","7","5","7","1"
    "Yes, I agree to participate,","Southeast University ","M.A in English ","Senior (Undergrad)","Female","> 25","Yes","No","3.01-3.50","No","No","Yes","Sometimes","Yes","Rarely","No","No","> 3 Hours","1","3","3","2","2","2","3","6","0","4","0","6","6","5","5","5","4"
    "Yes, I agree to participate,","Govt.Hazi Mohammad Mohsin College ","B.Sc honours at Department of Physics ","Sophomore (Undergrad)","Female","< 20","Yes","No","< 3.00","No","No","No","Sometimes","No","Rarely","Yes","Yes","> 3 Hours","1","1","2","2","2","1","3","7","0","3","1","7","1","7","5","7","0"
    "Yes, I agree to participate,","CU","history","Graduate student","Male","21-25","Yes","Yes","< 3.00","No","No","No","Sometimes","Yes","Never","No","No","1-3 Hours","3","1","2","3","3","3","3","6","1","4","1","6","4","6","5","5","6"
    "Yes, I agree to participate,","Abdul malek ukil medical College ","Mbbs","Junior (Undergrad)","Female","21-25","No","No","3.01-3.50","No","Yes","No","Sometimes","Yes","Never","Yes","No","> 3 Hours","0","1","3","2","2","1","3","6","6","6","7","7","7","7","5","0","0"
    "Yes, I agree to participate,","East Delta University ","CSE","Senior (Undergrad)","Female","21-25","No","Yes","3.01-3.50","No","No","No","Sometimes","No","Never","Yes","Yes","> 3 Hours","3","3","3","3","3","3","3","7","1","6","1","6","6","5","5","6","6"
    "Yes, I agree to participate,","University of Chittagong","Honours","Graduate student","Female","21-25","No","No","> 3.50","No","Yes","No","Never occurs","No","Never","No","No","> 3 Hours","1","2","1","2","1","3","3","1","5","4","7","7","7","7","5","1","1"
    "Yes, I agree to participate,","East Delta university ","CSE","Senior (Undergrad)","Male","21-25","No","No","3.01-3.50","No","No","No","Never occurs","Yes","Never","Yes","Yes","> 3 Hours","2","1","2","3","3","3","1","6","5","2","7","5","4","7","5","1","5"
    "Yes, I agree to participate,","Shamsun Nahar khan Nursing college,  Chattogram ","Bachelor of science  in Nursing ","Freshman (Undergrad)","Female","21-25","Yes","Yes","> 3.50","No","No","No","Sometimes","Yes","Rarely","Yes","No","> 3 Hours","3","3","3","3","3","2","3","2","5","7","2","5","6","5","5","3","2"
    "Yes, I agree to participate,","University of Chittagong","History","Graduate student","Female","21-25","No","No","> 3.50","No","No","No","Sometimes","Yes","Often","Yes","Yes","< 1 Hours","3","3","2","3","3","3","3","1","1","3","4","6","4","6","5","6","1"
    "Yes, I agree to participate,","East Delta University","CSE","Senior (Undergrad)","Male","> 25","Yes","No","< 3.00","No","No","No","Sometimes","No","Never","Yes","No","< 1 Hours","1","0","0","1","1","2","0","2","5","6","6","5","5","5","5","3","3"
    "Yes, I agree to participate,","Progoti Nursing college ","Bsc. In Nursing ","Freshman (Undergrad)","Male","> 25","Yes","Yes","3.01-3.50","No","No","Yes","Sometimes","No","Never","No","No","> 3 Hours","2","3","2","2","2","2","1","2","1","6","1","7","7","7","5","6","7"
    "Yes, I agree to participate,","Red crisent ","Diploma in nursing science and midwifery ","Freshman (Undergrad)","Female","21-25","No","No","3.01-3.50","No","No","No","Sometimes","No","Never","Yes","No","> 3 Hours","0","2","2","3","3","1","3","5","4","3","2","7","2","5","5","2","1"
    """

    data_io = io.StringIO(raw_text_data)
    df = pd.read_csv(
    data_io,
    engine="python",  # Helps handle inconsistent CSV formatting

    sep=",",
    quotechar='"',
    skipinitialspace=True
)

    anxiety_keywords = [
        "wind down", "over-react", "nervous energy", "agitated",
        "difficult to relax", "intolerant", "touchy"
    ]

# Automatically rename anxiety-related questions to simpler labels (Q1–Q7).
    rename_map = {} 
    q_counter = 1

    for col in df.columns:
        for keyword in anxiety_keywords:
            if keyword in col:
                rename_map[col] = f'Q{q_counter}'
                q_counter += 1
                break

    df.rename(columns=rename_map, inplace=True)
    required_cols = [f'Q{i}' for i in range(1, 8)]

# Convert anxiety question responses to numeric values. Missing or invalid entries are replaced with 0.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Total_Symptom_Score'] = df[required_cols].sum(axis=1)

# Function to classify total anxiety score into severity levels.
    def get_severity_corrected(score):
        if score <= 7:
            return 'Normal'
        elif score <= 10:
            return 'Mild'
        elif score <= 14:
            return 'Moderate'
        else:
            return 'Severe'

    df['Anxiety_Level'] = df['Total_Symptom_Score'].apply(get_severity_corrected)

    return df, required_cols


# Run data loading
df, required_cols = load_data()


# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────      
# DEFINING THE RULE-BASED ENGINE
# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────
'''Rule-based logic used to assign students into meaningful risk profiles.'''

def define_detailed_profiles(row):
    """
    Assigns a student risk profile based on
    anxiety level and related life stress factors.
    """
    # Thresholds
    SCORE_THRESHOLD = 10
    SOCIAL_MEDIA_THRESHOLD = '> 3 Hours'

    # Column mappings (normalized)
    acad_col = 'Are you happy with your academic Condition?'
    sm_col = 'How many hours do you spend on social media?'
    fin_col = 'Do you have financial problem in your family?'
    conflict_col = 'How often do you conflict with your friend?'
    bullied_col = 'Have you ever been bullied'
    breakup_col = 'Did you have a recent breakup?'
    violence_col = 'Violence in family?'

    # Logic
    if row['Total_Symptom_Score'] <= SCORE_THRESHOLD:
        return 'Low Risk (Healthy) Group'

    if row.get(acad_col) == 'No':
        return 'Academic Risk Group'

    if row.get(fin_col) == 'Yes':
        return 'Financial Stress Group'

    if row.get(bullied_col) == 'Yes' or (row.get(violence_col) not in ['Never', 'Rarely']):
        return 'Trauma/Bullying Related Group'

    # Safe checking for strings
    conflict_val = str(row.get(conflict_col, ''))
    if ('Most' in conflict_val or 'Often' in conflict_val) or row.get(breakup_col) == 'Yes':
        return 'Relationship Anxiety Group'

    if row.get(sm_col) == SOCIAL_MEDIA_THRESHOLD:
        return 'Digital Stress Group'

    return 'High Risk (Unclassified)'


# Applying the advanced model to the dataframe
df['Student_Profile'] = df.apply(define_detailed_profiles, axis=1)

# Recommendation Logic Dictionary
recommendation_logic = {
    "Academic Risk Group": {
        "focus": "Academic Dissatisfaction",
        "intervention": "1. Academic Counseling referral.\n2. Time management workshops.\n3. Peer-tutoring programs."
    },
    "Financial Stress Group": {
        "focus": "Financial Insecurity",
        "intervention": "1. Guidance to scholarship office.\n2. Part-time job placement support.\n3. Stress management specifically for uncertainty."
    },
    "Trauma/Bullying Related Group": {
        "focus": "Past Trauma / Bullying",
        "intervention": "1. URGENT referral to Psychological Services (CAPS).\n2. Safe space/Support group participation.\n3. Conflict resolution training."
    },
    "Relationship Anxiety Group": {
        "focus": "Social Conflicts / Breakup",
        "intervention": "1. Relationship counseling.\n2. Social skills training.\n3. Group therapy for emotional regulation."
    },
    "Digital Stress Group": {
        "focus": "Excessive Screen Time",
        "intervention": "1. Digital Detox program.\n2. Sleep hygiene education (screen-free bedroom).\n3. Mindfulness apps usage."
    },
    "Low Risk (Healthy) Group": {
        "focus": "Prevention",
        "intervention": "1. General wellbeing monitoring.\n2. Participation in social clubs to maintain health."
    },
    "High Risk (Unclassified)": {
        "focus": "Unknown High Anxiety",
        "intervention": "1. One-on-one deep interview required to identify root cause."
    }
}

# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ───
# STREAMLIT INTERACTIVE INTERFACE
# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ───

# Sidebar menu for switching between analysis and counselor tools.
st.sidebar.title("System Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Analysis Report (Static)", "Counselor Tool (Interactive)"])

if app_mode == "Analysis Report (Static)":
    st.title("Student Anxiety Analysis Report")

    # Data Dictionary Display
    with st.expander("📂 View Data Dictionary"):
        data_dict = {
            "Variable Name (Column)": [
                "Total_Symptom_Score", "Anxiety_Level", "Student_Profile", "Q1 - Q7",
                "Are you happy with your academic Condition?", "Do you have financial problem in your family?",
                "How many hours do you spend on social media?", "Living with family?", "Violence in family? / Bullied",
                "Predicted_Score_by_Group"
            ],
            "Description": [
                "Derived Sum: Sum of 7 anxiety-related questions (Range: 0-21).",
                "Derived Category: Clinical severity level (Normal, Mild, Moderate, Severe).",
                "Model Output: Segment assigned by Rule-Based algorithm (e.g., Academic Risk).",
                "Raw Data: Anxiety sub-scale questions from DASS-21.",
                "Risk Factor: Student satisfaction with academic performance.",
                "Risk Factor: Presence of financial difficulties in the family.",
                "Risk Factor: Daily social media usage duration (>3 hours considered risk).",
                "Demographic: Living situation (with family or not).",
                "Trauma Indicators: History of family violence or bullying.",
                "Evaluation Metric: Group mean score used for R-Squared calculation."
            ]
        }
        st.table(pd.DataFrame(data_dict))

    st.markdown("### Data-Driven Insights & Validation Metrics")

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    avg_score = df['Total_Symptom_Score'].mean()
    high_risk_count = df[df['Student_Profile'] != 'Low Risk (Healthy) Group'].shape[0]

    col1.metric("Average Anxiety Score", f"{avg_score:.2f}")
    col2.metric("Total Students", f"{len(df)}")
    col3.metric("High Risk Students", f"{high_risk_count}")

    # Evaluation Phase (R-Squared & Silhouette) inside Streamlit
    st.divider()
    st.subheader("Model Evaluation Metrics")

    # Calculate R2 dynamically
    group_means = df.groupby('Student_Profile')['Total_Symptom_Score'].mean()
    df['Predicted_Score'] = df['Student_Profile'].map(group_means)
    r2 = r2_score(df['Total_Symptom_Score'], df['Predicted_Score'])

    st.info(f"**R-Squared Score:** {r2:.3f}")
    st.caption(
        "This indicates how well our rule-based profiles explain the variance in anxiety scores. > 0.5 is considered strong for behavioral data.")

    # Visualizations
    st.divider()
    st.subheader("Student Profile Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    order_profiles = df['Student_Profile'].value_counts().index
    sns.countplot(y='Student_Profile', data=df, order=order_profiles, palette='Set2', edgecolor='black', ax=ax)
    plt.title('Distribution of Student Risk Profiles')
    st.pyplot(fig)

    st.subheader("Anxiety Score Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Total_Symptom_Score'], kde=True, color="skyblue", bins=10, ax=ax2)
    plt.title('Overall Anxiety Score Distribution')
    st.pyplot(fig2)

elif app_mode == "Counselor Tool (Interactive)":
    st.title("🎓 Guidance Counselor Decision Support System")
    st.markdown("""
    **Instructions:** Enter the student's survey responses below. 
    The system will automatically calculate the Anxiety Score and assign a Risk Profile.
    """)

    with st.form("student_assessment_form"):
        st.subheader("1. DASS-21 Anxiety Assessment")
        st.caption("Rate the student's symptoms (0: Did not apply to me, 3: Applied very much)")

        c1, c2 = st.columns(2)
        with c1:
            q1 = st.slider("Q1. Hard to wind down", 0, 3, 0)
            q2 = st.slider("Q2. Over-react to situations", 0, 3, 0)
            q3 = st.slider("Q3. Nervous energy", 0, 3, 0)
            q4 = st.slider("Q4. Getting agitated", 0, 3, 0)
        with c2:
            q5 = st.slider("Q5. Difficult to relax", 0, 3, 0)
            q6 = st.slider("Q6. Intolerant of interruptions", 0, 3, 0)
            q7 = st.slider("Q7. Rather touchy", 0, 3, 0)

        st.divider()
        st.subheader("2. Risk Factors & History")

        col_a, col_b = st.columns(2)

        with col_a:
            academic_happy = st.selectbox("Happy with Academic Condition?", ["Yes", "No"])
            financial_prob = st.selectbox("Financial Problems in Family?", ["No", "Yes"])
            social_media = st.selectbox("Social Media Usage", ["< 1 Hours", "1-3 Hours", "> 3 Hours"])

        with col_b:
            violence = st.selectbox("Violence in Family?", ["Never", "Rarely", "Sometimes", "Often"])
            bullied = st.selectbox("Ever been bullied?", ["No", "Yes"])
            conflict = st.selectbox("Conflict with Friends?",
                                    ["Never occurs", "Sometimes", "Often", "Most of the time"])
            breakup = st.selectbox("Recent Breakup?", ["No", "Yes"])

        submitted = st.form_submit_button("Assess Student Risk")

        if submitted:
            # Calculate Score
            total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7

            # Create a temporary dictionary for the model
            input_data = {
                'Total_Symptom_Score': total_score,
                'Are you happy with your academic Condition?': academic_happy,
                'How many hours do you spend on social media?': social_media,
                'Do you have financial problem in your family?': financial_prob,
                'Violence in family?': violence,
                'Have you ever been bullied': bullied,
                'How often do you conflict with your friend?': conflict,
                'Did you have a recent breakup?': breakup
            }

            # Run the Rule-Based Model
            risk_profile = define_detailed_profiles(input_data)

            # Display Results
            st.divider()
            st.markdown(f"### Assessment Result")

            # Color Coding
            if "Low Risk" in risk_profile:
                st.success(f"**Profile:** {risk_profile}")
            elif "High Risk" in risk_profile or "Trauma" in risk_profile:
                st.error(f"**Profile:** {risk_profile}")
            else:
                st.warning(f"**Profile:** {risk_profile}")

            st.metric("Total Anxiety Score", f"{total_score} / 21")

            # Show Recommendation
            rec = recommendation_logic.get(risk_profile, {"intervention": "Contact Supervisor."})

            st.info("📋 **Recommended Action Plan:**")
            st.write(rec['intervention'])

            st.markdown(f"**Focus Area:** {rec.get('focus', 'General')}")

# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# STATIC SCRIPT EXECUTION (Legacy Support)
# ────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──

# This part ensures that if you run the script normally (without streamlit), it still generates PNGs
if __name__ == "__main__" and not 'streamlit' in str(os.environ):
    # Fallback check for streamlit context
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx():
            pass  # Running in Streamlit, do nothing here
    except:
        # Running in standard Python CLI
        print("Running in CLI mode (Generating PNGs)...")
        # Generate charts for report
        plt.figure(figsize=(10, 6))
        order_profiles = df['Student_Profile'].value_counts().index
        sns.countplot(y='Student_Profile', data=df, order=order_profiles, palette='Set2', edgecolor='black')
        plt.title('Advanced Rule-Based Student Profiles')
        plt.tight_layout()
        plt.savefig('advanced_model_output.png')
        print(">> 'advanced_model_output.png' created.")

        # Calculate R2 for console output
        group_means = df.groupby('Student_Profile')['Total_Symptom_Score'].mean()
        df['Predicted_Score'] = df['Student_Profile'].map(group_means)
        r2 = r2_score(df['Total_Symptom_Score'], df['Predicted_Score'])
        print(f">> Model R-Squared: {r2:.3f}")

        print("\n--- FINAL PERSONALIZED RECOMMENDATIONS (DEVELOPMENT OUTPUT) ---\n")
        # Get top symptoms logic from previous files if needed for console output
        dass_items = required_cols
        group_symptom_profile = df.groupby('Student_Profile')[dass_items].mean().T
        top_symptoms_per_group = {
            group: group_symptom_profile[group].sort_values(ascending=False).head(3)
            for group in group_symptom_profile.columns
        }

        for group, details in recommendation_logic.items():
            if group in df['Student_Profile'].values:
                print(f"TARGET PROFILE: {group}")
                print(f"Main Stressor: {details['focus']}")
                if group in top_symptoms_per_group:
                    top_3 = top_symptoms_per_group[group].index.tolist()
                    print(f"Dominant Symptoms: {', '.join(top_3)}")
                print("ACTION PLAN:")
                print(details['intervention'])
                print("-" * 60)

        print("Script completed successfully.")

