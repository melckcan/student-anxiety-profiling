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

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# DATA UPLOADING AND PREPROCESSING
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──

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

# Converting string data to CSV format and uploading it to our DataFrame
data_io = io.StringIO(raw_text_data)
df = pd.read_csv(data_io)

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# DEFINING AND RENAMING ANXIETY QUESTIONS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──


anxiety_keywords = [
    "wind down", "over-react", "nervous energy", "agitated",
    "difficult to relax", "intolerant", "touchy"
]

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

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# DATA CLEANING AND HANDLING MISSING DATA
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# CALCULATING ANXIETY SCORES
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──

df['Total_Symptom_Score'] = df[required_cols].sum(axis=1)

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# CLASSIFICATION OF ANXIETY LEVELS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──

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

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
# DATA VISUALIZATION: ANXIETY DISTRIBUTION AND LEVELS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

order_levels = ['Normal', 'Mild', 'Moderate', 'Severe']

#Adjusting the seaborn visual style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 6))

# FIRST GRAPH - DISTRIBUTION OF ANXIETY SCORES
plt.subplot(1, 2, 1)
#Histogram
sns.histplot(df['Total_Symptom_Score'], kde=False, stat="density",
             color="skyblue", label="Student Scores", bins=10)

mu, std = norm.fit(df['Total_Symptom_Score'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label=f'Normal Dist. (μ={mu:.2f}, σ={std:.2f})')

plt.title('Distribution of Anxiety Scores (Corrected)', fontsize=14)
plt.xlabel('Total Symptom Score (0-21)', fontsize=12)
plt.legend()

# SECOND GRAPH - DISTRIBUTION OF ANXIETY LEVELS
plt.subplot(1, 2, 2)
ax = sns.countplot(x='Anxiety_Level', hue='Anxiety_Level', data=df, order=order_levels, palette="viridis", legend=False)
total = len(df)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        percentage = '{:.1f}%'.format(100 * height / total)
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                f'{int(height)}\n({percentage})',
                ha="center", fontsize=11, color='black')

plt.title('Student Anxiety Levels (Corrected Logic)', fontsize=14)
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.tight_layout()
plt.show()

print("EDA Analysis has been completed.")

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
# IDENTIFYING AND DETECTING RISK FACTORS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──
risk_factors = [
    "Are you happy with your academic Condition?",
    "Do you have financial problem in your family?",
    "How many hours do you spend on social media?",
    "Living with family?",
    "How often do you conflict with your friend?"
]

found_factors = []
for factor in risk_factors:
    for col in df.columns:
        if factor.lower() in col.lower().strip():
            found_factors.append(col)
            break

print(f"Factors to be analyzed were found: {len(found_factors)} factors.")

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
#VISUALIZING THE EFFECT OF RISK FACTORS ON ANXIETY LEVELS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

plt.figure(figsize=(18, 12))
sns.set(style="whitegrid")

custom_palette = {"Normal": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22", "Severe": "#e74c3c"}

for i, col in enumerate(found_factors[:4]):
    plt.subplot(2, 2, i + 1)

    ct = pd.crosstab(df[col], df['Anxiety_Level'], normalize='index') * 100
    ct = ct[['Normal', 'Mild', 'Moderate', 'Severe']]

    ct.plot(kind='bar', stacked=True, ax=plt.gca(), color=[custom_palette[x] for x in ct.columns], width=0.7,
            edgecolor='black')

    plt.title(f'Affecting Factor: {col.split("?")[0]}', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage(%)')
    plt.xlabel('')
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Anxiety Level', bbox_to_anchor=(1, 1), loc='upper left')

plt.tight_layout()
plt.show()

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
# DETAILED RISK ANALYSIS: AVERAGE SCORES
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

print("\n--- DETAILED RISK ANALYSIS RESULTS (AVERAGE SCORES) ---")
for col in found_factors:
    print(f"\n[{col}]")
    print(df.groupby(col)['Total_Symptom_Score'].mean().sort_values(ascending=False))

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
# ADVANCED RULE-BASED SEGMENTATION: STUDENT PROFILES
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

print("\n--- DEVELOPMENT PHASE: ADVANCED RULE-BASED SEGMENTATION ---")
print("Implementing hierarchical logic (Academic, Financial, Trauma, Relationship, Digital)...")


def define_detailed_profiles(row):
    
    SCORE_THRESHOLD = 10
    SOCIAL_MEDIA_THRESHOLD = '> 3 Hours'

   
    acad_col = 'Are you happy with your academic Condition?'
    sm_col = 'How many hours do you spend on social media?'
    fin_col = 'Do you have financial problem in your family?'
    conflict_col = 'How often do you conflict with your friend?'
    bullied_col = 'Have you ever been bullied'
    breakup_col = 'Did you have a recent breakup?'
    violence_col = 'Violence in family?'

    
    if row['Total_Symptom_Score'] <= SCORE_THRESHOLD:
        return 'Low Risk (Healthy) Group'

    
    if row[acad_col] == 'No':
        return 'Academic Risk Group'

    
    if row[fin_col] == 'Yes':
        return 'Financial Stress Group'

    
    if row[bullied_col] == 'Yes' or (row[violence_col] != 'Never' and row[violence_col] != 'Rarely'):
        return 'Trauma/Bullying Related Group'

    
    if ('Most' in str(row[conflict_col]) or 'Often' in str(row[conflict_col])) or row[breakup_col] == 'Yes':
        return 'Relationship Anxiety Group'

    
    if row[sm_col] == SOCIAL_MEDIA_THRESHOLD:
        return 'Digital Stress Group'

   
    return 'High Risk (Unclassified)'



df['Student_Profile'] = df.apply(define_detailed_profiles, axis=1)
risk_group_col = 'Student_Profile'
dass_items = required_cols

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─
# VISUALIZATON OF STUDENT PROFILES
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─

plt.figure(figsize=(12, 7))
order_profiles = df['Student_Profile'].value_counts().index
sns.countplot(y='Student_Profile', data=df, order=order_profiles, palette='Set2', edgecolor='black')
plt.title('Advanced Rule-Based Student Profiles (Development Phase)', fontsize=14)
plt.xlabel('Number of Students')
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('advanced_model_output.png')
print(">> 'advanced_model_output.png' has been created.")

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─
# PROFILE STATISTICS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─

print("\n--- NEW STUDENT PROFILES DISTRIBUTION ---")
print(df['Student_Profile'].value_counts())

print("\n--- AVERAGE ANXIETY SCORE PER GROUP ---")
print(df.groupby('Student_Profile')['Total_Symptom_Score'].mean().round(2))

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─
# IDENTIFYING THE DOMINANT SYMPTOMS
#          IN EACH GROUP
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ─

#Average symptom scores for each group
group_symptom_profile = df.groupby(risk_group_col)[dass_items].mean().T
#Find the top 3 symptoms for each group
top_symptoms_per_group = {
    group: group_symptom_profile[group].sort_values(ascending=False).head(3)
    for group in group_symptom_profile.columns
}


#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
# NETWORK ANALYSIS: CORRELATION BETWEEN SYMPTOMS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

#Correlation matrix
corr_matrix = df[dass_items].corr()

#Create the NetworkX graph
G = nx.Graph()
# Add strong correlations (>0.4) as graph edges
for i in corr_matrix.columns:
    for j in corr_matrix.columns:
        if i != j and abs(corr_matrix.loc[i, j]) >= 0.4:
            G.add_edge(i, j, weight=corr_matrix.loc[i, j])

print(f"\nGraph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

#Centrality measures: Which symptoms are most connected?
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
centrality_df = pd.DataFrame({
    "degree": degree_centrality,
    "betweenness": betweenness_centrality
}).sort_values("degree", ascending=False)

print("\n--- CENTRAL SYMPTOMS IN THE NETWORK ---")
print(centrality_df.round(3))

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───
# PERSONALIZED INTERVENTION RECOMMENDATIONS
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────୨ৎ───

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

print("\n--- FINAL PERSONALIZED RECOMMENDATIONS (DEVELOPMENT OUTPUT) ---\n")

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


#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────
# MODEL EVALUATION: SILHOUETTE SCORE
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ────

print("\n--- EVALUATION PHASE: SILHOUETTE & SENSITIVITY ---")


eval_df = df.copy()

#Convert categoric values to numbers
le = LabelEncoder()

feature_cols = [
    'Total_Symptom_Score',
    'Are you happy with your academic Condition?',
    'Do you have financial problem in your family?',
    'How many hours do you spend on social media?'
]

#Convert categorical columns to numeric
eval_df['Academic_Encoded'] = le.fit_transform(eval_df['Are you happy with your academic Condition?'])
eval_df['Financial_Encoded'] = le.fit_transform(eval_df['Do you have financial problem in your family?'])
eval_df['SocialMedia_Encoded'] = le.fit_transform(eval_df['How many hours do you spend on social media?'])

#Create feature matrix
X_eval = eval_df[['Total_Symptom_Score', 'Academic_Encoded', 'Financial_Encoded', 'SocialMedia_Encoded']]
labels = df['Student_Profile']

#Calculate silhouette score
#By this metric, we are measuring how well groups are separeted from each other
#Close to 1 = perfect separation, close to 0 = groups are mixed together
if len(labels.unique()) > 1:
    overall_score = silhouette_score(X_eval, labels)
    print(f"\n>> Model Silhouette Score: {overall_score:.3f}")

    if overall_score > 0.4:
        print("   (Interpretation: Groups are very well separated.)")
    elif overall_score > 0.2:
        print("   (Interpretation: Reasonable separation between groups.)")
    else:
        print("   (Interpretation: Groups share overlapping features - typical for rule-based logic.)")
else:
    print("Silhouette score could not be calculated (insufficient groups).")


#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────────୨ৎ──────────
# SENSITIVITY ANALYSIS: THE EFFECT OF THE THRESHOLD VALUE
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────────୨ৎ──────────

print("\n--- SENSITIVITY ANALYSIS (Threshold Stability) ---")


def run_scenario(threshold_val):
    """
    Recalculates profile distribution based on a new threshold.
    """
    temp_counts = {
        'Low Risk (Healthy) Group': 0,
        'High Risk (Risk Groups)': 0
    }

    for _, row in df.iterrows():
        
        if row['Total_Symptom_Score'] <= threshold_val:
            temp_counts['Low Risk (Healthy) Group'] += 1
        else:
            temp_counts['High Risk (Risk Groups)'] += 1

    return temp_counts



thresholds = range(7, 16)
sensitivity_results = []

for t in thresholds:
    res = run_scenario(t)
    total_students = sum(res.values())
    risk_ratio = (res['High Risk (Risk Groups)'] / total_students) * 100
    sensitivity_results.append({
        'Threshold': t,
        'Risk_Percentage': risk_ratio
    })

sens_df = pd.DataFrame(sensitivity_results)


plt.figure(figsize=(10, 6))
sns.lineplot(x='Threshold', y='Risk_Percentage', data=sens_df, marker='o', color='crimson', linewidth=2.5)

current_risk_ratio = sens_df[sens_df['Threshold'] == 10]['Risk_Percentage'].values[0]
plt.axvline(x=10, color='navy', linestyle='--', label=f'Current Threshold (10) -> {current_risk_ratio:.1f}% Risk')

plt.title('Sensitivity Analysis: Impact of Threshold on Risk Classification', fontsize=14)
plt.xlabel('Anxiety Score Threshold (Cut-off Point)', fontsize=12)
plt.ylabel('Percentage of Students in "High Risk" Group (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sensitivity_analysis.png')
print(">> 'sensitivity_analysis.png' has been created.")

#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────────୨ৎ──────────
# R² ANALYSIS: VARIANCE EXPLANATION
#────୨ৎ────────୨ৎ────────୨ৎ────────୨ৎ──────────୨ৎ──────────

print("\n--- R-SQUARED (R²) ANALYSIS (Variance Explanation) ---")

group_means = df.groupby('Student_Profile')['Total_Symptom_Score'].mean()

df['Predicted_Score_by_Group'] = df['Student_Profile'].map(group_means)


y_true = df['Total_Symptom_Score']
y_pred = df['Predicted_Score_by_Group']

r2 = r2_score(y_true, y_pred)

print(f">> Model R² Score: {r2:.3f}")
print(f"   (Interpretation: Our profiles explain {r2 * 100:.1f}% of the variance in anxiety scores.)")


if r2 > 0.5:
    print("   -> Strong Effect: The profiles are distinct indicators of anxiety severity.")
elif r2 > 0.3:
    print("   -> Moderate Effect: The profiles explain a significant portion of the stress.")
else:
    print("   -> Weak Effect: Anxiety varies greatly even within the same profile.")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=y_true, hue=df['Student_Profile'], palette='Set2', s=100, alpha=0.7)
plt.plot([0, 21], [0, 21], color='red', linestyle='--', label='Perfect Explanation Line')  # 45 degree line
plt.title(f'Variance Explanation (R² = {r2:.2f})\nActual Score vs. Group Mean', fontsize=14)
plt.xlabel('Predicted Score (Group Average)', fontsize=12)
plt.ylabel('Actual Student Score', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('r2_variance_analysis.png')
print(">> 'r2_variance_analysis.png' has been created.")

print("\nFull Analysis, Development, and Evaluation Completed Successfully.")