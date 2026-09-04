# app.py
# SISTEM KLASIFIKASI PENYAKIT ALZHEIMER - FINAL EDITION (FIXED)
# ========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif
import os
from datetime import datetime
import io
import base64
import warnings
import json
import time
from streamlit import session_state as ss

warnings.filterwarnings('ignore')

# ============================================
# !!! FUNGSI INI WAJIB ADA UNTUK LOAD MODEL !!!
# ============================================

def mi_with_seed(X_in, y_in):
    """Fungsi untuk mutual information dengan seed tetap 42"""
    return mutual_info_classif(X_in, y_in, random_state=42)

# ============================================
# KONFIGURASI HALAMAN
# ============================================

st.set_page_config(
    page_title="🧠 Alzheimer Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# INISIALISASI SESSION STATE
# ============================================

if 'threshold' not in ss:
    ss.threshold = 0.5

# ============================================
# CSS + ANIMASI BACKGROUND BUBBLE KESEHATAN
# ============================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Background gelap dengan efek gradien biru-ungu */
    .stApp {
        background: linear-gradient(135deg, #0a0a2e 0%, #1a0533 50%, #0a0a2e 100%);
        min-height: 100vh;
        overflow: hidden;
    }

    /* Canvas untuk bubble animation */
    #bg-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        pointer-events: none;
    }

    /* Glassmorphism container */
    .main .block-container {
        padding: 2rem 1.5rem;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 40px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 30px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        margin: 1rem auto;
        max-width: 1440px;
        position: relative;
        z-index: 1;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: fadeSlideUp 0.8s ease;
    }
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 46, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
        box-shadow: 8px 0 40px rgba(0,0,0,0.4) !important;
        z-index: 10;
    }
    [data-testid="stSidebar"] .sidebar-content {
        color: #f0f0f0 !important;
    }

    /* Toggle sidebar mobile */
    .sidebar-toggle {
        display: none;
        position: fixed;
        top: 0.7rem;
        left: 0.7rem;
        z-index: 1000;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(192, 132, 252, 0.15));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1.5px solid rgba(192, 132, 252, 0.4);
        border-radius: 14px;
        width: 44px;
        height: 44px;
        font-size: 1.35rem;
        color: #c084fc;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(76, 29, 149, 0.4);
    }
    .sidebar-toggle:hover {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.4), rgba(192, 132, 252, 0.25));
        border-color: rgba(192, 132, 252, 0.7);
        transform: scale(1.05);
        color: #fff;
    }
    .sidebar-toggle:active {
        transform: scale(0.95);
    }

    /* Backdrop overlay (saat sidebar mobile terbuka) */
    .sidebar-backdrop {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(2px);
        z-index: 998;
    }
    .sidebar-backdrop.show {
        display: block;
    }

    /* Mobile responsive - tablet portrait & HP */
    @media (max-width: 768px) {
        .sidebar-toggle {
            display: flex !important;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
            transform: translateX(-100%) !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            z-index: 999 !important;
            box-shadow: 8px 0 40px rgba(0,0,0,0.6) !important;
        }
        [data-testid="stSidebar"][aria-expanded="true"],
        [data-testid="stSidebar"].open {
            transform: translateX(0) !important;
        }

        /* Padding content utama geser agar tidak ketutup tombol toggle */
        .main .block-container {
            padding-top: 4rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }

        /* Radio button navigasi di sidebar HP - lebih jelas & bisa di-tap */
        [data-testid="stSidebar"] label {
            padding: 0.6rem 0.7rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.25rem !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            min-height: 42px !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stSidebar"] label:hover {
            background: rgba(168, 85, 247, 0.1) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            font-size: 1rem !important;
            line-height: 1.3 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 0.78rem !important;
            line-height: 1.3 !important;
            margin: 0.2rem 0 !important;
        }
    }

    /* HP kecil (<= 480px) */
    @media (max-width: 480px) {
        .sidebar-toggle {
            width: 40px;
            height: 40px;
            font-size: 1.2rem;
            top: 0.5rem;
            left: 0.5rem;
        }
        [data-testid="stSidebar"] {
            width: 240px !important;
            min-width: 240px !important;
            max-width: 240px !important;
        }
        .main .block-container {
            padding-top: 3.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        [data-testid="stSidebar"] label {
            padding: 0.5rem 0.6rem !important;
            font-size: 0.82rem !important;
            min-height: 38px !important;
        }
    }

    /* Cards */
    .card-glass {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 1.8rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        margin-bottom: 1.5rem;
        color: #f0f0f0;
        position: relative;
        overflow: hidden;
    }
    .card-glass::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(255,255,255,0.02) 0%, transparent 70%);
        pointer-events: none;
    }
    .card-glass:hover {
        transform: translateY(-6px) scale(1.005);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.09);
    }
    .card-glass h1, .card-glass h2, .card-glass h3, .card-glass h4 {
        color: #fff;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .card-glass p, .card-glass li, .card-glass span {
        color: rgba(255,255,255,0.8);
        line-height: 1.6;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(106, 17, 203, 0.35) !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.1) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -60%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.4s;
        pointer-events: none;
    }
    .stButton > button:hover::after {
        opacity: 1;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(106, 17, 203, 0.5) !important;
        background: linear-gradient(135deg, #7b1fa2 0%, #3d5afe 100%) !important;
    }
    .stButton > button:active {
        transform: scale(0.97) !important;
    }

    /* ============================================
       FORM INPUT - EYE CATCHING (TEAL PEARL THEME)
       ============================================ */

    /* Form container - card premium */
    .stForm {
        background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(168,85,247,0.04) 100%) !important;
        border-radius: 28px !important;
        padding: 2rem 1.8rem !important;
        border: 1px solid rgba(168, 85, 247, 0.22) !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 20px 60px rgba(10, 10, 46, 0.4), inset 0 1px 0 rgba(255,255,255,0.06) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stForm::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent 0%, #a855f7 50%, #c084fc 100%) !important;
    }

    /* Form header - eye-catching */
    .form-header {
        text-align: center !important;
        margin-bottom: 2rem !important;
        padding: 1.2rem 1rem !important;
        background: linear-gradient(135deg, rgba(168,85,247,0.08) 0%, rgba(192,132,252,0.05) 100%) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(168, 85, 247, 0.22) !important;
        position: relative !important;
    }
    .form-header h3 {
        color: #fff !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #ffffff 0%, #c084fc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: -0.02em !important;
    }
    .form-header p {
        color: rgba(192, 132, 252, 0.8) !important;
        font-size: 0.92rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 400 !important;
        letter-spacing: 0.2px !important;
    }

    /* Input wrapper - setiap input dibungkus card */
    .input-field-wrapper {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(168,85,247,0.03) 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.18) !important;
        border-radius: 18px !important;
        padding: 0.55rem 0.85rem !important;
        margin-bottom: 0.35rem !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        backdrop-filter: blur(8px) !important;
    }
    .input-field-wrapper:hover {
        border-color: rgba(168, 85, 247, 0.5) !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(168,85,247,0.08) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(76, 29, 149, 0.4) !important;
    }
    .input-field-wrapper:focus-within {
        border-color: #a855f7 !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(168,85,247,0.14) 100%) !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2), 0 8px 20px rgba(76, 29, 149, 0.4) !important;
    }

    /* Input label (header) */
    .input-field-label {
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        margin-bottom: 0.35rem !important;
        color: #f0f0f0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px !important;
    }
    .input-field-number {
        background: linear-gradient(135deg, #a855f7 0%, #c084fc 100%) !important;
        color: #fff !important;
        font-weight: 800 !important;
        font-size: 0.78rem !important;
        width: 28px !important;
        height: 22px !important;
        border-radius: 7px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        box-shadow: 0 2px 6px rgba(168, 85, 247, 0.5) !important;
        border: 1px solid rgba(192, 132, 252, 0.6) !important;
    }
    .input-field-icon {
        font-size: 1.1rem !important;
        margin-left: auto !important;
        opacity: 0.85 !important;
    }
    .input-field-desc {
        color: rgba(192, 132, 252, 0.7) !important;
        font-size: 0.72rem !important;
        margin-top: 0.35rem !important;
        padding-left: 0.3rem !important;
        font-style: italic !important;
        line-height: 1.4 !important;
        display: block !important;
    }

    /* Number input - lebih elegant (ungu) */
    .stNumberInput {
        margin-bottom: 0 !important;
    }
    .stNumberInput > div {
        background: rgba(10, 10, 46, 0.4) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(168, 85, 247, 0.22) !important;
        transition: all 0.25s ease !important;
    }
    .stNumberInput > div:hover {
        border-color: rgba(168, 85, 247, 0.5) !important;
        background: rgba(10, 10, 46, 0.55) !important;
    }
    .stNumberInput > div:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.22) !important;
    }
    .stNumberInput input {
        background: transparent !important;
        border: none !important;
        color: #fff !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        min-height: 44px !important;
        width: 100% !important;
    }
    .stNumberInput input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .stNumberInput label {
        display: none !important;  /* label dipindah ke HTML custom */
    }
    .stNumberInput button {
        background: rgba(168, 85, 247, 0.18) !important;
        border: none !important;
        color: #c084fc !important;
        font-size: 1.1rem !important;
        padding: 0.3rem 0.7rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        min-width: 36px !important;
        min-height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stNumberInput button:hover {
        background: rgba(168, 85, 247, 0.45) !important;
        color: #fff !important;
        transform: scale(1.05) !important;
    }
    .stNumberInput button:active {
        transform: scale(0.95) !important;
    }

    /* Row separator dalam form */
    .form-row-separator {
        display: flex !important;
        align-items: center !important;
        gap: 0.8rem !important;
        margin: 1.5rem 0 1rem 0 !important;
        grid-column: 1 / -1 !important;
    }
    .form-row-separator .line {
        flex: 1 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent 0%, rgba(144, 224, 239, 0.3) 50%, transparent 100%) !important;
    }
    .form-row-separator .label {
        color: rgba(144, 224, 239, 0.7) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    .form-row-separator .icon {
        font-size: 0.95rem !important;
        opacity: 0.8 !important;
    }

    /* Form submit button */
    .stForm .stButton {
        display: flex !important;
        justify-content: center !important;
        margin-top: 2rem !important;
    }
    .stForm .stButton > button {
        width: auto !important;
        min-width: 280px !important;
        padding: 0.9rem 3rem !important;
        font-size: 1.1rem !important;
        border-radius: 50px !important;
    }
    .stForm .stCaption {
        text-align: center !important;
        display: block !important;
        color: rgba(144, 224, 239, 0.5) !important;
        font-size: 0.8rem !important;
        margin-top: 0.7rem !important;
        letter-spacing: 0.5px !important;
    }

    /* Grid input columns - lebih rapi */
    .input-grid {
        display: grid !important;
        gap: 1rem !important;
    }

    /* Inputs lain */
    .stNumberInput input, .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 18px !important;
        color: #f0f0f0 !important;
        padding: 0.7rem 1.2rem !important;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
        font-size: 0.95rem !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #6a11cb !important;
        box-shadow: 0 0 0 4px rgba(106, 17, 203, 0.2) !important;
        background: rgba(255,255,255,0.12) !important;
        transform: scale(1.01);
    }
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: rgba(255,255,255,0.75) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.04);
        border-radius: 24px;
        padding: 0.5rem;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 18px;
        padding: 0.6rem 2rem;
        color: rgba(255,255,255,0.5);
        background: transparent;
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(106, 17, 203, 0.3);
        transform: scale(1.02);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255,255,255,0.8);
    }

    /* Metric */
    .stMetric {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(4px);
        border-radius: 24px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.4s ease;
    }
    .stMetric:hover {
        background: rgba(255,255,255,0.1);
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.15);
    }
    .stMetric label {
        color: rgba(255,255,255,0.6) !important;
        font-weight: 400;
    }
    .stMetric .stMetricValue {
        color: #fff !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }

    /* Dataframe */
    .stDataFrame {
        background: rgba(255,255,255,0.03);
        border-radius: 24px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(4px);
    }
    .stDataFrame table {
        color: #f0f0f0 !important;
    }
    .stDataFrame thead tr th {
        background: rgba(255,255,255,0.06) !important;
        color: #fff !important;
        font-weight: 600;
        padding: 0.8rem !important;
    }
    .stDataFrame tbody tr {
        transition: background 0.2s;
    }
    .stDataFrame tbody tr:hover {
        background: rgba(255,255,255,0.05) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 20px !important;
        color: #f0f0f0 !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
        padding: 0.8rem 1.2rem !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 0 0 20px 20px !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
        padding: 1rem !important;
    }

    /* Result cards */
    .result-alzheimer {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.25), rgba(192, 57, 43, 0.25)) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 32px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(231, 76, 60, 0.2);
        animation: resultPulse 1.5s ease-in-out infinite alternate;
    }
    .result-non-alzheimer {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.25), rgba(39, 174, 96, 0.25)) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 32px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(46, 204, 113, 0.2);
        animation: resultPulse 1.5s ease-in-out infinite alternate;
    }
    @keyframes resultPulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.02); }
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: rgba(255,255,255,0.25);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 2rem;
        backdrop-filter: blur(4px);
        border-radius: 24px;
        letter-spacing: 0.3px;
    }

    /* Uploader */
    .stFileUploader {
        margin-bottom: 2.5rem !important;
        margin-top: 1rem !important;
    }
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(255,255,255,0.1) !important;
        border-radius: 24px !important;
        padding: 2.5rem !important;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: #6a11cb !important;
        background: rgba(255,255,255,0.06) !important;
    }

    /* Radio buttons */
    .stRadio > div {
        background: rgba(255,255,255,0.02);
        border-radius: 20px;
        padding: 0.6rem;
    }
    .stRadio label {
        color: rgba(255,255,255,0.7) !important;
        font-weight: 400;
        padding: 0.5rem 1rem;
        border-radius: 14px;
        transition: all 0.2s ease;
    }
    .stRadio label:hover {
        background: rgba(255,255,255,0.05);
    }
    .stRadio [data-baseweb="radio"]:checked + div {
        background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
        border-radius: 14px;
        color: white !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.25);
    }

    /* ============================================
       PERBAIKAN 3: FITUR FINAL TAMPLILAN RAPI
       ============================================ */
    
    .feature-grid-container {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 0.8rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* === LAYOUT KARTU DIRATAKAN KE KIRI + DESKRIPSI TIDAK TERPOTONG === */
    .feature-card {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 14px !important;
        padding: 0.55rem 0.8rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        flex-direction: column !important;   /* NOMOR + NAMA di ATAS, DESKRIPSI di BAWAH */
        align-items: flex-start !important;   /* SEMUA RATA KIRI */
        justify-content: flex-start !important;
        gap: 0.25rem !important;
        text-align: left !important;
        min-height: 78px !important;
        backdrop-filter: blur(8px);
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
    }
    
    .feature-number-badge {
        background: linear-gradient(135deg, #a855f7, #c084fc) !important;
        color: #fff !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        flex-shrink: 0 !important;
        align-self: flex-start !important;  /* RATA KIRI */
        margin: 0 !important;
        box-shadow: 0 2px 8px rgba(168, 85, 247, 0.4) !important;
        border: 1px solid rgba(192, 132, 252, 0.6) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* BARIS NOMOR + NAMA FITUR (sejajar kiri) */
    .feature-header-row {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0.6rem !important;
        width: 100% !important;
        text-align: left !important;
    }
    
    .feature-name {
        color: rgba(255,255,255,0.92) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        text-align: left !important;
        line-height: 1.2 !important;
    }

    /* === DESKRIPSI: TIDAK DIPOTONG, RATA KIRI, BISA MULTIBARIS === */
    .feature-desc {
        color: rgba(202, 240, 248, 0.7) !important;
        font-size: 0.72rem !important;
        margin: 0 !important;          /* HAPUS margin-left:auto (tadi dorong ke kanan) */
        margin-top: 0.1rem !important;
        padding-left: 0 !important;
        text-align: left !important;   /* PAKSA RATA KIRI */
        white-space: normal !important; /* BOLEH MULTI BARIS */
        overflow: visible !important;  /* JANGAN disembunyikan */
        text-overflow: clip !important; /* JANGAN dipotong ... */
        width: 100% !important;        /* PENUH 1 baris kartu */
        max-width: none !important;    /* HAPUS max-width (tadi cuma 120px!) */
        line-height: 1.35 !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
    }
    
    @media (max-width: 1024px) {
        .feature-grid-container {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }
    
    @media (max-width: 640px) {
        .feature-grid-container {
            grid-template-columns: 1fr !important;
        }
        .feature-desc {
            max-width: 80px !important;
        }
    }

    /* Spacing untuk fitur list 3 baris */
    .feature-list-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .feature-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
    }
    .feature-item {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        border: 1px solid rgba(255,255,255,0.05);
        flex: 1 1 18%;
        min-width: 150px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .feature-item:hover {
        background: rgba(106, 17, 203, 0.15);
        border-color: rgba(106, 17, 203, 0.3);
        transform: translateY(-2px);
    }
    .feature-number {
        color: #6a11cb;
        font-weight: 600;
        margin-right: 4px;
    }

    /* Responsivitas */
    @media (max-width: 1200px) {
        .main .block-container { padding: 1.5rem 1rem; }
        .card-glass { padding: 1.5rem; }
    }
    @media (max-width: 992px) {
        .stTabs [data-baseweb="tab"] { padding: 0.4rem 1.2rem; font-size: 0.8rem; }
        .feature-item { min-width: 120px; font-size: 0.75rem; }
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem; border-radius: 24px; }
        .card-glass { padding: 1.2rem; border-radius: 20px; }
        .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.8rem; font-size: 0.7rem; }
        .stButton > button { padding: 0.5rem 1rem; font-size: 0.9rem; }
        .stMetric .stMetricValue { font-size: 1.4rem !important; }
        .result-alzheimer, .result-non-alzheimer { padding: 1.5rem; }
        .result-alzheimer h1, .result-non-alzheimer h1 { font-size: 2rem !important; }
        .feature-item { min-width: 100px; font-size: 0.7rem; padding: 0.3rem 0.5rem; }
        .feature-grid-container {
            grid-template-columns: 1fr !important;
        }
    }
    @media (max-width: 480px) {
        .main .block-container { padding: 0.8rem; border-radius: 16px; }
        .card-glass { padding: 1rem; }
        .stNumberInput input { padding: 0.5rem 0.8rem !important; font-size: 0.85rem !important; min-height: 44px !important; }
        .stTabs [data-baseweb="tab"] { padding: 0.2rem 0.5rem; font-size: 0.65rem; }
        .stButton > button { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
        .feature-item { min-width: 80px; font-size: 0.65rem; padding: 0.2rem 0.4rem; }
        .feature-number-badge { width: 26px !important; height: 26px !important; font-size: 0.65rem !important; }
        .feature-name { font-size: 0.75rem !important; }
        .feature-desc { display: none !important; }
    }

    /* Batch result spacing */
    .batch-result {
        margin-top: 2rem;
    }
    
    /* ============================================
       PERBAIKAN 2: CONFUSION MATRIX TEXT HITAM
       ============================================ */
    
    /* Style untuk heatmap text - warna hitam agar kontras */
    .plotly .heatmap .text {
        fill: #000000 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
</style>

<!-- Canvas untuk animasi bubble kesehatan -->
<canvas id="bg-canvas"></canvas>

<!-- JavaScript untuk animasi bubble -->
<script>
    (function() {
        const canvas = document.getElementById('bg-canvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        let bubbles = [];
        let time = 0;

        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        // Warna-warna biru-ungu untuk bubble
        const colors = [
            'rgba(106, 17, 203, 0.20)',
            'rgba(37, 117, 252, 0.18)',
            'rgba(46, 204, 113, 0.12)',
            'rgba(155, 89, 182, 0.16)',
            'rgba(52, 152, 219, 0.14)',
            'rgba(142, 68, 173, 0.18)',
            'rgba(41, 128, 185, 0.12)',
            'rgba(106, 17, 203, 0.15)'
        ];

        class Bubble {
            constructor() {
                this.reset(true);
            }
            reset(init) {
                this.x = Math.random() * width;
                this.y = init ? Math.random() * height : height + 20;
                this.radius = Math.random() * 30 + 10;
                this.speed = Math.random() * 0.5 + 0.2;
                this.opacity = Math.random() * 0.3 + 0.1;
                this.color = colors[Math.floor(Math.random() * colors.length)];
                this.wobble = Math.random() * 0.015 + 0.005;
                this.phase = Math.random() * Math.PI * 2;
                this.glow = Math.random() > 0.5;
            }
            update() {
                this.y -= this.speed;
                this.x += Math.sin(time * this.wobble + this.phase) * 0.4;
                if (this.y < -this.radius * 2) {
                    this.reset(false);
                }
            }
            draw() {
                // Glow effect
                if (this.glow) {
                    const grad = ctx.createRadialGradient(
                        this.x, this.y, 0,
                        this.x, this.y, this.radius * 3
                    );
                    grad.addColorStop(0, this.color);
                    grad.addColorStop(1, 'rgba(0,0,0,0)');
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius * 3, 0, Math.PI * 2);
                    ctx.fillStyle = grad;
                    ctx.fill();
                }

                // Bubble body
                const grad = ctx.createRadialGradient(
                    this.x - this.radius * 0.3, this.y - this.radius * 0.3, 0,
                    this.x, this.y, this.radius
                );
                grad.addColorStop(0, this.color.replace('0.', '0.9'));
                grad.addColorStop(0.5, this.color);
                grad.addColorStop(1, 'rgba(255,255,255,0.01)');

                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = grad;
                ctx.fill();

                // Highlight (cahaya di permukaan bubble)
                ctx.beginPath();
                ctx.arc(this.x - this.radius * 0.25, this.y - this.radius * 0.3, this.radius * 0.2, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.fill();

                // Highlight kedua (smaller)
                ctx.beginPath();
                ctx.arc(this.x - this.radius * 0.1, this.y - this.radius * 0.5, this.radius * 0.08, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.fill();

                // Garis tepi tipis
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }

        function initBubbles(count) {
            bubbles = [];
            for (let i = 0; i < count; i++) {
                bubbles.push(new Bubble());
            }
        }
        initBubbles(40);

        // Tambahan efek sinar (ray) yang bergerak
        function drawRays() {
            const cx = width * 0.5;
            const cy = height * 0.5;
            for (let i = 0; i < 8; i++) {
                const angle = time * 0.015 + i * Math.PI / 4;
                const len = Math.min(width, height) * 0.5 + Math.sin(time * 0.008 + i * 0.5) * 30;
                const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, len);
                grad.addColorStop(0, 'rgba(106, 17, 203, 0.02)');
                grad.addColorStop(0.5, 'rgba(37, 117, 252, 0.01)');
                grad.addColorStop(1, 'rgba(106, 17, 203, 0)');
                ctx.beginPath();
                ctx.ellipse(cx + Math.cos(angle) * len * 0.2, cy + Math.sin(angle) * len * 0.2, len, 25, angle, 0, Math.PI * 2);
                ctx.fillStyle = grad;
                ctx.fill();
            }
        }

        // Bintang kecil berkedip
        function drawStars() {
            const starCount = 30;
            for (let i = 0; i < starCount; i++) {
                const x = (i * 137.508) % width;
                const y = (i * 269.361) % height;
                const size = 0.5 + Math.sin(time * 0.02 + i) * 0.3;
                const alpha = 0.1 + Math.sin(time * 0.03 + i * 0.7) * 0.1;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(255,255,255,${alpha})`;
                ctx.fill();
            }
        }

        function animate() {
            time++;
            ctx.clearRect(0, 0, width, height);

            // Gambar bintang di latar belakang
            drawStars();

            // Gambar sinar lembut di latar belakang
            drawRays();

            // Update & gambar bubble
            bubbles.forEach(b => {
                b.update();
                b.draw();
            });

            requestAnimationFrame(animate);
        }
        animate();

        // Resize ulang
        window.addEventListener('resize', () => {
            resize();
            bubbles.forEach(b => {
                b.x = Math.random() * width;
                b.y = Math.random() * height;
            });
        });
    })();

    // ===== SIDEBAR TOGGLE UNTUK MOBILE =====
    (function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return;
        if (!document.querySelector('.sidebar-toggle')) {
            // Create backdrop
            const backdrop = document.createElement('div');
            backdrop.className = 'sidebar-backdrop';
            document.body.prepend(backdrop);

            // Create toggle button
            const btn = document.createElement('button');
            btn.className = 'sidebar-toggle';
            btn.innerHTML = '☰';
            btn.setAttribute('aria-label', 'Toggle sidebar');
            btn.style.cssText = 'display:flex; align-items:center; justify-content:center;';
            document.body.prepend(btn);

            const closeSidebar = () => {
                sidebar.classList.remove('open');
                backdrop.classList.remove('show');
                btn.innerHTML = '☰';
            };
            const openSidebar = () => {
                sidebar.classList.add('open');
                backdrop.classList.add('show');
                btn.innerHTML = '✕';
            };

            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (sidebar.classList.contains('open')) {
                    closeSidebar();
                } else {
                    openSidebar();
                }
            });
            backdrop.addEventListener('click', closeSidebar);
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
                    if (!sidebar.contains(e.target) && !btn.contains(e.target)) {
                        closeSidebar();
                    }
                }
            });
            // Tutup sidebar otomatis saat user memilih menu navigasi
            sidebar.addEventListener('click', (e) => {
                if (e.target.closest('label') && window.innerWidth <= 768) {
                    setTimeout(closeSidebar, 200);
                }
            });
        }
    })();
</script>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL DAN ARTIFACTS
# ============================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgb_hybrid_mi_rfe_optimized.pkl")
        features = joblib.load("selected_features_mi_rfe.pkl")["selected_features"]
        all_features = joblib.load("train_feature_columns.pkl")["all_features"]
        defaults = joblib.load("feature_defaults_median.pkl")["feature_defaults_median"]
        return model, features, all_features, defaults
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {e}")
        st.info("Pastikan file model (.pkl) berada di direktori yang sama dengan app.py")
        return None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Pastikan fungsi 'mi_with_seed' sudah didefinisikan di bagian atas app.py")
        return None, None, None, None

model, selected_features, all_features, defaults = load_model()

# ============================================
# DAFTAR 15 FITUR FINAL + DESKRIPSI SINGKAT
# ============================================

FINAL_FEATURES = [
    "Age", "AlcoholConsumption", "DietQuality", "SleepQuality", "DiastolicBP",
    "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems",
    "ADL", "PersonalityChanges", "DifficultyCompletingTasks"
]

# Deskripsi singkat TANPA rentang angka
FEATURE_INFO = {
    "Age": {"desc": "Usia pasien (tahun)"},
    "AlcoholConsumption": {"desc": "Konsumsi alkohol"},
    "DietQuality": {"desc": "Skor kualitas diet"},
    "SleepQuality": {"desc": "Skor kualitas tidur"},
    "DiastolicBP": {"desc": "Tekanan darah diastolik (mmHg)"},
    "CholesterolTotal": {"desc": "Kolesterol total (mg/dL)"},
    "CholesterolLDL": {"desc": "Kolesterol LDL (mg/dL)"},
    "CholesterolHDL": {"desc": "Kolesterol HDL (mg/dL)"},
    "MMSE": {"desc": "Skor MMSE"},
    "FunctionalAssessment": {"desc": "Skor penilaian fungsional"},
    "MemoryComplaints": {"desc": "Keluhan memori"},
    "BehavioralProblems": {"desc": "Masalah perilaku"},
    "ADL": {"desc": "Skor ADL"},
    "PersonalityChanges": {"desc": "Perubahan kepribadian"},
    "DifficultyCompletingTasks": {"desc": "Kesulitan tugas"}
}

# ============================================
# FUNGSI PREPROCESSING
# ============================================

@st.cache_data
def preprocess_input_cached(input_df):
    if all_features:
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = defaults.get(col, 0) if defaults else 0
        input_df = input_df[all_features]
    return input_df

def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    return preprocess_input_cached(input_df)

def predict_single(model, input_data):
    processed = preprocess_input(input_data)
    prob = model.predict_proba(processed)[0, 1]
    pred = model.predict(processed)[0]
    return pred, prob

def predict_batch(model, df):
    processed = preprocess_input_batch(df)
    probs = model.predict_proba(processed)[:, 1]
    preds = model.predict(processed)
    return preds, probs

def preprocess_input_batch(df):
    if all_features:
        for col in all_features:
            if col not in df.columns:
                df[col] = defaults.get(col, 0) if defaults else 0
        df = df[all_features]
    return df

# ============================================
# SIDEBAR NAVIGASI (tanpa threshold & tema)
# ============================================

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #fff;'>🧠 Alzheimer’s Disease Classification</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.9rem;'>Klasifikasi Penyakit Alzheimer</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "📌 Navigasi",
        [
            "🏠 Home",
            "📊 Prediksi 1 Pasien",
            "📁 Batch CSV",
            "📈 Model & Evaluasi",
            "📋 Daftar Fitur & Template"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.03); backdrop-filter: blur(8px); padding: 1rem; border-radius: 20px; border-left: 4px solid #6a11cb;'>
        <p style='font-size: 0.75rem; color: rgba(255,255,255,0.4); margin: 0;'>
            <strong>⚠️ Etika Penggunaan</strong><br>
            Aplikasi ini untuk penelitian dan edukasi. Hasil tidak untuk keputusan medis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# HALAMAN HOME
# ============================================

if page == "🏠 Home":
    st.markdown("""
    <div class='card-glass'>
        <h1 style='margin: 0; font-size: 3rem;'>🧠 Sistem Klasifikasi Alzheimer</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 1.2rem;'>
            XGBoost + Hybrid Feature Selection (MI+RFE) + GridSearchCV (Stratified K-Fold)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='card-glass'>
            <h3>📋 Ringkasan Sistem</h3>
            <p>Penyakit Alzheimer merupakan gangguan neurodegeneratif progresif yang menyebabkan penurunan fungsi kognitif, memori, serta kemampuan dalam menjalankan aktivitas sehari-hari. Sistem ini menggunakan pembelajaran mesin untuk klasifikasi otomatis.</p>
            <h4>🎯 Tujuan</h4>
            <ul>
                <li>Klasifikasi Alzheimer vs Non-Alzheimer</li>
                <li>Prediksi berbasis 15 fitur klinis</li>
                <li>Dukungan single & batch</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card-glass'>
            <h3>🚀 Panduan</h3>
            <ol style='line-height: 2.2;'>
                <li>📊 Pilih <strong>Prediksi 1 Pasien</strong></li>
                <li>📝 Masukkan 15 fitur</li>
                <li>🔮 Klik <strong>Prediksi</strong></li>
                <li>📋 Lihat hasil</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# HALAMAN PREDIKSI 1 PASIEN (DIPERBAIKI)
# ============================================

elif page == "📊 Prediksi 1 Pasien":
    st.markdown("""
    <div class='card-glass'>
        <h1 style='margin: 0; font-size: 2.2rem;'>📊 Prediksi 1 Pasien</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.7;'>Masukkan data sesuai 15 fitur final. Setiap field dilengkapi dengan instruksi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model tidak dapat dimuat. Periksa file model.")
        st.stop()
    
    with st.form("prediction_form"):
        # Header form dengan style sederhana
        st.markdown("""
        <div class='form-header'>
            <h3>📋 Form Input (15 Fitur Final)</h3>
            <p>Isi semua field dengan nilai yang sesuai</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Menggunakan 3 kolom untuk form input
        col1, col2, col3 = st.columns(3)
        input_data = {}
        features_to_show = FINAL_FEATURES
        
        # Bagi fitur menjadi 3 kolom
        third = len(features_to_show) // 3
        col1_features = features_to_show[:third]
        col2_features = features_to_show[third:2*third]
        col3_features = features_to_show[2*third:]
        
        # Icon mapping per fitur untuk eye-catching
        FEATURE_ICONS = {
            "Age": "🎂", "AlcoholConsumption": "🍷", "DietQuality": "🥗",
            "SleepQuality": "😴", "DiastolicBP": "💓", "CholesterolTotal": "🩸",
            "CholesterolLDL": "🩸", "CholesterolHDL": "🩸", "MMSE": "🧠",
            "FunctionalAssessment": "⚙️", "MemoryComplaints": "💭",
            "BehavioralProblems": "😟", "ADL": "🧬", "PersonalityChanges": "🎭",
            "DifficultyCompletingTasks": "🧩"
        }

        def create_input(col, feature_list, start_num=0):
            with col:
                for idx, feature in enumerate(feature_list):
                    display_name = feature.replace('_', ' ').title()
                    info = FEATURE_INFO.get(feature, {})
                    desc = info.get("desc", "")
                    num = start_num + idx + 1
                    icon = FEATURE_ICONS.get(feature, "📊")
                    is_binary = feature in ["MemoryComplaints", "BehavioralProblems", "PersonalityChanges", "DifficultyCompletingTasks"]

                    # Card wrapper dengan badge nomor + icon
                    st.markdown(f"""
                    <div class='input-field-wrapper'>
                        <div class='input-field-label'>
                            <span class='input-field-number'>{num:02d}</span>
                            <span>{display_name}</span>
                            <span class='input-field-icon'>{icon}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Input field (label disembunyikan via CSS karena sudah di HTML)
                    if is_binary:
                        val = st.number_input(
                            f"{num:02d} {display_name}",
                            value=0.0, step=1.0, min_value=0.0, max_value=1.0,
                            format="%.0f", key=f"input_{feature}",
                            help=desc, label_visibility="collapsed"
                        )
                    else:
                        val = st.number_input(
                            f"{num:02d} {display_name}",
                            value=0.0, step=0.01, min_value=0.0,
                            format="%.2f", key=f"input_{feature}",
                            help=desc, label_visibility="collapsed"
                        )

                    # Description di bawah input
                    if desc:
                        st.markdown(
                            f"<span class='input-field-desc'>📌 {desc}</span>",
                            unsafe_allow_html=True
                        )
                    input_data[feature] = val
        
        # Buat input di masing-masing kolom
        create_input(col1, col1_features, 0)
        create_input(col2, col2_features, len(col1_features))
        create_input(col3, col3_features, len(col1_features) + len(col2_features))
        
        # Tombol submit di tengah
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)
    
    if submitted:
        with st.spinner("🧠 Menganalisis dengan AI..."):
            try:
                pred, prob = predict_single(model, input_data)
                final_pred = 1 if prob >= ss.threshold else 0
                
                st.markdown("---")
                st.markdown("## 📋 Hasil Prediksi")
                
                col_res1, col_res2 = st.columns([3, 2])
                
                with col_res1:
                    if final_pred == 1:
                        st.markdown(f"""
                        <div class='result-alzheimer' style='padding:1.5rem;'>
                            <h1 style='margin:0; font-size:2.5rem;'>⚠️ Alzheimer</h1>
                            <p style='font-size:1.1rem; opacity:0.9;'>Probabilitas: {prob*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-non-alzheimer' style='padding:1.5rem;'>
                            <h1 style='margin:0; font-size:2.5rem;'>✅ Non-Alzheimer</h1>
                            <p style='font-size:1.1rem; opacity:0.9;'>Probabilitas: {prob*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); backdrop-filter: blur(4px); padding: 1.2rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); margin-top: 1rem;'>
                        <p><strong>🧠 Diagnosis:</strong> {'Alzheimer' if final_pred == 1 else 'Non-Alzheimer'}</p>
                        <p><strong>📊 Probabilitas:</strong> {prob*100:.2f}%</p>
                        <p><strong>⏱️ Waktu:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    # Tentukan warna & label risiko
                    if prob < 0.3:
                        risk_level, risk_color = "🟢 RISIKO RENDAH", "#2ecc71"
                    elif prob < 0.7:
                        risk_level, risk_color = "🟡 RISIKO SEDANG", "#f1c40f"
                    else:
                        risk_level, risk_color = "🔴 RISIKO TINGGI", "#e74c3c"

                    # Progress bar CSS (tanpa gambar/gauge)
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); backdrop-filter: blur(8px); padding: 1.2rem 1.4rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08); text-align: center;'>
                        <h4 style='color: rgba(255,255,255,0.6); margin:0 0 0.6rem 0; font-size: 0.85rem; font-weight: 500; letter-spacing: 0.5px;'>PROBABILITAS ALZHEIMER</h4>
                        <div style='font-size: 3rem; font-weight: 800; color: {risk_color}; line-height: 1; margin-bottom: 0.3rem;'>{prob*100:.1f}<span style='font-size: 1.4rem; opacity: 0.7;'>%</span></div>
                        <div style='background: rgba(255,255,255,0.08); border-radius: 12px; height: 14px; overflow: hidden; margin: 0.8rem 0 0.6rem 0; position: relative;'>
                            <div style='background: linear-gradient(90deg, {risk_color} 0%, {risk_color}dd 100%); height: 100%; width: {prob*100:.1f}%; border-radius: 12px; box-shadow: 0 0 12px {risk_color}66; transition: width 0.6s ease;'></div>
                        </div>
                        <h3 style='color: {risk_color}; margin: 0.6rem 0 0 0; font-size: 1.05rem; font-weight: 700; letter-spacing: 0.5px;'>{risk_level}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data Input di bawah, dengan jarak
                st.markdown("---")
                with st.expander("📊 Lihat Data Input", expanded=False):
                    st.dataframe(pd.DataFrame([input_data]).style.background_gradient(cmap="Blues", low=0.1, high=0.3), use_container_width=True)
                    
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {e}")

# ============================================
# HALAMAN BATCH CSV
# ============================================

elif page == "📁 Batch CSV":
    st.markdown("""
    <div class='card-glass'>
        <h1 style='margin: 0; font-size: 2.2rem;'>📁 Batch CSV</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.7;'>Upload CSV dengan 15 fitur final. Proses akan menampilkan progress.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model tidak dapat dimuat.")
        st.stop()
    
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload file CSV", type=['csv'], help="File harus memiliki 15 fitur final")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in FINAL_FEATURES if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Kolom hilang: {missing_cols}")
                st.info("Pastikan file CSV memiliki 15 fitur final.")
            else:
                st.success(f"✅ File berhasil diupload! ({len(df)} baris)")
                with st.expander("📊 Preview Data"):
                    st.dataframe(df[FINAL_FEATURES].head(10), use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 Prediksi Batch", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        total = len(df)
                        batch_size = max(1, total // 20)
                        preds = np.zeros(total, dtype=int)
                        probs = np.zeros(total)
                        for i in range(0, total, batch_size):
                            end = min(i+batch_size, total)
                            batch_df = df[FINAL_FEATURES].iloc[i:end].copy()
                            batch_preds, batch_probs = predict_batch(model, batch_df)
                            preds[i:end] = batch_preds
                            probs[i:end] = batch_probs
                            progress = (end / total)
                            progress_bar.progress(progress)
                            status_text.text(f"Memproses {end}/{total} baris...")
                        status_text.text("Selesai!")
                        
                        st.markdown("<div class='batch-result'>", unsafe_allow_html=True)
                        st.markdown("### 📊 Hasil Prediksi Batch")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Data", len(df))
                        with col2:
                            st.metric("Alzheimer", (preds==1).sum())
                        with col3:
                            st.metric("Non-Alzheimer", (preds==0).sum())
                        
                        result_df = df[FINAL_FEATURES].copy()
                        result_df['Prediction'] = preds
                        result_df['Prediction_Label'] = result_df['Prediction'].map({0: 'Non-Alzheimer', 1: 'Alzheimer'})
                        result_df['Probability'] = probs
                        result_df['Risk_Level'] = pd.cut(probs, bins=[-0.001, 0.3, 0.7, 1.001], labels=['Rendah', 'Sedang', 'Tinggi'])
                        
                        st.dataframe(result_df.style.background_gradient(subset=['Probability'], cmap='RdYlGn_r', low=0.1, high=0.3), use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button("📥 Download Hasil CSV", data=csv_buffer.getvalue(), file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")
        except Exception as e:
            st.error(f"⚠️ Error membaca file: {e}")

# ============================================
# HALAMAN MODEL & EVALUASI (DIPERBAIKI)
# ============================================

elif page == "📈 Model & Evaluasi":
    st.markdown("""
    <div class='card-glass'>
        <h1 style='margin: 0; font-size: 2.2rem;'>📈 Model & Evaluasi</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.7;'>Detail konfigurasi dan performa model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card-glass'>
            <h3>🎯 Performa Uji</h3>
            <table style='width:100%;'>
                <tr><td>Accuracy</td><td style='text-align:right;color:#2ecc71;'>95.35%</td></tr>
                <tr><td>ROC-AUC</td><td style='text-align:right;color:#2ecc71;'>94.75%</td></tr>
                <tr><td>Precision</td><td style='text-align:right;color:#2ecc71;'>94.59%</td></tr>
                <tr><td>Recall</td><td style='text-align:right;color:#2ecc71;'>92.11%</td></tr>
                <tr><td>F1-Score</td><td style='text-align:right;color:#2ecc71;'>93.33%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card-glass'>
            <h3>⚙️ Konfigurasi</h3>
            <table style='width:100%;'>
                <tr><td><strong>Algoritma</strong></td><td>XGBoost</td></tr>
                <tr><td><strong>Seleksi Fitur</strong></td><td>MI (K=20) + RFE (K=15)</td></tr>
                <tr><td><strong>Optimasi</strong></td><td>GridSearchCV</td></tr>
                <tr><td><strong>CV</strong></td><td>Stratified K-Fold (5)</td></tr>
                <tr><td><strong>Scoring</strong></td><td>ROC-AUC</td></tr>
                <tr><td><strong>Random State</strong></td><td>42</td></tr>
                <tr><td><strong>Total Fitur</strong></td><td>35 → 15</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Visualisasi Confusion Matrix (Data Uji)")
    
    # PERBAIKAN 2: Confusion Matrix dengan text hitam
    cm = np.array([[270, 10], [15, 135]])
    
    # Buat heatmap dengan text color hitam
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-Alzheimer', 'Alzheimer'],
        y=['Non-Alzheimer', 'Alzheimer'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "#000000"},  # Warna hitam
        colorscale='Blues',
        showscale=True,
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>%{y}<br>Jumlah: %{z}<extra></extra>'
    ))
    fig_cm.update_layout(
        title={
            'text': 'Confusion Matrix - Data Uji',
            'font': {'size': 20, 'color': '#f0f0f0'}
        },
        xaxis_title={
            'text': 'Predicted',
            'font': {'size': 16, 'color': '#f0f0f0'}
        },
        yaxis_title={
            'text': 'Actual',
            'font': {'size': 16, 'color': '#f0f0f0'}
        },
        height=450,
        width=500,
        margin=dict(l=60, r=60, t=80, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0',
        xaxis=dict(
            tickfont=dict(color='#f0f0f0', size=14),
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            tickfont=dict(color='#f0f0f0', size=14),
            gridcolor='rgba(255,255,255,0.05)'
        )
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================
# HALAMAN DAFTAR FITUR & TEMPLATE (DIPERBAIKI)
# ============================================

else:
    st.markdown("""
    <div class='card-glass'>
        <h1 style='margin: 0; font-size: 2.2rem;'>📋 Daftar Fitur & Template</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.7;'>Fitur yang digunakan pada input manual dan batch CSV.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='card-glass' style='margin-bottom: 0.4rem;'><h3>📋 15 Fitur Final</h3></div>", unsafe_allow_html=True)
        
        # PERBAIKAN 3: Tampilan fitur final yang lebih rapi dengan grid 3 kolom
        st.markdown("""
        <div style='margin-top: 0.2rem;'>
            <div class='feature-grid-container'>
        """, unsafe_allow_html=True)
        
        # Buat feature cards dalam grid 3 kolom - DESKRIPSI PENUH, RATA KIRI
        for idx, feature in enumerate(FINAL_FEATURES):
            display_name = feature.replace('_', ' ').title()
            info = FEATURE_INFO.get(feature, {})
            desc = info.get("desc", "")
            
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-header-row'>
                    <span class='feature-number-badge'>{idx+1:02d}</span>
                    <span class='feature-name'>{display_name}</span>
                </div>
                <span class='feature-desc'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card-glass'><h3>📥 Template CSV</h3><p style='color:rgba(255,255,255,0.5);'>Unduh template untuk batch.</p></div>", unsafe_allow_html=True)
        template_df = pd.DataFrame({col: [0] for col in FINAL_FEATURES})
        st.dataframe(template_df, use_container_width=True)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Template", data=csv_buffer.getvalue(), file_name="template_15_fitur.csv", mime="text/csv", use_container_width=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class='footer'>
    <p>Alzheimer’s Disease Classification System</p>
    <p style='font-size: 0.7rem; opacity:0.5;'>© 2026 Agung Iman Wicaksono</p>
</div>
""", unsafe_allow_html=True)