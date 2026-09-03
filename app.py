# app.py
# SISTEM KLASIFIKASI PENYAKIT ALZHEIMER - FINAL EDITION
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
import sys
from streamlit import session_state as ss

warnings.filterwarnings('ignore')

# ============================================
# FIX UNTUK VERCELL - PATH FILE .PKL
# ============================================

# Path absolut untuk Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fungsi untuk mendapatkan path file .pkl
def get_pkl_path(filename):
    """Mencari file .pkl di berbagai kemungkinan lokasi"""
    # 1. Cek di folder yang sama
    path1 = os.path.join(BASE_DIR, filename)
    if os.path.exists(path1):
        return path1
    
    # 2. Cek di folder api (jika ada)
    path2 = os.path.join(BASE_DIR, 'api', filename)
    if os.path.exists(path2):
        return path2
    
    # 3. Cek di folder src (jika ada)
    path3 = os.path.join(BASE_DIR, 'src', filename)
    if os.path.exists(path3):
        return path3
    
    # 4. Cek di current working directory
    path4 = os.path.join(os.getcwd(), filename)
    if os.path.exists(path4):
        return path4
    
    # 5. Kembalikan path default
    return path1

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
        top: 1rem;
        left: 1rem;
        z-index: 100;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 50%;
        width: 48px;
        height: 48px;
        font-size: 1.5rem;
        color: white;
        cursor: pointer;
        transition: 0.3s;
    }
    .sidebar-toggle:hover {
        background: rgba(255,255,255,0.2);
        transform: scale(1.05);
    }
    @media (max-width: 768px) {
        .sidebar-toggle {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stSidebar"] {
            width: 280px !important;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        [data-testid="stSidebar"].open {
            transform: translateX(0);
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

    /* Inputs */
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

    /* Caption untuk instruksi input (tanpa rentang) */
    .input-caption {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        font-style: italic;
        margin-top: -0.3rem;
        margin-bottom: 0.5rem;
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

    /* Uploader - beri jarak lebih dan geser ke bawah */
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
    }
    @media (max-width: 480px) {
        .main .block-container { padding: 0.8rem; border-radius: 16px; }
        .card-glass { padding: 1rem; }
        .stNumberInput input, .stTextInput input { padding: 0.5rem 0.8rem !important; font-size: 0.85rem !important; }
        .stTabs [data-baseweb="tab"] { padding: 0.2rem 0.5rem; font-size: 0.65rem; }
        .stButton > button { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
        .feature-item { min-width: 80px; font-size: 0.65rem; padding: 0.2rem 0.4rem; }
    }

    /* Batch result spacing */
    .batch-result {
        margin-top: 2rem;
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
            const btn = document.createElement('button');
            btn.className = 'sidebar-toggle';
            btn.innerHTML = '☰';
            btn.setAttribute('aria-label', 'Toggle sidebar');
            btn.style.cssText = 'display:flex; align-items:center; justify-content:center;';
            document.body.prepend(btn);
            btn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
                btn.innerHTML = sidebar.classList.contains('open') ? '✕' : '☰';
            });
            document.addEventListener('click', (e) => {
                if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
                    if (!sidebar.contains(e.target) && !btn.contains(e.target)) {
                        sidebar.classList.remove('open');
                        btn.innerHTML = '☰';
                    }
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
    "Age": {"desc": "Usia pasien"},
    "AlcoholConsumption": {"desc": "Konsumsi alkohol per minggu"},
    "DietQuality": {"desc": "Skor kualitas diet"},
    "SleepQuality": {"desc": "Skor kualitas tidur"},
    "DiastolicBP": {"desc": "Tekanan darah diastolik"},
    "CholesterolTotal": {"desc": "Kolesterol total"},
    "CholesterolLDL": {"desc": "Kolesterol LDL"},
    "CholesterolHDL": {"desc": "Kolesterol HDL"},
    "MMSE": {"desc": "Skor MMSE (Mini-Mental)"},
    "FunctionalAssessment": {"desc": "Skor penilaian fungsional"},
    "MemoryComplaints": {"desc": "Keluhan memori (0=Tidak, 1=Ya)"},
    "BehavioralProblems": {"desc": "Masalah perilaku (0=Tidak, 1=Ya)"},
    "ADL": {"desc": "Skor ADL (Aktivitas Harian)"},
    "PersonalityChanges": {"desc": "Perubahan kepribadian (0=Tidak, 1=Ya)"},
    "DifficultyCompletingTasks": {"desc": "Kesulitan tugas (0=Tidak, 1=Ya)"}
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
# HALAMAN PREDIKSI 1 PASIEN
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
        st.markdown("### 📋 Form Input (15 Fitur Final)")
        col1, col2, col3 = st.columns(3)
        input_data = {}
        features_to_show = FINAL_FEATURES
        third = len(features_to_show) // 3
        col1_features = features_to_show[:third]
        col2_features = features_to_show[third:2*third]
        col3_features = features_to_show[2*third:]
        
        def create_input(col, feature_list):
            with col:
                st.markdown("**Fitur Klinis**")
                for feature in feature_list:
                    display_name = feature.replace('_', ' ').title()
                    info = FEATURE_INFO.get(feature, {})
                    desc = info.get("desc", "")
                    val = st.number_input(
                        display_name,
                        value=0.0,
                        format="%.0f",
                        key=f"input_{feature}",
                        help=desc
                    )
                    if desc:
                        st.caption(f"📌 {desc}")
                    input_data[feature] = val
        
        create_input(col1, col1_features)
        create_input(col2, col2_features)
        create_input(col3, col3_features)
        
        st.markdown("---")
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
                            <p style='font-size:1.1rem; opacity:0.9;'>Probabilitas: {prob*100:.2f}% (Threshold: {ss.threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-non-alzheimer' style='padding:1.5rem;'>
                            <h1 style='margin:0; font-size:2.5rem;'>✅ Non-Alzheimer</h1>
                            <p style='font-size:1.1rem; opacity:0.9;'>Probabilitas: {prob*100:.2f}% (Threshold: {ss.threshold:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); backdrop-filter: blur(4px); padding: 1.2rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); margin-top: 1rem;'>
                        <p><strong>🧠 Diagnosis:</strong> {'Alzheimer' if final_pred == 1 else 'Non-Alzheimer'}</p>
                        <p><strong>📊 Probabilitas:</strong> {prob*100:.2f}%</p>
                        <p><strong>🎯 Threshold:</strong> {ss.threshold:.2f}</p>
                        <p><strong>⏱️ Waktu:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        title={'text': "Probabilitas Alzheimer", 'font': {'size': 16, 'color': '#f0f0f0'}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ccc'},
                            'bar': {'color': "#6a11cb"},
                            'bgcolor': "rgba(255,255,255,0.05)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.1)",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.3)'},
                                {'range': [30, 70], 'color': 'rgba(241, 196, 15, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "#e74c3c", 'width': 4},
                                'thickness': 0.75,
                                'value': ss.threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f0f0f0'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if prob < 0.3:
                        risk_level, risk_color = "🟢 RISIKO RENDAH", "#2ecc71"
                    elif prob < 0.7:
                        risk_level, risk_color = "🟡 RISIKO SEDANG", "#f1c40f"
                    else:
                        risk_level, risk_color = "🔴 RISIKO TINGGI", "#e74c3c"
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.03); backdrop-filter: blur(4px); padding: 0.8rem; border-radius: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.05); margin-top: 0.5rem;'>
                        <h4 style='color: rgba(255,255,255,0.6); margin:0;'>Interpretasi Risiko</h4>
                        <h2 style='color: {risk_color}; margin:0;'>{risk_level}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data Input di bawah, dengan jarak
                st.markdown("---")
                with st.expander("📊 Lihat Data Input", expanded=False):
                    st.dataframe(pd.DataFrame([input_data]).style.background_gradient(cmap="Blues", low=0.1, high=0.3), use_container_width=True)
                    
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {e}")

# ============================================
# HALAMAN BATCH CSV (dengan jarak)
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
    
    # Uploader dengan margin bawah yang cukup
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
                
                # Beri jarak sebelum tombol prediksi
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
                        
                        # Hasil dengan jarak
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
# HALAMAN MODEL & EVALUASI
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
    cm = np.array([[270, 10], [15, 135]])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-Alzheimer', 'Alzheimer'],
        y=['Non-Alzheimer', 'Alzheimer'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale='Blues',
        showscale=True,
        hoverongaps=False
    ))
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f0f0f0'
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================
# HALAMAN DAFTAR FITUR & TEMPLATE (3 baris rapi)
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
        st.markdown("<div class='card-glass'><h3>📋 15 Fitur Final</h3></div>", unsafe_allow_html=True)
        # Buat 3 baris: 5 fitur per baris
        row1 = FINAL_FEATURES[:5]
        row2 = FINAL_FEATURES[5:10]
        row3 = FINAL_FEATURES[10:15]
        
        # Baris 1
        cols1 = st.columns(5)
        for idx, feature in enumerate(row1):
            with cols1[idx]:
                st.markdown(f"""
                <div class='feature-item'>
                    <span class='feature-number'>{idx+1:02d}.</span> {feature}
                </div>
                """, unsafe_allow_html=True)
        
        # Baris 2
        cols2 = st.columns(5)
        for idx, feature in enumerate(row2):
            with cols2[idx]:
                st.markdown(f"""
                <div class='feature-item'>
                    <span class='feature-number'>{idx+6:02d}.</span> {feature}
                </div>
                """, unsafe_allow_html=True)
        
        # Baris 3
        cols3 = st.columns(5)
        for idx, feature in enumerate(row3):
            with cols3[idx]:
                st.markdown(f"""
                <div class='feature-item'>
                    <span class='feature-number'>{idx+11:02d}.</span> {feature}
                </div>
                """, unsafe_allow_html=True)
    
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
    <p>🧠 Alzheimer Prediction System</p>
    <p style='font-size: 0.7rem; opacity:0.5;'>Built with ❤️ by Agung Iman Wicaksono | Research Project</p>
</div>
""", unsafe_allow_html=True)