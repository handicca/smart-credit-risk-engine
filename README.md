
# Smart Credit AI

## Overview
Smart Credit AI adalah sistem machine learning untuk prediksi risiko kredit dan otomasi keputusan persetujuan pinjaman. Sistem ini menggunakan Logistic Regression untuk menghitung risk score calon peminjam dengan akurasi dan kecepatan tinggi.

## Fitur Utama
- **Risk Scoring**: Menghasilkan probabilitas gagal bayar (0-1)
- **Decision Policy**: Tiga kebijakan keputusan (Conservative, Balanced, Aggressive)
- **Feature Explainability**: Transparansi faktor-faktor yang mempengaruhi keputusan
- **Fast Inference**: Verifikasi dari hari menjadi detik

## Dataset
- **Target**: `loan_status` (Approved/Rejected)
- **Fitur Utama**: CIBIL score, pendapatan, jumlah pinjaman, aset, tenor pinjaman

## Model Performance
- **ROC-AUC**: 0.9727
- **Brier Score**: 0.0575
- **Recall (Rejected)**: ~91% (threshold 0.45)
- **Approval Rate**: ~62%

## Installation
```bash
pip install -r requirements.txt
```

## Penggunaan
```python
from inference import predict_credit_risk
result = predict_credit_risk(applicant_data, threshold=0.45)
# Output: {"decision": "Approved", "risk_score": 0.25, "message": "..."}
```

## Struktur Project
```
├── notebook.ipynb           # EDA, modeling, evaluation
├── data/
│   └── loan_approval_dataset.csv
├── smart_credit_model.joblib
├── smart_credit_metadata.joblib
├── requirements.txt
└── README.md
```