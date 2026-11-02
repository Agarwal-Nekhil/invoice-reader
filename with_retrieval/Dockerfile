FROM python:3.11-slim-bookworm

WORKDIR /app

# Tesseract and Poppler for the OCR path (pytesseract and pdf2image call them)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Uploaded PDFs are written here and deleted once the job ends
RUN mkdir -p /app/temp_invoices

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
