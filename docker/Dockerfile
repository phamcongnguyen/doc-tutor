FROM python:3.12-slim

WORKDIR /app

# Cài dependency trước để tận dụng layer cache của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY doctutor/ ./doctutor/

ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
