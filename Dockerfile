# Image de l'assistant RH : Python + bibliothèques + code
FROM python:3.12-slim

WORKDIR /app

# 1. PyTorch en version CPU : bien plus léger que la version GPU installée par défaut sous Linux
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Les bibliothèques du projet (couche mise en cache tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Le code et les données légères (sources, jeu de test)
COPY src/ src/
COPY data/ data/
COPY .streamlit/ .streamlit/

# Ollama tourne sur la machine hôte ; « host.docker.internal » est l'adresse de l'hôte vue depuis le conteneur
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Par défaut : l'API (docker-compose.yml lance aussi l'interface Streamlit avec la même image)
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]