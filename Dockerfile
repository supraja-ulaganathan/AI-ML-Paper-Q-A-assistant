FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the 21 papers and build the index once, while the image is built
RUN python setup_data.py

EXPOSE 10000
CMD ["python", "app.py"]