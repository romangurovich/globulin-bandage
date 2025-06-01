FROM python:3.10-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python3", "predict_nearby_aminos.py"] 