FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100

ENV VOXHEALTH_HOST=0.0.0.0
ENV VOXHEALTH_PORT=8100

CMD ["python3", "-m", "src.api.main"]
