# ML API - Iris Dataset

## Uruchomienie lokalne
```
pip install -r requirements.txt
python run.py
```

## Uruchomienie przez Docker
```
docker build -t iris-ml-api .
docker run -p 8000:8000 iris-ml-api
```

## Uruchomienie przez Docker Compose
```
docker-compose up --build
```

## Testowanie API (cURL)
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Endpointy:
- GET /        – powitanie
- POST /predict – predykcja
- GET /info    – info o modelu
- GET /health  – status serwera
