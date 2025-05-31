
Install fastapi 
> pip install fastapi uvicorn

Run fast api:
> uvicorn main:app --reload

Test:
> curl -X POST "http://127.0.0.1:8000/predict/?medinc=3.5"

Start Docker
> docker build -t linear-model .
> docker run -p 8000:80 linear-model

Start docker compose
> docker-compose up --build

