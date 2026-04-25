from fastapi import FastAPI
from config import settings
from routes import ingestion_routes

app = FastAPI()
app.include_router(ingestion_routes.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(message: str):
    client = settings.google_client
    
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=message
    )
    return {"response": response.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)