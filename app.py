from fastapi import fastapi
from pydantic import BaseModel
from app.model_loader import model_loader
import time

app = FASTAPI(title="NER Model Serving API")
model_loader = model_loader()

class TextRequest(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime: float

@app.on_event("startup")
async def startup_event():
    """ Load Model on Startup - Simulate S3 loading"""
    await model_loader.load_model("ner_model_v1.0")

@app.post("/predict")
async def predict(request: TextRequest):
    """ 
    Endpoint for NER Prediction 
    Simulates samsg ner task
    """
    start_time = time.time()
    try:
        # tokenize and predict
        tokens = request.text.split()
        predictions = model_loader.predict(tokens)

        # Format as entities
        entities = []
        for token, pred in zip(tokens, predictions):
            if pred != "O":  # No outside entity
                entities.append({
                    "text": token, 
                    "entity": pred,
                    "confidence": 0.95
                })
            
        return{
            "text": request.text,
            "entities": entities,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_version": request.model_version
        }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """ Health Check Endpoint """
        return HealthResponse(
            status = "healthy" if model_loader.model else "unhealthy",
            model_loaded = model_loader.model is not None,
            model_version = model_loader.current_version,
            uptime = model_loader.uptime()
        )

    @app.get("/model/info")
    async def model_info():
        """ Endpoint showing model architecture - SHOWS TF knowledge """
        if not model_loader.model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "framework": "TensorFlow 2.x",
            "model_type": "Transformer (BERT-based)",
            "task": "Named Entity Recogniton",
            "layers": len(model_loader.model.layers),
            "trainable_params": model_loader.model.count_params()
        }


