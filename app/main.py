from fastapi import FastAPI
from app.api import chat
from app.core.cors import setup_cors

app = FastAPI(
    title="AI Portfolio Backend",
    description="Backend API for AI-powered portfolio website",
    version="1.0.0",
)

# Configure CORS
setup_cors(app)

# Include chat router
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "AI Portfolio Backend API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

