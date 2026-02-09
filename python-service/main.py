from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import PORT
from routes import (
    stream_router,
    insights_router,
    library_router,
    mission_router,
    mindspace_router,
    queue_router,
    ai_router,
)

app = FastAPI(title="Cohesive Unified Backend")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(stream_router)
app.include_router(insights_router)
app.include_router(library_router)
app.include_router(mission_router)
app.include_router(mindspace_router)
app.include_router(queue_router)
app.include_router(ai_router)


@app.get("/")
async def root():
    """Root endpoint - service status."""
    return {"message": "Cohesive Unified Backend Running", "status": "healthy"}


if __name__ == "__main__":
    print(f"Starting unified backend on port: {PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
