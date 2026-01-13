from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.llm_gateway import LLMService
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Current AI Gateway")
llm_service = LLMService()

class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "groq"
    model: Optional[str] = None
    prompt_type: Optional[str] = None

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_completion(request: GenerateRequest):
    try:
        response = await llm_service.generate(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            prompt_type=request.prompt_type
        )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_insight(request: AnalyzeRequest):
    try:
        import asyncio
        # Run analysis and embedding in parallel
        analysis_task = llm_service.generate(
            prompt=request.text,
            provider="groq",
            prompt_type="auto_tag_capture"
        )
        embedding_task = llm_service.generate_embedding(text=request.text)
        
        analysis, embedding = await asyncio.gather(analysis_task, embedding_task)
        
        # Merge results
        if isinstance(analysis, dict):
            analysis["embedding"] = embedding
            
        return {"status": "success", "response": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_mission")
async def generate_mission(request: dict):
    # request body expected to have 'tasks' list
    tasks = request.get('tasks', [])
    if not tasks:
        return {"status": "success", "response": {"selected_ids": []}}
        
    try:
        response = await llm_service.generate(
            prompt=str(tasks),
            provider="groq",
            prompt_type="select_mission"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_schedule")
async def generate_schedule(request: dict):
    tasks = request.get('tasks', [])
    user_start_time = request.get('user_start_time', '09:00')
    current_date = request.get('current_date')
    
    if not tasks:
        return {"status": "success", "response": {"schedule": []}}
        
    try:
        # Wrap the complex input into a prompt string
        prompt_data = {
            "user_start_time": user_start_time,
            "current_date": current_date,
            "tasks": tasks
        }
        
        response = await llm_service.generate(
            prompt=str(prompt_data),
            provider="groq",
            prompt_type="generate_schedule"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"[AI] Schedule error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/breakdown")
async def breakdown_goal(request: AnalyzeRequest):
    goal = request.text
    print(f"[AI] Breaking down goal: {goal}")
    try:
        response = await llm_service.generate(
            prompt=goal,
            provider="groq",
            prompt_type="decompose_goal"
        )
        print(f"[AI] Breakdown success. Tasks count: {len(response.get('tasks', []))}")
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"[AI] Breakdown error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def generate_vector(request: AnalyzeRequest):
    try:
        vector = await llm_service.generate_embedding(text=request.text)
        return {"status": "success", "embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
