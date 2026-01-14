from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.llm_gateway import LLMService
import uvicorn
import os
import json
import asyncio
import numpy as np
from scipy.spatial.distance import cosine
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

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

@app.post("/mindspace/process")
async def process_mindspace_entry(request: dict):
    """
    Orchestrates Vision, Embedding, Clustering, and Sentiment for a Mindspace entry.
    """
    text = request.get("text", "")
    image_url = request.get("image_url")
    
    try:
        results = {}
        
        # 1. Vision Analysis if image provided
        if image_url:
            results["ai_description"] = await llm_service.generate(
                prompt=image_url,
                provider="groq",
                prompt_type="mindspace_vision"
            )
        
        # Effective content for further steps
        content_for_analysis = f"{text} {results.get('ai_description', '')}".strip()
        
        # 2. Parallel Embedding, Clustering, and Sentiment
        import asyncio
        embedding_task = llm_service.generate_embedding(text=content_for_analysis)
        
        # 2b. Clustering with Librarian Context
        existing_clusters = request.get("existing_clusters", [])
        existing_names = [c.get("name") for c in existing_clusters]
        
        clustering_prompt = f"Content: {content_for_analysis}\nExisting Sections: {json.dumps(existing_names)}"
        clustering_task = llm_service.generate(
            prompt=clustering_prompt,
            provider="groq",
            prompt_type="mindspace_clustering"
        )
        
        sentiment_task = llm_service.generate(
            prompt=content_for_analysis,
            provider="groq",
            prompt_type="mindspace_sentiment"
        )
        
        embedding, clusters_data, sentiment = await asyncio.gather(
            embedding_task, clustering_task, sentiment_task
        )
        
        suggested_tags = clusters_data.get("clusters", [])
        final_cluster_ids = []
        user_id = request.get("userId")

        for tag in suggested_tags:
            target_cluster_id = None
            
            # Exact match check
            exact_match = next((c for c in existing_clusters if c['name'].lower() == tag.lower()), None)
            
            if exact_match:
                target_cluster_id = exact_match['id']
            else:
                # SEMANTIC CHECK (Anti-fragmentation)
                tag_embedding = await llm_service.generate_embedding(tag)
                
                best_similarity = 0
                best_existing_id = None
                
                for cluster in existing_clusters:
                    if cluster.get('center_embedding'):
                        # cosine similarity
                        sim = 1 - cosine(tag_embedding, cluster['center_embedding'])
                        if sim > best_similarity:
                            best_similarity = sim
                            best_existing_id = cluster['id']
                
                if best_similarity > 0.85:
                    target_cluster_id = best_existing_id
                    print(f"[AI] Merging '{tag}' into existing cluster ID {target_cluster_id} (sim: {best_similarity:.2f})")
                else:
                    # Genuinely new concept, create it in Supabase
                    print(f"[AI] Creating new cluster: {tag}")
                    new_cluster = supabase.table('mindspace_clusters').insert({
                        'name': tag,
                        'user_id': user_id,
                        'summary': f"Items related to {tag}",
                        'center_embedding': tag_embedding
                    }).execute()
                    target_cluster_id = new_cluster.data[0]['id']

            if target_cluster_id:
                final_cluster_ids.append(target_cluster_id)
        
        results["embedding"] = embedding
        results["cluster_ids"] = final_cluster_ids
        results["tone"] = sentiment.get("tone", "Calm")
        
        return {"status": "success", "response": results}
    except Exception as e:
        print(f"[AI] Mindspace process error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mindspace/analyze_patterns")
async def analyze_patterns(request: dict):
    try:
        entries = request.get("entries", [])
        content = "\n---\n".join([f"Text: {e.get('content_text', '')}\nVision: {e.get('ai_description', '')}" for e in entries])
        
        response = await llm_service.generate(
            prompt=content,
            provider="groq",
            prompt_type="mindspace_patterns"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
