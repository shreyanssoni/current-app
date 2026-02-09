import asyncio
from fastapi import APIRouter, HTTPException

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service
from models.schemas import GenerateRequest, AnalyzeRequest, CopilotIgniteRequest

router = APIRouter()


@router.post("/generate")
async def generate_completion(request: GenerateRequest):
    """Generate completion from LLM."""
    llm_service = get_llm_service()

    try:
        response = await llm_service.generate(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            prompt_type=request.prompt_type,
        )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_insight(request: AnalyzeRequest):
    """Analyze an insight with AI."""
    llm_service = get_llm_service()

    try:
        # Run analysis and embedding in parallel
        analysis_task = llm_service.generate(
            prompt=request.text, provider="groq", prompt_type="auto_tag_capture"
        )
        embedding_task = llm_service.generate_embedding(text=request.text)

        analysis, embedding = await asyncio.gather(analysis_task, embedding_task)

        # Merge results
        if isinstance(analysis, dict):
            analysis["embedding"] = embedding

        return {"status": "success", "response": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def generate_vector(request: AnalyzeRequest):
    """Generate vector embedding."""
    llm_service = get_llm_service()

    try:
        vector = await llm_service.generate_embedding(text=request.text)
        return {"status": "success", "embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/breakdown")
async def breakdown_goal(request: AnalyzeRequest):
    """Break down a goal into subtasks."""
    llm_service = get_llm_service()
    goal = request.text
    print(f"[AI] Breaking down goal: {goal}")

    try:
        response = await llm_service.generate(prompt=goal, provider="groq", prompt_type="decompose_goal")
        print(f"[AI] Breakdown success. Tasks count: {len(response.get('tasks', []))}")
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"[AI] Breakdown error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_mission")
async def generate_mission_ai(request: dict):
    """Generate mission (select tasks)."""
    llm_service = get_llm_service()
    tasks = request.get("tasks", [])

    if not tasks:
        return {"status": "success", "response": {"selected_ids": []}}

    try:
        response = await llm_service.generate(
            prompt=str(tasks), provider="groq", prompt_type="select_mission"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_schedule")
async def generate_schedule_ai(request: dict):
    """Generate schedule for tasks."""
    llm_service = get_llm_service()
    tasks = request.get("tasks", [])
    user_start_time = request.get("user_start_time", "09:00")
    current_date = request.get("current_date")

    if not tasks:
        return {"status": "success", "response": {"schedule": []}}

    try:
        prompt_data = {
            "user_start_time": user_start_time,
            "current_date": current_date,
            "tasks": tasks,
        }

        response = await llm_service.generate(
            prompt=str(prompt_data), provider="groq", prompt_type="generate_schedule"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"[AI] Schedule error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mindspace/process")
async def process_mindspace_entry(request: dict):
    """Process mindspace entry (vision, embedding, clustering, sentiment)."""
    llm_service = get_llm_service()
    supabase = get_supabase_client()

    text = request.get("text", "")
    image_url = request.get("image_url")

    try:
        results = {}

        # 1. Vision Analysis if image provided
        if image_url:
            results["ai_description"] = await llm_service.generate(
                prompt=image_url, provider="groq", prompt_type="mindspace_vision"
            )

        # Effective content for further steps
        content_for_analysis = f"{text} {results.get('ai_description', '')}".strip()

        # 2. Parallel Embedding, Clustering, and Sentiment
        embedding_task = llm_service.generate_embedding(text=content_for_analysis)

        # 2b. Clustering with Librarian Context
        import json
        from services.llm_gateway import cosine_similarity

        existing_clusters = request.get("existing_clusters", [])
        existing_names = [c.get("name") for c in existing_clusters]

        clustering_prompt = f"Content: {content_for_analysis}\nExisting Sections: {json.dumps(existing_names)}"
        clustering_task = llm_service.generate(
            prompt=clustering_prompt, provider="groq", prompt_type="mindspace_clustering"
        )

        sentiment_task = llm_service.generate(
            prompt=content_for_analysis, provider="groq", prompt_type="mindspace_sentiment"
        )

        embedding, clusters_data, sentiment = await asyncio.gather(
            embedding_task, clustering_task, sentiment_task
        )

        suggested_tags = clusters_data.get("clusters", []) if isinstance(clusters_data, dict) else []
        final_cluster_ids = []
        user_id = request.get("userId")

        for tag in suggested_tags:
            target_cluster_id = None

            # Exact match check
            exact_match = next(
                (c for c in existing_clusters if c["name"].lower() == tag.lower()), None
            )

            if exact_match:
                target_cluster_id = exact_match["id"]
            else:
                # SEMANTIC CHECK (Anti-fragmentation)
                tag_embedding = await llm_service.generate_embedding(tag)

                best_similarity = 0
                best_existing_id = None

                for cluster in existing_clusters:
                    if cluster.get("center_embedding"):
                        sim = cosine_similarity(tag_embedding, cluster["center_embedding"])
                        if sim > best_similarity:
                            best_similarity = sim
                            best_existing_id = cluster["id"]

                if best_similarity > 0.85:
                    target_cluster_id = best_existing_id
                    print(
                        f"[AI] Merging '{tag}' into existing cluster ID {target_cluster_id} (sim: {best_similarity:.2f})"
                    )
                else:
                    # Genuinely new concept, create it in Supabase
                    print(f"[AI] Creating new cluster: {tag}")
                    new_cluster = (
                        supabase.table("mindspace_clusters")
                        .insert(
                            {
                                "name": tag,
                                "user_id": user_id,
                                "summary": f"Items related to {tag}",
                                "center_embedding": tag_embedding,
                            }
                        )
                        .execute()
                    )
                    if new_cluster.data:
                        target_cluster_id = new_cluster.data[0]["id"]

            if target_cluster_id:
                final_cluster_ids.append(target_cluster_id)

        results["embedding"] = embedding
        results["cluster_ids"] = final_cluster_ids
        results["tone"] = sentiment.get("tone", "Calm") if isinstance(sentiment, dict) else "Calm"

        return {"status": "success", "response": results}
    except Exception as e:
        print(f"[AI] Mindspace process error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mindspace/analyze_patterns")
async def analyze_patterns(request: dict):
    """Analyze patterns in mindspace entries."""
    llm_service = get_llm_service()

    try:
        entries = request.get("entries", [])
        content = "\n---\n".join(
            [f"Text: {e.get('content_text', '')}\nVision: {e.get('ai_description', '')}" for e in entries]
        )

        response = await llm_service.generate(
            prompt=content, provider="groq", prompt_type="mindspace_patterns"
        )
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/copilot/ignite")
async def copilot_ignite(request: CopilotIgniteRequest):
    """RAG-driven 'Active Co-Pilot' to break the cold start."""
    llm_service = get_llm_service()
    supabase = get_supabase_client()

    if not request.task_text or not request.userId:
        raise HTTPException(status_code=400, detail="task_text and userId are required")

    try:
        # 1. Embed Task Title
        embedding = await llm_service.generate_embedding(request.task_text)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to embed task text")

        # 2. Search Mindspace for Context
        search_results = supabase.rpc(
            "match_mindspace_entries",
            {
                "query_embedding": embedding,
                "match_threshold": 0.3,
                "match_count": 3,
                "p_user_id": request.userId,
            },
        ).execute()

        context_parts = []
        if search_results.data:
            for item in search_results.data:
                context_parts.append(item.get("content_text", ""))

        context_text = "\n---\n".join(context_parts) if context_parts else "No specific past context found."

        # 3. Generate Draft with RAG
        formatted_input = f"CONTEXT:\n{context_text}\n\nTASK: {request.task_text}"

        response = await llm_service.generate(
            prompt=formatted_input, provider="groq", prompt_type="copilot_ignite"
        )

        return {
            "status": "success",
            "draft": response.get("draft", "") if isinstance(response, dict) else response,
            "context_found": len(search_results.data) > 0 if search_results.data else False,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[AI] Copilot error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/alive")
async def alive():
    """Alive check endpoint."""
    return {"alive": True}
