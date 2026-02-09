from fastapi import APIRouter, Query, HTTPException

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service
from models.schemas import LibrarySparkRequest

router = APIRouter()


@router.get("/api/library/search")
async def search_library(
    userId: str = Query(default=TEST_USER_ID),
    q: str = Query(default=None),
    tag: str = Query(default=None),
):
    """Library search (semantic + tag filter)."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()

    try:
        if q:
            # Semantic search
            print(f'[LIBRARY] Semantic search for: "{q}"')
            query_vector = await llm_service.generate_embedding(q)

            if not query_vector:
                raise HTTPException(status_code=500, detail="Failed to generate embedding")

            # Perform Vector Search via RPC
            result = supabase.rpc(
                "match_insights",
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.5,
                    "match_count": 20,
                    "filter_user": userId,
                },
            ).execute()

            data = result.data or []

            counts = {
                "total": len(data),
                "ideas": len([i for i in data if i.get("type") == "IDEA"]),
                "tasks": len([i for i in data if i.get("type") == "TASK"]),
                "journals": len([i for i in data if i.get("type") == "JOURNAL"]),
                "principles": len([i for i in data if i.get("type") == "PRINCIPLE"]),
            }

            return {"results": data, "counts": counts}

        else:
            # Fallback to standard date-based retrieval
            query = (
                supabase.table("insights")
                .select("*")
                .eq("user_id", userId)
                .eq("status", "ARCHIVED")
                .order("created_at", desc=True)
                .limit(50)
            )

            if tag and tag != "ALL":
                query = query.contains("context_tags", [tag])

            result = query.execute()
            data = result.data or []

            counts = {
                "total": len(data),
                "ideas": len([i for i in data if i.get("type") == "IDEA"]),
                "tasks": len([i for i in data if i.get("type") == "TASK"]),
                "journals": len([i for i in data if i.get("type") == "JOURNAL"]),
                "principles": len([i for i in data if i.get("type") == "PRINCIPLE"]),
            }

            return {"results": data, "counts": counts}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Library search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/library/spark")
async def spark_idea(request: LibrarySparkRequest):
    """Spark idea (AI decomposition)."""
    llm_service = get_llm_service()

    if not request.content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        print(f'[SPARK] Generating tasks for: "{request.content}"')

        result = await llm_service.generate(
            prompt=request.content, provider="groq", prompt_type="spark_idea"
        )

        tasks = result.get("tasks", []) if isinstance(result, dict) else []

        print(f"[SPARK] Generated {len(tasks)} tasks")
        return {"tasks": tasks}

    except Exception as e:
        print(f"Spark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
