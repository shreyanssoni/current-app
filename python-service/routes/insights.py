from datetime import datetime
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service
from models.schemas import (
    InsightCaptureRequest,
    InsightMorphRequest,
    InsightRescheduleRequest,
    InsightDecomposeRequest,
)

router = APIRouter()


async def enrich_insight(insight_id: str, content: str):
    """Background task to enrich an insight with AI analysis."""
    llm_service = get_llm_service()
    supabase = get_supabase_client()

    try:
        print(f"[AI] Sending to AI: {insight_id}...")

        # Run analysis and embedding in parallel
        import asyncio

        analysis_task = llm_service.generate(
            prompt=content, provider="groq", prompt_type="auto_tag_capture"
        )
        embedding_task = llm_service.generate_embedding(text=content)

        analysis, embedding = await asyncio.gather(analysis_task, embedding_task)

        if analysis:
            tags = analysis.get("tags", [])
            insight_type = analysis.get("type", "JOURNAL")
            sentiment = analysis.get("sentiment", "NEUTRAL")

            print(f"[AI] AI Response for {insight_id}: [{', '.join(tags)}] (with embedding)")

            supabase.table("insights").update(
                {
                    "context_tags": tags,
                    "type": insight_type.upper(),
                    "sentiment": sentiment,
                    "embedding": embedding,
                    "meta_analysis": analysis,
                }
            ).eq("id", insight_id).execute()
        else:
            print(f"[AI] AI Failed for {insight_id}")

    except Exception as e:
        print(f"[AI] Critical Failure for {insight_id}: {e}")


@router.post("/api/insights/capture")
async def capture_insight(request: InsightCaptureRequest, background_tasks: BackgroundTasks):
    """Capture and enrich an insight."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    if not request.content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        print(f'[INSIGHT] Insight received: "{request.content[:30]}..."')

        # 1. Immediate Insert (RAW)
        result = (
            supabase.table("insights")
            .insert(
                {
                    "user_id": user_id,
                    "content": request.content,
                    "type": request.type.upper(),
                    "status": request.status,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            .execute()
        )

        if not result.data:
            raise Exception("Failed to insert insight")

        insight_id = result.data[0]["id"]

        # 2. Background Enrichment (AI)
        background_tasks.add_task(enrich_insight, insight_id, request.content)

        return {"status": "success", "insightId": insight_id}

    except Exception as e:
        print(f"Capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/insights/inbox")
async def get_inbox(userId: str = Query(default=TEST_USER_ID)):
    """Get enriched inbox."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("insights")
            .select("*")
            .eq("user_id", userId)
            .eq("status", "INBOX")
            .order("created_at", desc=True)
            .execute()
        )
        return {"insights": result.data or []}
    except Exception as e:
        print(f"Inbox fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/insights/{id}/morph")
async def morph_insight(id: str, request: InsightMorphRequest):
    """Morph insight (promote, save, or trash)."""
    supabase = get_supabase_client()

    if not request.action:
        raise HTTPException(status_code=400, detail="Action is required")

    try:
        update_data = {}

        if request.action == "promote":
            update_data = {"status": "ACTIVE", "type": "TASK"}
        elif request.action == "save":
            update_data = {"status": "ARCHIVED"}
        elif request.action == "trash":
            update_data = {"status": "TRASH"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        supabase.table("insights").update(update_data).eq("id", id).execute()
        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Morph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/insights/{id}/reschedule")
async def reschedule_insight(id: str, request: InsightRescheduleRequest):
    """Reschedule an insight."""
    supabase = get_supabase_client()

    try:
        final_start = request.start_at
        final_end = request.end_at

        if request.minutes:
            task_result = supabase.table("insights").select("start_at, end_at").eq("id", id).single().execute()
            task = task_result.data
            if task:
                start_dt = datetime.fromisoformat(task["start_at"].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(task["end_at"].replace("Z", "+00:00"))
                from datetime import timedelta

                final_start = (start_dt + timedelta(minutes=request.minutes)).isoformat()
                final_end = (end_dt + timedelta(minutes=request.minutes)).isoformat()

        supabase.table("insights").update(
            {
                "start_at": final_start,
                "end_at": final_end,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", id).execute()

        return {"success": True, "start_at": final_start, "end_at": final_end}

    except Exception as e:
        print(f"Reschedule error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/insights/{id}/complete")
async def complete_insight(id: str):
    """Complete an insight."""
    supabase = get_supabase_client()

    try:
        # Fetch current attributes to clear zone
        task_result = supabase.table("insights").select("json_attributes").eq("id", id).single().execute()
        task = task_result.data

        updated_attributes = dict(task.get("json_attributes") or {})
        updated_attributes.pop("zone", None)

        supabase.table("insights").update(
            {
                "status": "DONE",
                "json_attributes": updated_attributes,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", id).execute()

        return {"success": True}

    except Exception as e:
        print(f"Complete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/insights/decompose")
async def decompose_insight(request: InsightDecomposeRequest):
    """Decompose a task into subtasks (The Prism Feature)."""
    llm_service = get_llm_service()
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    if not request.id:
        raise HTTPException(status_code=400, detail="ID is required")

    try:
        # 1. Fetch original task
        original_result = supabase.table("insights").select("*").eq("id", request.id).single().execute()
        original = original_result.data

        if not original:
            raise Exception("Original task not found")

        # 2. Call AI breakdown service
        print(f'[DECOMPOSE] Goal: "{original["content"]}"')
        result = await llm_service.generate(
            prompt=original["content"], provider="groq", prompt_type="decompose_goal"
        )
        subtasks = result.get("tasks", []) if isinstance(result, dict) else []
        print(f"[DECOMPOSE] AI raw output subtasks: {subtasks}")

        if not subtasks:
            print("[DECOMPOSE] AI returned EMPTY list.")
            raise Exception("AI could not break down this goal")

        # 3. Archive the original task and mark as PROJECT
        supabase.table("insights").update(
            {"status": "ARCHIVED", "type": "project", "updated_at": datetime.utcnow().isoformat()}
        ).eq("id", request.id).execute()

        # 4. Insert new subtasks into the Dock
        entries = [
            {
                "user_id": user_id,
                "content": task_content,
                "type": "task",
                "status": "ACTIVE",
                "json_attributes": {"in_stream": False, "parent_id": request.id},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            for task_content in subtasks
        ]

        supabase.table("insights").insert(entries).execute()

        return {"success": True, "count": len(subtasks)}

    except Exception as e:
        print(f"Decomposition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
