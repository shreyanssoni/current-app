from datetime import datetime
from fastapi import APIRouter, HTTPException

from services.supabase import get_supabase_client
from models.schemas import QueueMoveRequest

router = APIRouter()


@router.patch("/api/queue/move")
async def move_queue(request: QueueMoveRequest):
    """Move task between zones."""
    supabase = get_supabase_client()

    if not request.id or not request.targetZone:
        raise HTTPException(status_code=400, detail="ID and targetZone are required")

    try:
        # 1. If moving to FOCUS, demote current FOCUS to NEXT
        if request.targetZone == "FOCUS":
            current_focus_result = (
                supabase.table("insights")
                .select("id, json_attributes")
                .eq("status", "ACTIVE")
                .eq("json_attributes->>zone", "FOCUS")
                .execute()
            )
            current_focus = current_focus_result.data or []

            for item in current_focus:
                updated_attr = dict(item.get("json_attributes") or {})
                updated_attr["zone"] = "NEXT"
                supabase.table("insights").update(
                    {"json_attributes": updated_attr, "updated_at": datetime.utcnow().isoformat()}
                ).eq("id", item["id"]).execute()

        # 2. Fetch and update the target task
        task_result = (
            supabase.table("insights").select("json_attributes").eq("id", request.id).single().execute()
        )
        task = task_result.data

        updated_attributes = dict(task.get("json_attributes") or {})
        updated_attributes["zone"] = request.targetZone

        supabase.table("insights").update(
            {"json_attributes": updated_attributes, "updated_at": datetime.utcnow().isoformat()}
        ).eq("id", request.id).execute()

        return {"success": True}

    except Exception as e:
        print(f"Move error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
