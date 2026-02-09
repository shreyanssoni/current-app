from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service
from models.schemas import (
    StreamLockRequest,
    StreamAddRequest,
    StreamToggleRequest,
    StreamUpdateRequest,
    SmartRescheduleRequest,
    TaskCompleteRequest,
    TaskSuggestRequest,
    TaskScheduleRequest,
)

router = APIRouter()


async def find_next_available_slot(user_id: str, duration_min: int = 30) -> dict:
    """Find the next available time slot for scheduling a task."""
    supabase = get_supabase_client()
    now = datetime.utcnow()
    today_str = now.strftime("%Y-%m-%d")

    # Fetch all today's stream tasks
    result = (
        supabase.table("insights")
        .select("id, start_at, end_at")
        .eq("user_id", user_id)
        .eq("status", "ACTIVE")
        .not_.is_("start_at", "null")
        .order("start_at", desc=False)
        .execute()
    )
    schedule = result.data or []

    search_start = now.timestamp() * 1000  # milliseconds

    # Ensure we start at least at 09:00 if it's earlier
    nine_am = datetime.strptime(f"{today_str}T09:00:00", "%Y-%m-%dT%H:%M:%S").timestamp() * 1000
    if search_start < nine_am:
        search_start = nine_am

    buffer = 5 * 60 * 1000  # 5 mins in ms
    duration_ms = duration_min * 60 * 1000

    potential_start = search_start

    for task in schedule:
        task_start = datetime.fromisoformat(task["start_at"].replace("Z", "+00:00")).timestamp() * 1000
        task_end = datetime.fromisoformat(task["end_at"].replace("Z", "+00:00")).timestamp() * 1000

        # If there's enough space between potentialStart and this task's start
        if task_start - potential_start >= duration_ms + buffer:
            return {
                "start": datetime.fromtimestamp(potential_start / 1000).isoformat() + "Z",
                "end": datetime.fromtimestamp((potential_start + duration_ms) / 1000).isoformat() + "Z",
            }

        # Move potentialStart to AFTER this task
        if task_end + buffer > potential_start:
            potential_start = task_end + buffer

    # Append to the end
    return {
        "start": datetime.fromtimestamp(potential_start / 1000).isoformat() + "Z",
        "end": datetime.fromtimestamp((potential_start + duration_ms) / 1000).isoformat() + "Z",
    }


@router.get("/api/stream")
async def get_stream(userId: str = Query(default=TEST_USER_ID)):
    """Fetch today's mission & dock with daily reset logic."""
    supabase = get_supabase_client()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        # 1. Daily Reset Check
        markers_result = (
            supabase.table("insights")
            .select("*")
            .eq("user_id", userId)
            .eq("type", "SYSTEM")
            .eq("content", "LAST_RESET")
            .order("created_at", desc=True)
            .execute()
        )
        markers = markers_result.data or []
        reset_status = markers[0] if markers else None
        last_reset = reset_status.get("json_attributes", {}).get("date") if reset_status else None

        print(f"[RESET] Status: today={today}, found={len(markers)}, lastReset={last_reset}")

        if last_reset != today:
            try:
                # Clean up duplicate markers
                if len(markers) > 1:
                    to_delete = [m["id"] for m in markers[1:]]
                    print(f"[RESET] Cleaning up {len(to_delete)} duplicate markers")
                    supabase.table("insights").delete().in_("id", to_delete).execute()

                print(f"[RESET] Triggering daily reset. Last was {last_reset}")

                # Get active tasks in stream
                active_result = (
                    supabase.table("insights")
                    .select("id, json_attributes")
                    .eq("user_id", userId)
                    .eq("status", "ACTIVE")
                    .execute()
                )
                active_in_stream = active_result.data or []
                stream_tasks = [
                    t for t in active_in_stream if t.get("json_attributes", {}).get("in_stream") is True
                ]

                # Clear in_stream and locked flags
                for task in stream_tasks:
                    attr = dict(task.get("json_attributes", {}))
                    attr.pop("in_stream", None)
                    attr.pop("locked", None)
                    supabase.table("insights").update(
                        {"json_attributes": attr, "updated_at": datetime.utcnow().isoformat()}
                    ).eq("id", task["id"]).execute()

                # Update or create reset marker
                if reset_status:
                    print(f"[RESET] Updating existing marker for {userId}")
                    supabase.table("insights").update(
                        {"json_attributes": {"date": today}, "updated_at": datetime.utcnow().isoformat()}
                    ).eq("user_id", userId).eq("type", "SYSTEM").eq("content", "LAST_RESET").execute()
                else:
                    print(f"[RESET] Inserting NEW marker for {userId}")
                    supabase.table("insights").insert(
                        {
                            "user_id": userId,
                            "type": "SYSTEM",
                            "content": "LAST_RESET",
                            "status": "ACTIVE",
                            "json_attributes": {"date": today},
                        }
                    ).execute()

                print(f"[RESET] Daily reset marker update attempted for {userId}. Date: {today}")
            except Exception as err:
                print(f"[RESET] CRITICAL: Failed inside daily reset logic: {err}")
        else:
            print(f"[RESET] No reset needed for {userId}. Last reset was {last_reset}")

        # 2. Fetch Tasks
        tasks_result = (
            supabase.table("insights")
            .select("*")
            .eq("user_id", userId)
            .eq("status", "ACTIVE")
            .order("start_at", desc=False, nullsfirst=False)
            .execute()
        )
        data = tasks_result.data or []

        actual_tasks = [t for t in data if t.get("type") != "SYSTEM"]
        result = {
            "stream": [t for t in actual_tasks if t.get("json_attributes", {}).get("in_stream") is True],
            "dock": [t for t in actual_tasks if not t.get("json_attributes", {}).get("in_stream")],
        }

        return result

    except Exception as e:
        print(f"Stream fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/stream/today")
async def get_today_plan(userId: str = Query(default=TEST_USER_ID)):
    """Get today's scheduled tasks."""
    supabase = get_supabase_client()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        result = (
            supabase.table("insights")
            .select("*")
            .eq("user_id", userId)
            .eq("status", "ACTIVE")
            .eq("json_attributes->>scheduled_date", today)
            .execute()
        )
        return {"plan": result.data or []}
    except Exception as e:
        print(f"Fetch plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stream/lock")
async def lock_stream(request: StreamLockRequest):
    """Lock the daily plan."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    if not request.tasks:
        raise HTTPException(status_code=400, detail="Missing tasks")

    try:
        stream_date = datetime.utcnow().strftime("%Y-%m-%d")
        result = (
            supabase.table("daily_streams")
            .upsert(
                {
                    "user_id": user_id,
                    "tasks": request.tasks,
                    "stream_date": stream_date,
                    "created_at": datetime.utcnow().isoformat(),
                },
                on_conflict="user_id, stream_date",
            )
            .execute()
        )
        return {"status": "success", "data": result.data}
    except Exception as e:
        print(f"Lock error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stream/add")
async def add_to_stream(request: StreamAddRequest):
    """Quick add directly to Dock or Stream."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    if not request.content:
        raise HTTPException(status_code=400, detail="Content is required")

    try:
        start_at = None
        end_at = None

        if request.in_stream:
            slot = await find_next_available_slot(user_id, 30)
            start_at = slot["start"]
            end_at = slot["end"]

        json_attributes = {"in_stream": request.in_stream}
        if request.duration:
            json_attributes["duration"] = request.duration

        result = (
            supabase.table("insights")
            .insert(
                {
                    "user_id": user_id,
                    "content": request.content,
                    "type": request.type.upper(),
                    "status": "ACTIVE",
                    "start_at": start_at,
                    "end_at": end_at,
                    "locked": request.in_stream,
                    "json_attributes": json_attributes,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
            .execute()
        )
        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"Queue add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/stream/toggle")
async def toggle_stream(request: StreamToggleRequest):
    """Toggle task between stream and dock."""
    supabase = get_supabase_client()

    if not request.id:
        raise HTTPException(status_code=400, detail="ID is required")

    try:
        task_result = supabase.table("insights").select("*").eq("id", request.id).single().execute()
        task = task_result.data

        start_at = task.get("start_at")
        end_at = task.get("end_at")

        if request.in_stream:
            # Find a new slot when moving to stream
            slot = await find_next_available_slot(task["user_id"], 30)
            start_at = slot["start"]
            end_at = slot["end"]
            print(f"[TOGGLE] Smart scheduling task {request.id}: {start_at} -> {end_at}")
        else:
            # Move to Dock - Clear timings
            start_at = None
            end_at = None
            print(f"[TOGGLE] Moving task {request.id} to dock, clearing timings")

        updated_attr = dict(task.get("json_attributes") or {})
        updated_attr["in_stream"] = request.in_stream
        updated_attr["locked"] = request.in_stream  # Auto-lock when entering stream
        updated_attr.pop("zone", None)

        supabase.table("insights").update(
            {
                "json_attributes": updated_attr,
                "start_at": start_at,
                "end_at": end_at,
                "locked": request.in_stream,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", request.id).execute()

        return {"success": True, "start_at": start_at, "end_at": end_at}
    except Exception as e:
        print(f"Stream toggle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/stream/{id}")
async def update_stream_task(id: str, request: StreamUpdateRequest):
    """Update a stream task."""
    supabase = get_supabase_client()

    allowed = ["start_at", "end_at", "content", "status", "json_attributes"]
    filtered_update = {k: v for k, v in request.model_dump().items() if k in allowed and v is not None}
    filtered_update["updated_at"] = datetime.utcnow().isoformat()

    try:
        result = supabase.table("insights").update(filtered_update).eq("id", id).execute()
        return result.data[0] if result.data else {}
    except Exception as e:
        print(f"Task update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/stream/{id}")
async def delete_stream_task(id: str):
    """Delete a task from stream/dock."""
    supabase = get_supabase_client()

    try:
        supabase.table("insights").delete().eq("id", id).execute()
        return {"success": True, "id": id}
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/stream/reschedule-smart")
async def smart_reschedule(request: SmartRescheduleRequest):
    """Smart reschedule a single task."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    try:
        start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999)

        schedule_result = (
            supabase.table("insights")
            .select("id, start_at, end_at")
            .eq("user_id", user_id)
            .eq("status", "ACTIVE")
            .gte("start_at", start_of_day.isoformat())
            .lte("end_at", end_of_day.isoformat())
            .order("start_at", desc=False)
            .execute()
        )
        schedule = schedule_result.data or []

        # Find first gap after NOW
        now = datetime.utcnow()
        remainder = 15 - (now.minute % 15)
        check_time = now + timedelta(minutes=remainder)

        work_end = now.replace(hour=23, minute=0, second=0, microsecond=0)

        found_slot = None

        while check_time < work_end:
            proposed_end = check_time + timedelta(minutes=request.duration)

            has_collision = False
            for t in schedule:
                if t["id"] == request.id:
                    continue
                t_start = datetime.fromisoformat(t["start_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                t_end = datetime.fromisoformat(t["end_at"].replace("Z", "+00:00")).replace(tzinfo=None)

                if (
                    (check_time >= t_start and check_time < t_end)
                    or (proposed_end > t_start and proposed_end <= t_end)
                    or (check_time <= t_start and proposed_end >= t_end)
                ):
                    has_collision = True
                    break

            if not has_collision:
                found_slot = {"start": check_time, "end": proposed_end}
                break

            check_time += timedelta(minutes=15)

        if found_slot:
            supabase.table("insights").update(
                {
                    "start_at": found_slot["start"].isoformat() + "Z",
                    "end_at": found_slot["end"].isoformat() + "Z",
                    "updated_at": datetime.utcnow().isoformat(),
                }
            ).eq("id", request.id).execute()
            return {"success": True, "slot": found_slot}
        else:
            raise HTTPException(status_code=409, detail="No available slots found for today.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Smart Reschedule error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/tasks/{taskId}/complete")
async def complete_task(taskId: str, request: TaskCompleteRequest):
    """Complete a specific task."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        stream_result = (
            supabase.table("daily_streams")
            .select("*")
            .eq("user_id", user_id)
            .gte("created_at", today.isoformat())
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not stream_result.data:
            raise HTTPException(status_code=404, detail="No active plan found for today")

        stream = stream_result.data[0]
        updated_tasks = [
            {**t, "status": "DONE"} if t.get("id") == taskId else t for t in stream.get("tasks", [])
        ]

        supabase.table("daily_streams").update({"tasks": updated_tasks}).eq("id", stream["id"]).execute()

        return {"status": "success", "taskId": taskId}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Task complete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/tasks/suggest")
async def suggest_task(request: TaskSuggestRequest):
    """Suggest task breakdown using AI."""
    llm_service = get_llm_service()
    final_context = request.taskContext or request.task or "No context provided"

    try:
        suggestion = await llm_service.generate(
            prompt=f'Task: "{final_context}", Resistance: {request.resistance or 50}',
            provider="groq",
            prompt_type="negotiate_resistance",
        )
        return {"suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/tasks/backlog")
async def get_backlog(userId: str = Query(default=TEST_USER_ID)):
    """Get unscheduled tasks (backlog)."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("insights")
            .select("*")
            .eq("user_id", userId)
            .eq("status", "ACTIVE")
            .eq("type", "TASK")
            .order("created_at", desc=True)
            .execute()
        )
        return {"backlog": result.data or []}
    except Exception as e:
        print(f"Backlog fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/tasks/{id}/schedule")
async def schedule_task(id: str, request: TaskScheduleRequest):
    """Schedule a task."""
    supabase = get_supabase_client()

    if not request.date or not request.time:
        raise HTTPException(status_code=400, detail="Date and time are required")

    try:
        insight_result = supabase.table("insights").select("json_attributes").eq("id", id).single().execute()
        insight = insight_result.data

        updated_attributes = {
            **(insight.get("json_attributes") or {}),
            "scheduled_date": request.date,
            "scheduled_time": request.time,
        }

        supabase.table("insights").update(
            {"json_attributes": updated_attributes, "status": "ACTIVE"}
        ).eq("id", id).execute()

        return {"success": True}
    except Exception as e:
        print(f"Schedule error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
