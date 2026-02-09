from datetime import datetime
from fastapi import APIRouter, HTTPException

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service
from models.schemas import MissionGenerateRequest, MissionCascadeRequest, MissionSwapRequest

router = APIRouter()


@router.post("/api/mission/generate")
async def generate_mission(request: MissionGenerateRequest):
    """Generate daily mission."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()
    user_id = request.userId or TEST_USER_ID

    try:
        # 1. Fetch all Dock tasks
        dock_result = (
            supabase.table("insights")
            .select("id, content, json_attributes")
            .eq("user_id", user_id)
            .eq("status", "ACTIVE")
            .execute()
        )

        dock_tasks = dock_result.data or []
        dock_only = [t for t in dock_tasks if not t.get("json_attributes", {}).get("in_stream")]

        if not dock_only:
            raise HTTPException(status_code=400, detail="No tasks in the Dock to generate from.")

        # 2. Call AI to select
        print(f"[MISSION] Generating for user {user_id} with {len(dock_only)} tasks.")
        mission_result = await llm_service.generate(
            prompt=str(dock_only), provider="groq", prompt_type="select_mission"
        )
        selected_ids = mission_result.get("selected_ids", []) if isinstance(mission_result, dict) else []
        print(f"[MISSION] AI selected IDs: {selected_ids}")

        if not selected_ids:
            print("[MISSION] AI returned no selections or error.")
            raise Exception("AI failed to select any tasks.")

        selected_tasks = [t for t in dock_only if t["id"] in selected_ids]

        # 3. Generate Schedule
        user_result = supabase.table("users").select("timezone").eq("id", user_id).single().execute()
        timezone = user_result.data.get("timezone", "UTC") if user_result.data else "UTC"
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        print(f"[MISSION] Scheduling for {user_id} @ {timezone}")

        schedule_input = {
            "user_start_time": "09:00",
            "current_date": today_str,
            "tasks": selected_tasks,
        }
        schedule_result = await llm_service.generate(
            prompt=str(schedule_input), provider="groq", prompt_type="generate_schedule"
        )
        schedule = schedule_result.get("schedule", []) if isinstance(schedule_result, dict) else []
        print(f"[MISSION] AI Schedule: {schedule}")

        # 4. Update tasks in DB
        committed_count = 0
        for item in schedule:
            task_id = item.get("id")
            start_time = item.get("start_time")
            end_time = item.get("end_time")

            start_utc = f"{today_str}T{start_time}:00Z"
            end_utc = f"{today_str}T{end_time}:00Z"

            target_task = next((dt for dt in selected_tasks if dt["id"] == task_id), None)
            if not target_task:
                continue

            attr = dict(target_task.get("json_attributes") or {})
            attr["in_stream"] = True
            attr["locked"] = True

            result = (
                supabase.table("insights")
                .update(
                    {
                        "json_attributes": attr,
                        "start_at": start_utc,
                        "end_at": end_utc,
                        "locked": True,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )
                .eq("id", task_id)
                .execute()
            )
            if result.data:
                committed_count += 1

        print(f"[MISSION] Successfully committed {committed_count} scheduled tasks.")
        return {"success": True, "count": committed_count}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Mission generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/mission/cascade")
async def cascade_mission(request: MissionCascadeRequest):
    """Cascade mission (push schedule back)."""
    supabase = get_supabase_client()
    user_id = request.userId or TEST_USER_ID

    try:
        # Fetch unfinished locked tasks in the stream
        tasks_result = (
            supabase.table("insights")
            .select("id, start_at, end_at")
            .eq("user_id", user_id)
            .eq("status", "ACTIVE")
            .eq("locked", True)
            .execute()
        )
        tasks = tasks_result.data or []

        if not tasks:
            return {"success": True, "count": 0}

        from datetime import timedelta

        for task in tasks:
            start_dt = datetime.fromisoformat(task["start_at"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(task["end_at"].replace("Z", "+00:00"))

            new_start = (start_dt + timedelta(minutes=request.minutes)).isoformat()
            new_end = (end_dt + timedelta(minutes=request.minutes)).isoformat()

            supabase.table("insights").update(
                {"start_at": new_start, "end_at": new_end, "updated_at": datetime.utcnow().isoformat()}
            ).eq("id", task["id"]).execute()

        return {"success": True, "count": len(tasks)}

    except Exception as e:
        print(f"Cascade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mission/swap")
async def swap_mission(request: MissionSwapRequest):
    """Swap mission task."""
    supabase = get_supabase_client()

    try:
        # 1. Get both tasks
        locked_result = (
            supabase.table("insights").select("*").eq("id", request.locked_id).single().execute()
        )
        dock_result = supabase.table("insights").select("*").eq("id", request.dock_id).single().execute()

        locked_task = locked_result.data
        dock_task = dock_result.data

        if not locked_task or not dock_task:
            raise Exception("Tasks not found")

        # 2. Unlock and move to Dock (Clear timings)
        locked_attr = dict(locked_task.get("json_attributes") or {})
        locked_attr.pop("in_stream", None)
        locked_attr.pop("locked", None)

        supabase.table("insights").update(
            {
                "json_attributes": locked_attr,
                "start_at": None,
                "end_at": None,
                "locked": False,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", request.locked_id).execute()

        # 3. Lock and move to Stream (Inherit timings)
        dock_attr = dict(dock_task.get("json_attributes") or {})
        dock_attr["in_stream"] = True
        dock_attr["locked"] = True

        supabase.table("insights").update(
            {
                "json_attributes": dock_attr,
                "start_at": locked_task["start_at"],
                "end_at": locked_task["end_at"],
                "locked": True,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", request.dock_id).execute()

        return {"success": True}

    except Exception as e:
        print(f"Mission swap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
