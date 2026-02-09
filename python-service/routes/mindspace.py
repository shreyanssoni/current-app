import json
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Form

from config import TEST_USER_ID
from services.supabase import get_supabase_client
from services.llm_gateway import get_llm_service, cosine_similarity
from models.schemas import MindspaceChatRequest

router = APIRouter()


@router.post("/api/mindspace/add")
async def add_mindspace(
    text: Optional[str] = Form(None),
    userId: str = Form(default=TEST_USER_ID),
    mediaType: str = Form(default="text"),
    imageUrl: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """Add entry to mindspace (processing pipeline)."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()
    final_image_url = imageUrl

    try:
        # A. Handle Image Upload if provided
        if image:
            print(f"[MINDSPACE] Uploading image: {image.filename}...")
            file_ext = image.filename.split(".")[-1] if image.filename else "jpg"
            file_name = f"{userId}_{int(datetime.utcnow().timestamp())}.{file_ext}"
            file_path = f"entries/{file_name}"

            file_content = await image.read()

            upload_result = supabase.storage.from_("mindspace_assets").upload(
                file_path, file_content, {"content-type": image.content_type or "image/jpeg", "upsert": "true"}
            )

            public_url = supabase.storage.from_("mindspace_assets").get_public_url(file_path)
            final_image_url = public_url

        # B. Fetch Existing Clusters (Context for Librarian)
        clusters_result = (
            supabase.table("mindspace_clusters")
            .select("id, name, center_embedding")
            .eq("user_id", userId)
            .limit(100)
            .execute()
        )
        existing_clusters = clusters_result.data or []

        # C. Process via AI (Vision, Embedding, Clustering, Tone)
        print(f"[MINDSPACE] Processing entry for user {userId} with {len(existing_clusters)} existing clusters...")

        ai_results = await process_mindspace_internally(
            text=text or "",
            image_url=final_image_url,
            user_id=userId,
            existing_clusters=existing_clusters,
            llm_service=llm_service,
            supabase=supabase,
        )

        # D. Storage: Save Entry
        entry_result = (
            supabase.table("mindspace_entries")
            .insert(
                {
                    "user_id": userId,
                    "content_text": text,
                    "media_url": final_image_url,
                    "media_type": "image" if final_image_url else "text",
                    "ai_description": ai_results.get("ai_description"),
                    "embedding": ai_results.get("embedding"),
                    "emotional_tone": ai_results.get("tone"),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            .execute()
        )
        entry = entry_result.data[0] if entry_result.data else None

        if not entry:
            raise Exception("Failed to create entry")

        # E. Link Entry <-> Clusters
        cluster_ids = ai_results.get("cluster_ids", [])
        if cluster_ids:
            print(f"[MINDSPACE] Linking entry {entry['id']} to clusters: {cluster_ids}")
            link_data = [{"entry_id": entry["id"], "cluster_id": cid} for cid in cluster_ids]
            supabase.table("entry_clusters").insert(link_data).execute()

        return {"success": True, "entry": entry}

    except Exception as e:
        print(f"[MINDSPACE] Add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_mindspace_internally(
    text: str,
    image_url: Optional[str],
    user_id: str,
    existing_clusters: list,
    llm_service,
    supabase,
) -> dict:
    """Internal mindspace processing (vision, embedding, clustering, sentiment)."""
    import asyncio

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
                new_cluster_result = (
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
                if new_cluster_result.data:
                    target_cluster_id = new_cluster_result.data[0]["id"]

        if target_cluster_id:
            final_cluster_ids.append(target_cluster_id)

    results["embedding"] = embedding
    results["cluster_ids"] = final_cluster_ids
    results["tone"] = sentiment.get("tone", "Calm") if isinstance(sentiment, dict) else "Calm"

    return results


@router.get("/api/mindspace/feed")
async def get_mindspace_feed(
    userId: str = Query(default=TEST_USER_ID),
    q: Optional[str] = Query(default=None),
):
    """Feed endpoint (chronological + resurfaced + search)."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()

    try:
        entries = []

        if q:
            print(f"[MINDSPACE] Searching for: {q}...")
            embedding = await llm_service.generate_embedding(q)

            if embedding:
                # Use Supabase RPC for vector similarity search
                search_result = supabase.rpc(
                    "match_mindspace_entries",
                    {
                        "query_embedding": embedding,
                        "match_threshold": 0.5,
                        "match_count": 50,
                        "p_user_id": userId,
                    },
                ).execute()
                entries = search_result.data or []
            else:
                # Fallback to text search
                print("[MINDSPACE] Embedding failed, falling back to text search.")
                text_result = (
                    supabase.table("mindspace_entries")
                    .select("*")
                    .eq("user_id", userId)
                    .or_(f"content_text.ilike.%{q}%,ai_description.ilike.%{q}%")
                    .order("created_at", desc=True)
                    .limit(50)
                    .execute()
                )
                entries = text_result.data or []
        else:
            # Standard feed: Fetch last 50 entries
            standard_result = (
                supabase.table("mindspace_entries")
                .select("*")
                .eq("user_id", userId)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            entries = standard_result.data or []

        # Fetch a "Resurfaced" memory (older than 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        resurfaced_result = (
            supabase.table("mindspace_entries")
            .select("*")
            .eq("user_id", userId)
            .lt("created_at", thirty_days_ago.isoformat())
            .limit(5)
            .execute()
        )
        resurfaced = resurfaced_result.data or []

        feed = list(entries)

        if resurfaced:
            import random

            memory = random.choice(resurfaced)
            memory["is_resurfaced"] = True
            # Inject at 10th position if possible
            if len(feed) >= 10:
                feed.insert(9, memory)
            elif feed:
                feed.append(memory)

        return {"feed": feed}

    except Exception as e:
        print(f"[MINDSPACE] Feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/mindspace/patterns")
async def get_mindspace_patterns(userId: str = Query(default=TEST_USER_ID)):
    """Patterns/Intelligence endpoint."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()

    try:
        # Fetch last 20 entries for analysis
        entries_result = (
            supabase.table("mindspace_entries")
            .select("content_text, ai_description, emotional_tone, created_at")
            .eq("user_id", userId)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        entries = entries_result.data

        if not entries:
            raise Exception("No entries")

        # Always generate new pattern analysis
        print(f"[MINDSPACE] Generating fresh patterns for user {userId}...")
        content = "\n---\n".join(
            [
                f"Text: {e.get('content_text', '')}\nVision: {e.get('ai_description', '')}"
                for e in entries
            ]
        )

        insight_text = await llm_service.generate(
            prompt=content, provider="groq", prompt_type="mindspace_patterns"
        )

        # Cache it for the user
        supabase.table("mindspace_insights").insert(
            {
                "user_id": userId,
                "insight_text": insight_text,
                "created_at": datetime.utcnow().isoformat(),
            }
        ).execute()

        return {"insight": insight_text}

    except Exception as e:
        print(f"[MINDSPACE] Patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/mindspace/clusters")
async def get_mindspace_clusters(userId: str = Query(default=TEST_USER_ID)):
    """Get clusters."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("mindspace_clusters")
            .select("*")
            .eq("user_id", userId)
            .order("name", desc=False)
            .execute()
        )
        return {"clusters": result.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/mindspace/clusters/{clusterId}/entries")
async def get_cluster_entries(clusterId: str):
    """Cluster filtered view."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("entry_clusters")
            .select("mindspace_entries(*)")
            .eq("cluster_id", clusterId)
            .execute()
        )
        entries = [d.get("mindspace_entries") for d in (result.data or []) if d.get("mindspace_entries")]
        return {"entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mindspace/chat")
async def mindspace_chat(request: MindspaceChatRequest):
    """RAG-based chat endpoint."""
    supabase = get_supabase_client()
    llm_service = get_llm_service()
    user_id = request.userId or TEST_USER_ID

    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # 1. Embed query
        embedding = await llm_service.generate_embedding(request.query)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to embed query")

        # 2. Semantic Search in Supabase
        search_result = supabase.rpc(
            "match_mindspace_entries",
            {
                "query_embedding": embedding,
                "match_threshold": 0.3,
                "match_count": 10,
                "p_user_id": user_id,
            },
        ).execute()

        if not search_result.data:
            context_text = "No relevant Mindspace entries found."
        else:
            # 3. Format Context
            context_parts = []
            for item in search_result.data:
                date_str = item.get("created_at", "Unknown Date")
                content = item.get("content_text") or "Image Insight"
                vision = item.get("ai_description") or ""
                part = f"[{date_str}] {content}\nVision: {vision}"
                context_parts.append(part)

            context_text = "\n\n".join(context_parts)

        # 4. Generate RAG Answer
        full_prompt = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {request.query}"

        answer = await llm_service.generate(
            prompt=full_prompt, provider="groq", prompt_type="mindspace_chat"
        )

        return {
            "status": "success",
            "answer": answer,
            "context_count": len(search_result.data) if search_result.data else 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[MINDSPACE] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/mindspace/{id}")
async def delete_mindspace_entry(id: str):
    """Delete mindspace entry."""
    supabase = get_supabase_client()

    try:
        supabase.table("mindspace_entries").delete().eq("id", id).execute()
        return {"success": True, "id": id}
    except Exception as e:
        print(f"Mindspace delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
