import express from 'express';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import multer from 'multer';
import { callAI, analyzeInsight, getEmbedding, decomposeGoal, generateMission, generateSchedule, processMindspace, analyzeMindspacePatterns, chatMindspace } from './services/aiService';
import { TEST_USER_ID } from './config';

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3001;

console.log(`Starting server on port: ${port}`);

// Supabase Init
const supabaseUrl = process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_KEY || '';
const supabase = createClient(supabaseUrl, supabaseKey);

app.use(express.json());

// Set up Multer for image uploads
const upload = multer({ storage: multer.memoryStorage() });

app.get('/', (req, res) => {
    res.send('Current Core Node Service Running');
});

// --- HELPER: Find Next Available Gap ---
async function findNextAvailableSlot(userId: string, durationMin: number = 30) {
    const now = new Date();
    const todayStr = now.toISOString().split('T')[0];

    // 1. Fetch all today's stream tasks
    const { data: schedule, error } = await supabase
        .from('insights')
        .select('id, start_at, end_at')
        .eq('user_id', userId)
        .eq('status', 'ACTIVE')
        .not('start_at', 'is', null)
        .order('start_at', { ascending: true });

    if (error) throw error;

    let searchStart = now.getTime();

    // Ensure we start at least at 09:00 if it's earlier, or just now
    const nineAM = new Date(`${todayStr}T09:00:00`).getTime();
    if (searchStart < nineAM) searchStart = nineAM;

    // Buffer of 5 mins between tasks if possible
    const buffer = 5 * 60000;
    const durationMs = durationMin * 60000;

    let potentialStart = searchStart;

    for (const task of (schedule || [])) {
        const taskStart = new Date(task.start_at).getTime();
        const taskEnd = new Date(task.end_at).getTime();

        // If there's enough space between potentialStart and this task's start
        if (taskStart - potentialStart >= durationMs + buffer) {
            return {
                start: new Date(potentialStart).toISOString(),
                end: new Date(potentialStart + durationMs).toISOString()
            };
        }

        // Otherwise, move potentialStart to AFTER this task
        if (taskEnd + buffer > potentialStart) {
            potentialStart = taskEnd + buffer;
        }
    }

    // If we reached here, just append to the end
    return {
        start: new Date(potentialStart).toISOString(),
        end: new Date(potentialStart + durationMs).toISOString()
    };
}

// Lock the daily plan
app.post('/api/stream/lock', async (req, res) => {
    const { userId = TEST_USER_ID, tasks } = req.body;

    if (!tasks) {
        return res.status(400).json({ error: 'Missing tasks' });
    }

    try {
        const streamDate = new Date().toISOString().split('T')[0];
        const { data, error } = await supabase
            .from('daily_streams')
            .upsert(
                {
                    user_id: userId,
                    tasks: tasks,
                    stream_date: streamDate,
                    created_at: new Date().toISOString()
                },
                { onConflict: 'user_id, stream_date' }
            )
            .select();

        if (error) throw error;
        res.json({ status: 'success', data });
    } catch (error: any) {
        console.error('Lock error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get today's plan
app.get('/api/stream/today', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;

    try {
        const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD

        const { data, error } = await supabase
            .from('insights')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'ACTIVE')
            .eq('json_attributes->>scheduled_date', today);

        if (error) throw error;
        res.json({ plan: data || [] });
    } catch (error: any) {
        console.error('Fetch plan error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Complete a specific task
app.patch('/api/tasks/:taskId/complete', async (req, res) => {
    const { taskId } = req.params;
    const { userId = TEST_USER_ID } = req.body;

    try {
        // 1. Get today's stream
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const { data: stream, error: fetchError } = await supabase
            .from('daily_streams')
            .select('*')
            .eq('user_id', userId)
            .gte('created_at', today.toISOString())
            .order('created_at', { ascending: false })
            .limit(1)
            .single();

        if (fetchError || !stream) {
            return res.status(404).json({ error: 'No active plan found for today' });
        }

        // 2. Map through tasks and update the matching one
        const updatedTasks = stream.tasks.map((t: any) => {
            if (t.id === taskId) {
                return { ...t, status: 'DONE' };
            }
            return t;
        });

        // 3. Update the stream
        const { error: updateError } = await supabase
            .from('daily_streams')
            .update({ tasks: updatedTasks })
            .eq('id', stream.id);

        if (updateError) throw updateError;

        res.json({ status: 'success', taskId });
    } catch (error: any) {
        console.error('Task complete error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Example route using the AI service
app.post('/api/tasks/suggest', async (req, res) => {
    const { taskContext, task, resistance } = req.body;
    const finalContext = taskContext || task || "No context provided";

    try {
        const suggestion = await callAI(
            `Task: "${finalContext}", Resistance: ${resistance || 50}`,
            'groq',
            undefined,
            'negotiate_resistance'
        );
        res.json({ suggestion });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// Insight Capture & Enrichment
app.post('/api/insights/capture', async (req, res) => {
    const { content, userId = TEST_USER_ID, type = 'raw', status = 'INBOX' } = req.body;

    if (!content) {
        return res.status(400).json({ error: 'Content is required' });
    }

    try {
        console.log(`📝 Insight received: "${content.substring(0, 30)}..."`);

        // 1. Immediate Insert (RAW)
        const { data, error } = await supabase
            .from('insights')
            .insert([
                {
                    user_id: userId,
                    content: content,
                    type: type.toUpperCase(),
                    status: status,
                    created_at: new Date().toISOString()
                }
            ])
            .select('id');

        if (error) throw error;
        if (!data || data.length === 0) throw new Error('Failed to insert insight');
        const insightId = data[0].id;

        // 2. Background Enrichment (AI)
        (async () => {
            try {
                console.log(`🤖 Sending to AI: ${insightId}...`);
                const aiResponse = await analyzeInsight(content);

                if (aiResponse) {
                    const { tags, type, sentiment, embedding } = aiResponse;
                    console.log(`✅ AI Response for ${insightId}: [${tags.join(', ')}] (with embedding)`);

                    const { error: updateError } = await supabase
                        .from('insights')
                        .update({
                            context_tags: tags,
                            type: type.toUpperCase(),
                            sentiment: sentiment,
                            embedding: embedding, // Save the vector
                            meta_analysis: aiResponse
                        })
                        .eq('id', insightId)
                        .select('id');

                    if (updateError) console.error('Enrichment Update Error:', updateError);
                } else {
                    console.log(`❌ AI Failed for ${insightId}`);
                }
            } catch (aiErr: any) {
                console.error(`❌ AI Critical Failure for ${insightId}:`, aiErr.message);
            }
        })();

        // Respond immediately to the frontend
        res.json({ status: 'success', insightId });

    } catch (error: any) {
        console.error('Capture error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get Enriched Inbox
app.get('/api/insights/inbox', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;

    try {
        const { data, error } = await supabase
            .from('insights')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'INBOX')
            .order('created_at', { ascending: false });

        if (error) throw error;
        res.json({ insights: data });
    } catch (error: any) {
        console.error('Inbox fetch error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Library Search (Semantic + Tag Filter)
app.get('/api/library/search', async (req, res) => {
    const { userId = TEST_USER_ID, q, tag } = req.query;

    try {
        if (q) {
            // 1. Get embedding for query
            console.log(`🔍 Semantic search for: "${q}"`);
            const queryVector = await getEmbedding(q as string);

            if (!queryVector) {
                return res.status(500).json({ error: 'Failed to generate embedding' });
            }

            // 2. Perform Vector Search via RPC
            const { data, error } = await supabase.rpc('match_insights', {
                query_embedding: queryVector,
                match_threshold: 0.5,
                match_count: 20,
                filter_user: userId
            });

            if (error) throw error;

            // Calculate counts
            const counts = {
                total: data?.length || 0,
                ideas: data?.filter((i: any) => i.type === 'IDEA').length || 0,
                tasks: data?.filter((i: any) => i.type === 'TASK').length || 0,
                journals: data?.filter((i: any) => i.type === 'JOURNAL').length || 0,
                principles: data?.filter((i: any) => i.type === 'PRINCIPLE').length || 0
            };

            res.json({ results: data, counts });
        } else {
            // 3. Fallback to standard date-based retrieval
            let query = supabase
                .from('insights')
                .select('*')
                .eq('user_id', userId)
                .eq('status', 'ARCHIVED')
                .order('created_at', { ascending: false }) // Strict DESC ordering
                .limit(50);

            if (tag && tag !== 'ALL') {
                query = query.contains('context_tags', [tag]);
            }

            const { data, error } = await query;
            if (error) throw error;

            // Calculate counts
            const counts = {
                total: data?.length || 0,
                ideas: data?.filter((i: any) => i.type === 'IDEA').length || 0,
                tasks: data?.filter((i: any) => i.type === 'TASK').length || 0,
                journals: data?.filter((i: any) => i.type === 'JOURNAL').length || 0,
                principles: data?.filter((i: any) => i.type === 'PRINCIPLE').length || 0
            };

            res.json({ results: data, counts });
        }
    } catch (error: any) {
        console.error('Library search error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Spark Idea (AI Decomposition)
app.post('/api/library/spark', async (req, res) => {
    const { content } = req.body;

    if (!content) {
        return res.status(400).json({ error: 'Content is required' });
    }

    try {
        console.log(`[SPARK] Generating tasks for: "${content}"`);

        // Call AI Service via Helper
        const result = await callAI(content, 'groq', undefined, 'spark_idea');

        // Result is expected to be { tasks: [...] }
        const data = result;
        const tasks = data.tasks || [];

        console.log(`[SPARK] Generated ${tasks.length} tasks`);
        res.json({ tasks });

    } catch (error: any) {
        console.error('Spark error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get Unscheduled Tasks (Backlog)
app.get('/api/tasks/backlog', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;

    try {
        const { data, error } = await supabase
            .from('insights')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'ACTIVE') // Promoted
            .eq('type', 'TASK')     // Must be a task
            .order('created_at', { ascending: false });

        if (error) throw error;

        // Filter out tasks that are already in ANY daily stream if needed.
        // For now, we'll return all ACTIVE tasks since we don't have a robust 'scheduled' flag in the insights table yet.
        // We can refine this later by checking if the task ID exists in any daily_streams tasks array.

        res.json({ backlog: data || [] });
    } catch (error: any) {
        console.error('Backlog fetch error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Morph Insight (Promote, Save, or Trash)
app.patch('/api/insights/:id/morph', async (req, res) => {
    const { id } = req.params;
    const { action } = req.body;

    if (!action) {
        return res.status(400).json({ error: 'Action is required' });
    }

    try {
        let updateData: any = {};

        switch (action) {
            case 'promote':
                updateData = { status: 'ACTIVE', type: 'TASK' };
                break;
            case 'save':
                updateData = { status: 'ARCHIVED' }; // Keep existing type (IDEA, PRINCIPLE, etc)
                break;
            case 'trash':
                updateData = { status: 'TRASH' };
                break;
            default:
                return res.status(400).json({ error: 'Invalid action' });
        }

        const { error } = await supabase
            .from('insights')
            .update(updateData)
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true });
    } catch (error: any) {
        console.error('Morph error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Schedule Task
app.patch('/api/tasks/:id/schedule', async (req, res) => {
    const { id } = req.params;
    const { date, time } = req.body;

    if (!date || !time) {
        return res.status(400).json({ error: 'Date and time are required' });
    }

    try {
        // 1. Get current insight to preserve other attributes
        const { data: insight, error: fetchError } = await supabase
            .from('insights')
            .select('json_attributes')
            .eq('id', id)
            .single();

        if (fetchError) throw fetchError;

        const updatedAttributes = {
            ...(insight.json_attributes || {}),
            scheduled_date: date,
            scheduled_time: time
        };

        const { error: updateError } = await supabase
            .from('insights')
            .update({
                json_attributes: updatedAttributes,
                status: 'ACTIVE' // Ensure it's active
            })
            .eq('id', id);

        if (updateError) throw updateError;
        res.json({ success: true });
    } catch (error: any) {
        console.error('Schedule error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Day Stream - Fetch Today's Mission & The Dock
app.get('/api/stream', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;

    try {
        // 1. Daily Reset Check
        const today = new Date().toISOString().split('T')[0];

        const { data: markers, error: resetError } = await supabase
            .from('insights')
            .select('*')
            .eq('user_id', userId)
            .eq('type', 'SYSTEM')
            .eq('content', 'LAST_RESET')
            .order('created_at', { ascending: false });

        if (resetError) console.error('[RESET] Query error:', resetError);

        const resetStatus = (markers && markers.length > 0) ? markers[0] : null;
        const lastReset = resetStatus?.json_attributes?.date;

        console.log(`[RESET] Status: today=${today}, found=${markers?.length}, latest_row=${JSON.stringify(resetStatus)}, lastReset=${lastReset}`);

        if (lastReset !== today) {
            try {
                // If we found multiple markers, clean them up to prevent "death spiral"
                if (markers && markers.length > 1) {
                    const toDelete = markers.slice(1).map(m => m.id);
                    console.log(`[RESET] Cleaning up ${toDelete.length} duplicate markers`);
                    await supabase.from('insights').delete().in('id', toDelete);
                }

                console.log(`[RESET] Triggering daily reset. Last was ${lastReset}`);

                const { data: activeInStream } = await supabase
                    .from('insights')
                    .select('id, json_attributes')
                    .eq('user_id', userId)
                    .eq('status', 'ACTIVE');

                const streamTasks = activeInStream?.filter(t => t.json_attributes?.in_stream === true) || [];

                if (streamTasks.length > 0) {
                    for (const task of streamTasks) {
                        const attr = { ...task.json_attributes };
                        delete attr.in_stream;
                        delete attr.locked;
                        await supabase
                            .from('insights')
                            .update({ json_attributes: attr, updated_at: new Date().toISOString() })
                            .eq('id', task.id);
                    }
                }

                if (resetStatus) {
                    console.log(`[RESET] Updating existing marker for ${userId}`);
                    const { error: upError } = await supabase
                        .from('insights')
                        .update({
                            json_attributes: { date: today },
                            updated_at: new Date().toISOString()
                        })
                        .eq('user_id', userId)
                        .eq('type', 'SYSTEM')
                        .eq('content', 'LAST_RESET');
                    if (upError) console.error('[RESET] Marker update failed:', upError);
                } else {
                    console.log(`[RESET] Inserting NEW marker for ${userId}`);
                    const { error: inError } = await supabase
                        .from('insights')
                        .insert([{
                            user_id: userId,
                            type: 'SYSTEM',
                            content: 'LAST_RESET',
                            status: 'ACTIVE',
                            json_attributes: { date: today }
                        }]);
                    if (inError) console.error('[RESET] Marker insert failed:', inError);
                    const { data: verifyMarker } = await supabase.from('insights').select('*').eq('user_id', userId).eq('type', 'system').eq('content', 'LAST_RESET').maybeSingle();
                    console.log(`[RESET] Post-update verification:`, verifyMarker?.json_attributes);
                }
                console.log(`[RESET] Daily reset marker update attempted for ${userId}. Date: ${today}`);
            } catch (err) {
                console.error('[RESET] CRITICAL: Failed inside daily reset logic:', err);
            }
        } else {
            console.log(`[RESET] No reset needed for ${userId}. Last reset was ${lastReset}`);
        }

        // 2. Fetch Tasks
        const { data, error } = await supabase
            .from('insights')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'ACTIVE')
            .order('start_at', { ascending: true, nullsFirst: false });

        if (error) throw error;

        const actualTasks = data?.filter(t => t.type !== 'SYSTEM') || [];
        const result = {
            stream: actualTasks.filter(t => t.json_attributes?.in_stream === true) || [],
            dock: actualTasks.filter(t => !t.json_attributes?.in_stream) || []
        };

        res.json(result);
    } catch (error: any) {
        console.error('Stream fetch error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Toggle Task in Stream
app.patch('/api/stream/toggle', async (req, res) => {
    const { id, in_stream } = req.body;

    if (!id) {
        return res.status(400).json({ error: 'ID is required' });
    }

    try {
        const { data: task } = await supabase
            .from('insights')
            .select('*')
            .eq('id', id)
            .single();

        let start_at = task?.start_at;
        let end_at = task?.end_at;

        if (in_stream) {
            // Always find a new slot when moving to stream via toggle (Smart Schedule)
            const slot = await findNextAvailableSlot(task.user_id, 30);
            start_at = slot.start;
            end_at = slot.end;
            console.log(`[TOGGLE] Smart scheduling task ${id}: ${start_at} -> ${end_at}`);
        } else {
            // Move to Dock - Clear timings
            start_at = null;
            end_at = null;
            console.log(`[TOGGLE] Moving task ${id} to dock, clearing timings`);
        }

        const updatedAttr = {
            ...(task?.json_attributes || {}),
            in_stream: !!in_stream,
            locked: !!in_stream // Auto-lock when entering stream
        };

        delete updatedAttr.zone;

        const { error } = await supabase
            .from('insights')
            .update({
                json_attributes: updatedAttr,
                start_at: start_at,
                end_at: end_at,
                locked: !!in_stream,
                updated_at: new Date().toISOString()
            })
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true, start_at, end_at });
    } catch (error: any) {
        console.error('Stream toggle error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Quick Add directly to Dock
app.post('/api/stream/add', async (req, res) => {
    const { content, type = 'task', userId = TEST_USER_ID, in_stream = false, duration } = req.body;

    if (!content) {
        return res.status(400).json({ error: 'Content is required' });
    }

    try {
        let start_at = null;
        let end_at = null;

        if (in_stream) {
            const slot = await findNextAvailableSlot(userId, 30);
            start_at = slot.start;
            end_at = slot.end;
        }

        const { data, error } = await supabase
            .from('insights')
            .insert([
                {
                    user_id: userId,
                    content: content,
                    type: type.toUpperCase(),
                    status: 'ACTIVE',
                    start_at: start_at,
                    end_at: end_at,
                    locked: !!in_stream,
                    json_attributes: {
                        in_stream: !!in_stream,
                        ...(duration ? { duration } : {})
                    },
                    created_at: new Date().toISOString(),
                    updated_at: new Date().toISOString()
                }
            ])
            .select();

        if (error) throw error;
        res.json(data[0]);
    } catch (error: any) {
        console.error('Queue add error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Reschedule Task
app.patch('/api/insights/:id/reschedule', async (req, res) => {
    const { id } = req.params;
    const { start_at, end_at, minutes } = req.body;

    try {
        let finalStart = start_at;
        let finalEnd = end_at;

        if (minutes) {
            const { data: task } = await supabase.from('insights').select('start_at, end_at').eq('id', id).single();
            if (task) {
                finalStart = new Date(new Date(task.start_at).getTime() + minutes * 60000).toISOString();
                finalEnd = new Date(new Date(task.end_at).getTime() + minutes * 60000).toISOString();
            }
        }

        const { error } = await supabase
            .from('insights')
            .update({
                start_at: finalStart,
                end_at: finalEnd,
                updated_at: new Date().toISOString()
            })
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true, start_at: finalStart, end_at: finalEnd });
    } catch (error: any) {
        console.error('Reschedule error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Delete Task from Stream/Dock
app.delete('/api/stream/:id', async (req, res) => {
    const { id } = req.params;

    try {
        const { error } = await supabase
            .from('insights')
            .delete()
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true, id });
    } catch (error: any) {
        console.error('Delete error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Delete Mindspace Entry
app.delete('/api/mindspace/:id', async (req, res) => {
    const { id } = req.params;

    try {
        const { error } = await supabase
            .from('mindspace_entries')
            .delete()
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true, id });
    } catch (error: any) {
        console.error('Mindspace delete error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Move Task between zones
app.patch('/api/queue/move', async (req, res) => {
    const { id, targetZone } = req.body;

    if (!id || !targetZone) {
        return res.status(400).json({ error: 'ID and targetZone are required' });
    }

    try {
        // 1. If moving to FOCUS, demote current FOCUS to NEXT
        if (targetZone === 'FOCUS') {
            const { data: currentFocus } = await supabase
                .from('insights')
                .select('id, json_attributes')
                .eq('status', 'ACTIVE')
                .eq('json_attributes->>zone', 'FOCUS');

            if (currentFocus && currentFocus.length > 0) {
                for (const item of currentFocus) {
                    const updatedAttr = { ...item.json_attributes, zone: 'NEXT' };
                    await supabase
                        .from('insights')
                        .update({ json_attributes: updatedAttr, updated_at: new Date().toISOString() })
                        .eq('id', item.id);
                }
            }
        }

        // 2. Fetch and update the target task
        const { data: task, error: fetchError } = await supabase
            .from('insights')
            .select('json_attributes')
            .eq('id', id)
            .single();

        if (fetchError) throw fetchError;

        const updatedAttributes = {
            ...(task.json_attributes || {}),
            zone: targetZone
        };

        const { error: updateError } = await supabase
            .from('insights')
            .update({
                json_attributes: updatedAttributes,
                updated_at: new Date().toISOString()
            })
            .eq('id', id);

        if (updateError) throw updateError;
        res.json({ success: true });
    } catch (error: any) {
        console.error('Move error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Complete Task
app.patch('/api/insights/:id/complete', async (req, res) => {
    const { id } = req.params;

    try {
        // Fetch current attributes to clear zone
        const { data: task } = await supabase
            .from('insights')
            .select('json_attributes')
            .eq('id', id)
            .single();

        const updatedAttributes = { ...(task?.json_attributes || {}) };
        delete updatedAttributes.zone;

        const { error } = await supabase
            .from('insights')
            .update({
                status: 'DONE',
                json_attributes: updatedAttributes,
                updated_at: new Date().toISOString()
            })
            .eq('id', id);

        if (error) throw error;
        res.json({ success: true });
    } catch (error: any) {
        console.error('Complete error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Generate Daily Mission
app.post('/api/mission/generate', async (req, res) => {
    const { userId = TEST_USER_ID } = req.body;

    try {
        // 1. Fetch all Dock tasks
        const { data: dockTasks, error } = await supabase
            .from('insights')
            .select('id, content, json_attributes')
            .eq('user_id', userId)
            .eq('status', 'active');

        if (error) throw error;
        const dockOnly = (dockTasks as any[])?.filter(t => !t.json_attributes?.in_stream) || [];

        if (dockOnly.length === 0) {
            return res.status(400).json({ error: 'No tasks in the Dock to generate from.' });
        }

        // 2. Call AI to select
        console.log(`[MISSION] Generating for user ${userId} with ${dockOnly.length} tasks.`);
        const selectedIds = await generateMission(dockOnly);
        console.log(`[MISSION] AI selected IDs:`, selectedIds);

        if (!selectedIds || selectedIds.length === 0) {
            console.warn('[MISSION] AI returned no selections or error.');
            throw new Error('AI failed to select any tasks.');
        }

        const selectedTasks = dockOnly.filter(t => selectedIds.includes(t.id));

        // 3. Generate Schedule
        // Fetch user timezone (default to UTC for now)
        const { data: user } = await supabase.from('users').select('timezone').eq('id', userId).single();
        const timezone = user?.timezone || 'UTC';
        const todayStr = new Date().toISOString().split('T')[0];

        console.log(`[MISSION] Scheduling for ${userId} @ ${timezone}`);
        const schedule = await generateSchedule(selectedTasks, "09:00", todayStr);
        console.log(`[MISSION] AI Schedule:`, schedule);

        // 4. Update tasks in DB
        let committedCount = 0;
        for (const item of schedule) {
            const { id, start_time, end_time } = item;

            // Helper to convert local time + date + timezone to UTC
            // e.g. 2023-10-27T09:00:00 within 'America/New_York'
            const startUTC = new Date(`${todayStr}T${start_time}:00`).toISOString();
            const endUTC = new Date(`${todayStr}T${end_time}:00`).toISOString();

            const targetTask = selectedTasks.find(dt => dt.id === id);
            if (!targetTask) continue;

            const attr = {
                ...(targetTask.json_attributes || {}),
                in_stream: true,
                locked: true
            };

            const { error: updateError } = await supabase
                .from('insights')
                .update({
                    json_attributes: attr,
                    start_at: startUTC,
                    end_at: endUTC,
                    locked: true,
                    updated_at: new Date().toISOString()
                })
                .eq('id', id);

            if (!updateError) committedCount++;
        }

        console.log(`[MISSION] Successfully committed ${committedCount} scheduled tasks.`);
        res.json({ success: true, count: committedCount });
    } catch (error: any) {
        console.error('Mission generation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Cascade Mission (Push schedule back)
app.patch('/api/mission/cascade', async (req, res) => {
    const { userId = TEST_USER_ID, minutes = 15 } = req.body;

    try {
        const now = new Date().toISOString();

        // Fetch unfinished locked tasks in the stream
        const { data: tasks, error } = await supabase
            .from('insights')
            .select('id, start_at, end_at')
            .eq('user_id', userId)
            .eq('status', 'ACTIVE')
            .eq('locked', true);
        // Removed .gt('end_at', now) to allow cascading past due tasks

        if (error) throw error;
        if (!tasks || tasks.length === 0) return res.json({ success: true, count: 0 });

        for (const task of tasks) {
            const newStart = new Date(new Date(task.start_at).getTime() + minutes * 60000).toISOString();
            const newEnd = new Date(new Date(task.end_at).getTime() + minutes * 60000).toISOString();

            await supabase
                .from('insights')
                .update({
                    start_at: newStart,
                    end_at: newEnd,
                    updated_at: new Date().toISOString()
                })
                .eq('id', task.id);
        }

        res.json({ success: true, count: tasks.length });
    } catch (error: any) {
        console.error('Cascade error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Swap Mission Task
app.post('/api/mission/swap', async (req, res) => {
    const { locked_id, dock_id } = req.body;

    try {
        // 1. Get both tasks
        const { data: lockedTask } = await supabase.from('insights').select('*').eq('id', locked_id).single();
        const { data: dockTask } = await supabase.from('insights').select('*').eq('id', dock_id).single();

        if (!lockedTask || !dockTask) throw new Error('Tasks not found');

        // 2. Unlock and move to Dock (Clear timings)
        const lockedAttr = { ...(lockedTask?.json_attributes || {}) };
        delete lockedAttr.in_stream;
        delete lockedAttr.locked;
        await supabase.from('insights').update({
            json_attributes: lockedAttr,
            start_at: null,
            end_at: null,
            locked: false,
            updated_at: new Date().toISOString()
        }).eq('id', locked_id);

        // 3. Lock and move to Stream (Inherit timings)
        const dockAttr = { ...(dockTask?.json_attributes || {}), in_stream: true, locked: true };
        await supabase.from('insights').update({
            json_attributes: dockAttr,
            start_at: lockedTask.start_at,
            end_at: lockedTask.end_at,
            locked: true,
            updated_at: new Date().toISOString()
        }).eq('id', dock_id);

        res.json({ success: true });
    } catch (error: any) {
        console.error('Mission swap error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Decompose Task (The Prism Feature)
app.post('/api/insights/decompose', async (req, res) => {
    const { id, userId = TEST_USER_ID } = req.body;

    if (!id) {
        return res.status(400).json({ error: 'ID is required' });
    }

    try {
        // 1. Fetch original task
        const { data: original, error: fetchError } = await supabase
            .from('insights')
            .select('*')
            .eq('id', id)
            .single();

        if (fetchError || !original) throw new Error('Original task not found');

        // 2. Call AI breakdown service
        console.log(`[DECOMPOSE] Goal: "${original.content}"`);
        const subtasks = await decomposeGoal(original.content);
        console.log(`[DECOMPOSE] AI raw output subtasks:`, subtasks);

        if (!subtasks || subtasks.length === 0) {
            console.warn('[DECOMPOSE] AI returned EMPTY list.');
            throw new Error('AI could not break down this goal');
        }

        // 3. Archive the original task and mark as PROJECT
        await supabase
            .from('insights')
            .update({
                status: 'archived',
                type: 'project',
                updated_at: new Date().toISOString()
            })
            .eq('id', id);

        // 4. Insert new subtasks into the Dock (status='active', in_stream=false)
        const entries = subtasks.map((taskContent: string) => ({
            user_id: userId,
            content: taskContent,
            type: 'task',
            status: 'active',
            json_attributes: { in_stream: false, parent_id: id },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
        }));

        const { error: insertError } = await supabase
            .from('insights')
            .insert(entries);

        if (insertError) {
            console.error('[DECOMPOSE] Bulk insert failed:', insertError);
            throw insertError;
        }

        res.json({ success: true, count: subtasks.length });

    } catch (error: any) {
        console.error('Decomposition error:', error);
        res.status(500).json({ error: error.message });
    }
});

// --- MINDSPACE MODULE ---

// 1. Add Entry (Processing Pipeline)
app.post('/api/mindspace/add', upload.single('image'), async (req, res) => {
    const { text, userId = TEST_USER_ID, mediaType = 'text' } = req.body;
    let imageUrl = req.body.imageUrl; // Fallback for direct URL

    try {
        // A. Handle Image Upload if provided
        if (req.file) {
            console.log(`[MINDSPACE] Uploading image: ${req.file.originalname}...`);
            const fileExt = req.file.originalname.split('.').pop();
            const fileName = `${userId}_${Date.now()}.${fileExt}`;
            const filePath = `entries/${fileName}`;

            const { data: uploadData, error: uploadError } = await supabase.storage
                .from('mindspace_assets')
                .upload(filePath, req.file.buffer, {
                    contentType: req.file.mimetype,
                    upsert: true
                });

            if (uploadError) {
                console.error('[STORAGE] Upload failed:', uploadError);
                throw uploadError;
            }

            const { data: { publicUrl } } = supabase.storage
                .from('mindspace_assets')
                .getPublicUrl(filePath);

            imageUrl = publicUrl;
        }

        // B. Fetch Existing Clusters (Context for Librarian)
        const { data: existingClusters } = await supabase
            .from('mindspace_clusters')
            .select('id, name, center_embedding')
            .eq('user_id', userId)
            .limit(100);

        // C. Process via AI (Vision, Embedding, Clustering, Tone)
        console.log(`[MINDSPACE] Processing entry for user ${userId} with ${existingClusters?.length || 0} existing clusters...`);
        const aiResults = await processMindspace(text || "", imageUrl, userId, existingClusters || []);

        // C. Storage: Save Entry
        const { data: entry, error: entryError } = await supabase
            .from('mindspace_entries')
            .insert({
                user_id: userId,
                content_text: text,
                media_url: imageUrl,
                media_type: imageUrl ? 'image' : 'text',
                ai_description: aiResults.ai_description,
                embedding: aiResults.embedding,
                emotional_tone: aiResults.tone,
                created_at: new Date().toISOString()
            })
            .select()
            .single();

        if (entryError) throw entryError;

        // D. Link Entry <-> Clusters (Python already assigned/created them)
        if (aiResults.cluster_ids && aiResults.cluster_ids.length > 0) {
            console.log(`[MINDSPACE] Linking entry ${entry.id} to clusters:`, aiResults.cluster_ids);
            const linkData = aiResults.cluster_ids.map((cid: string | number) => ({
                entry_id: entry.id,
                cluster_id: cid
            }));
            const { error: linkError } = await supabase
                .from('entry_clusters')
                .insert(linkData);
            if (linkError) console.error('[MINDSPACE] Cluster linking error:', linkError);
        }

        res.json({ success: true, entry });
    } catch (error: any) {
        console.error('[MINDSPACE] Add error:', error);
        res.status(500).json({ error: error.message });
    }
});

// 2. Feed Endpoint (Chronological + Resurfaced + Search)
app.get('/api/mindspace/feed', async (req, res) => {
    const { userId = TEST_USER_ID, q } = req.query;

    try {
        let queryBuilder = supabase
            .from('mindspace_entries')
            .select('*')
            .eq('user_id', userId);

        let entries;

        if (q) {
            console.log(`[MINDSPACE] Searching for: ${q}...`);
            const embedding = await getEmbedding(q as string);

            if (embedding) {
                // Use Supabase RPC for vector similarity search
                const { data: searchResults, error: searchError } = await supabase.rpc('match_mindspace_entries', {
                    query_embedding: embedding,
                    match_threshold: 0.5,
                    match_count: 50,
                    p_user_id: userId
                });

                if (searchError) throw searchError;
                entries = searchResults;
            } else {
                // Fallback to text search if embedding service is down or failed
                console.warn('[MINDSPACE] Embedding failed, falling back to text search.');
                const { data: textResults, error: textError } = await queryBuilder
                    .or(`content_text.ilike.%${q}%,ai_description.ilike.%${q}%`)
                    .order('created_at', { ascending: false })
                    .limit(50);

                if (textError) throw textError;
                entries = textResults;
            }
        } else {
            // Standard feed: Fetch last 50 entries
            const { data: standardResults, error } = await queryBuilder
                .order('created_at', { ascending: false })
                .limit(50);

            if (error) throw error;
            entries = standardResults;
        }

        // Fetch a "Resurfaced" memory (older than 30 days)
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

        const { data: resurfaced } = await supabase
            .from('mindspace_entries')
            .select('*')
            .eq('user_id', userId) // Ensure resurfaced memory is for the current user
            .lt('created_at', thirtyDaysAgo.toISOString())
            .limit(5); // Get a few to pick one

        let feed: any[] = [...(entries || [])];

        if (resurfaced && resurfaced.length > 0) {
            const memory = resurfaced[Math.floor(Math.random() * resurfaced.length)];
            // Inject at 10th position if possible
            if (feed.length >= 10) {
                feed.splice(9, 0, { ...memory, is_resurfaced: true });
            } else if (feed.length > 0) {
                feed.push({ ...memory, is_resurfaced: true });
            }
        }

        res.json({ feed });
    } catch (error: any) {
        console.error('[MINDSPACE] Feed error:', error);
        res.status(500).json({ error: error.message });
    }
});

// 3. Patterns/Intelligence Endpoint
app.get('/api/mindspace/patterns', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;
    try {
        // Fetch last 20 entries for analysis for SPECIFIC user
        const { data: entries, error } = await supabase
            .from('mindspace_entries')
            .select('content_text, ai_description, emotional_tone, created_at')
            .eq('user_id', userId)
            .order('created_at', { ascending: false })
            .limit(20);

        if (error || !entries) throw error || new Error('No entries');

        /* 
        // Cache check removed as per user request for fresh generation on every click
        const { data: cached } = await supabase
            .from('mindspace_insights')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false })
            .limit(1)
            .single();

        const oneHourAgo = new Date(Date.now() - 3600000);
        if (cached && new Date(cached.created_at) > oneHourAgo) {
            return res.json({ insight: cached.insight_text });
        }
        */

        // Always generate new pattern analysis
        console.log(`[MINDSPACE] Generating fresh patterns for user ${userId}...`);
        const insightText = await analyzeMindspacePatterns(entries);

        // Cache it for the user
        await supabase
            .from('mindspace_insights')
            .insert({
                user_id: userId,
                insight_text: insightText,
                created_at: new Date().toISOString()
            });

        res.json({ insight: insightText });
    } catch (error: any) {
        console.error('[MINDSPACE] Patterns error:', error);
        res.status(500).json({ error: error.message });
    }
});

// 4. Clusters Endpoint
app.get('/api/mindspace/clusters', async (req, res) => {
    const { userId = TEST_USER_ID } = req.query;
    try {
        const { data: clusters, error } = await supabase
            .from('mindspace_clusters')
            .select('*')
            .eq('user_id', userId)
            .order('name', { ascending: true });

        if (error) throw error;
        res.json({ clusters });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// 5. Cluster Filtered View
app.get('/api/mindspace/clusters/:clusterId/entries', async (req, res) => {
    const { clusterId } = req.params;
    try {
        const { data, error } = await supabase
            .from('entry_clusters')
            .select('mindspace_entries(*)')
            .eq('cluster_id', clusterId);

        if (error) throw error;
        const entries = data.map(d => d.mindspace_entries).filter(e => e !== null);
        res.json({ entries });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// 5. Chat Endpoint (RAG)
app.post('/api/mindspace/chat', async (req, res) => {
    const { query, userId = TEST_USER_ID } = req.body;
    try {
        const result = await chatMindspace(query, userId);
        res.json(result);
    } catch (error: any) {
        console.error('[MINDSPACE] Chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`Core service listening at http://0.0.0.0:${port}`);
});
