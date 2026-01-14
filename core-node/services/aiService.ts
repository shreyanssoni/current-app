import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

interface AIResponse {
    status: string;
    response: string;
}

/**
 * Sends a generic request to the Python AI Gateway Service
 * @param prompt The prompt to send to the LLM
 * @param provider The model provider (e.g., 'groq', 'ollama', 'openai')
 * @param model Optional specific model name
 * @param promptType Optional prompt strategy (e.g., 'negotiate_resistance')
 */
export async function callAI(prompt: string, provider: string = 'groq', model?: string, promptType?: string): Promise<any> {
    try {
        const response = await axios.post<AIResponse>(`${AI_SERVICE_URL}/generate`, {
            prompt,
            provider,
            model,
            prompt_type: promptType
        });

        if (response.data.status === 'success') {
            return response.data.response;
        } else {
            throw new Error('AI Service returned an unsuccessful status');
        }
    } catch (error: any) {
        console.error('Error calling AI Service:', error.message);
        throw new Error(`Failed to communicate with AI Service: ${error.message}`);
    }
}

/**
 * Specifically calls the enrichment endpoint
 */
export async function analyzeInsight(text: string): Promise<any> {
    try {
        const response = await axios.post<AIResponse>(`${AI_SERVICE_URL}/analyze`, {
            text
        });
        if (response.data.status === 'success') {
            return response.data.response;
        }
        throw new Error('Analysis failed');
    } catch (error: any) {
        console.error('Error in analyzeInsight:', error.message);
        throw error;
    }
}

/**
 * Gets a vector embedding for a piece of text
 */
export async function getEmbedding(text: string): Promise<number[] | null> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/embed`, {
            text
        });
        if (response.data.status === 'success') {
            return response.data.embedding;
        }
        return null;
    } catch (error: any) {
        console.error('Error in getEmbedding:', error.message);
        return null;
    }
}

/**
 * Breaks down a goal into sub-tasks (The Prism Feature)
 */
export async function decomposeGoal(text: string): Promise<string[]> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/breakdown`, {
            text
        });
        if (response.data.status === 'success') {
            return response.data.response?.tasks || [];
        }
        return [];
    } catch (error: any) {
        console.error('Error in decomposeGoal:', error.message);
        throw error;
    }
}

/**
 * Selects 3-5 tasks from the dock for a daily mission
 */
export async function generateMission(tasks: any[]): Promise<string[]> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/generate_mission`, {
            tasks
        });
        if (response.data.status === 'success') {
            return response.data.response?.selected_ids || [];
        }
        return [];
    } catch (error: any) {
        console.error('Error in generateMission:', error.message);
        throw error;
    }
}

/**
 * Generates a time-blocked schedule for selected tasks
 */
export async function generateSchedule(tasks: any[], startTime: string, curDate: string): Promise<any[]> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/generate_schedule`, {
            tasks,
            user_start_time: startTime,
            current_date: curDate
        });
        if (response.data.status === 'success') {
            return response.data.response?.schedule || [];
        }
        return [];
    } catch (error: any) {
        console.error('Error in generateSchedule:', error.message);
        throw error;
    }
}
/**
 * Orchestrates Vision, Embedding, Clustering, and Sentiment for a Mindspace entry.
 */
export async function processMindspace(text: string, imageUrl?: string, userId?: string, existingClusters: any[] = []): Promise<any> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/mindspace/process`, {
            text,
            image_url: imageUrl,
            userId,
            existing_clusters: existingClusters
        });
        if (response.data.status === 'success') {
            return response.data.response;
        }
        throw new Error('Mindspace processing failed');
    } catch (error: any) {
        console.error('Error in processMindspace:', error.message);
        throw error;
    }
}

/**
 * Analyzes recent entries for recurring themes and patterns.
 */
export async function analyzeMindspacePatterns(entries: any[]): Promise<string> {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/mindspace/analyze_patterns`, {
            entries
        });
        if (response.data.status === 'success') {
            return response.data.response || "Continue capturing your thoughts to surface patterns.";
        }
        return "Insight calculation in progress...";
    } catch (error: any) {
        console.error('Error in analyzeMindspacePatterns:', error.message);
        return "Mindspace is reflecting on your thoughts...";
    }
}
