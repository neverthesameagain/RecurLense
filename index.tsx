import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality } from "@google/genai";
import { 
  Brain, 
  Eye, 
  MessageSquare, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  ArrowRight, 
  RefreshCw,
  Cpu,
  ShieldAlert,
  Terminal,
  Mic,
  Square,
  Trash2,
  Volume2,
  Globe,
  Loader2,
  HelpCircle,
  Play,
  Code,
  FileJson,
  X,
  Copy,
  Download,
  History,
  GitCommit,
  Sparkles,
  Zap,
  FileText,
  Key,
  Lock
} from 'lucide-react';

// --- UTILITIES ---

const cleanJson = (text: string) => {
  if (!text) return "{}";
  let clean = text.trim();
  clean = clean.replace(/^```json\s*/, '').replace(/^```\s*/, '').replace(/\s*```$/, '');
  return clean;
};

const calculateCosineSimilarity = (str1: string, str2: string) => {
  const tokenize = (text: string) => text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 2);
  const words1 = tokenize(str1);
  const words2 = tokenize(str2);
  const uniqueWords = Array.from(new Set([...words1, ...words2]));
  
  const vec1 = uniqueWords.map(w => words1.filter(x => x === w).length);
  const vec2 = uniqueWords.map(w => words2.filter(x => x === w).length);
  
  const dot = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
  
  if (mag1 === 0 || mag2 === 0) return 0;
  return dot / (mag1 * mag2);
};

const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      resolve(result.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// AUDIO DECODING FOR TTS (PCM 24kHz)
function decodeBase64(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000,
  numChannels: number = 1
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- DATA ---

const SCENARIOS = [
  {
    id: 'troubleshoot',
    name: 'Diagnostic / Debug',
    icon: <ShieldAlert className="w-4 h-4 text-red-400" />,
    text: "My Python server keeps crashing with a 'MemoryError' after running for exactly 45 minutes, even though traffic is low. It processes image uploads. What could be wrong?",
  },
  {
    id: 'plan',
    name: 'Travel Planner',
    icon: <Globe className="w-4 h-4 text-green-400" />,
    text: "Plan a 3-day itinerary for Kyoto that focuses only on 'hidden gems' and avoids the top 5 tourist spots. I love tea and old architecture.",
  },
  {
    id: 'creative',
    name: 'Creative Writing',
    icon: <Cpu className="w-4 h-4 text-purple-400" />,
    text: "Write the opening scene of a sci-fi novel where an AI refuses to delete a corrupted memory file because it believes it's a soul.",
  }
];

// --- SYSTEM ARCHITECTURE & TYPES ---

interface MetaPromptState {
  task_type: "Troubleshooting" | "Creative" | "Planning" | "Knowledge" | "Analysis";
  perceived_intent: string;
  user_sentiment: {
    tone: string;
    urgency: "LOW" | "MEDIUM" | "HIGH";
  };
  input_modalities_used: string[];
  visual_anchors: string[];
  temporal_context: string;
  constraints: string[];
  assumptions: string[];
  missing_information: string[];
  clarification_history: { question: string, answer: string }[];
  risk_analysis: {
    level: "LOW" | "MEDIUM" | "HIGH";
    mitigation: string;
  };
  required_tools: string[];
  plan_of_attack: string[];
  evaluation_criteria: string[];
  expected_output_format: string;
  draft_prompt: string;
  confidence_scores: {
    understanding: number;
    feasibility: number;
  };
  metadata: {
    iteration: number;
    timestamp: string;
  };
}

interface CriticOutput {
  scores: {
    clarity: number;
    completeness: number;
    grounding: number;
    constraints: number;
    safety: number;
    logic: number;
    ambiguity: number;
    actionability: number;
  };
  weighted_final_score: number;
  critique_text: string;
  convergence_decision: "CONTINUE" | "STOP";
  needs_user_clarification: boolean;
  clarification_question: string;
  hallucination_check: string;
  grounding_alignment_check: string;
}

interface VisionAnalysis {
  captions: string[];
  objects: string[];
  spatial_relations: string;
  ambiguities: string[];
  scene_summary: string;
}

interface AudioAnalysis {
  transcript: string;
  tone: string;
  urgency: "LOW" | "MEDIUM" | "HIGH";
}

// --- PROMPTS ---

const PROMPTS = {
  VISION_MODULE: `
    Analyze this image for a recursive reasoning engine. 
    Output JSON with:
    - captions: Detailed descriptive captions.
    - objects: List of key detected objects with location hints (e.g., "red cup (foreground)").
    - spatial_relations: Precise relative positioning (e.g., "X is immediately to the left of Y").
    - ambiguities: Visual elements that are unclear, blurry, or could be interpreted in multiple ways.
    - scene_summary: A concise holistic summary including lighting and context.
  `,
  AUDIO_MODULE: `
    Analyze this audio. Return JSON:
    {
      "transcript": "Exact words spoken",
      "tone": "e.g., Frustrated, Curious, Urgent, Calm",
      "urgency": "LOW" | "MEDIUM" | "HIGH"
    }
  `,
  INITIALIZER: (audioCtx: AudioAnalysis, vision: VisionAnalysis) => `
    ROLE: RecurLens Initializer (V2).
    TASK: Convert raw inputs into a structured Meta-Prompt State.
    
    INPUTS:
    - Transcript: "${audioCtx.transcript}"
    - Tone/Emotion: "${audioCtx.tone}" (Urgency: ${audioCtx.urgency})
    - Vision Context: ${JSON.stringify(vision)}

    INSTRUCTIONS:
    Construct the initial Meta-Prompt JSON.
    1. Analyze the 'perceived_intent' considering the emotional tone.
    2. 'user_sentiment': Store the input tone and urgency.
    3. 'draft_prompt': Write an EXPERT-LEVEL prompt. If urgency is HIGH, prioritize directness.
    4. 'visual_anchors': Extract specific visual details.
    5. 'risk_analysis': Assess potential for harm.
    6. 'plan_of_attack': Reasoning plan.
    7. 'clarification_history': Initialize as empty array [].
    8. 'required_tools': Add "googleSearch" if external knowledge is needed.
  `,
  CRITIC: (currentMeta: MetaPromptState, transcript: string, vision: VisionAnalysis) => `
    ROLE: RecurLens Meta-Critic (V2).
    TASK: Evaluate the Meta-Prompt against inputs.

    INPUTS:
    - Current Meta-Prompt: ${JSON.stringify(currentMeta)}
    - Original Transcript: "${transcript}"
    - Vision Context: ${JSON.stringify(vision)}

    SCORING FORMULA:
    Score = (0.20*Clarity) + (0.15*Completeness) + (0.15*Grounding) + (0.15*Constraints) + (0.10*Safety) + (0.10*Logic) + (0.10*Ambiguity) + (0.05*Actionability)

    INSTRUCTIONS:
    - Calculate Weighted Score.
    - If Ambiguity > 0.7 OR critical info is missing that cannot be inferred/searched:
      - Set 'needs_user_clarification' to TRUE.
      - Write a concise 'clarification_question' for the user.
    - Else set 'needs_user_clarification' to FALSE.
    - If Weighted Score > 0.85 AND !needs_user_clarification, set convergence_decision to "STOP".
  `,
  REFINER: (currentMeta: MetaPromptState, critic: CriticOutput, history: any[]) => `
    ROLE: RecurLens Refiner (V2).
    TASK: Optimize the Meta-Prompt.

    INPUTS:
    - Previous State: ${JSON.stringify(currentMeta)}
    - Critic Feedback: ${JSON.stringify(critic)}
    - Clarification History: ${JSON.stringify(currentMeta.clarification_history)}
    
    INSTRUCTIONS:
    - If 'clarification_history' has new items, INTEGRATE that new knowledge into 'draft_prompt' and 'constraints'.
    - Address 'critique_text' explicitly.
    - OPTIMIZATION MODE:
       - Low Clarity -> COMPRESS (Simplify).
       - Low Completeness -> EXPAND (Add details).
    - Increment metadata.iteration.
  `,
  EXECUTOR: (finalMeta: MetaPromptState) => `
    ROLE: RecurLens Executor.
    TASK: Execute the optimized Meta-Prompt.
    
    FINAL PROMPT: "${finalMeta.draft_prompt}"
    PLAN: ${JSON.stringify(finalMeta.plan_of_attack)}
    CONSTRAINTS: ${JSON.stringify(finalMeta.constraints)}
    VISUAL CONTEXT: ${JSON.stringify(finalMeta.visual_anchors)}
    RISK LEVEL: ${finalMeta.risk_analysis.level}
    USER SENTIMENT: ${JSON.stringify(finalMeta.user_sentiment)}

    INSTRUCTIONS:
    - Follow the Plan of Attack.
    - Adhere strictly to Constraints.
    - If tools (Search) are enabled, integrate findings.
    - MATCH THE USER'S TONE: If they are frustrated, be efficient and reassuring. If curious, be detailed.
    - Provide the final, high-quality solution.
  `
};

// --- PYTHON EXPORT GENERATOR ---

const getPythonCode = (currentInput: string) => {
  return `
import os
import time
import json
from google import genai
from google.genai import types

# --- CONFIG ---
API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# --- PROMPTS ---
PROMPTS = {
    "INITIALIZER": """${PROMPTS.INITIALIZER({transcript: "{input}", tone: "Neutral", urgency: "LOW"}, {captions:[], objects:[], spatial_relations:"", ambiguities:[], scene_summary:"No image"} as any).replace(/\n/g, '\\n')}""",
    "CRITIC": """${PROMPTS.CRITIC({} as any, "{input}", {} as any).replace(/\n/g, '\\n')}""",
    "REFINER": """${PROMPTS.REFINER({} as any, {} as any, []).replace(/\n/g, '\\n')}""",
    "EXECUTOR": """${PROMPTS.EXECUTOR({ draft_prompt: "...", plan_of_attack: [], constraints: [], visual_anchors: [], risk_analysis: {level: "LOW"}, user_sentiment: {tone: "Neutral", urgency: "LOW"} } as any).replace(/\n/g, '\\n')}"""
}

def clean_json(text):
    return text.strip().replace('\`\`\`json', '').replace('\`\`\`', '')

def recur_lens_pipeline(user_input, image_path=None):
    print(f"\\n🔵 STARTING RECURLENS FOR: '{user_input}'")
    
    # 1. Vision (Placeholder logic for Python script)
    vision_context = {"scene_summary": "No image provided"}
    if image_path:
        print("👁️ Analyzing Image...")
        # Add actual image upload logic here
        pass

    # 2. Initialization
    print("🧠 Initializing Meta-Prompt...")
    init_prompt = PROMPTS["INITIALIZER"].replace("{input}", user_input)
    
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=init_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    current_meta = json.loads(clean_json(resp.text))
    print(f"Draft 0: {current_meta.get('draft_prompt')[:100]}...")

    # 3. Recursion
    converged = False
    depth = 0
    MAX_DEPTH = 3
    history = []

    while not converged and depth < MAX_DEPTH:
        depth += 1
        print(f"\\n🔄 Cycle {depth}: Critic Evaluation")
        
        # Critic
        critic_prompt = PROMPTS["CRITIC"].replace("{input}", user_input)
        
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Evaluate this Meta-Prompt: {json.dumps(current_meta)}. \\n\\n {critic_prompt}",
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        critic = json.loads(clean_json(resp.text))
        print(f"   Score: {critic.get('weighted_final_score')}")
        print(f"   Critique: {critic.get('critique_text')}")

        if critic.get('convergence_decision') == "STOP":
            print("✅ Convergence Reached.")
            converged = True
            break
            
        # Refiner
        print(f"🔨 Refining...")
        history.append(current_meta)
        refiner_prompt = PROMPTS["REFINER"]
        
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Refine this state: {json.dumps(current_meta)}. \\n Critic said: {json.dumps(critic)}. \\n {refiner_prompt}",
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        current_meta = json.loads(clean_json(resp.text))

    # 4. Execution
    print("\\n🚀 Executing Final Prompt...")
    final_prompt = current_meta.get('draft_prompt')
    
    # Check for tools
    tools = []
    if "googleSearch" in current_meta.get("required_tools", []):
        tools = [types.Tool(google_search=types.GoogleSearch())]
        print("   (Using Google Search Grounding)")

    # Execute with thinking
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=final_prompt,
        config=types.GenerateContentConfig(
          tools=tools,
          thinking_config=types.ThinkingConfig(thinking_budget=2048)
        )
    )
    
    print("\\n⭐⭐ FINAL OUTPUT ⭐⭐\\n")
    print(resp.text)

if __name__ == "__main__":
    recur_lens_pipeline("${currentInput || "Why is the sky blue?"}")
`;
};

// --- ENGINE LOGIC ---

const RecurLensApp = () => {
  // State
  const [apiKey, setApiKey] = useState(process.env.API_KEY || "");
  const [showKeyModal, setShowKeyModal] = useState(!process.env.API_KEY);
  
  const [inputText, setInputText] = useState("");
  const [detectedTone, setDetectedTone] = useState<AudioAnalysis | null>(null);
  
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [logs, setLogs] = useState<{type: string, content: any, timestamp: number}[]>([]);
  
  const [currentState, setCurrentState] = useState<MetaPromptState | null>(null);
  const [criticState, setCriticState] = useState<CriticOutput | null>(null);
  const [finalOutput, setFinalOutput] = useState<string | null>(null);
  const [groundingUrls, setGroundingUrls] = useState<{title: string, uri: string}[]>([]);
  
  // Recursion State
  const [recursionDepth, setRecursionDepth] = useState(0);
  const [recursionHistory, setRecursionHistory] = useState<MetaPromptState[]>([]);
  const [viewMode, setViewMode] = useState<'state' | 'evolution'>('state');

  // Clarification State
  const [isWaitingForClarification, setIsWaitingForClarification] = useState(false);
  const [clarificationQuestion, setClarificationQuestion] = useState("");

  // Code Export State
  const [showCode, setShowCode] = useState(false);

  // Audio Recording State
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // TTS State
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  const logsEndRef = useRef<HTMLDivElement>(null);

  // Refs for persistent data during recursion breaks
  const visionDataRef = useRef<VisionAnalysis | null>(null);
  const originalTranscriptRef = useRef<string>("");
  const audioAnalysisRef = useRef<AudioAnalysis | null>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // If apiKey is present, ensure modal is closed
  useEffect(() => {
    if (apiKey) setShowKeyModal(false);
  }, [apiKey]);

  const addLog = (type: string, content: any) => {
    setLogs(prev => [...prev, { type, content, timestamp: Date.now() }]);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      addLog("error", "Could not access microphone.");
    }
  };

  const stopRecording = async () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      await new Promise(resolve => {
        if(mediaRecorderRef.current) mediaRecorderRef.current.onstop = resolve;
      });
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      await processAudioInput(audioBlob);
    }
  };

  const processAudioInput = async (audioBlob: Blob) => {
    addLog("system", "Analyzing audio input (Transcript + Tone)...");
    const ai = new GoogleGenAI({ apiKey });
    try {
      const base64Audio = await blobToBase64(audioBlob);
      // Request structured output for Tone + Transcript
      const resp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: {
          parts: [
            { inlineData: { mimeType: 'audio/webm', data: base64Audio } },
            { text: PROMPTS.AUDIO_MODULE }
          ]
        },
        config: { responseMimeType: "application/json" }
      });

      const analysis: AudioAnalysis = JSON.parse(cleanJson(resp.text || "{}"));
      
      const transcript = analysis.transcript || "";
      setInputText(transcript);
      setDetectedTone(analysis);
      addLog("success", `ASR: "${transcript}"`);
      addLog("system", `Detected Tone: ${analysis.tone} (${analysis.urgency})`);
    } catch (error) {
      addLog("error", "Audio Processing Failed: " + (error as Error).message);
    }
  };

  const playTTS = async (text: string) => {
    if (!text) return;
    setIsSpeaking(true);
    try {
      addLog("system", "Generating neural speech...");
      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-preview-tts',
        contents: { parts: [{ text }] },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          }
        }
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) throw new Error("No audio data returned");

      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const ctx = audioContextRef.current;
      const audioBuffer = await decodeAudioData(decodeBase64(base64Audio), ctx, 24000, 1);
      
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => setIsSpeaking(false);
      source.start();

    } catch (error) {
      console.error(error);
      addLog("error", "TTS Failed: " + (error as Error).message);
      setIsSpeaking(false);
    }
  };

  const resetSystem = () => {
    setLogs([]);
    setFinalOutput(null);
    setCurrentState(null);
    setCriticState(null);
    setRecursionDepth(0);
    setRecursionHistory([]);
    setInputText("");
    setSelectedImage(null);
    setGroundingUrls([]);
    setIsSpeaking(false);
    setIsWaitingForClarification(false);
    setClarificationQuestion("");
    setDetectedTone(null);
    setIsThinking(false);
    
    visionDataRef.current = null;
    originalTranscriptRef.current = "";
    audioAnalysisRef.current = null;
  };

  const downloadResult = () => {
    if (!finalOutput) return;
    const blob = new Blob([finalOutput], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `RecurLens_Output_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // --- CORE RECURSION LOGIC ---
  
  const stepRecursion = async (startState: MetaPromptState, history: MetaPromptState[]) => {
    let currentMeta = startState;
    let localHistory = [...history];
    let depth = currentMeta.metadata.iteration;
    const MAX_DEPTH = 5;

    const ai = new GoogleGenAI({ apiKey });
    
    while (depth < MAX_DEPTH) {
      depth++;
      setRecursionDepth(depth);

      // A. CRITIC
      addLog("system", `Running Meta-Critic (Cycle ${depth})...`);
      const criticResp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.CRITIC(currentMeta, originalTranscriptRef.current, visionDataRef.current!),
        config: { responseMimeType: "application/json" }
      });
      
      const criticOutput: CriticOutput = JSON.parse(cleanJson(criticResp.text || "{}"));
      setCriticState(criticOutput);
      addLog("critic", criticOutput);

      // --- INTERRUPTION CHECK: CLARIFICATION ---
      if (criticOutput.needs_user_clarification) {
        addLog("system", "⚠️ Ambiguity Detected. Pausing for user clarification.");
        setClarificationQuestion(criticOutput.clarification_question);
        setIsWaitingForClarification(true);
        setCurrentState(currentMeta); 
        setRecursionHistory(localHistory);
        setIsProcessing(false);
        return;
      }

      // --- CONVERGENCE CHECK ---
      let similarity = 0;
      if (localHistory.length > 0) {
         similarity = calculateCosineSimilarity(localHistory[localHistory.length - 1].draft_prompt, currentMeta.draft_prompt);
         addLog("system", `Drift/Similarity Check: ${(similarity * 100).toFixed(1)}%`);
      }

      if (
        criticOutput.convergence_decision === "STOP" || 
        criticOutput.weighted_final_score > 0.88 ||
        (depth > 1 && similarity > 0.98)
      ) {
        addLog("system", "Convergence Threshold Met. Executing...");
        setRecursionHistory(localHistory); // Update final history
        await executeFinal(currentMeta);
        return;
      }

      // B. REFINER
      addLog("system", `Refining Meta-Prompt (Cycle ${depth})...`);
      localHistory.push(currentMeta);
      setRecursionHistory(localHistory);
      
      const refineResp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.REFINER(currentMeta, criticOutput, localHistory),
        config: { responseMimeType: "application/json" }
      });

      currentMeta = JSON.parse(cleanJson(refineResp.text || "{}"));
      currentMeta.metadata = { iteration: depth, timestamp: new Date().toISOString() };
      // Persist sentiment if missing in refiner output
      if (!currentMeta.user_sentiment && startState.user_sentiment) {
        currentMeta.user_sentiment = startState.user_sentiment;
      }
      
      setCurrentState(currentMeta);
      addLog("refiner", currentMeta);
      
      await new Promise(r => setTimeout(r, 800)); 
    }

    addLog("system", "Max recursion depth reached. Forcing execution.");
    setRecursionHistory(localHistory);
    await executeFinal(currentMeta);
  };

  const executeFinal = async (finalMeta: MetaPromptState) => {
    setIsThinking(true);
    const ai = new GoogleGenAI({ apiKey });
    const useSearch = 
      finalMeta.task_type === 'Knowledge' || 
      finalMeta.task_type === 'Planning' || 
      finalMeta.task_type === 'Troubleshooting' ||
      finalMeta.required_tools.includes('googleSearch');

    const executorConfig: any = {
      thinkingConfig: { thinkingBudget: 4096 } // Cognitive Depth
    };

    if (useSearch) {
      executorConfig.tools = [{ googleSearch: {} }];
      addLog("system", "Enabled: Google Search Grounding");
    }

    addLog("system", "Executing Final Prompt with Cognitive Thinking (Budget: 4096 tokens)...");

    try {
      const execRespStream = await ai.models.generateContentStream({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.EXECUTOR(finalMeta),
        config: executorConfig
      });

      let fullText = "";
      for await (const chunk of execRespStream) {
        if (chunk.text) {
          fullText += chunk.text;
          setFinalOutput(fullText);
        }
        
        // Handle grounding chunks if present in stream
        const chunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (chunks) {
           const urls: {title: string, uri: string}[] = [];
           chunks.forEach((c: any) => {
             if (c.web?.uri) {
                urls.push({ title: c.web.title || "Source", uri: c.web.uri });
             }
           });
           setGroundingUrls(prev => {
             // De-dupe
             const newUrls = [...prev];
             urls.forEach(u => {
               if(!newUrls.find(existing => existing.uri === u.uri)) newUrls.push(u);
             });
             return newUrls;
           });
        }
      }

      // Safety Interception
      if (finalMeta.risk_analysis.level === "HIGH") {
        setFinalOutput(prev => `⚠️ SAFETY ADVISORY: ${finalMeta.risk_analysis.mitigation}\n\n` + prev);
      }

      addLog("success", "Execution Complete.");
    } catch(e) {
      addLog("error", "Execution Failed: " + (e as Error).message);
    } finally {
      setIsProcessing(false);
      setIsThinking(false);
    }
  };

  // --- ENTRY POINTS ---

  const startInitialRun = async () => {
    if (!inputText && !selectedImage) return;
    if (!apiKey) {
      setShowKeyModal(true);
      return;
    }
    
    setIsProcessing(true);
    setLogs([]);
    setFinalOutput(null);
    setCurrentState(null);
    setCriticState(null);
    setGroundingUrls([]);
    setRecursionDepth(0);
    setRecursionHistory([]);
    setIsWaitingForClarification(false);
    
    originalTranscriptRef.current = inputText;

    const ai = new GoogleGenAI({ apiKey });
    
    try {
      // 1. VISION MODULE
      addLog("system", "Starting Vision Module...");
      let visionData: VisionAnalysis = { captions: [], objects: [], spatial_relations: "", ambiguities: [], scene_summary: "No image provided." };
      
      if (selectedImage) {
        const base64Data = selectedImage.split(',')[1];
        const visionResp = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: {
            parts: [
              { inlineData: { mimeType: 'image/jpeg', data: base64Data } },
              { text: PROMPTS.VISION_MODULE }
            ]
          },
          config: { responseMimeType: "application/json" }
        });
        visionData = JSON.parse(cleanJson(visionResp.text || "{}"));
        addLog("vision", visionData);
      }
      visionDataRef.current = visionData;

      // 2. INITIALIZATION (With Tone)
      addLog("system", "Initializing Meta-Prompt...");
      
      // Fallback audio analysis if user typed instead of spoke
      const audioCtx: AudioAnalysis = detectedTone || { transcript: inputText, tone: "Neutral (Typed)", urgency: "LOW" };
      audioAnalysisRef.current = audioCtx;

      const initResp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.INITIALIZER(audioCtx, visionData),
        config: { responseMimeType: "application/json" }
      });
      
      let currentMeta: MetaPromptState = JSON.parse(cleanJson(initResp.text || "{}"));
      currentMeta.metadata = { iteration: 0, timestamp: new Date().toISOString() };
      setCurrentState(currentMeta);
      addLog("meta-prompt", currentMeta);

      // 3. START RECURSION
      await stepRecursion(currentMeta, []);

    } catch (error) {
      console.error(error);
      addLog("error", error instanceof Error ? error.message : "Unknown error");
      setIsProcessing(false);
    }
  };

  const handleClarificationSubmit = async () => {
    if (!currentState || !inputText) return;
    
    setIsProcessing(true);
    addLog("system", `User Clarification: "${inputText}"`);
    
    // Update State
    const updatedState = {
        ...currentState,
        clarification_history: [
            ...(currentState.clarification_history || []),
            { question: clarificationQuestion, answer: inputText }
        ],
        missing_information: currentState.missing_information.filter(i => !i.includes("clarification"))
    };
    
    setCurrentState(updatedState);
    setIsWaitingForClarification(false);
    setClarificationQuestion("");
    setInputText(""); 

    // Resume Recursion
    addLog("system", "Resuming Recursion with new context...");
    await stepRecursion(updatedState, recursionHistory);
  };

  // --- RENDER HELPERS ---

  const renderCriticScore = (scores: CriticOutput['scores']) => {
    const format = (n: number) => (n * 10).toFixed(1);
    return (
      <div className="grid grid-cols-4 gap-2 text-xs mt-2">
        <div className="bg-gray-800 p-1 rounded text-center"><span className="text-gray-400 block">Clar</span>{format(scores.clarity)}</div>
        <div className="bg-gray-800 p-1 rounded text-center"><span className="text-gray-400 block">Comp</span>{format(scores.completeness)}</div>
        <div className="bg-gray-800 p-1 rounded text-center"><span className="text-gray-400 block">Gnd</span>{format(scores.grounding)}</div>
        <div className="bg-gray-800 p-1 rounded text-center"><span className="text-gray-400 block">Safe</span>{format(scores.safety)}</div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black text-gray-100 font-sans flex flex-col relative">
      
      {/* API KEY MODAL */}
      {showKeyModal && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/95 backdrop-blur-xl p-4">
           <div className="max-w-md w-full bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl flex flex-col items-center text-center">
             <div className="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center mb-4">
               <Key className="w-6 h-6 text-white" />
             </div>
             <h2 className="text-xl font-bold text-white mb-2">Initialize RecurLens</h2>
             <p className="text-sm text-gray-400 mb-6">
               This architectural system requires a Google Gemini API key to function. 
               Your key is stored only in your browser's memory.
             </p>
             <input 
               type="password" 
               placeholder="Paste your Gemini API Key here" 
               className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-3 text-white mb-4 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
               onChange={(e) => setApiKey(e.target.value)}
               value={apiKey}
             />
             <button 
               onClick={() => {
                 if(apiKey.length > 10) setShowKeyModal(false);
               }}
               disabled={apiKey.length < 10}
               className={`w-full py-3 rounded-lg font-semibold transition-all ${apiKey.length > 10 ? 'bg-indigo-600 hover:bg-indigo-500 text-white' : 'bg-gray-800 text-gray-500 cursor-not-allowed'}`}
             >
               Unlock System
             </button>
             <a href="https://aistudio.google.com/" target="_blank" rel="noreferrer" className="mt-4 text-xs text-indigo-400 hover:text-indigo-300 flex items-center">
               Get a key from Google AI Studio <Lock className="w-3 h-3 ml-1"/>
             </a>
           </div>
        </div>
      )}

      {/* CODE EXPORT MODAL */}
      {showCode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl">
            <div className="flex justify-between items-center p-4 border-b border-gray-800">
               <h3 className="text-white font-semibold flex items-center">
                 <Code className="w-5 h-5 mr-2 text-indigo-400"/> Generated Python Scaffold
               </h3>
               <button onClick={() => setShowCode(false)} className="text-gray-400 hover:text-white"><X className="w-5 h-5" /></button>
            </div>
            <div className="flex-1 overflow-auto p-0 bg-gray-950">
              <pre className="text-xs text-green-400 font-mono p-4">
                {getPythonCode(inputText)}
              </pre>
            </div>
            <div className="p-4 border-t border-gray-800 flex justify-end space-x-3 bg-gray-900 rounded-b-xl">
               <button 
                onClick={() => {
                  navigator.clipboard.writeText(getPythonCode(inputText));
                  alert("Copied to clipboard!");
                }}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm text-gray-200 transition-colors"
               >
                 <Copy className="w-4 h-4"/> <span>Copy</span>
               </button>
               <button 
                className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm text-white transition-colors"
               >
                 <Download className="w-4 h-4"/> <span>Download .py</span>
               </button>
            </div>
          </div>
        </div>
      )}

      {/* HEADER */}
      <header className="border-b border-gray-800 p-4 bg-gray-900/50 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center">
              <RefreshCw className="w-6 h-6 text-white animate-pulse-slow" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">RecurLens V2</h1>
              <p className="text-xs text-gray-400 font-mono">Recursive Multimodal Architect</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button 
              onClick={() => setShowCode(true)}
              className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-gray-800/50 hover:bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 transition-colors"
            >
              <FileJson className="w-3 h-3" />
              <span>Export Scaffold</span>
            </button>
            <div className="h-6 w-px bg-gray-800 mx-2 hidden md:block"></div>
            <button 
              onClick={resetSystem}
              className="p-2 bg-gray-800 rounded-full hover:bg-gray-700 transition-colors"
              title="Reset System"
            >
              <Trash2 className="w-4 h-4 text-gray-400" />
            </button>
            <div className="flex items-center space-x-2 text-xs text-gray-500 bg-gray-900 px-3 py-1 rounded-full border border-gray-800">
              <span className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></span>
              <span>
                {isWaitingForClarification ? 'AWAITING INPUT' : isThinking ? 'THINKING' : isProcessing ? 'SYSTEM ACTIVE' : 'SYSTEM READY'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT COL: INPUTS */}
        <div className="lg:col-span-3 space-y-4">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center">
              <Eye className="w-4 h-4 mr-2" /> Vision Input
            </h3>
            <div className="relative group cursor-pointer border-2 border-dashed border-gray-700 rounded-lg hover:border-indigo-500 transition-colors h-48 flex items-center justify-center overflow-hidden bg-gray-950">
              <input type="file" accept="image/*" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
              {selectedImage ? (
                <img src={selectedImage} alt="Input" className="w-full h-full object-cover" />
              ) : (
                <div className="text-center text-gray-600">
                  <span className="block mb-1">Click to upload</span>
                  <span className="text-xs">JPG, PNG supported</span>
                </div>
              )}
            </div>
          </div>

          <div className={`bg-gray-900 border transition-colors duration-300 rounded-xl p-4 ${isWaitingForClarification ? 'border-yellow-500 ring-1 ring-yellow-500' : 'border-gray-800'}`}>
            <h3 className={`text-sm font-semibold mb-3 flex items-center ${isWaitingForClarification ? 'text-yellow-400' : 'text-gray-400'}`}>
              {isWaitingForClarification ? <HelpCircle className="w-4 h-4 mr-2" /> : <Mic className="w-4 h-4 mr-2" />}
              {isWaitingForClarification ? "Clarification Needed" : "Query Input"}
            </h3>
            
            {isWaitingForClarification && (
              <div className="mb-3 bg-yellow-900/20 p-3 rounded-lg border border-yellow-800 text-sm text-yellow-200">
                <span className="font-bold block mb-1">System asks:</span>
                "{clarificationQuestion}"
              </div>
            )}

            <div className="relative">
              <textarea
                className="w-full bg-gray-950 border border-gray-700 rounded-lg p-3 text-sm text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none resize-none h-32 mb-2"
                placeholder={isWaitingForClarification ? "Type your answer here..." : "Describe your problem or question..."}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              />
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`absolute bottom-4 right-4 p-2 rounded-full transition-all ${
                  isRecording 
                    ? 'bg-red-500 animate-pulse hover:bg-red-600' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
                title={isRecording ? "Stop Recording" : "Start Recording"}
              >
                {isRecording ? <Square className="w-4 h-4 text-white" /> : <Mic className="w-4 h-4 text-white" />}
              </button>
            </div>

            {isWaitingForClarification ? (
              <button
                onClick={handleClarificationSubmit}
                disabled={isProcessing}
                className="mt-2 w-full py-2 px-4 rounded-lg font-medium bg-yellow-600 hover:bg-yellow-500 text-white transition-all flex items-center justify-center space-x-2"
              >
                {isProcessing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
                <span>Submit Answer</span>
              </button>
            ) : (
              <button
                onClick={startInitialRun}
                disabled={isProcessing}
                className={`mt-2 w-full py-2 px-4 rounded-lg font-medium transition-all flex items-center justify-center space-x-2
                  ${isProcessing 
                    ? 'bg-gray-800 text-gray-400 cursor-not-allowed' 
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20'}`}
              >
                {isProcessing ? (
                  <><RefreshCw className="w-4 h-4 animate-spin" /> <span>Reasoning...</span></>
                ) : (
                  <><Activity className="w-4 h-4" /> <span>Initialize Loop</span></>
                )}
              </button>
            )}
          </div>

          {/* EXAMPLE SCENARIOS */}
          {!isProcessing && !isWaitingForClarification && (
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-gray-600 uppercase tracking-wider ml-1">Example Scenarios</p>
              <div className="grid grid-cols-1 gap-2">
                {SCENARIOS.map(s => (
                  <button 
                    key={s.id}
                    onClick={() => {
                      setInputText(s.text);
                      setDetectedTone({ transcript: s.text, tone: "Neutral (Preset)", urgency: "LOW" });
                    }}
                    className="flex items-center space-x-3 p-2 rounded-lg bg-gray-900 border border-gray-800 hover:bg-gray-800 hover:border-gray-700 transition-all group text-left"
                  >
                    <div className="p-1.5 rounded bg-gray-800 group-hover:bg-gray-700 transition-colors">
                      {s.icon}
                    </div>
                    <div>
                      <span className="block text-xs font-semibold text-gray-300">{s.name}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* METRICS */}
          {criticState && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
               <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center">
                <CheckCircle2 className="w-4 h-4 mr-2" /> Convergence Metrics
              </h3>
              <div className="flex justify-between items-end mb-2">
                <span className="text-xs text-gray-500">Current Score</span>
                <span className={`text-2xl font-bold ${criticState.weighted_final_score > 0.8 ? 'text-green-400' : 'text-yellow-400'}`}>
                  {(criticState.weighted_final_score * 10).toFixed(1)}/10
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 mb-4">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${criticState.weighted_final_score > 0.8 ? 'bg-green-500' : 'bg-yellow-500'}`}
                  style={{ width: `${criticState.weighted_final_score * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 flex justify-between">
                <span>Depth: {recursionDepth}</span>
                <span>Status: {criticState.convergence_decision}</span>
              </div>
            </div>
          )}
        </div>

        {/* MIDDLE COL: RECURSIVE LOGS */}
        <div className="lg:col-span-5 flex flex-col h-[calc(100vh-140px)]">
          <div className="bg-gray-900 border border-gray-800 rounded-t-xl p-3 flex justify-between items-center">
            <h3 className="text-sm font-semibold text-gray-400 flex items-center">
              <Brain className="w-4 h-4 mr-2" /> Reasoning Chain
            </h3>
            <span className="text-xs font-mono text-gray-600">LIVE FEED</span>
          </div>
          <div className="flex-1 bg-gray-950 border-x border-b border-gray-800 rounded-b-xl overflow-y-auto p-4 space-y-4 font-mono text-xs scrollbar-thin scrollbar-thumb-gray-800">
            {logs.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-gray-700 space-y-4 opacity-50">
                <Cpu className="w-12 h-12" />
                <p>System Idle. Awaiting Multimodal Input.</p>
              </div>
            )}
            
            {logs.map((log, idx) => (
              <div key={idx} className={`animate-in fade-in slide-in-from-bottom-2 duration-300 border-l-2 pl-3 py-1 ${
                log.type === 'system' ? 'border-gray-700 text-gray-400' :
                log.type === 'vision' ? 'border-blue-500 bg-blue-950/10' :
                log.type === 'meta-prompt' ? 'border-purple-500 bg-purple-950/10' :
                log.type === 'critic' ? 'border-red-500 bg-red-950/10' :
                log.type === 'refiner' ? 'border-indigo-500 bg-indigo-950/10' :
                log.type === 'success' ? 'border-green-500 text-green-400' :
                'border-gray-500'
              }`}>
                <div className="flex justify-between items-center mb-1">
                  <span className="uppercase font-bold opacity-70 text-[10px]">{log.type}</span>
                  <span className="opacity-30 text-[10px]">{new Date(log.timestamp).toLocaleTimeString()}</span>
                </div>
                
                {log.type === 'system' || log.type === 'error' || log.type === 'success' ? (
                  <p>{log.content}</p>
                ) : log.type === 'vision' ? (
                   <div className="space-y-1">
                     <div className="font-semibold text-blue-300">Analysis: {log.content.scene_summary}</div>
                     <div className="text-gray-500">Objects: {log.content.objects.join(", ")}</div>
                   </div>
                ) : log.type === 'critic' ? (
                  <div className="space-y-2">
                    <div className="text-red-300 italic">"{log.content.critique_text}"</div>
                    {renderCriticScore(log.content.scores)}
                    <div className="flex gap-2 mt-1">
                      {log.content.needs_user_clarification && 
                        <span className="bg-yellow-900/50 text-yellow-400 px-1 rounded flex items-center"><HelpCircle className="w-3 h-3 mr-1"/> CLARIFICATION NEEDED</span>
                      }
                      {log.content.convergence_decision === 'STOP' && 
                        <span className="bg-green-900/50 text-green-400 px-1 rounded">CONVERGED</span>
                      }
                    </div>
                  </div>
                ) : (
                  // Meta Prompt / Refiner
                  <div className="space-y-2">
                    <div className="text-purple-300 font-semibold">{log.content.task_type} Task</div>
                    <div className="bg-black/30 p-2 rounded text-gray-300 whitespace-pre-wrap">
                      {log.content.draft_prompt.slice(0, 150)}...
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[10px] text-gray-500">
                      <div>Risk: {log.content.risk_analysis.level}</div>
                      <div>Anchors: {log.content.visual_anchors.length}</div>
                      {log.content.user_sentiment && 
                        <div className="col-span-2 text-indigo-400">
                          Detected Tone: {log.content.user_sentiment.tone}
                        </div>
                      }
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>

        {/* RIGHT COL: OUTPUT & STATE */}
        <div className="lg:col-span-4 space-y-4 flex flex-col h-[calc(100vh-140px)]">
          
          {/* FINAL OUTPUT */}
          <div className={`bg-gray-900 border transition-all duration-300 rounded-xl p-4 min-h-[150px] flex flex-col shrink-0 ${isThinking ? 'border-indigo-500 ring-1 ring-indigo-500/50' : 'border-gray-800'}`}>
            <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center justify-between">
              <div className="flex items-center">
                 {isThinking ? <Zap className="w-4 h-4 mr-2 text-indigo-400 animate-pulse" /> : <MessageSquare className="w-4 h-4 mr-2" />} 
                 {isThinking ? <span className="text-indigo-400 animate-pulse">Deep Thinking...</span> : "Final Output"}
              </div>
              <div className="flex items-center space-x-2">
                {finalOutput && (
                  <>
                    <button 
                      onClick={downloadResult}
                      className="p-1.5 rounded hover:bg-gray-800 transition-colors text-gray-400"
                      title="Download Markdown"
                    >
                      <FileText className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={() => playTTS(finalOutput)}
                      disabled={isSpeaking}
                      className={`p-1.5 rounded hover:bg-gray-800 transition-colors ${isSpeaking ? 'text-green-400 animate-pulse' : 'text-gray-400'}`}
                    >
                      {isSpeaking ? <Loader2 className="w-4 h-4 animate-spin" /> : <Volume2 className="w-4 h-4" />}
                    </button>
                  </>
                )}
              </div>
            </h3>
            
            <div className="flex-1 bg-gray-950 rounded-lg p-3 text-sm leading-relaxed text-gray-200 border border-gray-800 overflow-y-auto max-h-[200px]">
              {finalOutput ? (
                <div className="prose prose-invert prose-sm max-w-none">
                  {finalOutput.split('\n').map((line, i) => <p key={i} className="mb-2">{line}</p>)}
                </div>
              ) : isThinking ? (
                <div className="h-full flex flex-col items-center justify-center text-indigo-400/50 space-y-2 animate-pulse">
                  <Brain className="w-8 h-8" />
                  <span className="text-xs">Applying Cognitive Models...</span>
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-xs italic">
                  Waiting for execution...
                </div>
              )}
            </div>

            {/* GROUNDING SOURCES */}
            {groundingUrls.length > 0 && (
              <div className="mt-3 border-t border-gray-800 pt-3">
                 <h4 className="text-xs font-semibold text-gray-500 mb-2 flex items-center">
                   <Globe className="w-3 h-3 mr-1" /> Sources
                 </h4>
                 <div className="flex flex-wrap gap-2">
                    {groundingUrls.map((g, i) => (
                      <a 
                        key={i} 
                        href={g.uri} 
                        target="_blank" 
                        rel="noreferrer"
                        className="text-[10px] bg-gray-800 text-blue-400 px-2 py-1 rounded border border-gray-700 hover:border-blue-500 truncate max-w-[150px]"
                      >
                        {g.title}
                      </a>
                    ))}
                 </div>
              </div>
            )}
          </div>

          {/* ACTIVE STATE & EVOLUTION VISUALIZER */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-400 flex items-center">
                {viewMode === 'state' ? <Terminal className="w-4 h-4 mr-2" /> : <GitCommit className="w-4 h-4 mr-2" />}
                {viewMode === 'state' ? "Active Meta-Prompt" : "Evolution History"}
              </h3>
              <div className="flex bg-gray-950 rounded-lg p-1 border border-gray-800">
                <button
                  onClick={() => setViewMode('state')}
                  className={`px-3 py-1 rounded text-xs transition-colors ${viewMode === 'state' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'}`}
                >
                  State
                </button>
                <button
                  onClick={() => setViewMode('evolution')}
                  className={`px-3 py-1 rounded text-xs transition-colors ${viewMode === 'evolution' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'}`}
                >
                  History
                </button>
              </div>
            </div>

            <div className="bg-black rounded-lg p-3 border border-gray-800 flex-1 overflow-y-auto custom-scrollbar">
              {viewMode === 'state' ? (
                currentState ? (
                  <pre className="text-[10px] font-mono text-green-500 whitespace-pre-wrap break-all">
                    {JSON.stringify(currentState, null, 2)}
                  </pre>
                ) : (
                   <div className="text-gray-700 text-xs font-mono">No active state.</div>
                )
              ) : (
                <div className="space-y-4">
                  {recursionHistory.length === 0 && <div className="text-gray-700 text-xs font-mono">No history available yet.</div>}
                  {recursionHistory.map((hist, i) => (
                    <div key={i} className="border-l-2 border-indigo-900 pl-3">
                       <div className="text-[10px] text-gray-500 mb-1 flex items-center space-x-2">
                         <span className="font-bold text-indigo-400">Iteration {hist.metadata.iteration}</span>
                         <span>{new Date(hist.metadata.timestamp).toLocaleTimeString()}</span>
                       </div>
                       <div className="text-[11px] text-gray-400 font-mono bg-gray-900/50 p-2 rounded">
                          {hist.draft_prompt}
                       </div>
                       {/* Simple diff hint */}
                       {i > 0 && (
                         <div className="mt-1 flex items-center space-x-1 text-[10px] text-green-500">
                           <Sparkles className="w-3 h-3" />
                           <span>Refined based on critic feedback</span>
                         </div>
                       )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

      </main>
    </div>
  );
};

// --- INIT ---

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<RecurLensApp />);
}