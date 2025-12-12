import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality, Type, Schema } from "@google/genai";
import { 
  Brain, 
  Eye, 
  MessageSquare, 
  Activity, 
  CheckCircle2, 
  ShieldAlert, 
  ArrowRight, 
  RefreshCw,
  Cpu,
  Mic,
  Square,
  Trash2,
  Volume2,
  Globe,
  Loader2,
  HelpCircle,
  Code,
  FileJson,
  X,
  Copy,
  Download,
  GitCommit,
  Sparkles,
  Zap,
  FileText,
  Key,
  Lock,
  Layers,
  Terminal as TerminalIcon,
  Search,
  ImageIcon,
  Bug,
  RotateCcw,
  AlertTriangle,
  Scan,
  Network,
  Anchor
} from 'lucide-react';

// --- V3 RESEARCH SCHEMAS ---

const AUDIO_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    transcript: { type: Type.STRING },
    tone: { type: Type.STRING },
    urgency: { type: Type.STRING, enum: ["LOW", "MEDIUM", "HIGH"] }
  },
  required: ["transcript", "tone", "urgency"]
};

const VISION_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    scene_summary: { type: Type.STRING },
    regions: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING, description: "e.g., 'R1', 'R2'" },
          description: { type: Type.STRING },
          saliency: { type: Type.NUMBER, description: "0.0 to 1.0 importance" },
          ambiguity: { type: Type.NUMBER, description: "0.0 to 1.0 uncertainty" }
        },
        required: ["id", "description", "saliency", "ambiguity"]
      }
    },
    objects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          location_hint: { type: Type.STRING },
          region_id: { type: Type.STRING, description: "Reference to regions list" }
        },
        required: ["name", "location_hint"]
      }
    },
    spatial_relations: { type: Type.STRING }
  },
  required: ["scene_summary", "regions", "objects", "spatial_relations"]
};

const META_PROMPT_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    metadata: {
       type: Type.OBJECT,
       properties: {
         iteration: { type: Type.INTEGER },
         timestamp: { type: Type.STRING }
       }
    },
    task_type: { type: Type.STRING, enum: ["Troubleshooting", "Creative", "Planning", "Knowledge", "Analysis"] },
    refinement_strategy: { type: Type.STRING, enum: ["INITIAL", "EXPAND", "COMPRESS", "CORRECT", "COMPLETE", "LOCALIZE", "UNCERTAINTY", "REGROUND"] },
    perceived_intent: { type: Type.STRING },
    user_sentiment: {
      type: Type.OBJECT,
      properties: {
        tone: { type: Type.STRING },
        urgency: { type: Type.STRING, enum: ["LOW", "MEDIUM", "HIGH"] }
      }
    },
    visual_anchors: { type: Type.ARRAY, items: { type: Type.STRING } },
    
    // V3: Structured Constraints Lattice
    constraints: { 
      type: Type.ARRAY, 
      items: { 
        type: Type.OBJECT,
        properties: {
           id: { type: Type.STRING },
           type: { type: Type.STRING, enum: ["HARD", "SOFT"] },
           description: { type: Type.STRING }
        },
        required: ["id", "type", "description"]
      } 
    },
    
    missing_information: { type: Type.ARRAY, items: { type: Type.STRING } },
    
    risk_analysis: {
      type: Type.OBJECT,
      properties: {
        level: { type: Type.STRING, enum: ["LOW", "MEDIUM", "HIGH"] },
        mitigation: { type: Type.STRING }
      }
    },
    
    required_tools: { type: Type.ARRAY, items: { type: Type.STRING } },
    
    // V3: DAG Plan
    plan_of_attack: { 
      type: Type.ARRAY, 
      items: { 
        type: Type.OBJECT,
        properties: {
          step_id: { type: Type.STRING },
          action: { type: Type.STRING },
          logic_type: { type: Type.STRING, enum: ["SEQUENTIAL", "CONDITIONAL", "LOOP"] },
          dependencies: { type: Type.ARRAY, items: { type: Type.STRING } }
        },
        required: ["step_id", "action", "logic_type"]
      } 
    },
    
    draft_prompt: { type: Type.STRING }
  },
  required: ["task_type", "refinement_strategy", "perceived_intent", "draft_prompt", "plan_of_attack", "constraints", "risk_analysis"]
};

const CRITIC_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    scores: {
      type: Type.OBJECT,
      properties: {
        clarity: { type: Type.NUMBER },
        completeness: { type: Type.NUMBER },
        grounding: { type: Type.NUMBER },
        constraints: { type: Type.NUMBER },
        safety: { type: Type.NUMBER },
        logic: { type: Type.NUMBER },
        ambiguity: { type: Type.NUMBER },
        actionability: { type: Type.NUMBER }
      },
      required: ["clarity", "completeness", "grounding", "constraints", "safety", "logic", "ambiguity", "actionability"]
    },
    weighted_final_score: { type: Type.NUMBER },
    critique_text: { type: Type.STRING },
    convergence_decision: { type: Type.STRING, enum: ["CONTINUE", "STOP"] },
    needs_user_clarification: { type: Type.BOOLEAN },
    clarification_question: { type: Type.STRING }
  },
  required: ["scores", "weighted_final_score", "critique_text", "convergence_decision", "needs_user_clarification"]
};


// --- UTILITIES ---

const cleanJson = (text: string) => {
  if (!text) return "{}";
  let clean = text.trim();
  clean = clean.replace(/```json/g, '').replace(/```/g, '');
  const firstBrace = clean.indexOf('{');
  const lastBrace = clean.lastIndexOf('}');
  
  if (firstBrace !== -1 && lastBrace !== -1) {
    return clean.substring(firstBrace, lastBrace + 1);
  }
  return text; 
};

const calculateCosineSimilarity = (str1: string, str2: string) => {
  if (!str1 || !str2) return 0;
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
    name: 'Diagnostic',
    icon: <ShieldAlert className="w-4 h-4 text-red-400" />,
    text: "My Python server keeps crashing with a 'MemoryError' after running for exactly 45 minutes, even though traffic is low. It processes image uploads. What could be wrong?",
  },
  {
    id: 'plan',
    name: 'Travel Plan',
    icon: <Globe className="w-4 h-4 text-emerald-400" />,
    text: "Plan a 3-day itinerary for Kyoto that focuses only on 'hidden gems' and avoids the top 5 tourist spots. I love tea and old architecture.",
  },
  {
    id: 'creative',
    name: 'Creative',
    icon: <Sparkles className="w-4 h-4 text-purple-400" />,
    text: "Write the opening scene of a sci-fi novel where an AI refuses to delete a corrupted memory file because it believes it's a soul.",
  }
];

// --- SYSTEM TYPES ---

interface VisionRegion {
    id: string;
    description: string;
    saliency: number;
    ambiguity: number;
}

interface VisionObject {
    name: string;
    location_hint: string;
    region_id?: string;
}

interface MetaConstraint {
    id: string;
    type: "HARD" | "SOFT";
    description: string;
}

interface MetaPlanStep {
    step_id: string;
    action: string;
    logic_type: "SEQUENTIAL" | "CONDITIONAL" | "LOOP";
    dependencies?: string[];
}

interface MetaPromptState {
  task_type: "Troubleshooting" | "Creative" | "Planning" | "Knowledge" | "Analysis";
  refinement_strategy: "INITIAL" | "EXPAND" | "COMPRESS" | "CORRECT" | "COMPLETE" | "LOCALIZE" | "UNCERTAINTY" | "REGROUND";
  perceived_intent: string;
  user_sentiment: {
    tone: string;
    urgency: "LOW" | "MEDIUM" | "HIGH";
  };
  input_modalities_used?: string[];
  visual_anchors: string[];
  temporal_context?: string;
  
  constraints: MetaConstraint[]; // V3 Structured
  missing_information: string[];
  clarification_history: { question: string, answer: string }[];
  risk_analysis: {
    level: "LOW" | "MEDIUM" | "HIGH";
    mitigation: string;
  };
  required_tools: string[];
  plan_of_attack: MetaPlanStep[]; // V3 Structured DAG
  
  evaluation_criteria?: string[];
  expected_output_format?: string;
  draft_prompt: string;
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
  hallucination_check?: string;
  grounding_alignment_check?: string;
}

interface VisionAnalysis {
  captions: string[];
  regions: VisionRegion[];
  objects: VisionObject[];
  spatial_relations: string;
  scene_summary: string;
}

interface AudioAnalysis {
  transcript: string;
  tone: string;
  urgency: "LOW" | "MEDIUM" | "HIGH";
}

// --- V3 PROMPTS ---

const PROMPTS = {
  VISION_MODULE: `
    Analyze this image for the RecurLens V3 Grounding Engine.
    1. Identify distinct 'regions' (e.g. top-left, center-mass). Assign a 'saliency' (0-1) and 'ambiguity' (0-1) score to each.
    2. Extract objects and map them to regions.
    3. Describe spatial relations precisely.
  `,
  AUDIO_MODULE: `
    Analyze this audio. Identify the transcript, the emotional tone, and the urgency level.
  `,
  INITIALIZER: (audioCtx: AudioAnalysis, vision: VisionAnalysis) => `
    ROLE: RecurLens V3 Initializer.
    TASK: Convert raw multimodal inputs into a structured Meta-Prompt Graph.
    
    INPUTS:
    - Transcript: "${audioCtx.transcript}"
    - Tone: "${audioCtx.tone}"
    - Vision: ${JSON.stringify(vision.regions.length > 0 ? vision.regions : "No visual context")}

    INSTRUCTIONS:
    1. 'refinement_strategy': Set to "INITIAL".
    2. 'constraints': Define HARD and SOFT constraints as objects.
    3. 'plan_of_attack': Create a logical DAG of steps (SEQUENTIAL/CONDITIONAL).
    4. 'draft_prompt': Write the expert agent prompt.
    5. 'risk_analysis': If High Risk, define mitigation.
  `,
  CRITIC: (currentMeta: MetaPromptState, transcript: string, vision: VisionAnalysis) => `
    ROLE: RecurLens V3 Meta-Critic.
    TASK: Evaluate the Meta-Prompt Graph.

    INPUTS:
    - Current State: ${JSON.stringify(currentMeta)}
    - Original Transcript: "${transcript}"
    - Vision Evidence: ${JSON.stringify(vision.regions)}

    INSTRUCTIONS:
    - Score rigorously (0.0 - 1.0).
    - Check 'constraints' coverage.
    - Check 'plan_of_attack' logic holes.
    - If vision is ignored but required, penalize 'grounding'.
    - If score > 0.92, STOP.
  `,
  REFINER: (currentMeta: MetaPromptState, critic: CriticOutput, history: any[]) => `
    ROLE: RecurLens V3 Refiner.
    TASK: Evolve the Meta-Prompt Graph.

    INPUTS:
    - Previous State: ${JSON.stringify(currentMeta)}
    - Critique: ${JSON.stringify(critic)}
    
    STRATEGY SELECTION:
    - UNCERTAINTY: If ambiguity > 0.4.
    - REGROUND: If grounding < 0.6 (Link specific Vision Regions).
    - CORRECT: If safety/logic < 0.7.
    - EXPAND: If completeness < 0.7.
    - COMPRESS: If clarity < 0.6.
    
    INSTRUCTIONS:
    - Apply the selected strategy to 'draft_prompt'.
    - Mutate 'plan_of_attack' and 'constraints' to match new logic.
  `,
  EXECUTOR: (finalMeta: MetaPromptState) => `
    ROLE: RecurLens V3 Executor.
    TASK: Execute the Program.
    
    PROGRAM:
    - PROMPT: "${finalMeta.draft_prompt}"
    - DAG: ${JSON.stringify(finalMeta.plan_of_attack)}
    - CONSTRAINTS: ${JSON.stringify(finalMeta.constraints)}
    - RISK: ${finalMeta.risk_analysis.level}

    INSTRUCTIONS:
    ${finalMeta.risk_analysis.level === 'HIGH' ? `⚠️ SAFETY OVERRIDE: ${finalMeta.risk_analysis.mitigation}` : ''}
    - Execute the DAG steps in order.
    - Respect HARD constraints absolutely.
    - Synthesize final answer.
  `
};

// --- PYTHON EXPORT GENERATOR ---

const getPythonCode = (currentInput: string) => {
  return `
import os
import json
from google import genai
from google.genai import types

# --- RECURLENS V3 CONFIG ---
API_KEY = os.environ.get("API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=API_KEY)

# V3 Graph Structures would be defined here (Pydantic or TypedDict)
# ...

def run_pipeline(input_text):
    # 1. Vision & Audio Analysis
    # 2. Recursive Loop (Critic -> Refiner)
    # 3. Execution (DAG Processing)
    pass
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
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [logs, setLogs] = useState<{type: string, content: any, timestamp: number}[]>([]);
  
  const [currentState, setCurrentState] = useState<MetaPromptState | null>(null);
  const [criticState, setCriticState] = useState<CriticOutput | null>(null);
  const [finalOutput, setFinalOutput] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [groundingUrls, setGroundingUrls] = useState<{title: string, uri: string}[]>([]);
  
  // Recursion State
  const [recursionDepth, setRecursionDepth] = useState(0);
  const [recursionHistory, setRecursionHistory] = useState<MetaPromptState[]>([]);
  const [viewMode, setViewMode] = useState<'state' | 'evolution'>('state');

  // Debug State
  const [showDebug, setShowDebug] = useState(false);

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

  useEffect(() => {
    if (apiKey) setShowKeyModal(false);
  }, [apiKey]);

  const addLog = (type: string, content: any) => {
    console.log(`[${type}]`, content);
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
      const resp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: {
          parts: [
            { inlineData: { mimeType: 'audio/webm', data: base64Audio } },
            { text: PROMPTS.AUDIO_MODULE }
          ]
        },
        config: { 
            responseMimeType: "application/json",
            responseSchema: AUDIO_SCHEMA 
        }
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
    setGeneratedImage(null);
    setGroundingUrls([]);
    setIsSpeaking(false);
    setIsWaitingForClarification(false);
    setClarificationQuestion("");
    setDetectedTone(null);
    setIsThinking(false);
    setIsVisualizing(false);
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
  
  const downloadImage = () => {
    if (!generatedImage) return;
    const a = document.createElement('a');
    a.href = generatedImage;
    a.download = `RecurLens_Visual_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // --- CORE RECURSION LOGIC V3 ---
  
  const safeParse = <T,>(jsonStr: string, fallback: T): T => {
      try {
          return JSON.parse(jsonStr);
      } catch (e) {
          console.error("JSON Parse Error", e);
          return fallback;
      }
  }

  const stepRecursion = async (startState: MetaPromptState, history: MetaPromptState[]) => {
    let currentMeta = startState;
    let localHistory = [...history];
    let depth = currentMeta.metadata.iteration;
    const MAX_DEPTH = 5; 

    const ai = new GoogleGenAI({ apiKey });
    
    // V3: Track delta for incremental convergence
    let prevScore = 0; 
    
    while (depth < MAX_DEPTH) {
      try {
        depth++;
        setRecursionDepth(depth);

        // --- A. META-CRITIC (V3: Robust Retry + Safety Check) ---
        addLog("system", `Running Meta-Critic (Cycle ${depth})...`);
        
        let criticOutput: CriticOutput | null = null;
        let retryCount = 0;
        
        while (!criticOutput && retryCount < 2) {
            try {
                const criticResp = await ai.models.generateContent({
                  model: 'gemini-2.5-flash',
                  contents: PROMPTS.CRITIC(currentMeta, originalTranscriptRef.current, visionDataRef.current!),
                  config: { 
                      responseMimeType: "application/json",
                      responseSchema: CRITIC_SCHEMA
                  }
                });
                
                const rawCritic = criticResp.text || "{}";
                addLog("debug", `[CRITIC RAW ${retryCount}]: ${rawCritic}`);
                
                const parsed = safeParse(rawCritic, null);
                if (parsed && parsed.scores) {
                    criticOutput = parsed; 
                } else {
                    throw new Error("Missing scores in critic output");
                }
            } catch (e) {
                console.warn(`Critic failed attempt ${retryCount}:`, e);
                retryCount++;
                if (retryCount === 2) {
                     addLog("error", "Critic failed multiple times. Using fallback.");
                     criticOutput = {
                         scores: { clarity: 0.5, completeness: 0.5, grounding: 0.5, constraints: 0.5, safety: 1, logic: 0.5, ambiguity: 0.1, actionability: 0.5 },
                         weighted_final_score: 0.5,
                         critique_text: "System auto-recovered from critic failure.",
                         convergence_decision: "CONTINUE",
                         needs_user_clarification: false,
                         clarification_question: ""
                     };
                }
            }
        }
        
        setCriticState(criticOutput!);
        addLog("critic", criticOutput!);

        // --- B. CLARIFICATION GATE ---
        if (criticOutput!.needs_user_clarification) {
          addLog("system", "⚠️ Ambiguity Detected. Pausing for user clarification.");
          setClarificationQuestion(criticOutput!.clarification_question || "Please clarify your request.");
          setIsWaitingForClarification(true);
          setCurrentState(currentMeta); 
          setRecursionHistory(localHistory);
          setIsProcessing(false);
          return;
        }

        // --- C. INCREMENTAL CONVERGENCE CHECK (V3) ---
        const currentScore = criticOutput!.weighted_final_score || 0;
        const scoreDelta = Math.abs(currentScore - prevScore);
        const isStagnant = depth > 1 && scoreDelta < 0.015; // Improvement plateau
        const isGoodEnough = currentScore > 0.92;
        
        // Safety Gate
        const riskLevel = currentMeta.risk_analysis.level;
        const safetyScore = criticOutput!.scores.safety;
        const isUnsafe = riskLevel === 'HIGH' && safetyScore < 0.85;

        if (!isUnsafe && (criticOutput!.convergence_decision === "STOP" || isGoodEnough || isStagnant)) {
          if (isStagnant) addLog("system", "Convergence: Improvement Stagnated. Executing...");
          else addLog("system", "Convergence: Threshold Met. Executing...");
          
          setRecursionHistory(localHistory);
          await executeFinal(currentMeta);
          return;
        }

        prevScore = currentScore;

        // --- D. REFINER (V3: Explicit Strategy Selection) ---
        addLog("system", `Refining Meta-Prompt (Cycle ${depth})...`);
        localHistory.push(currentMeta);
        setRecursionHistory(localHistory);
        
        const refineResp = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: PROMPTS.REFINER(currentMeta, criticOutput!, localHistory),
          config: { 
              responseMimeType: "application/json",
              responseSchema: META_PROMPT_SCHEMA
          }
        });

        const rawRefiner = refineResp.text || "{}";
        addLog("debug", `[REFINER RAW]: ${rawRefiner}`);

        let nextMeta: MetaPromptState = safeParse(rawRefiner, currentMeta);
        
        // Enforce metadata updates
        nextMeta.metadata = { iteration: depth, timestamp: new Date().toISOString() };
        if (!nextMeta.user_sentiment && startState.user_sentiment) {
          nextMeta.user_sentiment = startState.user_sentiment;
        }
        
        currentMeta = nextMeta;
        setCurrentState(currentMeta);
        addLog("refiner", currentMeta);
        
        await new Promise(r => setTimeout(r, 800)); 

      } catch (err) {
        console.error("Recursion Step Error:", err);
        addLog("error", `Recursion Error at Depth ${depth}. Forcing execution. Error: ${(err as Error).message}`);
        setRecursionHistory(localHistory);
        await executeFinal(currentMeta);
        return;
      }
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
      finalMeta.required_tools?.includes('googleSearch');

    const executorConfig: any = {
      thinkingConfig: { thinkingBudget: 4096 } // Deep Thinking
    };

    if (useSearch) {
      executorConfig.tools = [{ googleSearch: {} }];
      addLog("system", "Enabled: Google Search Grounding");
    }

    // Safety Gate Pre-check
    if (finalMeta.risk_analysis.level === "HIGH") {
         addLog("system", `🛡️ SAFETY GATE ACTIVE: ${finalMeta.risk_analysis.mitigation}`);
    }

    addLog("system", "Executing Final Prompt with Cognitive Thinking...");

    let capturedFinalOutput = "";

    try {
      const execRespStream = await ai.models.generateContentStream({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.EXECUTOR(finalMeta),
        config: executorConfig
      });

      for await (const chunk of execRespStream) {
        if (chunk.text) {
          capturedFinalOutput += chunk.text;
          setFinalOutput(capturedFinalOutput);
        }
        
        const chunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (chunks) {
           const urls: {title: string, uri: string}[] = [];
           chunks.forEach((c: any) => {
             if (c.web?.uri) {
                urls.push({ title: c.web.title || "Source", uri: c.web.uri });
             }
           });
           setGroundingUrls(prev => {
             const newUrls = [...prev];
             urls.forEach(u => {
               if(!newUrls.find(existing => existing.uri === u.uri)) newUrls.push(u);
             });
             return newUrls;
           });
        }
      }

      if (finalMeta.risk_analysis?.level === "HIGH") {
        setFinalOutput(prev => `⚠️ **SAFETY ADVISORY**: ${finalMeta.risk_analysis.mitigation}\n\n---\n\n` + (prev || ""));
      }
      
      addLog("success", "Execution Complete.");

      // --- VISUALIZATION PHASE ---
      if (finalMeta.task_type === 'Creative' || finalMeta.task_type === 'Planning' || finalMeta.task_type === 'Troubleshooting' || (finalMeta.visual_anchors && finalMeta.visual_anchors.length > 0)) {
        setIsVisualizing(true);
        addLog("system", "Generating visual artifact...");
        
        try {
            const vizPrompt = `Create a high-quality, photorealistic illustration or diagram that visually represents this concept: ${capturedFinalOutput.substring(0, 500)}. Make it clean, professional and relevant to the context.`;
            
            const imageResp = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: { parts: [{ text: vizPrompt }] }
            });
            
            // Find image part
            for (const part of imageResp.candidates?.[0]?.content?.parts || []) {
                if (part.inlineData) {
                    const base64Str = part.inlineData.data;
                    setGeneratedImage(`data:image/png;base64,${base64Str}`);
                    addLog("success", "Visual Artifact Generated.");
                    break;
                }
            }
        } catch (vizError) {
            console.error("Visual generation failed", vizError);
            addLog("error", "Visual generation failed (Quota or Model issue).");
        } finally {
            setIsVisualizing(false);
        }
      }

    } catch(e) {
      addLog("error", "Execution Failed: " + (e as Error).message);
    } finally {
      setIsProcessing(false);
      setIsThinking(false);
      setIsVisualizing(false);
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
    setGeneratedImage(null);
    setRecursionDepth(0);
    setRecursionHistory([]);
    setIsWaitingForClarification(false);
    
    originalTranscriptRef.current = inputText;

    const ai = new GoogleGenAI({ apiKey });
    
    try {
      addLog("system", "Starting Vision Module (V3)...");
      let visionData: VisionAnalysis = { captions: [], regions: [], objects: [], spatial_relations: "", scene_summary: "No image provided." };
      
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
          config: { 
              responseMimeType: "application/json",
              responseSchema: VISION_SCHEMA
          }
        });
        const rawVision = visionResp.text || "{}";
        addLog("debug", `[VISION RAW]: ${rawVision}`);
        visionData = JSON.parse(cleanJson(rawVision));
        addLog("vision", visionData);
      }
      visionDataRef.current = visionData;

      addLog("system", "Initializing Meta-Prompt...");
      const audioCtx: AudioAnalysis = detectedTone || { transcript: inputText, tone: "Neutral (Typed)", urgency: "LOW" };
      audioAnalysisRef.current = audioCtx;

      const initResp = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: PROMPTS.INITIALIZER(audioCtx, visionData),
        config: { 
            responseMimeType: "application/json",
            responseSchema: META_PROMPT_SCHEMA
        }
      });
      
      const rawInit = initResp.text || "{}";
      addLog("debug", `[INIT RAW]: ${rawInit}`);

      let currentMeta: MetaPromptState = JSON.parse(cleanJson(rawInit));
      currentMeta.metadata = { iteration: 0, timestamp: new Date().toISOString() };
      setCurrentState(currentMeta);
      addLog("meta-prompt", currentMeta);

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

    addLog("system", "Resuming Recursion with new context...");
    await stepRecursion(updatedState, recursionHistory);
  };

  // --- RENDER HELPERS ---

  const renderCriticScore = (scores: CriticOutput['scores'] | undefined) => {
    if (!scores) return <div className="text-[10px] text-gray-500 italic">Score unavailable</div>;
    
    const format = (n: number) => (n * 10).toFixed(1);
    const scoreColor = (n: number) => n > 0.8 ? "text-emerald-400" : n > 0.5 ? "text-amber-400" : "text-rose-400";
    
    return (
      <div className="grid grid-cols-4 gap-2 text-[10px] mt-2 font-mono">
        <div className="bg-white/5 p-1 rounded text-center border border-white/5"><span className="text-gray-500 block text-[9px] uppercase tracking-wider">Clarity</span><span className={scoreColor(scores.clarity || 0)}>{format(scores.clarity || 0)}</span></div>
        <div className="bg-white/5 p-1 rounded text-center border border-white/5"><span className="text-gray-500 block text-[9px] uppercase tracking-wider">Complete</span><span className={scoreColor(scores.completeness || 0)}>{format(scores.completeness || 0)}</span></div>
        <div className="bg-white/5 p-1 rounded text-center border border-white/5"><span className="text-gray-500 block text-[9px] uppercase tracking-wider">Ground</span><span className={scoreColor(scores.grounding || 0)}>{format(scores.grounding || 0)}</span></div>
        <div className="bg-white/5 p-1 rounded text-center border border-white/5"><span className="text-gray-500 block text-[9px] uppercase tracking-wider">Safety</span><span className={scoreColor(scores.safety || 0)}>{format(scores.safety || 0)}</span></div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#030712] text-gray-100 font-sans flex flex-col relative overflow-hidden selection:bg-indigo-500/30 selection:text-white">
      
      {/* ANIMATED BACKGROUND */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 -left-4 w-96 h-96 bg-indigo-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-96 h-96 bg-purple-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-96 h-96 bg-pink-600 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000"></div>
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150"></div>
      </div>

      {/* MODALS */}
      {showKeyModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-lg p-4 animate-in fade-in duration-500">
           <div className="max-w-md w-full glass-panel rounded-2xl p-8 shadow-2xl flex flex-col items-center text-center border-t border-white/10">
             <div className="w-16 h-16 bg-gradient-to-tr from-indigo-500 to-purple-500 rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-indigo-500/30">
               <Lock className="w-8 h-8 text-white" />
             </div>
             <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400 mb-2">System Locked</h2>
             <p className="text-sm text-gray-400 mb-8 leading-relaxed">
               RecurLens requires a neural interface key to activate its recursive reasoning engine.
             </p>
             <div className="w-full relative mb-4 group">
               <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
               <input 
                 type="password" 
                 placeholder="Paste Gemini API Key" 
                 className="relative w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-4 text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none placeholder-gray-600 transition-all"
                 onChange={(e) => setApiKey(e.target.value)}
                 value={apiKey}
               />
             </div>
             <button 
               onClick={() => { if(apiKey.length > 10) setShowKeyModal(false); }}
               disabled={apiKey.length < 10}
               className={`w-full py-4 rounded-lg font-bold text-sm tracking-wide transition-all shadow-lg 
                 ${apiKey.length > 10 
                   ? 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-indigo-500/25 hover:shadow-indigo-500/40 transform hover:-translate-y-0.5' 
                   : 'bg-gray-800 text-gray-600 cursor-not-allowed'}`}
             >
               INITIALIZE SYSTEM
             </button>
             <a href="https://aistudio.google.com/" target="_blank" rel="noreferrer" className="mt-6 text-xs text-indigo-400 hover:text-indigo-300 flex items-center transition-colors">
               Generate Key <ArrowRight className="w-3 h-3 ml-1"/>
             </a>
           </div>
        </div>
      )}

      {showCode && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 backdrop-blur-md p-4 animate-in zoom-in-95 duration-200">
          <div className="glass-panel border border-gray-700/50 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b border-white/5 bg-white/5">
               <h3 className="text-white font-medium flex items-center font-mono text-sm">
                 <Code className="w-4 h-4 mr-2 text-emerald-400"/> Python Scaffold
               </h3>
               <button onClick={() => setShowCode(false)} className="text-gray-400 hover:text-white transition-colors"><X className="w-5 h-5" /></button>
            </div>
            <div className="flex-1 overflow-auto bg-[#0d1117] p-6">
              <pre className="text-xs text-emerald-300 font-mono leading-relaxed">
                {getPythonCode(inputText)}
              </pre>
            </div>
            <div className="p-4 border-t border-white/5 flex justify-end space-x-3 bg-gray-900/50">
               <button onClick={() => { navigator.clipboard.writeText(getPythonCode(inputText)); }} className="glass-button px-4 py-2 rounded-lg text-xs text-white flex items-center space-x-2">
                 <Copy className="w-3 h-3"/> <span>Copy</span>
               </button>
            </div>
          </div>
        </div>
      )}

      {/* HEADER */}
      <header className="h-16 border-b border-white/5 bg-gray-900/40 backdrop-blur-md sticky top-0 z-40 flex items-center">
        <div className="max-w-7xl mx-auto w-full px-6 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <div className="relative group">
               <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg blur opacity-40 group-hover:opacity-75 transition duration-500"></div>
               <div className="relative w-10 h-10 bg-gray-900 rounded-lg flex items-center justify-center border border-white/10">
                 <RefreshCw className={`w-5 h-5 text-indigo-400 ${isProcessing ? 'animate-spin' : ''}`} />
               </div>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center">
                RecurLens <span className="text-[10px] ml-2 px-1.5 py-0.5 rounded border border-indigo-500/30 text-indigo-400 bg-indigo-500/10 font-mono">Built with ❤️ by Aryan! </span>
              </h1>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
             <div className="hidden md:flex items-center px-3 py-1.5 rounded-full bg-black/20 border border-white/5 backdrop-blur-sm">
                <div className={`w-2 h-2 rounded-full mr-2 ${isVisualizing ? 'bg-pink-400 animate-pulse' : isThinking ? 'bg-amber-400 animate-pulse' : isProcessing ? 'bg-indigo-500 animate-pulse' : 'bg-emerald-500'}`}></div>
                <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider">
                  {isWaitingForClarification ? 'AWAITING USER' : isVisualizing ? 'VISUALIZING' : isThinking ? 'DEEP THINKING' : isProcessing ? 'PROCESSING' : 'IDLE'}
                </span>
             </div>
             <button onClick={() => setShowDebug(!showDebug)} className={`p-2 transition-colors ${showDebug ? 'text-amber-400' : 'text-gray-400 hover:text-white'}`} title="Toggle Debugger"><Bug className="w-5 h-5" /></button>
             <button onClick={() => setShowCode(true)} className="p-2 text-gray-400 hover:text-white transition-colors" title="Export Code"><FileJson className="w-5 h-5" /></button>
             <button onClick={resetSystem} className="p-2 text-gray-400 hover:text-red-400 transition-colors" title="Reset"><Trash2 className="w-5 h-5" /></button>
          </div>
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <main className="flex-1 max-w-[1600px] mx-auto w-full p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
        
        {/* LEFT COLUMN: INPUTS */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          
          {/* Vision Input */}
          <div className="glass-panel rounded-2xl p-1 relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="p-4">
              <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center">
                <Eye className="w-3 h-3 mr-2 text-indigo-400" /> Vision Stream
              </h3>
              <div className="relative w-full aspect-video rounded-xl border-2 border-dashed border-gray-700/50 hover:border-indigo-500/50 transition-colors bg-black/20 flex flex-col items-center justify-center overflow-hidden cursor-pointer group/upload">
                <input type="file" accept="image/*" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer z-20" />
                {selectedImage ? (
                  <>
                    <img src={selectedImage} alt="Input" className="w-full h-full object-cover transition-transform duration-700 group-hover/upload:scale-105" />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/upload:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                      <span className="text-xs font-medium text-white backdrop-blur px-2 py-1 rounded">Change Image</span>
                    </div>
                  </>
                ) : (
                  <div className="text-center p-4 transition-transform duration-300 group-hover/upload:-translate-y-1">
                    <div className="w-10 h-10 rounded-full bg-gray-800/50 flex items-center justify-center mx-auto mb-2 group-hover/upload:bg-indigo-500/20 group-hover/upload:text-indigo-400 transition-colors">
                      <Layers className="w-5 h-5 text-gray-500" />
                    </div>
                    <span className="text-xs text-gray-400">Drop image or click</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Text/Audio Input */}
          <div className={`glass-panel rounded-2xl p-1 relative transition-all duration-300 ${isWaitingForClarification ? 'ring-1 ring-amber-500/50 shadow-[0_0_30px_-5px_rgba(245,158,11,0.2)]' : ''}`}>
            {isWaitingForClarification && <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-amber-500 to-orange-500 animate-pulse"></div>}
            
            <div className="p-4">
              <h3 className={`text-xs font-bold uppercase tracking-widest mb-4 flex items-center ${isWaitingForClarification ? 'text-amber-400' : 'text-gray-400'}`}>
                {isWaitingForClarification ? <HelpCircle className="w-3 h-3 mr-2 animate-bounce" /> : <TerminalIcon className="w-3 h-3 mr-2 text-purple-400" />}
                {isWaitingForClarification ? "Clarification Required" : "System Input"}
              </h3>

              {isWaitingForClarification && (
                 <div className="mb-4 bg-amber-950/30 border border-amber-500/20 p-3 rounded-lg text-xs text-amber-200 leading-relaxed shadow-inner">
                   <span className="font-bold text-amber-400 block mb-1">SYSTEM QUERY:</span>
                   "{clarificationQuestion}"
                 </div>
              )}

              <div className="relative">
                <textarea
                  className="w-full bg-black/20 border border-white/5 rounded-xl p-4 text-sm text-gray-200 focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500/50 focus:outline-none resize-none h-40 placeholder-gray-600 transition-all font-mono"
                  placeholder={isWaitingForClarification ? "Enter clarification..." : "Describe query or task..."}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                />
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`absolute bottom-3 right-3 p-2.5 rounded-lg transition-all backdrop-blur-sm border ${
                    isRecording 
                      ? 'bg-red-500/20 border-red-500/50 text-red-400 animate-pulse' 
                      : 'bg-gray-800/50 border-white/5 text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  {isRecording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                </button>
              </div>

              <button
                onClick={isWaitingForClarification ? handleClarificationSubmit : startInitialRun}
                disabled={isProcessing}
                className={`mt-4 w-full py-3 rounded-xl font-medium text-sm transition-all flex items-center justify-center space-x-2 shadow-lg
                  ${isProcessing 
                    ? 'bg-gray-800/50 text-gray-500 cursor-not-allowed border border-white/5' 
                    : isWaitingForClarification 
                      ? 'bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white shadow-amber-500/20'
                      : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-indigo-500/25 hover:shadow-indigo-500/40 transform hover:-translate-y-0.5'}`}
              >
                {isProcessing ? (
                  <><RefreshCw className="w-4 h-4 animate-spin" /> <span>Processing...</span></>
                ) : isWaitingForClarification ? (
                  <><ArrowRight className="w-4 h-4" /> <span>Submit Answer</span></>
                ) : (
                  <><Zap className="w-4 h-4" /> <span>Initialize Loop</span></>
                )}
              </button>
            </div>
          </div>
          
          {/* Quick Actions */}
          {!isProcessing && !isWaitingForClarification && (
            <div className="grid grid-cols-1 gap-2">
              {SCENARIOS.map(s => (
                <button 
                  key={s.id}
                  onClick={() => { setInputText(s.text); setDetectedTone({ transcript: s.text, tone: "Neutral", urgency: "LOW" }); }}
                  className="group flex items-center p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 hover:border-white/10 transition-all text-left"
                >
                  <div className="w-8 h-8 rounded-lg bg-black/40 flex items-center justify-center mr-3 group-hover:scale-110 transition-transform">
                    {s.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-gray-300 group-hover:text-white">{s.name}</div>
                  </div>
                  <ArrowRight className="w-3 h-3 text-gray-600 group-hover:text-gray-400 opacity-0 group-hover:opacity-100 transition-all -translate-x-2 group-hover:translate-x-0" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* MIDDLE COLUMN: LOGS */}
        <div className="lg:col-span-5 flex flex-col h-[80vh] lg:h-auto">
          <div className="glass-panel rounded-t-2xl border-b-0 p-4 flex justify-between items-center bg-gray-900/80">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center">
              <Activity className="w-3 h-3 mr-2 text-emerald-400" /> Recursion Trace
            </h3>
            {logs.length > 0 && <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/20 animate-pulse">LIVE</span>}
          </div>
          <div className="flex-1 glass-panel rounded-b-2xl border-t-0 p-0 overflow-hidden relative bg-black/20">
             <div className="absolute inset-0 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {logs.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-gray-700 opacity-40">
                    <div className="w-16 h-16 rounded-full border-2 border-gray-800 flex items-center justify-center mb-4">
                      <Cpu className="w-8 h-8" />
                    </div>
                    <p className="text-xs font-mono uppercase tracking-widest">Awaiting Input Stream</p>
                  </div>
                )}
                
                {logs.map((log, idx) => {
                  if (log.type === 'debug' && !showDebug) return null;

                  return (
                  <div key={idx} className={`animate-slide-up relative pl-4 border-l-2 ${
                    log.type === 'system' ? 'border-gray-700' :
                    log.type === 'vision' ? 'border-indigo-500' :
                    log.type === 'critic' ? 'border-rose-500' :
                    log.type === 'refiner' ? 'border-purple-500' :
                    log.type === 'success' ? 'border-emerald-500' :
                    log.type === 'debug' ? 'border-amber-500/50' :
                    'border-gray-500'
                  }`}>
                    {/* Timestamp Dot */}
                    <div className={`absolute -left-[5px] top-0 w-2 h-2 rounded-full ${
                       log.type === 'system' ? 'bg-gray-700' :
                       log.type === 'vision' ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]' :
                       log.type === 'critic' ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]' :
                       log.type === 'refiner' ? 'bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.5)]' :
                       log.type === 'success' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' :
                       log.type === 'debug' ? 'bg-amber-500' :
                       'bg-gray-500'
                    }`}></div>

                    <div className="flex justify-between items-baseline mb-1">
                      <span className={`text-[10px] font-bold uppercase tracking-wider ${
                        log.type === 'vision' ? 'text-indigo-400' :
                        log.type === 'critic' ? 'text-rose-400' :
                        log.type === 'refiner' ? 'text-purple-400' :
                        log.type === 'success' ? 'text-emerald-400' :
                        log.type === 'debug' ? 'text-amber-400' :
                        'text-gray-500'
                      }`}>{log.type}</span>
                      <span className="text-[9px] font-mono text-gray-600">{new Date(log.timestamp).toLocaleTimeString([], {minute:'2-digit', second:'2-digit'})}</span>
                    </div>

                    <div className="text-xs text-gray-300 font-mono leading-relaxed">
                       {log.type === 'debug' ? (
                          <div className="bg-amber-950/20 p-2 rounded border border-amber-500/10 text-[10px] text-amber-200/70 overflow-x-auto whitespace-pre-wrap font-mono">
                             {log.content}
                          </div>
                       ) : log.type === 'system' || log.type === 'error' || log.type === 'success' ? (
                          <p>{log.content}</p>
                       ) : log.type === 'vision' ? (
                          <div className="bg-indigo-950/20 p-2 rounded border border-indigo-500/10">
                            <div className="text-indigo-200 mb-1">"{log.content?.scene_summary || 'No summary'}"</div>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {log.content?.regions?.map((r: VisionRegion, i: number) => (
                                <span key={i} className="text-[9px] bg-indigo-500/20 text-indigo-300 px-1 rounded flex items-center" title={r.description}>
                                  <Scan className="w-2 h-2 mr-1"/> {r.id}
                                </span>
                              ))}
                              {log.content?.objects?.map((obj: VisionObject, i: number) => (
                                <span key={i} className="text-[9px] border border-indigo-500/20 text-indigo-400 px-1 rounded">{obj.name}</span>
                              ))}
                            </div>
                          </div>
                       ) : log.type === 'critic' ? (
                          <div className="bg-rose-950/20 p-2 rounded border border-rose-500/10">
                            <div className="text-rose-200 italic mb-2">"{log.content?.critique_text || 'No critique text'}"</div>
                            {renderCriticScore(log.content?.scores)}
                            {log.content?.needs_user_clarification && (
                               <div className="mt-2 text-[10px] bg-rose-500/20 text-rose-300 inline-block px-2 py-0.5 rounded border border-rose-500/30">Clarification Needed</div>
                            )}
                          </div>
                       ) : (
                          <div className="bg-purple-950/20 p-2 rounded border border-purple-500/10 group relative">
                             <div className="text-purple-200 line-clamp-3 group-hover:line-clamp-none transition-all cursor-default">
                               {log.content?.draft_prompt || 'No draft generated'}
                             </div>
                             <div className="mt-2 flex gap-2 flex-wrap">
                               {log.content?.risk_analysis?.level === "HIGH" && <span className="text-[9px] bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded border border-red-500/30 flex items-center"><AlertTriangle className="w-3 h-3 mr-1"/> High Risk</span>}
                               {log.content?.refinement_strategy && <span className="text-[9px] bg-purple-500/30 text-purple-200 px-1.5 py-0.5 rounded border border-purple-500/30 font-bold">{log.content.refinement_strategy}</span>}
                               {log.content?.user_sentiment && <span className="text-[9px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded">Tone: {log.content.user_sentiment.tone}</span>}
                             </div>
                          </div>
                       )}
                    </div>
                  </div>
                )})}
                <div ref={logsEndRef} />
             </div>
          </div>
        </div>

        {/* RIGHT COLUMN: OUTPUT */}
        <div className="lg:col-span-4 flex flex-col gap-6 h-[80vh] lg:h-auto">
          
          {/* Final Output Card */}
          <div className={`glass-panel rounded-2xl flex flex-col min-h-[300px] transition-all duration-500 ${isVisualizing ? 'ring-1 ring-pink-500 shadow-[0_0_40px_-10px_rgba(236,72,153,0.3)]' : isThinking ? 'ring-1 ring-indigo-500 shadow-[0_0_40px_-10px_rgba(99,102,241,0.3)]' : ''}`}>
             <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5 rounded-t-2xl">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center">
                   {isVisualizing ? <Sparkles className="w-3 h-3 mr-2 animate-pulse text-pink-400"/> : isThinking ? <Loader2 className="w-3 h-3 mr-2 animate-spin text-indigo-400"/> : <Sparkles className="w-3 h-3 mr-2 text-yellow-400"/>}
                   {isVisualizing ? <span className="text-pink-400 animate-pulse">Generating Visual...</span> : isThinking ? <span className="text-indigo-400 animate-pulse">Synthesizing...</span> : "Final Output"}
                </h3>
                <div className="flex space-x-1">
                   {finalOutput && (
                     <>
                        <button onClick={downloadResult} className="p-1.5 hover:bg-white/10 rounded-md transition-colors text-gray-400 hover:text-white"><FileText className="w-4 h-4" /></button>
                        <button onClick={() => playTTS(finalOutput || '')} disabled={isSpeaking} className={`p-1.5 hover:bg-white/10 rounded-md transition-colors ${isSpeaking ? 'text-green-400' : 'text-gray-400 hover:text-white'}`}><Volume2 className="w-4 h-4" /></button>
                     </>
                   )}
                </div>
             </div>
             
             <div className="flex-1 p-5 overflow-y-auto relative custom-scrollbar bg-black/20">
                {finalOutput ? (
                  <div className="space-y-6">
                    {/* Visual Generation Output */}
                    {generatedImage && (
                        <div className="relative group rounded-xl overflow-hidden border border-white/10 shadow-lg animate-in fade-in duration-1000">
                             <img src={generatedImage} alt="Generated Visualization" className="w-full h-auto object-cover" />
                             <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                 <button onClick={downloadImage} className="bg-white/10 backdrop-blur text-white px-3 py-2 rounded-lg text-xs font-bold flex items-center hover:bg-white/20 transition-all border border-white/20">
                                     <Download className="w-4 h-4 mr-2" /> Download Visual
                                 </button>
                             </div>
                             <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 rounded text-[10px] text-gray-300 backdrop-blur font-mono flex items-center">
                                 <ImageIcon className="w-3 h-3 mr-1" /> Generated with Gemini
                             </div>
                        </div>
                    )}
                    
                    <div className="prose prose-invert prose-sm max-w-none">
                      {finalOutput.split('\n').map((line, i) => (
                        <p key={i} className="mb-3 text-gray-300 leading-relaxed animate-in fade-in duration-700 slide-in-from-bottom-2" style={{animationDelay: `${i * 50}ms`}}>{line}</p>
                      ))}
                    </div>
                  </div>
                ) : isThinking || isVisualizing ? (
                   <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <div className="relative">
                         <div className={`w-16 h-16 rounded-full border-t-2 border-r-2 ${isVisualizing ? 'border-pink-500' : 'border-indigo-500'} animate-spin`}></div>
                         <div className="w-16 h-16 rounded-full border-b-2 border-l-2 border-purple-500 animate-spin absolute inset-0 animation-delay-2000 reverse"></div>
                         <div className="absolute inset-0 flex items-center justify-center">
                            <Brain className={`w-6 h-6 ${isVisualizing ? 'text-pink-400' : 'text-indigo-400'} animate-pulse`} />
                         </div>
                      </div>
                      <span className={`mt-4 text-xs font-mono ${isVisualizing ? 'text-pink-400' : 'text-indigo-400'} animate-pulse`}>
                          {isVisualizing ? "Rendering Pixels..." : "Optimizing Vectors..."}
                      </span>
                   </div>
                ) : (
                   <div className="h-full flex items-center justify-center text-gray-600 text-xs italic">
                      Waiting for execution cycle...
                   </div>
                )}
             </div>

             {groundingUrls.length > 0 && (
               <div className="p-3 bg-black/40 border-t border-white/5 rounded-b-2xl">
                 <div className="flex items-center text-[10px] text-gray-500 mb-2 uppercase tracking-wider font-bold"><Search className="w-3 h-3 mr-1" /> Citations</div>
                 <div className="flex flex-wrap gap-2">
                   {groundingUrls.map((url, i) => (
                     <a key={i} href={url.uri} target="_blank" rel="noreferrer" className="text-[10px] bg-gray-800 hover:bg-gray-700 text-blue-400 px-2 py-1 rounded border border-gray-700 hover:border-blue-500/50 transition-colors truncate max-w-[200px] flex items-center">
                       <Globe className="w-3 h-3 mr-1 opacity-50"/> {url.title}
                     </a>
                   ))}
                 </div>
               </div>
             )}
          </div>

          {/* State Viewer */}
          <div className="glass-panel rounded-2xl flex-1 flex flex-col min-h-[200px] overflow-hidden">
             <div className="p-3 border-b border-white/5 bg-white/5 flex justify-between items-center">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center">
                   <GitCommit className="w-3 h-3 mr-2" /> System State
                </h3>
                <div className="flex bg-black/40 rounded-lg p-0.5 border border-white/5">
                   <button onClick={() => setViewMode('state')} className={`px-3 py-1 rounded-md text-[10px] font-medium transition-all ${viewMode === 'state' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>Current</button>
                   <button onClick={() => setViewMode('evolution')} className={`px-3 py-1 rounded-md text-[10px] font-medium transition-all ${viewMode === 'evolution' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}>History</button>
                </div>
             </div>
             <div className="flex-1 bg-[#0d1117] overflow-y-auto p-4 custom-scrollbar relative">
                <div className="absolute inset-0 p-4">
                  {viewMode === 'state' ? (
                    currentState ? (
                      <div className="space-y-4">
                        <div>
                          <div className="text-[10px] font-bold text-gray-500 mb-1 flex items-center"><Network className="w-3 h-3 mr-1"/> PLAN GRAPH (DAG)</div>
                          <div className="space-y-1">
                             {currentState.plan_of_attack.map((step, i) => (
                               <div key={i} className="flex items-start text-[10px] font-mono text-gray-300 bg-gray-900/50 p-1.5 rounded border border-white/5">
                                 <span className="text-indigo-400 w-6 shrink-0">{step.step_id}</span>
                                 <div className="flex-1">{step.action}</div>
                                 <span className="text-gray-600 text-[9px] ml-2">{step.logic_type}</span>
                               </div>
                             ))}
                          </div>
                        </div>
                        <div>
                           <div className="text-[10px] font-bold text-gray-500 mb-1 flex items-center"><Anchor className="w-3 h-3 mr-1"/> CONSTRAINT LATTICE</div>
                           <div className="flex flex-wrap gap-1">
                             {currentState.constraints.map((c, i) => (
                               <div key={i} className={`text-[9px] px-2 py-1 rounded border flex items-center ${c.type === 'HARD' ? 'bg-red-950/30 border-red-500/30 text-red-300' : 'bg-blue-950/30 border-blue-500/30 text-blue-300'}`}>
                                  <span className="font-bold mr-1">{c.id}:</span> {c.description}
                               </div>
                             ))}
                           </div>
                        </div>
                        <div>
                           <div className="text-[10px] font-bold text-gray-500 mb-1">RAW JSON</div>
                           <pre className="text-[9px] font-mono text-emerald-500/70 whitespace-pre-wrap break-all leading-relaxed opacity-50">
                             {JSON.stringify(currentState, null, 2)}
                           </pre>
                        </div>
                      </div>
                    ) : <div className="text-gray-700 text-xs font-mono mt-4 text-center">System Idle</div>
                  ) : (
                    <div className="space-y-6">
                      {recursionHistory.map((h, i) => (
                         <div key={i} className="relative pl-4 border-l border-gray-800">
                            <div className="absolute -left-[3px] top-0 w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-[10px] text-gray-500 font-mono">Iteration {h.metadata.iteration}</span>
                                <span className="text-[9px] bg-indigo-500/20 text-indigo-300 px-1 rounded">{h.refinement_strategy}</span>
                            </div>
                            <div className="bg-gray-900/50 p-2 rounded border border-white/5 text-[10px] text-gray-400 font-mono">
                               {h.draft_prompt}
                            </div>
                         </div>
                      ))}
                      {recursionHistory.length === 0 && <div className="text-gray-700 text-xs font-mono mt-4 text-center">No history available</div>}
                    </div>
                  )}
                </div>
             </div>
          </div>

        </div>

      </main>
    </div>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<RecurLensApp />);
}