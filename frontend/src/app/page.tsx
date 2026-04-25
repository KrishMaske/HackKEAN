"use client";

import { useState, useEffect } from "react";

interface MaskArea {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface SceneResult {
  success: boolean;
  final_selection: string;
  reasoning_log: string[];
  mask_area?: MaskArea;
  error?: string;
}

export default function SceneShiftUI() {
  // --- State Management ---
  const [mounted, setMounted] = useState(false);
  const [userInterest, setUserInterest] = useState("Gym Bro");
  const [sceneId, setSceneId] = useState("stranger_things_83");
  const [guardrails, setGuardrails] = useState(true);
  const [data, setData] = useState<SceneResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState<string>("default");

  // --- Hydration Fix + Thread ID Init ---
  useEffect(() => {
    setMounted(true);
    // Generate or restore a stable session ID so the graph can persist memory
    // across scene changes (Stranger Things → The Office etc.)
    let id = sessionStorage.getItem("sceneshift_thread_id");
    if (!id) {
      id = `user_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      sessionStorage.setItem("sceneshift_thread_id", id);
    }
    setThreadId(id);
  }, []);

  // --- API Handshake ---
  const handleGenerate = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/generate-scene?guardrails=${guardrails}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_interest: userInterest,
          scene_id: sceneId,
          thread_id: threadId,
        }),
      });
      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error("Failed to connect to backend:", err);
    } finally {
      setLoading(false);
    }
  };

  if (!mounted) return null;

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white p-8 font-sans">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* --- Left Column: Control Panel --- */}
        <div className="space-y-6 bg-[#141414] p-6 rounded-xl border border-white/10">
          <h1 className="text-2xl font-bold tracking-tighter text-blue-500">SCENESHIFT v0.1</h1>

          <div className="space-y-4">
            <div>
              <label className="block text-xs uppercase text-gray-500 mb-2">User Interest</label>
              <input
                type="text"
                value={userInterest}
                onChange={(e) => setUserInterest(e.target.value)}
                className="w-full bg-black border border-white/20 p-3 rounded focus:border-blue-500 outline-none"
              />
            </div>

            <div>
              <label className="block text-xs uppercase text-gray-500 mb-2">Target Scene</label>
              <select
                value={sceneId}
                onChange={(e) => setSceneId(e.target.value)}
                className="w-full bg-black border border-white/20 p-3 rounded outline-none"
              >
                <option value="stranger_things_83">Stranger Things (1983)</option>
                <option value="the_office_05">The Office (2005)</option>
                <option value="succession_20">Succession (2020)</option>
              </select>
            </div>

            <div className="flex items-center gap-3 p-3 bg-black/50 rounded border border-white/5">
              <input
                type="checkbox"
                checked={guardrails}
                onChange={(e) => setGuardrails(e.target.checked)}
                className="w-4 h-4 accent-blue-500"
              />
              <span className="text-sm">Enable Historical Guardrails</span>
            </div>

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded transition-all disabled:opacity-50"
            >
              {loading ? "PROCESSING..." : "GENERATE IN-SCENE OBJECT"}
            </button>
          </div>
        </div>

        {/* --- Center/Right Column: Video & Terminal --- */}
        <div className="lg:col-span-2 space-y-6">

          {/* Video Container with Target Zone Overlay */}
          <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-white/10 shadow-2xl">
            {/* The actual video served via your static route */}
            <video
              key={sceneId}
              autoPlay
              loop
              muted
              className="w-full h-full object-cover opacity-60"
            >
              <source src={`http://localhost:8000/input/${sceneId === 'stranger_things_83' ? 'STRANGER_THINGS_CLIP.mp4' : 'OFFICE_CLIP.mp4'}`} type="video/mp4" />
            </video>

            {/* AI Target Zone Overlay */}
            {data?.mask_area && (
              <div
                style={{
                  position: 'absolute',
                  border: '2px dashed #00FF00',
                  left: `${data.mask_area.x}px`,
                  top: `${data.mask_area.y}px`,
                  width: `${data.mask_area.w}px`,
                  height: `${data.mask_area.h}px`,
                  transition: 'all 0.5s ease-in-out'
                }}
                className="bg-green-500/10 flex items-start justify-start p-1"
              >
                <span className="text-[10px] font-mono text-green-400 bg-black/80 px-1">AI_TARGET_ZONE</span>
              </div>
            )}

            {/* Selection Result Overlay */}
            {data?.final_selection && (
              <div className="absolute bottom-6 left-6 bg-blue-600 px-4 py-2 rounded shadow-lg animate-pulse">
                <p className="text-xs uppercase font-bold text-blue-100">Injected Object</p>
                <p className="text-lg font-bold">{data.final_selection}</p>
              </div>
            )}
          </div>

          {/* Reasoning Terminal */}
          <div className="bg-black border border-white/10 rounded-xl p-6 h-64 overflow-y-auto font-mono text-sm">
            <h2 className="text-gray-500 mb-4 border-b border-white/10 pb-2 text-xs uppercase tracking-widest">Agent Reasoning Log</h2>
            {data?.reasoning_log ? (
              data.reasoning_log.map((log, i) => (
                <div key={i} className="mb-2 text-blue-400">
                  <span className="text-gray-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
                  {log}
                </div>
              ))
            ) : (
              <div className="text-gray-700 italic">Waiting for orchestrator signal...</div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}