'use client';
import { useState, useEffect } from 'react';

export default function InteractiveVisionMark() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [confidence, setConfidence] = useState(0);

  // Simulate interactive "Live" tracking data
  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        setConfidence(Math.floor(Math.random() * (99 - 92 + 1) + 92));
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  const runPipeline = () => {
    setIsProcessing(true);
    setProgress(0);
    setLogs(["Initializing pipeline..."]);

    const steps = ["Identifying Object...", "Mapping Temporal Vectors...", "SAM 2 Propagation...", "Finalizing Mask..."];

    steps.forEach((step, i) => {
      setTimeout(() => {
        setLogs(prev => [...prev, `[SYSTEM] ${step}`]);
        setProgress((prev) => prev + 25);
        if (i === steps.length - 1) setIsProcessing(false);
      }, (i + 1) * 1500);
    });
  };

  return (
    <main className="min-h-screen bg-[#0a0a0c] text-slate-200 p-6 font-sans selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6">

        {/* --- LEFT: CONTROL PANEL --- */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <div className="bg-[#141417] border border-white/5 p-6 rounded-3xl shadow-2xl">
            <div className="flex items-center gap-2 mb-8">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-ping"></div>
              <h2 className="font-bold text-sm uppercase tracking-widest text-blue-500">Input Engine</h2>
            </div>

            <div className="space-y-4">
              <div className="group">
                <label className="text-[10px] font-bold text-slate-500 uppercase mb-2 block group-focus-within:text-blue-400 transition-colors">Target Object</label>
                <input
                  type="text"
                  placeholder="e.g. Orange Car"
                  className="w-full bg-black/40 border border-white/10 rounded-2xl p-4 outline-none focus:border-blue-500/50 focus:ring-4 focus:ring-blue-500/10 transition-all text-sm"
                />
              </div>

              <button
                onClick={runPipeline}
                disabled={isProcessing}
                className="w-full relative group overflow-hidden bg-blue-600 py-4 rounded-2xl font-black text-white transition-all active:scale-95 disabled:opacity-50"
              >
                <span className="relative z-10">{isProcessing ? 'ANALYZING...' : 'START EXTRACTION'}</span>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-blue-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              </button>
            </div>
          </div>

          {/* REAL-TIME LOGS */}
          <div className="bg-black rounded-3xl p-5 border border-white/5 h-64 flex flex-col shadow-inner">
            <div className="flex justify-between items-center mb-4">
              <span className="text-[10px] font-mono text-slate-500">TERMINAL_FEED</span>
              {isProcessing && <span className="text-[10px] font-mono text-green-500 animate-pulse">● LIVE</span>}
            </div>
            <div className="font-mono text-[11px] space-y-2 overflow-y-auto scrollbar-hide">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2 text-green-400/80">
                  <span className="text-slate-700">[{new Date().toLocaleTimeString([], { hour12: false })}]</span>
                  <span>{log}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* --- RIGHT: INTERACTIVE VIEWER --- */}
        <div className="col-span-12 lg:col-span-8 bg-[#141417] border border-white/5 rounded-[2rem] overflow-hidden flex flex-col">

          {/* Top Video Area */}
          <div className="relative flex-grow bg-black group cursor-crosshair">
            {/* Visualizer Overlay */}
            <div className="absolute inset-0 z-10 opacity-30 pointer-events-none"
              style={{ backgroundImage: 'radial-gradient(circle, #3b82f6 1px, transparent 1px)', backgroundSize: '30px 30px' }}></div>

            <div className="absolute inset-0 flex items-center justify-center">
              {/* Replace with your video tags here */}
              <div className="text-slate-800 font-black text-6xl italic select-none uppercase tracking-tighter opacity-20">
                {isProcessing ? 'Processing Stream...' : 'Ready for Feed'}
              </div>
            </div>

            {/* Interactive Confidence Badge */}
            {isProcessing && (
              <div className="absolute top-6 right-6 z-20 bg-black/80 backdrop-blur-md border border-white/10 p-4 rounded-2xl shadow-2xl animate-in zoom-in-95">
                <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">AI Confidence</p>
                <p className="text-3xl font-mono font-black text-blue-500">{confidence}%</p>
              </div>
            )}
          </div>

          {/* Bottom Controls Area */}
          <div className="p-8 bg-[#18181b] border-t border-white/5">
            <div className="flex items-center gap-6">
              <div className="flex-grow">
                <div className="flex justify-between mb-2">
                  <span className="text-[10px] font-bold text-slate-500 uppercase">Temporal Progress</span>
                  <span className="text-[10px] font-bold text-blue-500 uppercase">{progress}%</span>
                </div>
                <div className="w-full bg-black rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-blue-500 h-full transition-all duration-700 ease-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex gap-2">
                <button className="p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors">
                  <div className="w-4 h-4 border-2 border-slate-400 rounded-sm"></div>
                </button>
                <button className="p-3 bg-blue-500/20 text-blue-400 rounded-xl font-bold text-xs px-6 border border-blue-500/20">
                  EXPORT MASK (JSON)
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}