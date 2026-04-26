'use client';
import { useState, useEffect } from 'react';

export default function SpotlightApp() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        setConfidence(Math.floor(Math.random() * (99 - 94 + 1) + 94));
      }, 400);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  const scrollToApp = () => {
    document.getElementById('app-section')?.scrollIntoView({ behavior: 'smooth' });
  };

  const runPipeline = () => {
    setIsProcessing(true);
    setProgress(0);
    setLogs(["[SYSTEM] Spotlight Neural Engine Active..."]);
    const steps = ["Scanning Frames...", "Isolating Object Geometry...", "Temporal Mask Synthesis...", "Finalizing Analytics..."];
    steps.forEach((step, i) => {
      setTimeout(() => {
        setLogs(prev => [...prev, `[LOG] ${step}`]);
        setProgress((prev) => Math.min(prev + 25, 100));
        if (i === steps.length - 1) setIsProcessing(false);
      }, (i + 1) * 1200);
    });
  };

  return (
    <div className="bg-[#050505] text-slate-200 scroll-smooth selection:bg-yellow-500/30">
      
      {/* SECTION 1: HERO */}
      <section className="h-screen flex flex-col items-center justify-center relative overflow-hidden px-6">
        {/* Spotlight Visual Effect */}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-yellow-500/10 blur-[120px] rounded-full pointer-events-none"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full pointer-events-none"></div>
        
        <div className="z-10 text-center max-w-4xl">
          <div className="inline-block px-4 py-1 mb-6 border border-yellow-500/20 rounded-full bg-yellow-500/5 text-yellow-500 text-[10px] font-bold tracking-[0.2em] uppercase">
            AI-Powered Temporal Masking
          </div>
          <h1 className="text-8xl md:text-9xl font-black tracking-tighter text-white mb-6">
            SPOT<span className="text-yellow-500">LIGHT.</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-400 mb-10 leading-relaxed max-w-xl mx-auto font-light">
            We find the signal in the noise. Automated product tracking and 
            <span className="text-white"> pixel-perfect masks</span> for any video content.
          </p>
          <button 
            onClick={scrollToApp}
            className="px-12 py-5 bg-white text-black hover:bg-yellow-500 rounded-full font-black text-sm uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-[0_0_40px_rgba(255,255,255,0.1)]"
          >
            Enter Workspace
          </button>
        </div>
      </section>

      {/* SECTION 2: WORKSPACE */}
      <section id="app-section" className="min-h-screen p-6 md:p-12 bg-[#0a0a0a] border-t border-white/5">
        <div className="max-w-7xl mx-auto grid grid-cols-12 gap-8">
          
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <div className="bg-[#111111] border border-white/5 p-8 rounded-[2.5rem]">
              <h2 className="text-[10px] font-black text-yellow-500 uppercase tracking-[0.2em] mb-8">Detection Parameters</h2>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-[10px] text-slate-500 uppercase font-bold ml-1">Object Identifier</label>
                  <input type="text" placeholder="e.g. KFC Bucket" className="w-full bg-black border border-white/10 rounded-2xl p-4 outline-none focus:border-yellow-500/50 transition-all text-sm font-medium" />
                </div>
                <button onClick={runPipeline} disabled={isProcessing} className="w-full bg-yellow-500 py-4 rounded-2xl font-black text-black hover:bg-yellow-400 disabled:opacity-50 transition-all uppercase text-xs tracking-widest">
                  {isProcessing ? 'Processing...' : 'Run Extraction'}
                </button>
              </div>
            </div>

            <div className="bg-black rounded-[2.5rem] p-6 border border-white/5 h-64 font-mono text-[10px] overflow-y-auto text-yellow-500/60 leading-loose">
              {logs.map((l, i) => <div key={i} className="animate-in fade-in slide-in-from-left-2">{l}</div>)}
              {isProcessing && <div className="w-1 h-3 bg-yellow-500 animate-pulse inline-block"></div>}
            </div>
          </div>

          <div className="col-span-12 lg:col-span-8 bg-[#111111] border border-white/5 rounded-[3rem] overflow-hidden flex flex-col shadow-2xl">
            <div className="relative flex-grow bg-black flex items-center justify-center">
              {isProcessing && (
                <div className="absolute top-8 right-8 z-20 bg-black/90 backdrop-blur-xl border border-yellow-500/20 p-6 rounded-[2rem] shadow-2xl">
                  <p className="text-[9px] text-slate-500 font-bold uppercase mb-1 tracking-widest">Model Confidence</p>
                  <p className="text-5xl font-mono font-black text-yellow-500 tracking-tighter">{confidence}%</p>
                </div>
              )}
              <div className="flex flex-col items-center gap-4 opacity-10">
                <div className="w-16 h-16 border-4 border-slate-700 rounded-full border-t-yellow-500 animate-spin"></div>
                <div className="text-slate-500 font-bold text-xs uppercase tracking-[0.4em]">Waiting for Stream</div>
              </div>
            </div>
            <div className="p-10 bg-[#0d0d0d] border-t border-white/5">
              <div className="flex justify-between mb-4 text-[9px] font-black text-slate-500 uppercase tracking-[0.3em]">
                <span>Neural Propagation Map</span>
                <span className="text-yellow-500">{progress}%</span>
              </div>
              <div className="w-full bg-black rounded-full h-1.5 overflow-hidden">
                <div className="bg-yellow-500 h-full transition-all duration-700 ease-in-out" style={{width: `${progress}%`}}></div>
              </div>
            </div>
          </div>

        </div>
      </section>
    </div>
  );
}
