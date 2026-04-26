'use client';
import { useState, useEffect } from 'react';

export default function SpotlightFinal() {
  const [view, setView] = useState('pipeline');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showData, setShowData] = useState(false);
  const [target, setTarget] = useState('');

  const scrollToApp = () => {
    document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' });
  };

  const runAnalysis = () => {
    setIsProcessing(true);
    setShowData(false);
    setTimeout(() => {
      setIsProcessing(false);
      setShowData(true);
    }, 2500);
  };

  const cokeData = {
    exposure: "7.4s", prominence: "Maximum", value: "$112,000", visibility: "94%",
    logs: ["Target: Coke Can", "Subject: Eleven", "Status: Solo Hero Shot", "Temporal Stability: 99.5%"]
  };

  const kfcData = {
    exposure: "11.2s", prominence: "High", value: "$84,000", visibility: "85%",
    logs: ["Target: KFC Bucket", "Cuts Detected: 3", "Context: Family Dinner", "Temporal Stability: 98.2%"]
  };

  const activeData = target.toLowerCase().includes('coke') ? cokeData : kfcData;

  return (
    <div className="bg-[#050505] text-slate-200 scroll-smooth selection:bg-yellow-500/30">
      
      {/* SECTION 1: HERO LANDING */}
      <section className="h-screen flex flex-col items-center justify-center relative overflow-hidden px-6">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-yellow-500/10 via-transparent to-transparent opacity-50 animate-pulse"></div>
        <div className="z-10 text-center">
          <div className="inline-block px-4 py-1 mb-6 border border-yellow-500/30 rounded-full bg-yellow-500/5 text-yellow-500 text-[10px] font-black tracking-[0.3em] uppercase">
            Computer Vision for Shoppable Media
          </div>
          <h1 className="text-8xl md:text-[10rem] font-black tracking-tighter text-white leading-none mb-6">
            SPOT<span className="text-yellow-500 italic">LIGHT.</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-400 mb-10 max-w-2xl mx-auto font-light tracking-tight">
            We quantify the unquantifiable. Precise temporal masking for <span className="text-white">brand placement intelligence</span>.
          </p>
          <button onClick={scrollToApp} className="px-12 py-5 bg-yellow-500 text-black rounded-full font-black text-xs uppercase tracking-[0.2em] hover:bg-white transition-all hover:scale-105 shadow-[0_0_50px_rgba(234,179,8,0.2)]">
            Open Pipeline
          </button>
        </div>
      </section>

      {/* SECTION 2: WORKSPACE & ANALYTICS */}
      <section id="workspace" className="min-h-screen bg-[#0a0a0a] border-t border-white/5 flex flex-col">
        {/* NAV HUD */}
        <nav className="flex justify-between items-center px-12 py-8 sticky top-0 bg-black/80 backdrop-blur-md z-40 border-b border-white/5">
          <span className="text-xl font-black text-white">SPOT<span className="text-yellow-500 italic font-light">LIGHT</span></span>
          <div className="flex gap-2">
            <button onClick={() => setView('pipeline')} className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest rounded-full transition-all ${view === 'pipeline' ? 'bg-yellow-500 text-black' : 'text-slate-500 bg-white/5'}`}>Extraction</button>
            <button onClick={() => setView('analytics')} className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest rounded-full transition-all ${view === 'analytics' ? 'bg-yellow-500 text-black' : 'text-slate-500 bg-white/5'}`}>Analytical View</button>
          </div>
        </nav>

        <div className="p-12 max-w-[1600px] mx-auto w-full flex-grow">
          {view === 'pipeline' ? (
            <div className="grid grid-cols-12 gap-8 animate-in fade-in duration-500">
              <div className="col-span-12 lg:col-span-8 space-y-6">
                <div className="aspect-video bg-black rounded-[2.5rem] border border-white/5 relative overflow-hidden shadow-2xl flex items-center justify-center">
                  {isProcessing ? (
                    <div className="text-center">
                      <div className="w-12 h-12 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                      <p className="text-[10px] font-black text-yellow-500 tracking-[0.4em] uppercase">Neural Mapping Active</p>
                    </div>
                  ) : (
                    <p className="text-[10px] font-black text-slate-700 tracking-[0.4em] uppercase">Ready for Feed Ingest</p>
                  )}
                </div>
                <div className="grid grid-cols-4 gap-4">
                  {['Temporal Stability', 'Confidence', 'Mask Density', 'Scene Logic'].map((s, i) => (
                    <div key={i} className="bg-[#111] p-4 border-t border-white/5 text-center rounded-2xl">
                      <p className="text-[8px] text-slate-500 uppercase font-black mb-1">{s}</p>
                      <p className="text-sm font-bold text-white tracking-widest">{showData ? "NOMINAL" : "---"}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="col-span-12 lg:col-span-4 space-y-6">
                <div className="bg-[#111] p-8 rounded-[2.5rem] border border-white/5">
                  <h3 className="text-[10px] font-black text-yellow-500 uppercase tracking-widest mb-6">Target Identifier</h3>
                  <input 
                    type="text" 
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder="e.g. KFC Bucket or Coke Can" 
                    className="w-full bg-black border border-white/10 rounded-2xl p-4 mb-4 outline-none focus:border-yellow-500 transition-all text-xs"
                  />
                  <button onClick={runAnalysis} className="w-full bg-yellow-500 text-black py-4 rounded-2xl font-black uppercase text-[10px] tracking-widest hover:bg-white transition-all">
                    Start Neural Scan
                  </button>
                </div>
                <div className="bg-black rounded-[2.5rem] p-6 border border-white/5 h-64 font-mono text-[10px] text-yellow-500/60 leading-relaxed overflow-y-auto">
                   {showData ? activeData.logs.map((l, i) => <div key={i}>{`> ${l}`}</div>) : <div className="opacity-20 italic">Awaiting neural initialization...</div>}
                </div>
              </div>
            </div>
          ) : (
            <div className="animate-in zoom-in-95 duration-500">
              <div className="grid grid-cols-4 gap-6 mb-8">
                {[
                  { l: "Exposure Value", v: showData ? activeData.value : "$0", s: "Market Equivalent" },
                  { l: "On-Screen Duration", v: showData ? activeData.exposure : "0s", s: "Aggregate Time" },
                  { l: "Visual Prominence", v: showData ? activeData.prominence : "N/A", s: "Subjective Score" },
                  { l: "Platform Visibility", v: showData ? activeData.visibility : "0%", s: "Temporal Sync" }
                ].map((c, i) => (
                  <div key={i} className="bg-[#111] p-8 rounded-[2.5rem] border border-white/5">
                    <p className="text-[9px] text-slate-500 font-black uppercase tracking-widest mb-2">{c.l}</p>
                    <p className="text-4xl font-light text-white tracking-tighter mb-1">{c.v}</p>
                    <p className="text-[9px] text-yellow-500 font-bold uppercase">{c.s}</p>
                  </div>
                ))}
              </div>
              <div className="bg-[#111] p-12 rounded-[3rem] border border-white/5 min-h-[400px] flex items-center justify-center text-center">
                 <div>
                    <h4 className="text-xs font-black uppercase tracking-[0.5em] text-slate-600 mb-8 italic">Revenue Attribution Matrix</h4>
                    <div className="flex items-end gap-3 h-48">
                       {[40, 70, 50, 90, 100, 60, 80, 40, 95, 30].map((h, i) => (
                         <div key={i} className="w-8 bg-white/5 rounded-t-sm relative group">
                            <div className="absolute bottom-0 w-full bg-yellow-500 transition-all duration-1000" style={{height: showData ? `${h}%` : '0%'}}></div>
                         </div>
                       ))}
                    </div>
                    <p className="mt-8 text-[10px] text-slate-500 font-bold uppercase tracking-widest">Temporal Attention Delta (Peak: Frame 112)</p>
                 </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
