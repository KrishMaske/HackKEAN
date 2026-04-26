'use client';
import { useState } from 'react';

export default function SpotlightExecutive() {
  const [view, setView] = useState('pipeline'); // pipeline or analytics
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const runPipeline = () => {
    setIsProcessing(true);
    setProgress(0);
    setTimeout(() => { setProgress(100); setIsProcessing(false); }, 2500);
  };

  return (
    <div className="bg-[#050505] text-slate-200 min-h-screen font-sans selection:bg-yellow-500/30">
      
      {/* GLOBAL HUD */}
      <nav className="flex justify-between items-center px-12 py-6 border-b border-white/5 sticky top-0 bg-black/50 backdrop-blur-xl z-50">
        <h1 className="text-2xl font-black tracking-tighter text-white">SPOT<span className="text-yellow-500 italic">LIGHT</span></h1>
        <div className="flex gap-4">
          <button 
            onClick={() => setView('pipeline')}
            className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest transition-all ${view === 'pipeline' ? 'bg-yellow-500 text-black' : 'text-slate-500 border border-white/10'}`}
          >
            Extraction Pipeline
          </button>
          <button 
            onClick={() => setView('analytics')}
            className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest transition-all ${view === 'analytics' ? 'bg-yellow-500 text-black' : 'text-slate-500 border border-white/10 hover:text-white'}`}
          >
            Analytical View (ROI)
          </button>
        </div>
      </nav>

      <main className="p-12">
        {view === 'pipeline' ? (
          /* PIPELINE VIEW */
          <div className="max-w-[1400px] mx-auto grid grid-cols-12 gap-8 animate-in fade-in duration-500">
            <div className="col-span-12 lg:col-span-8 space-y-6">
              <div className="aspect-video bg-[#111] rounded-sm border border-white/10 flex items-center justify-center relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent opacity-60"></div>
                {isProcessing ? (
                  <div className="text-center z-10">
                    <div className="w-16 h-1 w-48 bg-white/10 mx-auto mb-4 overflow-hidden">
                      <div className="h-full bg-yellow-500 transition-all duration-[2500ms]" style={{width: `${progress}%`}}></div>
                    </div>
                    <p className="text-[10px] font-bold text-yellow-500 tracking-[0.4em] animate-pulse uppercase">Neural Mapping In Progress</p>
                  </div>
                ) : (
                  <div className="text-center z-10 opacity-20">
                    <p className="text-xs font-black uppercase tracking-[0.5em]">Feed Ingest Ready</p>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-3 gap-6">
                {['Detection Confidence: 98.4%', 'Temporal Continuity: Stable', 'Object ID: #KFC_2026'].map((t, i) => (
                  <div key={i} className="bg-[#0f0f0f] p-4 border-l-2 border-yellow-500">
                    <p className="text-[9px] font-bold text-slate-500 uppercase">{t.split(':')[0]}</p>
                    <p className="text-sm font-medium text-white">{t.split(':')[1]}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="col-span-12 lg:col-span-4 bg-[#111] p-8 border border-white/10 flex flex-col">
              <h2 className="text-xs font-black uppercase tracking-[0.2em] mb-8 border-b border-white/5 pb-4">Targeting Engine</h2>
              <div className="space-y-6 flex-grow">
                <div>
                  <label className="text-[9px] text-slate-500 font-black uppercase block mb-2">Primary Descriptor</label>
                  <input type="text" placeholder="KFC Original Bucket" className="w-full bg-black border border-white/10 p-4 text-xs outline-none focus:border-yellow-500 transition-all" />
                </div>
                <button onClick={runPipeline} className="w-full bg-yellow-500 text-black py-5 font-black uppercase text-[11px] tracking-[0.2em] hover:bg-white active:scale-95 transition-all">
                  Generate Masks
                </button>
              </div>
              <p className="mt-8 text-[9px] text-slate-600 leading-relaxed uppercase tracking-tighter italic font-bold">
                Note: All masks are exported in JSON & After Effects compatible formats.
              </p>
            </div>
          </div>
        ) : (
          /* ANALYTICAL VIEW (ROI DASHBOARD) */
          <div className="max-w-[1400px] mx-auto animate-in zoom-in-95 duration-500">
            <header className="mb-12">
              <h2 className="text-4xl font-light text-white mb-2 tracking-tight">Campaign <span className="font-black">Revenue Attribution</span></h2>
              <p className="text-slate-500 text-sm">Real-time performance metrics for "The Bear: S3" Product Placements</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
              {[
                { label: "Total Exposure Value", val: "$142,800", color: "text-white" },
                { label: "Direct Revenue Lift", val: "+18.4%", color: "text-yellow-500" },
                { label: "Unique Brand Views", val: "2.4M", color: "text-white" },
                { label: "Cost Per Active Second", val: "$0.04", color: "text-white" }
              ].map((card, i) => (
                <div key={i} className="bg-[#111] p-8 border border-white/5 shadow-2xl relative overflow-hidden group">
                  <div className="absolute right-0 top-0 w-1 h-full bg-yellow-500 opacity-0 group-hover:opacity-100 transition-all"></div>
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">{card.label}</p>
                  <p className={`text-4xl font-light ${card.color}`}>{card.val}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-12 gap-8">
              {/* Fake Graph Visual */}
              <div className="col-span-12 lg:col-span-8 bg-[#111] p-8 border border-white/5 min-h-[400px]">
                 <div className="flex justify-between items-center mb-8">
                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Attention Retension Over Time</p>
                    <div className="flex gap-4 text-[9px] font-bold">
                      <span className="flex items-center gap-1"><div className="w-2 h-2 bg-yellow-500"></div> PRODUCT</span>
                      <span className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-700"></div> SCENE AVG</span>
                    </div>
                 </div>
                 <div className="w-full h-64 flex items-end gap-2">
                    {[40, 60, 45, 90, 85, 30, 50, 75, 95, 60, 40, 80].map((h, i) => (
                      <div key={i} className="flex-grow bg-white/5 relative group cursor-pointer">
                        <div className="absolute bottom-0 w-full bg-yellow-500/40 group-hover:bg-yellow-500 transition-all" style={{height: `${h}%`}}></div>
                      </div>
                    ))}
                 </div>
                 <p className="mt-8 text-xs text-slate-500 italic text-center">Peak retention correlates with Frame 242-280 (High Saturation Masking)</p>
              </div>

              <div className="col-span-12 lg:col-span-4 space-y-6">
                <div className="bg-yellow-500 p-8 text-black">
                  <h4 className="font-black uppercase text-[10px] tracking-widest mb-2">Executive Summary</h4>
                  <p className="text-sm font-bold leading-tight mb-4 text-black/80">Placement performed 40% above benchmark due to 4.2s of "unobstructed" screen time.</p>
                  <button className="w-full border-2 border-black py-3 font-black uppercase text-[10px] hover:bg-black hover:text-white transition-all">Download PDF Report</button>
                </div>
                <div className="bg-[#111] p-8 border border-white/5">
                   <p className="text-[10px] font-black text-slate-500 uppercase mb-4">Competitor Comparison</p>
                   <div className="space-y-4">
                      <div className="flex justify-between text-xs font-bold"><span>McDonald's</span> <span className="text-slate-500">Low Visibility</span></div>
                      <div className="flex justify-between text-xs font-bold text-yellow-500"><span>Spotlight (KFC)</span> <span>High Prominence</span></div>
                   </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
