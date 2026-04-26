"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

type TimelinePoint = {
  second: number;
  timestamp_ms: number;
  frame_index: number;
  found: boolean;
  confidence: number;
  bbox: [number, number, number, number] | null;
  screen_coverage: number;
  position: string;
  movement_from_previous_px: number;
  status: string;
};

type Analytics = {
  show_id: string;
  product: string;
  video: {
    url: string | null;
    width: number;
    height: number;
    fps: number;
    frame_count: number;
    duration_seconds: number;
  };
  summary: {
    detected_seconds: number;
    sampled_seconds: number;
    visibility_rate: number;
    first_seen_second: number | null;
    last_seen_second: number | null;
    average_screen_coverage: number;
    max_screen_coverage: number;
    average_confidence: number;
    continuous_segments: Array<{ start_second: number; end_second: number; duration_seconds: number }>;
  };
  scene_understanding: {
    headline: string;
    key_moments: Array<{ second: number; type: string; label: string }>;
    interaction_insights: string[];
  };
  marketing: {
    insights: string[];
    optimizations: string[];
  };
  timeline: TimelinePoint[];
  masked_video_url?: string;
};

const API_BASE = "http://localhost:8000";

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

export default function SpotlightDashboard() {
  const [view, setView] = useState('pipeline'); // pipeline or analytics
  const [showId, setShowId] = useState("orange_car");
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSecond, setCurrentSecond] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Upload States
  const [showUpload, setShowUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadShowId, setUploadShowId] = useState("");
  const [uploadTarget, setUploadTarget] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const loadAnalytics = async (id = showId) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/analytics/${id}`);
      const payload = await response.json();
      if (!response.ok) {
        if (response.status === 404) {
          setError("Analytics not ready. Ingestion may still be in progress.");
        } else {
          throw new Error(payload.detail || "Failed to load analytics");
        }
        setAnalytics(null);
        return;
      }
      setAnalytics(payload);
      setError(null);
      setCurrentSecond(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analytics");
      setAnalytics(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile || !uploadShowId) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", uploadFile);
    formData.append("show_id", uploadShowId);
    formData.append("target_object", uploadTarget);

    try {
      const response = await fetch(`${API_BASE}/ingest/video`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Upload failed");
      
      setShowUpload(false);
      setShowId(uploadShowId);
      setLoading(true);
      setError("Analyzing scene... Marketing agents are calculating ROI impact.");
      
      const pollInterval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/analytics/${uploadShowId}`);
          if (res.ok) {
            const data = await res.json();
            setAnalytics(data);
            setLoading(false);
            setError(null);
            clearInterval(pollInterval);
          }
        } catch (e) { }
      }, 5000);

      setTimeout(() => clearInterval(pollInterval), 600000);

    } catch (err) {
      alert("Upload failed: " + (err instanceof Error ? err.message : "Unknown error"));
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const currentPoint = useMemo(() => {
    if (!analytics) return null;
    return analytics.timeline.find((point) => point.second === currentSecond) || analytics.timeline[0] || null;
  }, [analytics, currentSecond]);

  const overlayStyle = useMemo(() => {
    if (!analytics || !currentPoint?.bbox) return null;
    const [x0, y0, x1, y1] = currentPoint.bbox;
    return {
      left: `${(x0 / analytics.video.width) * 100}%`,
      top: `${(y0 / analytics.video.height) * 100}%`,
      width: `${((x1 - x0) / analytics.video.width) * 100}%`,
      height: `${((y1 - y0) / analytics.video.height) * 100}%`,
    };
  }, [analytics, currentPoint]);

  const videoUrl = analytics?.video.url ? `${API_BASE}${analytics.video.url}` : null;
  const maskedVideoUrl = analytics?.show_id ? `${API_BASE}/masks/${analytics.show_id}_mask.mp4` : null;

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
                {videoUrl ? (
                  <video 
                    ref={videoRef} 
                    src={videoUrl} 
                    className="w-full h-full object-contain" 
                    controls 
                    muted 
                    onTimeUpdate={(e) => setCurrentSecond(Math.floor(e.currentTarget.currentTime))}
                  />
                ) : (
                  <div className="text-center z-10 opacity-20">
                    <p className="text-xs font-black uppercase tracking-[0.5em]">{loading ? 'Loading Feed...' : 'Awaiting Video Input'}</p>
                  </div>
                )}
                {overlayStyle && currentPoint?.found && (
                  <div 
                    style={overlayStyle} 
                    className="absolute border-2 border-yellow-500 bg-yellow-500/10 shadow-[0_0_30px_rgba(234,179,8,0.3)] pointer-events-none"
                  >
                    <div className="absolute -top-6 left-0 bg-yellow-500 text-[8px] font-black text-black px-1.5 py-0.5 rounded-sm uppercase tracking-wider">
                      {analytics?.product} | {pct(currentPoint.confidence)}
                    </div>
                  </div>
                )}
                {error && (
                  <div className="absolute inset-0 bg-black/80 flex items-center justify-center p-8 text-center">
                    <p className="text-xs font-bold text-yellow-500 uppercase tracking-widest">{error}</p>
                  </div>
                )}
              </div>
              
              <div className="grid grid-cols-3 gap-6">
                {[
                  { label: 'Exposure Stability', val: analytics ? pct(analytics.summary.visibility_rate) : '0%' },
                  { label: 'Temporal Status', val: loading ? 'Syncing...' : 'Stable' },
                  { label: 'System Logic', val: analytics ? 'Marketing Active' : 'Idle' }
                ].map((t, i) => (
                  <div key={i} className="bg-[#0f0f0f] p-4 border-l-2 border-yellow-500">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">{t.label}</p>
                    <p className="text-sm font-medium text-white">{t.val}</p>
                  </div>
                ))}
              </div>

              {/* Mask Rendering Preview */}
              {analytics && (
                <div className="bg-[#111] p-6 border border-white/5 rounded-sm">
                   <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Neural Mask Preview</h3>
                   <div className="aspect-video bg-black/50 border border-white/10 rounded overflow-hidden">
                      {maskedVideoUrl ? <video src={maskedVideoUrl} controls muted className="w-full h-full object-contain" /> : <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-700 font-bold uppercase tracking-widest">Rendering...</div>}
                   </div>
                </div>
              )}
            </div>

            <div className="col-span-12 lg:col-span-4 bg-[#111] p-8 border border-white/10 flex flex-col">
              <h2 className="text-xs font-black uppercase tracking-[0.2em] mb-8 border-b border-white/5 pb-4">Targeting Engine</h2>
              <div className="space-y-6 flex-grow">
                <div>
                  <label className="text-[9px] text-slate-500 font-black uppercase block mb-2 tracking-widest">Show Identifier</label>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={showId}
                      onChange={(e) => setShowId(e.target.value)}
                      placeholder="e.g. orange_car" 
                      className="flex-grow bg-black border border-white/10 p-4 text-xs outline-none focus:border-yellow-500 transition-all text-white" 
                    />
                    <button onClick={() => loadAnalytics()} className="bg-white/5 border border-white/10 px-4 hover:bg-white/10 transition-all">
                      <span className="text-[10px] font-black uppercase text-white">Load</span>
                    </button>
                  </div>
                </div>
                <div className="pt-4 border-t border-white/5">
                  <button 
                    onClick={() => setShowUpload(true)} 
                    className="w-full bg-yellow-500 text-black py-5 font-black uppercase text-[11px] tracking-[0.2em] hover:bg-white active:scale-95 transition-all shadow-[0_0_20px_rgba(234,179,8,0.2)]"
                  >
                    Ingest New Scene
                  </button>
                </div>
                
                {analytics && (
                  <div className="space-y-4 pt-8">
                     <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Metadata</p>
                     <div className="text-xs text-slate-300 space-y-2">
                        <p><span className="text-yellow-500/50 mr-2">PRODUCT:</span> {analytics.product}</p>
                        <p><span className="text-yellow-500/50 mr-2">DURATION:</span> {analytics.video.duration_seconds}s</p>
                        <p><span className="text-yellow-500/50 mr-2">EXPOSURE:</span> {analytics.summary.detected_seconds}s</p>
                     </div>
                  </div>
                )}
              </div>
              <p className="mt-8 text-[9px] text-slate-600 leading-relaxed uppercase tracking-tighter italic font-bold">
                Note: Neural masks are cached in the Scene Vault for instant retrieval.
              </p>
            </div>
          </div>
        ) : (
          /* ANALYTICAL VIEW (ROI DASHBOARD) */
          <div className="max-w-[1400px] mx-auto animate-in zoom-in-95 duration-500">
            <header className="mb-12">
              <h2 className="text-4xl font-light text-white mb-2 tracking-tight">Campaign <span className="font-black">Revenue Attribution</span></h2>
              <p className="text-slate-500 text-sm">{analytics?.scene_understanding.headline || 'Real-time performance metrics'}</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
              {[
                { label: "Visibility Score", val: analytics ? pct(analytics.summary.visibility_rate) : "0%", color: "text-white" },
                { label: "Active Airtime", val: analytics ? `${analytics.summary.detected_seconds}s` : "0s", color: "text-yellow-500" },
                { label: "Peak Frame Cov.", val: analytics ? pct(analytics.summary.max_screen_coverage) : "0%", color: "text-white" },
                { label: "Confidence", val: analytics ? pct(analytics.summary.average_confidence) : "0%", color: "text-white" }
              ].map((card, i) => (
                <div key={i} className="bg-[#111] p-8 border border-white/5 shadow-2xl relative overflow-hidden group">
                  <div className="absolute right-0 top-0 w-1 h-full bg-yellow-500 opacity-0 group-hover:opacity-100 transition-all"></div>
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">{card.label}</p>
                  <p className={`text-4xl font-light ${card.color}`}>{card.val}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-12 gap-8">
              {/* Engagement Timeline */}
              <div className="col-span-12 lg:col-span-8 bg-[#111] p-8 border border-white/5 min-h-[400px]">
                 <div className="flex justify-between items-center mb-8">
                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Attention Retention Analysis</p>
                    <div className="flex gap-4 text-[9px] font-bold">
                      <span className="flex items-center gap-1"><div className="w-2 h-2 bg-yellow-500"></div> PRODUCT EXPOSURE</span>
                      <span className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-700"></div> NO DETECTION</span>
                    </div>
                 </div>
                 <div className="w-full h-64 flex items-end gap-1.5">
                    {analytics?.timeline.map((point, i) => (
                      <div 
                        key={i} 
                        onClick={() => { setView('pipeline'); if (videoRef.current) videoRef.current.currentTime = point.second; }}
                        className={`flex-grow relative group cursor-pointer transition-all ${point.found ? 'bg-yellow-500/20' : 'bg-white/5'}`}
                      >
                        <div 
                          className="absolute bottom-0 w-full bg-yellow-500 opacity-60 group-hover:opacity-100 transition-all" 
                          style={{height: point.found ? `${point.screen_coverage * 400 + 10}%` : '0%'}}
                        ></div>
                      </div>
                    ))}
                 </div>
                 <p className="mt-8 text-xs text-slate-500 italic text-center">Interactive segments: click any bar to jump to scene telemetry.</p>
              </div>

              <div className="col-span-12 lg:col-span-4 space-y-6">
                <div className="bg-yellow-500 p-8 text-black">
                  <h4 className="font-black uppercase text-[10px] tracking-widest mb-2">Executive Strategy</h4>
                  <div className="space-y-4 mb-8">
                    {analytics?.marketing.optimizations.slice(0, 2).map((opt, i) => (
                      <p key={i} className="text-sm font-bold leading-tight text-black/80 border-b border-black/10 pb-2">
                        {opt}
                      </p>
                    )) || <p className="text-sm font-bold">Awaiting agent analysis...</p>}
                  </div>
                  <button className="w-full border-2 border-black py-3 font-black uppercase text-[10px] hover:bg-black hover:text-white transition-all">Download ROI Report</button>
                </div>
                <div className="bg-[#111] p-8 border border-white/5">
                   <p className="text-[10px] font-black text-slate-500 uppercase mb-4">Strategic Insights</p>
                   <div className="space-y-3">
                      {analytics?.marketing.insights.map((insight, i) => (
                        <div key={i} className="flex gap-3 text-xs font-bold items-start leading-snug">
                          <span className="text-yellow-500">●</span>
                          <span className="text-slate-300">{insight}</span>
                        </div>
                      ))}
                   </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Upload Modal (Spotlight Style) */}
      {showUpload && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/90 backdrop-blur-md animate-in fade-in zoom-in-95 duration-300">
           <div className="relative w-full max-w-lg bg-[#0a0a0a] border border-white/10 p-12 shadow-[0_0_50px_rgba(0,0,0,1)]">
              <button onClick={() => setShowUpload(false)} className="absolute top-6 right-6 text-slate-500 hover:text-white transition-all">✕</button>
              <h2 className="text-xs font-black uppercase tracking-[0.4em] text-yellow-500 mb-12">Neural Feed Ingestion</h2>
              <form onSubmit={handleUpload} className="space-y-8">
                <div className="space-y-2">
                  <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Show Identifier</label>
                  <input required value={uploadShowId} onChange={e => setUploadShowId(e.target.value)} placeholder="e.g. Bear_S3_E01" className="w-full bg-black border border-white/10 p-4 text-xs outline-none focus:border-yellow-500 transition-all text-white" />
                </div>
                <div className="space-y-2">
                  <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Target Product</label>
                  <input value={uploadTarget} onChange={e => setUploadTarget(e.target.value)} placeholder="e.g. KFC Bucket" className="w-full bg-black border border-white/10 p-4 text-xs outline-none focus:border-yellow-500 transition-all text-white" />
                </div>
                <div className="space-y-2">
                  <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Video Source</label>
                  <input type="file" required accept="video/*" onChange={e => setUploadFile(e.target.files?.[0] || null)} className="w-full text-[10px] text-slate-500 file:bg-white/5 file:border-none file:text-white file:px-4 file:py-2 file:mr-4 file:cursor-pointer hover:file:bg-white/10" />
                </div>
                <button type="submit" disabled={isUploading} className="w-full bg-yellow-500 text-black py-5 font-black uppercase text-[11px] tracking-[0.2em] hover:bg-white active:scale-95 transition-all mt-4">
                  {isUploading ? "Neural Mapping..." : "Initialize Pipeline"}
                </button>
              </form>
           </div>
        </div>
      )}
    </div>
  );
}
