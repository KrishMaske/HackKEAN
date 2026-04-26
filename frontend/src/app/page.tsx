"use client";
import { useState } from 'react';
import { Play, Layers, RefreshCcw, ShieldCheck, ChevronDown, Rocket, Info, Plus, Box, Image as ImageIcon } from 'lucide-react';

const SCENES = [
  { id: 'tokyo', name: 'Tokyo District 03', path: '/scenes/tokyo_preview.png', color: '#1ce783' },
  { id: 'cyber', name: 'Night City - Alley', path: '/scenes/cyber_preview.png', color: '#f0abfc' },
];

const ASSETS = [
  { id: 'car', name: 'Future Hypercar', type: 'Vehicle', detail: 'Carbon Fiber / Gloss' },
  { id: 'drone', name: 'Security Drone', type: 'Prop', detail: 'Matte Stealth' },
];

export default function VisualistDashboard() {
  const [activeScene, setActiveScene] = useState(SCENES[0]);
  const [activeAsset, setActiveAsset] = useState(ASSETS[0]);
  const [currentTab, setCurrentTab] = useState('editor');
  const [isRendering, setIsRendering] = useState(false);
  const [result, setResult] = useState(null);

  const scrollToApp = () => document.getElementById('app-interface')?.scrollIntoView({ behavior: 'smooth' });

  const triggerRender = async () => {
    setIsRendering(true);
    setTimeout(() => { // Simulated for demo if backend is silent
      setIsRendering(false);
      setResult("success");
    }, 3000);
  };

  return (
    <div className="bg-[#0b0c0f] text-white snap-y snap-mandatory overflow-y-scroll h-screen selection:bg-[#1ce783]/30">
      
      {/* LANDING */}
      <section className="h-screen w-full flex flex-col items-center justify-center relative snap-start bg-black">
        <div className="z-10 text-center space-y-4">
          <h1 className="text-8xl font-black tracking-tighter uppercase italic">SCENE<span className="text-[#1ce783]">SHIFT</span></h1>
          <button onClick={scrollToApp} className="mt-10 px-12 py-4 bg-[#1ce783] text-black font-black rounded-sm hover:scale-105 transition-transform tracking-widest uppercase text-sm">Launch Studio</button>
        </div>
      </section>

      {/* STUDIO */}
      <section id="app-interface" className="h-screen w-full flex flex-col snap-start bg-[#0b0c0f] relative overflow-hidden">
        <nav className="w-full px-12 py-8 flex justify-between items-center z-20 bg-gradient-to-b from-black/80 to-transparent">
          <div className="flex items-center gap-10">
            <h2 className="text-2xl font-black italic tracking-tighter text-[#1ce783]">SCENESHIFT</h2>
            <div className="hidden md:flex gap-8 text-[11px] font-black tracking-[0.2em] text-zinc-400 uppercase">
              <span onClick={() => setCurrentTab('editor')} className={`${currentTab === 'editor' ? 'text-white border-b-2 border-[#1ce783]' : 'hover:text-white'} pb-1 cursor-pointer transition-all`}>Editor</span>
              <span onClick={() => setCurrentTab('assets')} className={`${currentTab === 'assets' ? 'text-white border-b-2 border-[#1ce783]' : 'hover:text-white'} pb-1 cursor-pointer transition-all`}>Assets</span>
            </div>
          </div>
        </nav>

        <div className="flex-1 flex items-center px-12 gap-12 max-w-[1600px] mx-auto w-full">
          {/* VIEWPORT */}
          <div className="flex-[2] aspect-video bg-black rounded-sm overflow-hidden border border-zinc-800 relative group shadow-2xl">
            <img src={activeScene.path} className={`w-full h-full object-cover transition-all duration-1000 ${isRendering ? 'scale-110 blur-3xl opacity-20' : 'scale-100 opacity-100'}`} />
            
            {currentTab === 'editor' && !isRendering && (
              <div className="absolute border-2 border-[#1ce783] animate-pulse" style={{ left: '20%', top: '50%', width: '300px', height: '150px' }}>
                <div className="absolute -top-6 left-0 text-[#1ce783] text-[9px] font-black uppercase tracking-widest">Target: {activeAsset.name}</div>
              </div>
            )}

            {isRendering && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60">
                <RefreshCcw className="animate-spin text-[#1ce783] mb-4" size={40} />
                <p className="font-mono text-[#1ce783] text-[10px] tracking-[0.5em] uppercase">Syncing Physics...</p>
              </div>
            )}
          </div>

          {/* SIDE PANEL */}
          <div className="flex-1 space-y-8 min-w-[300px]">
            {currentTab === 'editor' ? (
              <>
                <div>
                  <h3 className="text-5xl font-black uppercase tracking-tight italic leading-none">{activeScene.name}</h3>
                  <p className="text-zinc-500 text-sm mt-4 font-medium italic">Active Asset: <span className="text-white">{activeAsset.name}</span></p>
                </div>
                <div className="space-y-3">
                  <button onClick={triggerRender} disabled={isRendering} className="w-full flex items-center justify-center gap-3 py-5 bg-white text-black font-black uppercase text-xs tracking-[0.2em] hover:bg-[#1ce783] transition-colors">
                    <Play size={14} fill="black" /> {isRendering ? "Processing..." : "Generate Asset"}
                  </button>
                </div>
              </>
            ) : (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4">
                <h3 className="text-3xl font-black uppercase tracking-tight italic">Asset Library</h3>
                <div className="grid gap-4">
                  {ASSETS.map(asset => (
                    <div 
                      key={asset.id} 
                      onClick={() => setActiveAsset(asset)}
                      className={`p-4 border rounded-sm cursor-pointer transition-all ${activeAsset.id === asset.id ? 'border-[#1ce783] bg-[#1ce783]/10' : 'border-zinc-800 hover:border-zinc-600'}`}
                    >
                      <p className="text-[10px] font-black text-[#1ce783] uppercase mb-1">{asset.type}</p>
                      <h4 className="font-bold">{asset.name}</h4>
                      <p className="text-xs text-zinc-500">{asset.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* BOTTOM GALLERY */}
        <div className="w-full px-12 py-10 bg-gradient-to-t from-black to-transparent">
          <h4 className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500 mb-6">Switch Environment</h4>
          <div className="flex gap-5">
            {SCENES.map(scene => (
              <div 
                key={scene.id}
                onClick={() => setActiveScene(scene)}
                className={`w-56 h-32 rounded-sm border transition-all cursor-pointer relative overflow-hidden group ${activeScene.id === scene.id ? 'border-[#1ce783]' : 'border-zinc-800 opacity-40 hover:opacity-100'}`}
              >
                <img src={scene.path} className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-black/40 group-hover:bg-transparent transition-all"></div>
                <p className="absolute bottom-3 left-3 text-[9px] font-black uppercase tracking-widest">{scene.name}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
