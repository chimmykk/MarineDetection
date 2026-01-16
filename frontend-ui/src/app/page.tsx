"use client";

import React, { useState, useRef, useEffect } from "react";
import Head from "next/head";
import { Upload, Camera, Waves, Shield, ArrowRight, CheckCircle2, Loader2, Maximize2, RefreshCcw, Filter, Fish, Bug, Heart } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Detection {
  class_name: string;
  confidence: number;
  category: string;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface ProcessResult {
  enhanced_image: string;
  annotated_image: string | null;
  detections: Detection[];
  processing_time: number;
  category_filter: string;
}

// Category colors and icons
const CATEGORY_CONFIG: Record<string, { color: string; bgColor: string; icon: React.ReactNode }> = {
  marine: { color: 'text-cyan-400', bgColor: 'bg-cyan-400/20', icon: <Waves size={12} /> },
  species: { color: 'text-orange-400', bgColor: 'bg-orange-400/20', icon: <Fish size={12} /> },
  disease: { color: 'text-red-400', bgColor: 'bg-red-400/20', icon: <Bug size={12} /> },
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("enhancement", "combined");
    formData.append("confidence", "0.25");
    formData.append("detection", "true");
    formData.append("category", categoryFilter);

    try {
      const response = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Processing failed");

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend. Ensure it is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Group detections by category
  const groupedDetections = result?.detections.reduce((acc, det) => {
    const cat = det.category || 'unknown';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(det);
    return acc;
  }, {} as Record<string, Detection[]>) || {};

  return (
    <div className="min-h-screen ocean-gradient text-white selection:bg-ocean-primary/30">
      <Head>
        <title>Underwater Detection - 23 Classes</title>
      </Head>

      {/* Navigation */}
      <nav className="p-6 flex justify-between items-center border-b border-white/10 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="bg-ocean-primary p-2 rounded-lg">
            <Waves className="text-[#0a192f]" size={24} />
          </div>
          <span className="text-2xl font-bold tracking-tighter">Unified Marine Detector</span>
        </div>
        <div className="flex gap-4 text-xs">
          <span className="px-2 py-1 rounded bg-cyan-400/20 text-cyan-400">7 Marine</span>
          <span className="px-2 py-1 rounded bg-orange-400/20 text-orange-400">12 Species</span>
          <span className="px-2 py-1 rounded bg-red-400/20 text-red-400">4 Disease</span>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-2 gap-12 items-start">
        {/* Left Content */}
        <div className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ocean-primary/10 border border-ocean-primary/20 text-ocean-primary text-xs font-bold uppercase tracking-widest">
              <Shield size={14} /> 23-Class Detection Model
            </div>
            <h1 className="text-4xl font-black leading-tight">
              Unified Marine Life Detection
            </h1>
            <p className="text-lg text-white/60 max-w-xl leading-relaxed">
              Detect marine life, identify fish species, and diagnose fish diseases - all with a single model.
            </p>
          </motion.div>

          {/* Category Filter */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-4 rounded-xl"
          >
            <div className="flex items-center gap-2 mb-3">
              <Filter size={16} className="text-ocean-primary" />
              <span className="text-sm font-bold">Detection Category</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[
                { value: 'all', label: 'All Classes', count: 23 },
                { value: 'marine', label: 'Marine Life', count: 7 },
                { value: 'species', label: 'Fish Species', count: 12 },
                { value: 'disease', label: 'Fish Disease', count: 4 },
              ].map((cat) => (
                <button
                  key={cat.value}
                  onClick={() => setCategoryFilter(cat.value)}
                  className={`p-3 rounded-lg text-xs font-bold transition-all ${
                    categoryFilter === cat.value
                      ? 'bg-ocean-primary text-[#0a192f]'
                      : 'bg-white/5 hover:bg-white/10'
                  }`}
                >
                  <div>{cat.label}</div>
                  <div className="text-[10px] opacity-70">{cat.count} classes</div>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Upload Area */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className={`glass-card p-10 rounded-3xl border-2 border-dashed transition-all ${
              file ? 'border-ocean-primary/50' : 'border-white/10 hover:border-ocean-primary/30'
            }`}
          >
            {!preview ? (
              <div 
                className="flex flex-col items-center gap-4 cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="w-20 h-20 bg-ocean-primary/10 rounded-full flex items-center justify-center text-ocean-primary animate-pulse">
                  <Upload size={32} />
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold">Drop underwater imagery here</p>
                  <p className="text-sm text-white/40">Supporting JPG, PNG formats</p>
                </div>
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef} 
                  onChange={handleFileChange}
                  accept="image/*"
                />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative aspect-video rounded-xl overflow-hidden shadow-2xl group">
                  <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-4">
                    <button 
                      onClick={() => {setPreview(null); setFile(null); setResult(null);}}
                      className="p-3 bg-red-500/80 rounded-full hover:scale-110 transition-transform"
                    >
                      <RefreshCcw size={20} />
                    </button>
                  </div>
                </div>
                <button 
                  onClick={handleUpload}
                  disabled={loading}
                  className="w-full py-4 bg-ocean-primary text-[#0a192f] font-black rounded-xl hover:shadow-[0_0_30px_rgba(100,255,218,0.5)] transition-all flex items-center justify-center gap-2 text-lg"
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin" /> Analyzing Image...
                    </>
                  ) : (
                    <>
                      Detect ({categoryFilter === 'all' ? '23 classes' : categoryFilter}) <ArrowRight size={20} />
                    </>
                  )}
                </button>
              </div>
            )}
          </motion.div>

          {/* Class Reference */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-4 rounded-xl"
          >
            <h3 className="text-sm font-bold mb-3">Detectable Classes</h3>
            <div className="space-y-3">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Waves size={14} className="text-cyan-400" />
                  <span className="text-xs font-bold text-cyan-400">Marine Life</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'].map(c => (
                    <span key={c} className="px-2 py-0.5 bg-cyan-400/10 text-cyan-400 rounded text-[10px]">{c}</span>
                  ))}
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Fish size={14} className="text-orange-400" />
                  <span className="text-xs font-bold text-orange-400">Fish Species</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {['surgeonfish', 'triggerfish', 'jack', 'spadefish', 'wrasse', 'snapper', 'angelfish', 'damselfish', 'parrotfish', 'tuna', 'grouper', 'moorish_idol'].map(c => (
                    <span key={c} className="px-2 py-0.5 bg-orange-400/10 text-orange-400 rounded text-[10px]">{c}</span>
                  ))}
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Bug size={14} className="text-red-400" />
                  <span className="text-xs font-bold text-red-400">Fish Disease</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {['bacterial_gill_disease', 'bacterial_red_disease', 'bacterial_disease', 'healthy_fish'].map(c => (
                    <span key={c} className="px-2 py-0.5 bg-red-400/10 text-red-400 rounded text-[10px]">{c}</span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Right Results */}
        <div className="h-full flex flex-col items-center justify-start min-h-[500px]">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div 
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center space-y-4 opacity-30 px-12 mt-20"
              >
                <div className="w-32 h-32 border-4 border-ocean-primary/20 rounded-full mx-auto flex items-center justify-center">
                  <Camera size={48} />
                </div>
                <p className="text-lg font-medium">System Ready</p>
                <p className="text-sm">Unified model loaded with 23 classes.</p>
              </motion.div>
            ) : (
              <motion.div 
                key="result"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="w-full space-y-6"
              >
                <div className="flex justify-between items-center px-4">
                  <div className="flex gap-2">
                    <button 
                      onClick={() => setShowAnnotated(false)}
                      className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${!showAnnotated ? 'bg-ocean-primary text-[#0a192f]' : 'bg-white/5'}`}
                    >
                      ENHANCED
                    </button>
                    <button 
                      onClick={() => setShowAnnotated(true)}
                      className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${showAnnotated ? 'bg-ocean-accent text-white' : 'bg-white/5'}`}
                    >
                      DETECTION
                    </button>
                  </div>
                  <div className="text-[10px] text-white/40 uppercase tracking-widest font-black">
                    {result.processing_time.toFixed(2)}s
                  </div>
                </div>

                <div className="relative glass-card p-2 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] group">
                  <img 
                    src={showAnnotated && result.annotated_image ? result.annotated_image : result.enhanced_image} 
                    alt="Processed" 
                    className="w-full rounded-xl"
                  />
                </div>

                {/* Detection Stats by Category */}
                <div className="grid grid-cols-3 gap-3">
                  {['marine', 'species', 'disease'].map(cat => {
                    const config = CATEGORY_CONFIG[cat];
                    const count = groupedDetections[cat]?.length || 0;
                    return (
                      <div key={cat} className={`glass-card p-3 rounded-xl ${config.bgColor}`}>
                        <div className="flex items-center gap-1 mb-1">
                          {config.icon}
                          <span className={`text-[10px] font-bold uppercase ${config.color}`}>{cat}</span>
                        </div>
                        <div className={`text-2xl font-black ${config.color}`}>{count}</div>
                      </div>
                    );
                  })}
                </div>

                {/* Detection List Grouped by Category */}
                <div className="glass-card p-4 rounded-2xl space-y-4 max-h-[350px] overflow-y-auto custom-scrollbar">
                  <h3 className="text-sm font-bold flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-ocean-primary" /> 
                    Detections ({result.detections.length})
                  </h3>
                  
                  {Object.entries(groupedDetections).map(([category, dets]) => {
                    const config = CATEGORY_CONFIG[category] || { color: 'text-white', bgColor: 'bg-white/10', icon: null };
                    return (
                      <div key={category} className="space-y-2">
                        <div className={`flex items-center gap-2 ${config.color}`}>
                          {config.icon}
                          <span className="text-xs font-bold uppercase">{category}</span>
                          <span className="text-[10px] opacity-60">({dets.length})</span>
                        </div>
                        {dets.map((det, idx) => (
                          <div key={idx} className={`flex justify-between items-center ${config.bgColor} p-3 rounded-lg border border-white/5`}>
                            <span className="font-bold capitalize text-sm">{det.class_name.replace(/_/g, ' ')}</span>
                            <span className={`${config.color} font-mono text-sm`}>{(det.confidence * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    );
                  })}
                  
                  {result.detections.length === 0 && (
                    <div className="text-white/20 text-center py-4 text-sm italic">
                      No objects detected.
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-20 p-8 border-t border-white/5 text-center text-white/30 text-xs">
        <p>Unified Marine Detector - 23 Classes (Marine Life, Fish Species, Fish Disease)</p>
      </footer>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(100, 255, 218, 0.2);
          border-radius: 10px;
        }
      `}</style>
    </div>
  );
}
