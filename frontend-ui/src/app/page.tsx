"use client";

import React, { useState, useRef } from "react";
import Head from "next/head";
import { Upload, Camera, Waves, Shield, ArrowRight, CheckCircle2, Loader2, Maximize2, RefreshCcw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Detection {
  class_name: string;
  confidence: number;
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
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [showAnnotated, setShowAnnotated] = useState(true);
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

    try {
      // Assuming FastAPI is running on localhost:8000
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

  return (
    <div className="min-h-screen ocean-gradient text-white selection:bg-ocean-primary/30">
      <Head>
        <title>Underwater Detection</title>
      </Head>

      {/* Navigation */}
      <nav className="p-6 flex justify-between items-center border-b border-white/10 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="bg-ocean-primary p-2 rounded-lg">
            <Waves className="text-[#0a192f]" size={24} />
          </div>
          <span className="text-2xl font-bold tracking-tighter">Underwater Processing</span>
        </div>
        <div className="flex gap-6 text-sm font-medium text-white/70">
          <a href="#" className="hover:text-ocean-primary transition-colors">Platform</a>
          <a href="#" className="hover:text-ocean-primary transition-colors">Technology</a>
          <a href="#" className="hover:text-ocean-primary transition-colors">Mission</a>
        </div>
        <button className="bg-white/5 hover:bg-white/10 px-4 py-2 rounded-full border border-white/20 transition-all">
          Dashboard
        </button>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-2 gap-12 items-center">
        {/* Left Content */}
        <div className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ocean-primary/10 border border-ocean-primary/20 text-ocean-primary text-xs font-bold uppercase tracking-widest">
              <Shield size={14} /> Next-Gen Maritime Intelligence
            </div>
              Underwater Image Enhancement and Marine Life Detection.
            <p className="text-xl text-white/60 max-w-xl leading-relaxed">
              Image enhancement and object detection system. Restore clarity to murky waters and identify marine species.
            </p>
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
                  <p className="text-sm text-white/40">Supporting JPG, PNG and RAW formats</p>
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
                      Execute Pipeline <ArrowRight size={20} />
                    </>
                  )}
                </button>
              </div>
            )}
          </motion.div>
        </div>

        {/* Right Results */}
        <div className="h-full flex flex-col items-center justify-center min-h-[500px]">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div 
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center space-y-4 opacity-30 px-12"
              >
                <div className="w-32 h-32 border-4 border-ocean-primary/20 rounded-full mx-auto flex items-center justify-center">
                  <Camera size={48} />
                </div>
                <p className="text-lg font-medium">System Ready</p>
                <p className="text-sm">Inference models loaded on CPU.</p>
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
                    Processing Latency: {result.processing_time.toFixed(2)}s
                  </div>
                </div>

                <div className="relative glass-card p-2 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] group">
                  <img 
                    src={showAnnotated && result.annotated_image ? result.annotated_image : result.enhanced_image} 
                    alt="Processed" 
                    className="w-full rounded-xl"
                  />
                  <div className="absolute top-6 right-6 p-2 bg-black/60 rounded-lg backdrop-blur-md">
                    <Maximize2 size={16} />
                  </div>
                </div>

                {/* Detection Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="glass-card p-4 rounded-xl space-y-1">
                    <span className="text-[10px] text-white/40 font-bold uppercase">Detected Objects</span>
                    <div className="text-2xl font-black text-ocean-primary">{result.detections.length}</div>
                  </div>
                  <div className="glass-card p-4 rounded-xl space-y-1">
                    <span className="text-[10px] text-white/40 font-bold uppercase">Average Confidence</span>
                    <div className="text-2xl font-black text-ocean-accent">
                      {result.detections.length > 0 
                        ? `${(result.detections.reduce((acc, d) => acc + d.confidence, 0) / result.detections.length * 100).toFixed(1)}%`
                        : "N/A"
                      }
                    </div>
                  </div>
                </div>

                {/* Detection List */}
                <div className="glass-card p-6 rounded-2xl space-y-4 max-h-[200px] overflow-y-auto custom-scrollbar">
                  <h3 className="text-sm font-bold flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-ocean-primary" /> Detections
                  </h3>
                  <div className="space-y-2">
                    {result.detections.map((det, idx) => (
                      <div key={idx} className="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-white/5 hover:border-white/20 transition-all">
                        <span className="font-bold capitalize">{det.class_name}</span>
                        <span className="text-ocean-primary font-mono text-sm">{(det.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                    {result.detections.length === 0 && (
                      <div className="text-white/20 text-center py-4 text-sm italic">
                        No objects identified.
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-20 p-12 border-t border-white/5 text-center text-white/30 text-xs">
        <div className="max-w-4xl mx-auto space-y-4">
          <p>© 2024 Marine Analysis Tool. All rights reserved.</p>
          <div className="flex justify-center gap-8">
            <a href="#" className="hover:text-ocean-primary transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-ocean-primary transition-colors">Documentation</a>
            <a href="#" className="hover:text-ocean-primary transition-colors">Settings</a>
          </div>
        </div>
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
