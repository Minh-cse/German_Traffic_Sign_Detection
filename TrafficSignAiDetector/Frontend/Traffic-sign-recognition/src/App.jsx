import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Camera, Users, Info, Play, AlertTriangle, Check, Activity, User, Navigation2Icon } from 'lucide-react';

// ─── System Metrics Component ───────────────────────────────────────────────
function SystemMetrics() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/metrics");
        const data = await res.json();
        setMetrics(data);
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const barColor = (pct) => {
    if (pct > 80) return '#ef4444';
    if (pct > 50) return '#f59e0b';
    return '#69c280';
  };

  const gpuBarColor = (pct) => {
    if (pct > 80) return '#ef4444';
    if (pct > 50) return '#f59e0b';
    return '#7F77DD';
  };

  if (!metrics) return (
    <div className="bg-white rounded-xl shadow-xl p-4 text-xs text-slate-400 animate-pulse flex items-center gap-2">
      <Activity size={14} className="text-slate-300" />
      Loading system metrics...
    </div>
  );

  const MetricRow = ({ label, percent, sublabel, color }) => (
    <div>
      <div className="flex justify-between items-center text-xs mb-1">
        <span className="text-slate-500 font-medium">{label}</span>
        <div className="flex items-center gap-1.5">
          {sublabel && <span className="text-slate-400">{sublabel}</span>}
          <span className="font-bold text-slate-700">{percent}%</span>
        </div>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
        <div
          className="h-1.5 rounded-full transition-all duration-700"
          style={{ width: `${Math.min(percent, 100)}%`, background: color }}
        />
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-xl shadow-xl p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2">
        <div className="w-7 h-7 bg-blue-50 rounded-full flex items-center justify-center">
          <Activity size={14} className="text-blue-500" />
        </div>
        <h3 className="font-bold text-sm text-slate-700">System Usage</h3>
        <div className="ml-auto flex items-center gap-1.5 text-xs text-green-500">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse inline-block"></span>
          Live
        </div>
      </div>

      {/* CPU */}
      <MetricRow
        label="CPU"
        percent={metrics.cpu_percent}
        color={barColor(metrics.cpu_percent)}
      />

      {/* RAM */}
      <MetricRow
        label="RAM"
        percent={metrics.ram_percent}
        sublabel={`${metrics.ram_used_gb}/${metrics.ram_total_gb}GB`}
        color={barColor(metrics.ram_percent)}
      />

      {/* GPU */}
      {metrics.gpu_available ? (
        <>
          <MetricRow
            label="GPU"
            percent={metrics.gpu_percent}
            color={gpuBarColor(metrics.gpu_percent)}
          />
          <MetricRow
            label="GPU Mem"
            percent={metrics.gpu_memory_percent}
            sublabel={`${metrics.gpu_memory_used_gb}/${metrics.gpu_memory_total_gb}GB`}
            color={gpuBarColor(metrics.gpu_memory_percent)}
          />
        </>
      ) : (
        <div className="flex items-center gap-1.5 text-xs text-slate-400 pt-1">
          <span className="w-1.5 h-1.5 rounded-full bg-slate-300 inline-block"></span>
          GPU not available
        </div>
      )}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────
const App = () => {
  const [activeTab, setActiveTab] = useState('detection');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [detections, setDetections] = useState([]);
  const [processedImage, setProcessedImage] = useState(null);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);

  const captureAndDetect = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const blob = await fetch(imageSrc).then(res => res.blob());
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post("http://localhost:8000/predict", formData);

      if (response.data && response.data.detections) {
        const top3 = response.data.detections.slice(0, 3);
        const formatted = top3.map((det, index) => ({
          id: Date.now() + index,
          type: det.class_name,
          confidence: `${Math.round(det.confidence * 100)}%`,
          time: "Just now",
          color: index === 0 ? "text-[#69c280]" : "text-slate-400"
        }));
        setDetections(prev => [...formatted, ...prev].slice(0, 5));
      }
    } catch (error) {
      console.error("Detection error:", error);
    }
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post("http://localhost:8000/predict", formData);

      setIsCameraActive(false);

      if (response.data?.processed_image) {
        setProcessedImage(`data:image/jpeg;base64,${response.data.processed_image}`);
      } else {
        setProcessedImage(null);
      }

      if (response.data?.detections?.length) {
        const top3 = response.data.detections.slice(0, 3);
        const formatted = top3.map((det, index) => ({
          id: Date.now() + index,
          type: det.class_name,
          confidence: `${Math.round(det.confidence * 100)}%`,
          time: "Just now",
          color: index === 0 ? "text-[#69c280]" : "text-slate-400"
        }));
        setDetections(prev => [...formatted, ...prev].slice(0, 5));
      }
    } catch (err) {
      console.error("Upload failed:", err.response?.data || err.message);
      alert(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  useEffect(() => {
    let interval;
    if (isCameraActive) {
      interval = setInterval(() => captureAndDetect(), 2000);
    }
    return () => clearInterval(interval);
  }, [isCameraActive, captureAndDetect]);

  const toggleCamera = () => setIsCameraActive(!isCameraActive);
  const clearProcessed = () => { setProcessedImage(null); setDetections([]); };

  const renderContent = () => {
    switch (activeTab) {

      // ── DETECTION TAB ──────────────────────────────────────────────────────
      case 'detection':
        return (
          <div className="space-y-6">
            <div className="text-center text-white mb-6 animate-fly-in" style={{ animationDelay: '0ms' }}>
              <h2 className="text-xl font-semibold opacity-90">Traffic Sign Detection</h2>
              <p className="text-sm opacity-75">Point your camera at traffic signs and watch our AI detect them in real-time!</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* ── Camera Feed ── */}
              <div className="lg:col-span-2 animate-fly-in" style={{ animationDelay: '100ms' }}>
                <div className="bg-slate-900 rounded-lg aspect-video relative overflow-hidden shadow-xl border-4 border-white/20">
                  {processedImage ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                      <img src={processedImage} alt="Processed result" className="w-full h-full object-contain" />
                      <div className="absolute top-4 right-4 z-30">
                        <button onClick={clearProcessed} className="bg-white text-slate-700 px-3 py-2 rounded-lg font-semibold shadow-md hover:bg-slate-100">
                          Close
                        </button>
                      </div>
                    </div>
                  ) : !isCameraActive ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 z-20 bg-slate-900/60">
                      <Camera size={64} className="mb-4 opacity-50" />
                      <p className="mb-3 text-sm font-medium">Camera is off</p>
                      <div className="flex items-center gap-3">
                        <button
                          onClick={toggleCamera}
                          className="flex items-center gap-2 bg-[#69c280] hover:bg-[#5ab370] text-white px-4 py-2 rounded-lg font-semibold transition"
                        >
                          <Play size={16} />
                          <span>Start Camera</span>
                        </button>
                        <button
                          onClick={() => fileInputRef.current.click()}
                          className="bg-white text-slate-700 px-4 py-2 rounded-lg font-semibold shadow-md hover:bg-slate-100 transition"
                        >
                          Upload Image
                        </button>
                      </div>
                      <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                    </div>
                  ) : (
                    <div className="absolute inset-0">
                      <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="w-full h-full object-cover"
                        videoConstraints={{ facingMode: { ideal: "environment" } }}
                      />
                      <div className="absolute top-4 left-4 z-30">
                        <div className="bg-red-500 text-white px-3 py-1 rounded-full text-xs flex items-center gap-2 shadow-sm animate-pulse">
                          <div className="w-2 h-2 bg-white rounded-full"></div>
                          LIVE FEED
                        </div>
                      </div>
                      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-30">
                        <button
                          onClick={toggleCamera}
                          className="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded-lg font-semibold shadow-lg"
                        >
                          Stop Camera
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Feature Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 animate-fly-in" style={{ animationDelay: '200ms' }}>
                  <div className="bg-white/90 backdrop-blur-sm p-4 rounded-xl shadow-lg flex flex-col items-center text-center transition-transform hover:-translate-y-1">
                    <div className="w-10 h-10 bg-[#69c280]/20 rounded-full flex items-center justify-center text-[#69c280] mb-3">
                      <Camera size={20} />
                    </div>
                    <h3 className="font-semibold text-slate-700 text-sm">Real-time Detection</h3>
                  </div>
                  <div className="bg-white/90 backdrop-blur-sm p-4 rounded-xl shadow-lg flex flex-col items-center text-center transition-transform hover:-translate-y-1">
                    <div className="w-10 h-10 bg-[#ebe46c]/30 rounded-full flex items-center justify-center text-yellow-700 mb-3">
                      <Check size={20} />
                    </div>
                    <h3 className="font-semibold text-slate-700 text-sm">High Accuracy</h3>
                  </div>
                  <div className="bg-white/90 backdrop-blur-sm p-4 rounded-xl shadow-lg flex flex-col items-center text-center transition-transform hover:-translate-y-1">
                    <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center text-red-600 mb-3">
                      <AlertTriangle size={20} />
                    </div>
                    <h3 className="font-semibold text-slate-700 text-sm">Instant Alerts</h3>
                  </div>
                </div>
              </div>

              {/* ── Sidebar: System Metrics + Recent Detections ── */}
              <div className="space-y-4 animate-fly-in" style={{ animationDelay: '300ms' }}>

                {/* System Metrics */}
                <SystemMetrics />

                {/* Recent Detections */}
                <div className="bg-white rounded-xl shadow-xl p-6 flex flex-col min-h-[300px]">
                  <div className="flex items-center gap-2 mb-4 text-slate-700">
                    <div className="w-8 h-8 bg-[#69c280]/20 rounded-full flex items-center justify-center text-[#69c280]">
                      <Activity size={16} />
                    </div>
                    <h3 className="font-bold">Recent Detections</h3>
                  </div>

                  <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                    {detections.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center py-8">
                        <AlertTriangle size={40} className="mb-3 opacity-20" />
                        <p className="text-sm">No detections yet</p>
                        <p className="text-xs opacity-75 mt-1">Start the camera to begin</p>
                      </div>
                    ) : (
                      detections.map((det, index) => (
                        <div key={`${det.id}-${index}`} className="flex items-center p-3 bg-slate-50 rounded-lg border border-slate-100 animate-slide-in">
                          <div className={`p-2 rounded-lg bg-white shadow-sm mr-3 ${det.color}`}>
                            <Navigation2Icon size={20} />
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-semibold text-slate-700">{det.type}</p>
                            <div className="flex justify-between items-center mt-1">
                              <span className="text-xs text-slate-500">{det.time}</span>
                              <span className="text-xs font-bold text-[#69c280] bg-[#69c280]/10 px-2 py-0.5 rounded-full">
                                {det.confidence}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

              </div>
              {/* ── End Sidebar ── */}

            </div>
          </div>
        );

      // ── ABOUT US TAB ───────────────────────────────────────────────────────
      case 'about-us':
        return (
          <div className="max-w-5xl mx-auto">
            <div className="bg-white/95 backdrop-blur rounded-2xl shadow-xl p-8 md:p-12 animate-fly-in">
              <div className="text-center mb-10">
                <h2 className="text-3xl font-bold text-slate-800 mb-4">Meet the Team</h2>
                <p className="text-slate-600 max-w-2xl mx-auto">
                  We are a group of passionate developers and AI researchers dedicated to making roads safer through intelligent technology.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-8">
                {["Phạm Đức Thiện", "Dương Phạm Ngọc Minh", "Hồ Đức Nhật Hoàng"].map((member, index) => {
                  let roleTitle = "";
                  if (["Phạm Đức Thiện"].includes(member)) {
                    roleTitle = "AI Engineer";
                  } else if (member === "Dương Phạm Ngọc Minh") {
                    roleTitle = "Frontend Developer";
                  } else {
                    roleTitle = "Reporter";
                  }
                  return (
                    <div key={member} className="group relative w-full md:w-[calc(33.333%-2rem)] min-w-[250px] animate-fly-in" style={{ animationDelay: `${(index + 1) * 100}ms` }}>
                      <div className="bg-gradient-to-br from-slate-100 to-slate-200 rounded-2xl p-6 text-center transition-all duration-300 hover:shadow-lg hover:-translate-y-2 h-full">
                        <div className="w-24 h-24 bg-white mx-auto rounded-full mb-4 shadow-md flex items-center justify-center text-slate-300 group-hover:scale-110 transition-transform">
                          <User size={40} />
                        </div>
                        <h3 className="font-bold text-lg text-slate-800">{member}</h3>
                        <p className="text-[#69c280] text-sm font-medium mb-2">{roleTitle}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        );

      // ── ABOUT PROJECT TAB ──────────────────────────────────────────────────
      case 'about-project':
        return (
          <div className="max-w-4xl mx-auto animate-fly-in">
            <div className="bg-white/95 backdrop-blur rounded-2xl shadow-xl overflow-hidden">
              <div className="bg-slate-900 p-8 text-white relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-[#69c280] rounded-full mix-blend-multiply filter blur-3xl opacity-20 -translate-y-1/2 translate-x-1/2"></div>
                <div className="relative z-10 animate-fly-in" style={{ animationDelay: '100ms' }}>
                  <h2 className="text-3xl font-bold mb-4">Project Overview</h2>
                  <p className="text-slate-300 max-w-2xl">
                    TrafficSignal is a state-of-the-art computer vision system designed to identify and classify road signs in real-time using lightweight neural networks.
                  </p>
                </div>
              </div>

              <div className="p-8 md:p-12">
                <div className="grid md:grid-cols-2 gap-12">
                  <div className="space-y-6 animate-fly-in" style={{ animationDelay: '200ms' }}>
                    <h3 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                      <Activity className="text-[#69c280]" />
                      Technical Stack
                    </h3>
                    <ul className="space-y-4">
                      <li className="flex items-start gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-[#69c280] mt-2"></div>
                        <div>
                          <span className="font-semibold text-slate-700 block">YOLOv8 Architecture</span>
                          <span className="text-sm text-slate-500">Utilized for high-speed object detection with 99.2% mAP.</span>
                        </div>
                      </li>
                      <li className="flex items-start gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-[#69c280] mt-2"></div>
                        <div>
                          <span className="font-semibold text-slate-700 block">FastAPI</span>
                          <span className="text-sm text-slate-500">Enables high-performance, server-side APIs with automatic validation, async support, and low latency.</span>
                        </div>
                      </li>
                      <li className="flex items-start gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-[#69c280] mt-2"></div>
                        <div>
                          <span className="font-semibold text-slate-700 block">React, Tailwind and Vite</span>
                          <span className="text-sm text-slate-500">For a responsive, accessible, and performant user interface.</span>
                        </div>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-slate-50 rounded-xl p-6 border border-slate-100 animate-fly-in" style={{ animationDelay: '300ms' }}>
                    <h3 className="text-xl font-bold text-slate-800 mb-4">Key Capabilities</h3>
                    <div className="space-y-4">
                      {[
                        { label: 'Speed Limit Detection', pct: 98 },
                        { label: 'Stop Sign Recognition', pct: 99.5 },
                        { label: 'Low Light Performance', pct: 92 },
                      ].map(({ label, pct }) => (
                        <div key={label}>
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-600">{label}</span>
                            <span className="font-bold text-[#69c280]">{pct}% Accuracy</span>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2 mt-1">
                            <div className="bg-[#69c280] h-2 rounded-full" style={{ width: `${pct}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#69c280] via-[#ebe46c] to-[#69c280] font-sans text-slate-800">

      {/* Header */}
      <header className="px-6 pt-6 pb-2 max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <div className="bg-white p-2 rounded-full shadow-lg">
            <Navigation2Icon size={24} className="text-[#69c280]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white drop-shadow-sm">TrafficSignAI</h1>
            <p className="text-xs text-white/90 font-medium tracking-wide">Smart Detection System</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex flex-wrap gap-4">
          {[
            { id: 'detection', label: 'Detection', Icon: Camera },
            { id: 'about-us', label: 'About Us', Icon: Users },
            { id: 'about-project', label: 'About Project', Icon: Info },
          ].map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold transition-all shadow-sm ${
                activeTab === id
                  ? 'bg-white text-[#69c280] translate-y-0 shadow-md'
                  : 'bg-white/30 text-white hover:bg-white/40 hover:-translate-y-0.5'
              }`}
            >
              <Icon size={18} /> {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="max-w-7xl mx-auto p-6 pt-4">
        {renderContent()}
      </main>

      <footer className="text-center text-emerald-900/60 py-6 text-sm">
        <p>&copy; 2025 TrafficSignal AI. All rights reserved.</p>
      </footer>

      <style jsx>{`
        @keyframes flyInUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .animate-fly-in {
          animation: flyInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          opacity: 0;
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-10px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        .animate-slide-in {
          animation: slideIn 0.3s ease-out forwards;
        }
        .custom-scrollbar::-webkit-scrollbar       { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track  { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb  { background-color: #cbd5e1; border-radius: 20px; }
      `}</style>
    </div>
  );
};

export default App;