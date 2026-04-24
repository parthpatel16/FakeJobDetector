import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [activeTab, setActiveTab] = useState('text'); 
  const [activeView, setActiveView] = useState('home'); 
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewLocal, setImagePreviewLocal] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Company verification state
  const [companyReport, setCompanyReport] = useState(null);
  const [companyLoading, setCompanyLoading] = useState(false);
  const [showInputGuide, setShowInputGuide] = useState(false);

  // Chatbot state
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatAvailable, setChatAvailable] = useState(false);
  const chatEndRef = useRef(null);
  const chatInputRef = useRef(null);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Check chatbot availability on mount
  useEffect(() => {
    fetch("http://127.0.0.1:5000/chat/status")
      .then(r => r.json())
      .then(data => setChatAvailable(data.available))
      .catch(() => setChatAvailable(false));
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, chatLoading]);

  // Focus chat input when opened
  useEffect(() => {
    if (chatOpen) chatInputRef.current?.focus();
  }, [chatOpen]);

  // When results arrive and chat is empty, add a welcome message
  useEffect(() => {
    if (result && chatMessages.length === 0) {
      const welcome = result.prediction === 'Fake'
        ? `ΓÜá∩╕Å This job posting was flagged as **potentially fraudulent** (${result.confidence}% confidence). I strongly advise caution.\n\nI can help you understand why it was flagged, or answer any career questions. What would you like to know?`
        : `Γ£à This job posting appears **legitimate** (${result.confidence}% confidence).\n\nI can help you check if you're eligible, create a skill roadmap, or prepare for the interview. Ask me anything!`;
      
      setChatMessages([{ role: 'assistant', content: welcome }]);
    }
  }, [result]);

  // Async company verification ΓÇö fires after prediction result arrives
  useEffect(() => {
    if (result && result.company_name) {
      setCompanyLoading(true);
      setCompanyReport(null);
      fetch("http://127.0.0.1:5000/company-verify", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company_name: result.company_name,
          job_text: result.extracted_text || '',
        }),
      })
        .then(r => r.json())
        .then(data => setCompanyReport(data.company_info))
        .catch(err => {
          console.error('Company verification failed:', err);
          setCompanyReport({ name: result.company_name, source: 'error', verdict: 'Verification request failed.' });
        })
        .finally(() => setCompanyLoading(false));
    } else if (result && !result.company_name) {
      // No company name found at all
      setCompanyReport({
        name: 'Unknown',
        trust_score: 0,
        verdict: 'No company name detected in the job posting. This is often a sign of fraudulent listings.',
        red_flags: ['No company name provided ΓÇö legitimate employers always identify themselves.'],
        trust_breakdown: [{ factor: 'Company Identity', status: 'fail', detail: 'No company name found in posting' }],
        source: 'none',
      });
    }
  }, [result]);

  const resetState = () => {
    setResult(null);
    setError(null);
    setCompanyReport(null);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreviewLocal(reader.result);
      reader.readAsDataURL(file);
      resetState();
    }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    resetState();
    setChatMessages([]); // Reset chat for new analysis

    const API_BASE = "http://127.0.0.1:5000";

    try {
      let response;
      if (activeTab === 'text') {
        if (!inputText.trim()) throw new Error("Please paste a job description first.");
        response = await fetch(`${API_BASE}/predict-text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputText }),
        });
      } else if (activeTab === 'image') {
        if (!selectedFile) throw new Error("Please upload an image first.");
        const formData = new FormData();
        formData.append('image', selectedFile);
        response = await fetch(`${API_BASE}/predict-image`, {
          method: 'POST',
          body: formData,
        });
      } else if (activeTab === 'url') {
        if (!inputUrl.trim()) throw new Error("Please enter a job URL first.");
        response = await fetch(`${API_BASE}/predict-url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: inputUrl }),
        });
      }

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Analysis failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleChatSend = async (forcedMsg = null) => {
    const userMsg = typeof forcedMsg === 'string' ? forcedMsg.trim() : chatInput.trim();
    if (!userMsg || chatLoading) return;
    
    if (typeof forcedMsg !== 'string') setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);

    try {
      // Build history for the API (exclude the welcome message from Gemini history to prevent role alternance error)
      const historyForApi = chatMessages
        .filter((_, i) => i !== 0) // Skip first welcome assistant message
        .map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', content: m.content }));

      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          job_context: result || null,
          history: historyForApi,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Chat failed');
      
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.reply, suggestions: data.suggestions }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: `Γ¥î Error: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChatKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChatSend();
    }
  };

  const handleSuggestionClick = (text) => {
    handleChatSend(text);
  };

  const handleTryExample = (type = 'generic') => {
    setActiveTab('text');
    if (type === 'indian_fake') {
      setInputText("Company: QuickPay Solutions Pvt Ltd\nJob Title: Data Entry Operator\nLocation: Work From Home\nSalary: INR 15,000 - 25,000 per month\nExperience: No experience needed\n\nUrgent Work from Home Opportunity! Earn 5000-10000 INR daily by simple data entry and SMS sending. No experience or interview required. Apply on WhatsApp +91-700-xxx-xxxx. To start immediately, a security deposit of 999 INR is required for background verification using your Aadhaar/PAN card.");
    } else {
      setInputText("Company: GlobalTech Innovations Inc\nJob Title: Marketing Associate\nLocation: Remote\nSalary: $3,500 - $5,000/month\nExperience: 2+ years\n\nExceptional opportunity for an ambitious person! Work from home and earn $10,000 monthly with no prior experience. Immediate hiring. We require your bank details for direct deposit bonus setup. No interview needed, just apply now! Contact us at hr@gmail.com");
    }
    resetState();
    setChatMessages([]);
  };

  // --- Simple Markdown Renderer ---
  const renderMarkdown = (text) => {
    if (!text) return null;
    // Process line by line
    const lines = text.split('\n');
    const elements = [];
    let inList = false;
    let listItems = [];

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(<ul key={`ul-${elements.length}`} className="list-disc list-inside space-y-1 my-2">{listItems}</ul>);
        listItems = [];
        inList = false;
      }
    };

    const formatInline = (line) => {
      // Bold
      line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      // Italic
      line = line.replace(/\*(.+?)\*/g, '<em>$1</em>');
      // Code
      line = line.replace(/`(.+?)`/g, '<code class="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-xs font-mono">$1</code>');
      return line;
    };

    lines.forEach((line, i) => {
      const trimmed = line.trim();
      
      // Headers
      if (trimmed.startsWith('### ')) {
        flushList();
        elements.push(<h4 key={i} className="font-black text-sm mt-3 mb-1 text-on-surface dark:text-white" dangerouslySetInnerHTML={{ __html: formatInline(trimmed.slice(4)) }} />);
      } else if (trimmed.startsWith('## ')) {
        flushList();
        elements.push(<h3 key={i} className="font-black text-base mt-4 mb-2 text-on-surface dark:text-white" dangerouslySetInnerHTML={{ __html: formatInline(trimmed.slice(3)) }} />);
      } else if (trimmed.startsWith('# ')) {
        flushList();
        elements.push(<h2 key={i} className="font-black text-lg mt-4 mb-2 text-on-surface dark:text-white" dangerouslySetInnerHTML={{ __html: formatInline(trimmed.slice(2)) }} />);
      }
      // Bullet points
      else if (trimmed.match(/^[-*ΓÇó]\s/)) {
        inList = true;
        listItems.push(<li key={i} className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: formatInline(trimmed.slice(2)) }} />);
      }
      // Numbered lists
      else if (trimmed.match(/^\d+\.\s/)) {
        flushList();
        const content = trimmed.replace(/^\d+\.\s/, '');
        elements.push(<div key={i} className="flex gap-2 text-sm my-0.5"><span className="text-primary font-black shrink-0">{trimmed.match(/^\d+/)[0]}.</span><span dangerouslySetInnerHTML={{ __html: formatInline(content) }} /></div>);
      }
      // Empty line
      else if (trimmed === '') {
        flushList();
        elements.push(<div key={i} className="h-2" />);
      }
      // Normal text
      else {
        flushList();
        elements.push(<p key={i} className="text-sm leading-relaxed" dangerouslySetInnerHTML={{ __html: formatInline(trimmed) }} />);
      }
    });
    flushList();
    return elements;
  };

  // --- Sub-Components ---
  const PipelineCard = ({ step }) => (
    <motion.div 
      initial={{ opacity: 0, x: -20 }} 
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: step.step * 0.1 }}
      className="relative"
    >
      <div className="flex gap-4">
        <div className="flex flex-col items-center">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-indigo-600 flex items-center justify-center shadow-lg shadow-primary/20 shrink-0">
            <span className="material-symbols-outlined text-white text-lg">{step.icon}</span>
          </div>
          {step.step < 5 && <div className="w-0.5 h-full bg-gradient-to-b from-primary/30 to-transparent min-h-[20px]"></div>}
        </div>
        <div className="pb-6 flex-1">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-[10px] font-black uppercase tracking-widest text-primary">Step {step.step}</span>
            <span className="px-2 py-0.5 bg-green-500/10 text-green-600 dark:text-green-400 text-[9px] font-black uppercase tracking-widest rounded-full">Done</span>
          </div>
          <h4 className="text-base font-black text-on-surface dark:text-white tracking-tight">{step.name}</h4>
          <p className="text-sm text-on-surface-variant dark:text-slate-400 mt-1 leading-relaxed">{step.description}</p>
          <p className="text-xs font-bold text-primary/80 dark:text-primary/60 mt-2 bg-primary/5 dark:bg-primary/10 px-3 py-1.5 rounded-lg inline-block">{step.detail}</p>
          
          {step.checks_passed && (
            <div className="mt-3 space-y-1.5">
              {step.checks_passed.map((c, i) => (
                <div key={`p-${i}`} className="flex items-center gap-2 text-xs">
                  <span className="material-symbols-outlined text-green-500 text-sm">check_circle</span>
                  <span className="text-green-700 dark:text-green-400 font-medium">{c}</span>
                </div>
              ))}
              {step.checks_failed.map((c, i) => (
                <div key={`f-${i}`} className="flex items-center gap-2 text-xs">
                  <span className="material-symbols-outlined text-red-500 text-sm">cancel</span>
                  <span className="text-red-700 dark:text-red-400 font-bold">{c}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );

  const JobDetailRow = ({ icon, label, value }) => (
    <div className="flex items-start gap-3 py-2">
      <span className="material-symbols-outlined text-primary text-lg mt-0.5">{icon}</span>
      <div>
        <p className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant dark:text-slate-500">{label}</p>
        <p className="text-sm font-bold text-on-surface dark:text-white">{value}</p>
      </div>
    </div>
  );

  // Trust score color helper
  const getTrustColor = (score) => {
    if (score === null || score === undefined) return { bg: 'bg-slate-500', text: 'text-slate-500', label: 'N/A' };
    if (score >= 70) return { bg: 'bg-green-500', text: 'text-green-500', label: 'HIGH TRUST' };
    if (score >= 40) return { bg: 'bg-amber-500', text: 'text-amber-500', label: 'MODERATE' };
    return { bg: 'bg-red-500', text: 'text-red-500', label: 'LOW TRUST' };
  };

  const getStatusIcon = (status) => {
    if (status === 'pass') return { icon: 'check_circle', color: 'text-green-500' };
    if (status === 'warn') return { icon: 'warning', color: 'text-amber-500' };
    return { icon: 'cancel', color: 'text-red-500' };
  };

  // Quick suggestion buttons for the chatbot
  const chatSuggestions = result ? [
    result.prediction === 'Fake' ? "Why is this flagged as fake?" : "Am I eligible for this job?",
    "Create a learning roadmap for this role",
    "What skills do I need?",
    "Help me prepare for the interview",
  ] : [
    "How does FakeJobDetector work?",
    "What are common job scam signs?",
    "Tips for safe job searching",
  ];

  return (
    <div className="min-h-screen bg-background dark:bg-slate-950 text-on-background dark:text-slate-100 selection:bg-primary-container selection:text-on-primary-container transition-colors duration-500">
      {/* TopNavBar */}
      <nav className="fixed top-0 w-full z-50 glass-nav dark:bg-slate-900/80 shadow-sm border-none backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-10">
            <span 
              onClick={() => setActiveView('home')}
              className="text-2xl font-black tracking-tighter text-slate-900 dark:text-white font-headline cursor-pointer hover:scale-105 transition-transform"
            >
              FakeJob<span className="text-primary">Detector</span>
            </span>
            <div className="hidden md:flex gap-8 items-center">
              {['home', 'about'].map((view) => (
                <button
                  key={view}
                  onClick={() => setActiveView(view)}
                  className={`text-sm font-bold tracking-tight transition-all duration-300 uppercase ${
                    activeView === view 
                    ? 'text-primary border-b-2 border-primary' 
                    : 'text-on-surface-variant dark:text-slate-400 hover:text-primary'
                  }`}
                >
                  {view}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="p-3 rounded-full bg-surface-container-high dark:bg-slate-800 hover:scale-110 transition-all duration-200"
            >
              <span className="material-symbols-outlined text-on-surface-variant dark:text-slate-200">
                {isDarkMode ? 'light_mode' : 'dark_mode'}
              </span>
            </button>
          </div>
        </div>
      </nav>

      <main className="pt-32 pb-20">
        {activeView === 'home' && (
          <>
            {/* Hero Section */}
            <section className="max-w-4xl mx-auto px-6 text-center mb-16">
              <motion.span 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="inline-block px-5 py-2 mb-8 text-[10px] font-black tracking-[0.2em] text-primary uppercase bg-primary-fixed dark:bg-primary/20 rounded-full"
              >
                Advanced Phishing Detection
              </motion.span>
              <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-5xl md:text-6xl font-black tracking-tighter text-on-surface dark:text-white mb-8 leading-[1.1]"
              >
                Don't get <span className="text-primary italic">Scammed.</span>
              </motion.h1>
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-xl text-on-surface-variant dark:text-slate-300 font-body max-w-2xl mx-auto leading-relaxed"
              >
                Our AI analyzes linguistic patterns, requests for money, and suspicious contact methods to verify job postings instantly.
              </motion.p>
            </section>

            {/* Input Switcher Tabs */}
            <section className="max-w-2xl mx-auto px-6 mb-10 flex justify-center gap-4">
              {['text', 'image', 'url'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => { setActiveTab(tab); resetState(); }}
                  className={`px-8 py-3 rounded-xl text-xs font-black tracking-widest uppercase transition-all duration-300 ${
                    activeTab === tab 
                    ? 'bg-primary text-white shadow-xl shadow-primary/30' 
                    : 'bg-surface-container-high dark:bg-slate-800 text-on-surface-variant dark:text-slate-400 hover:bg-surface-container-highest'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </section>

            {/* Main Input Section */}
            <section className="max-w-4xl mx-auto px-6 mb-20">
              <div className="bg-surface-container-low dark:bg-slate-900 rounded-3xl p-10 shadow-2xl shadow-primary/5 border border-slate-200/50 dark:border-slate-800 mt-2">
                {/* Input Guide Panel */}
                <div className="mb-6">
                  <button
                    onClick={() => setShowInputGuide(!showInputGuide)}
                    className="flex items-center gap-2 text-xs font-bold text-primary hover:text-primary/80 transition-colors group"
                  >
                    <span className="material-symbols-outlined text-sm transition-transform duration-200" style={{ transform: showInputGuide ? 'rotate(180deg)' : 'rotate(0deg)' }}>expand_more</span>
                    <span className="material-symbols-outlined text-sm">info</span>
                    Tips for better results ΓÇö What to include
                  </button>
                  <AnimatePresence>
                    {showInputGuide && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-4 p-6 bg-primary/5 dark:bg-primary/10 rounded-2xl border border-primary/10 dark:border-primary/20">
                          <p className="text-xs font-bold text-on-surface dark:text-white mb-4 uppercase tracking-widest">≡ƒôï For the most accurate analysis, include:</p>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {[
                              { icon: 'domain', label: 'Company Name', hint: 'Who is hiring?' },
                              { icon: 'badge', label: 'Job Title / Role', hint: 'What position?' },
                              { icon: 'description', label: 'Job Description', hint: 'Responsibilities & duties' },
                              { icon: 'checklist', label: 'Requirements', hint: 'Skills, qualifications, experience' },
                              { icon: 'payments', label: 'Salary / CTC', hint: 'Compensation or pay range' },
                              { icon: 'location_on', label: 'Location', hint: 'Office, remote, or hybrid?' },
                              { icon: 'mail', label: 'Contact Info', hint: 'Email, phone, or website' },
                            ].map((item, i) => (
                              <div key={i} className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/50 dark:hover:bg-slate-800/50 transition-colors">
                                <span className="material-symbols-outlined text-primary text-lg">{item.icon}</span>
                                <div>
                                  <p className="text-xs font-black text-on-surface dark:text-white">{item.label}</p>
                                  <p className="text-[10px] text-on-surface-variant dark:text-slate-400">{item.hint}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                          <p className="text-[10px] text-on-surface-variant dark:text-slate-500 mt-4 italic">≡ƒÆí The more details you provide, the more accurate and comprehensive our analysis will be.</p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <div className="relative">
                  {activeTab === 'text' && (
                    <textarea
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      className="w-full h-72 p-8 bg-surface-container-lowest dark:bg-slate-800 rounded-2xl border-none focus:ring-4 focus:ring-primary/20 font-body text-on-surface dark:text-white placeholder:text-outline/50 transition-all duration-300 resize-none shadow-inner"
                      placeholder="Paste the job title and description here..."
                    />
                  )}
                  {activeTab === 'image' && (
                    <div className="w-full min-h-[18rem]">
                      <div 
                        onClick={() => fileInputRef.current.click()}
                        className={`w-full flex flex-col items-center justify-center p-8 bg-surface-container-lowest dark:bg-slate-800 rounded-2xl border-4 border-dashed border-primary-fixed-dim dark:border-slate-700 cursor-pointer hover:bg-surface-container-lowest/50 transition-all group ${imagePreviewLocal ? 'py-4' : 'h-72'}`}
                      >
                        <input type="file" hidden ref={fileInputRef} onChange={handleFileChange} accept="image/*" />
                        {imagePreviewLocal ? (
                          <div className="w-full space-y-4">
                            <img src={imagePreviewLocal} alt="Uploaded preview" className="max-h-60 mx-auto rounded-xl object-contain border border-slate-200 dark:border-slate-700 shadow-lg" />
                            <p className="text-center text-sm font-bold text-primary">{selectedFile?.name}</p>
                            <p className="text-center text-[10px] text-outline uppercase tracking-widest">Click to change image</p>
                          </div>
                        ) : (
                          <>
                            <span className="material-symbols-outlined text-6xl text-primary-fixed-dim dark:text-slate-600 group-hover:scale-110 transition-transform mb-6">image</span>
                            <p className="font-bold text-on-surface-variant dark:text-slate-300">Upload a screenshot of the job posting</p>
                            <p className="text-[10px] text-outline mt-3 italic uppercase tracking-widest">Supports JPG, PNG (AI OCR Auto-Extraction)</p>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                  {activeTab === 'url' && (
                    <div className="w-full h-72 flex flex-col items-center justify-center p-8 bg-surface-container-lowest dark:bg-slate-800 rounded-2xl shadow-inner">
                      <span className="material-symbols-outlined text-6xl text-primary-fixed-dim dark:text-slate-600 mb-8">link</span>
                      <input type="url" value={inputUrl} onChange={(e) => setInputUrl(e.target.value)} placeholder="https://example.com/job-post-url" className="w-full max-w-lg p-5 bg-surface-container-low dark:bg-slate-700 rounded-2xl border-none focus:ring-4 focus:ring-primary/20 text-center text-on-surface dark:text-white" />
                      <p className="text-[10px] text-outline mt-6 uppercase tracking-widest">We will scrape visible text from the page for analysis.</p>
                    </div>
                  )}

              {isAnalyzing && (
                <div className="absolute inset-0 bg-surface-container-lowest/80 backdrop-blur-sm rounded-lg flex flex-col items-center justify-center">
                  <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                  <p className="mt-4 font-bold text-primary animate-pulse tracking-widest uppercase text-xs">Processing Intelligence...</p>
                </div>
              )}
            </div>

            {error && (
              <p className="mt-4 text-center text-error font-medium text-sm flex items-center justify-center gap-2">
                <span className="material-symbols-outlined text-sm">error</span> {error}
              </p>
            )}

                <div className="mt-10 flex flex-col md:flex-row items-stretch gap-6 justify-center">
                  <button onClick={handleAnalyze} disabled={isAnalyzing} className="w-full md:w-64 px-10 py-5 bg-gradient-to-br from-primary to-indigo-700 text-white font-black tracking-widest uppercase rounded-2xl hover:scale-105 active:scale-95 transition-all duration-200 shadow-2xl shadow-primary/30 disabled:opacity-50 disabled:scale-100">
                    Analyze Now
                  </button>
                  <div className="flex gap-4 w-full md:w-auto">
                    <button onClick={() => handleTryExample('generic')} className="flex-1 px-8 py-5 bg-surface-container-highest dark:bg-slate-800 text-primary dark:text-slate-200 font-bold rounded-2xl hover:bg-surface-container-high transition-all duration-200 text-xs tracking-tighter">
                      Global Example
                    </button>
                    <button onClick={() => handleTryExample('indian_fake')} className="flex-1 px-8 py-5 bg-surface-container-highest dark:bg-slate-800 text-primary dark:text-slate-200 font-bold rounded-2xl border border-primary/20 hover:bg-surface-container-high transition-all duration-200 text-xs tracking-tighter">
                      Indian Example
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {/* ===== RESULTS SECTION ===== */}
            <AnimatePresence>
              {result && (
                <motion.section initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="max-w-7xl mx-auto px-6 mb-24 space-y-8">
                  {/* Verdict Banner */}
                  <div className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-10 shadow-2xl border-none relative overflow-hidden">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
                      <div className="flex items-center gap-8">
                        <div className={`w-20 h-20 rounded-[1.5rem] flex items-center justify-center shrink-0 ${result.prediction === 'Fake' ? 'bg-red-500 shadow-xl shadow-red-500/30' : 'bg-green-500 shadow-xl shadow-green-500/30'}`}>
                          <span className="material-symbols-outlined text-4xl text-white">{result.prediction === 'Fake' ? 'dangerous' : 'verified_user'}</span>
                        </div>
                        <div>
                          <h2 className="text-3xl md:text-4xl font-black text-on-surface dark:text-white tracking-tighter">{result.prediction === 'Fake' ? 'Fraud Detected' : 'Likely Genuine'}</h2>
                          <p className="text-on-surface-variant dark:text-slate-400 text-sm font-bold uppercase tracking-[0.2em] mt-1">Neural Analysis Complete</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="bg-surface-container-low dark:bg-slate-800 p-6 rounded-[1.5rem] text-center min-w-[180px] border border-slate-200/30 dark:border-slate-700">
                          <span className="block text-xs font-black text-on-surface-variant dark:text-slate-400 uppercase tracking-widest mb-1">Confidence</span>
                          <span className={`text-4xl font-black tracking-tighter ${result.prediction === 'Fake' ? 'text-red-500' : 'text-green-500'}`}>{result.confidence}%</span>
                        </div>
                        {chatAvailable && (
                          <button
                            onClick={() => setChatOpen(true)}
                            className="px-6 py-4 bg-gradient-to-br from-violet-500 to-purple-700 text-white font-black text-xs tracking-widest uppercase rounded-2xl hover:scale-105 active:scale-95 transition-all duration-200 shadow-xl shadow-purple-500/30 flex items-center gap-2"
                          >
                            <span className="material-symbols-outlined text-lg">smart_toy</span>
                            Ask AI
                          </button>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* ===== MAIN CENTER CARDS: Company & Job Details ===== */}
                  <div className="flex flex-col gap-8 max-w-5xl mx-auto">
                    {/* Image Preview (if applicable) */}
                    {(result.image_preview || imagePreviewLocal) && activeTab === 'image' && (
                      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-xl border border-slate-200/20 dark:border-slate-800">
                        <h3 className="text-lg font-black text-on-surface dark:text-white mb-4 flex items-center gap-2">
                          <span className="material-symbols-outlined text-primary">photo_camera</span>Uploaded Image
                        </h3>
                        <img src={result.image_preview || imagePreviewLocal} alt="Scanned document" className="w-full rounded-2xl border border-slate-200 dark:border-slate-700 shadow-md object-contain max-h-72" />
                        <p className="text-[10px] uppercase tracking-widest text-center mt-3 text-on-surface-variant dark:text-slate-500 font-black">OCR Processed via Tesseract v5</p>
                      </motion.div>
                    )}

                    {/* Company Due Diligence Card ΓÇö CENTER/MAIN */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-2xl border border-slate-200/20 dark:border-slate-800 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-bl from-primary/10 to-transparent rounded-bl-full"></div>
                      <h3 className="text-xl font-black text-on-surface dark:text-white mb-6 flex items-center gap-2">
                        <span className="material-symbols-outlined text-primary text-2xl">verified_user</span>Company Verification
                      </h3>

                      {/* Loading State */}
                      {companyLoading && (
                        <div className="space-y-4 animate-pulse">
                          <div className="flex items-center gap-4">
                            <div className="w-14 h-14 rounded-2xl bg-slate-200 dark:bg-slate-700"></div>
                            <div className="flex-1 space-y-2">
                              <div className="h-5 bg-slate-200 dark:bg-slate-700 rounded-lg w-3/4"></div>
                              <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-lg w-1/2"></div>
                            </div>
                          </div>
                          <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-full"></div>
                          <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-5/6"></div>
                          <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-4/6"></div>
                          <p className="text-[10px] text-primary font-bold uppercase tracking-widest text-center mt-4 animate-pulse">≡ƒöì Running AI due-diligence check...</p>
                        </div>
                      )}

                      {/* Company Report */}
                      {!companyLoading && companyReport && (
                        <div className="space-y-5">
                          {/* Header: Name + Trust Score */}
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-4">
                              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${companyReport.trust_score !== null && companyReport.trust_score !== undefined ? getTrustColor(companyReport.trust_score).bg + '/20' : 'bg-slate-200 dark:bg-slate-700'}`}>
                                <span className={`text-2xl font-black ${companyReport.trust_score !== null && companyReport.trust_score !== undefined ? getTrustColor(companyReport.trust_score).text : 'text-slate-400'}`}>
                                  {companyReport.name && companyReport.name !== 'Unknown' ? companyReport.name.charAt(0).toUpperCase() : '?'}
                                </span>
                              </div>
                              <div>
                                <p className="text-lg font-black text-on-surface dark:text-white tracking-tight">{companyReport.name || 'Unknown'}</p>
                                {companyReport.industry && companyReport.industry !== 'Unknown' && (
                                  <p className="text-[10px] font-bold text-on-surface-variant dark:text-slate-400 uppercase tracking-widest">{companyReport.industry}</p>
                                )}
                              </div>
                            </div>
                            {companyReport.trust_score !== null && companyReport.trust_score !== undefined && (
                              <div className="text-center">
                                <div className={`w-16 h-16 rounded-2xl ${getTrustColor(companyReport.trust_score).bg}/10 border-2 ${getTrustColor(companyReport.trust_score).bg === 'bg-green-500' ? 'border-green-500/30' : getTrustColor(companyReport.trust_score).bg === 'bg-amber-500' ? 'border-amber-500/30' : 'border-red-500/30'} flex items-center justify-center`}>
                                  <span className={`text-xl font-black ${getTrustColor(companyReport.trust_score).text}`}>{companyReport.trust_score}</span>
                                </div>
                                <p className={`text-[8px] font-black uppercase tracking-widest mt-1 ${getTrustColor(companyReport.trust_score).text}`}>{getTrustColor(companyReport.trust_score).label}</p>
                              </div>
                            )}
                          </div>

                          {/* Quick Info Row */}
                          {(companyReport.location || companyReport.size || companyReport.founded) && (
                            <div className="flex flex-wrap gap-2">
                              {companyReport.location && companyReport.location !== 'Unknown' && (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-bold bg-slate-100 dark:bg-slate-800 rounded-lg"><span className="material-symbols-outlined text-xs text-primary">location_on</span>{companyReport.location}</span>
                              )}
                              {companyReport.size && companyReport.size !== 'Unknown' && (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-bold bg-slate-100 dark:bg-slate-800 rounded-lg"><span className="material-symbols-outlined text-xs text-primary">group</span>{companyReport.size}</span>
                              )}
                              {companyReport.founded && companyReport.founded !== 'Unknown' && (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-bold bg-slate-100 dark:bg-slate-800 rounded-lg"><span className="material-symbols-outlined text-xs text-primary">calendar_today</span>{companyReport.founded}</span>
                              )}
                              {companyReport.registration_type && companyReport.registration_type !== 'Unknown' && (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-bold bg-slate-100 dark:bg-slate-800 rounded-lg"><span className="material-symbols-outlined text-xs text-primary">gavel</span>{companyReport.registration_type}</span>
                              )}
                            </div>
                          )}

                          {/* Trust Breakdown Table */}
                          {companyReport.trust_breakdown && companyReport.trust_breakdown.length > 0 && (
                            <div className="bg-white/30 dark:bg-slate-800/30 rounded-2xl p-4 border border-slate-200/30 dark:border-slate-700/30">
                              <p className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant dark:text-slate-500 mb-3">Trust Breakdown</p>
                              <div className="space-y-2">
                                {companyReport.trust_breakdown.map((item, i) => {
                                  const statusStyle = getStatusIcon(item.status);
                                  return (
                                    <div key={i} className="flex items-center gap-3 py-1">
                                      <span className={`material-symbols-outlined text-sm ${statusStyle.color}`}>{statusStyle.icon}</span>
                                      <span className="text-xs font-bold text-on-surface dark:text-white min-w-[120px]">{item.factor}</span>
                                      <span className="text-xs text-on-surface-variant dark:text-slate-400 flex-1">{item.detail}</span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* Red Flags */}
                          {companyReport.red_flags && companyReport.red_flags.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-[10px] font-black uppercase tracking-widest text-red-500">≡ƒÜ¿ Red Flags</p>
                              {companyReport.red_flags.map((flag, i) => (
                                <div key={i} className="flex gap-2 items-start bg-red-500/5 p-2.5 rounded-xl border border-red-500/10">
                                  <span className="material-symbols-outlined text-red-500 text-sm mt-0.5">warning</span>
                                  <p className="text-xs font-bold text-red-700 dark:text-red-400">{flag}</p>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Positive Signs */}
                          {companyReport.positive_signs && companyReport.positive_signs.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-[10px] font-black uppercase tracking-widest text-green-500">Γ£à Positive Signs</p>
                              {companyReport.positive_signs.map((sign, i) => (
                                <div key={i} className="flex gap-2 items-start bg-green-500/5 p-2.5 rounded-xl border border-green-500/10">
                                  <span className="material-symbols-outlined text-green-500 text-sm mt-0.5">check_circle</span>
                                  <p className="text-xs font-bold text-green-700 dark:text-green-400">{sign}</p>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Verdict */}
                          {companyReport.verdict && (
                            <div className={`p-4 rounded-2xl border ${
                              companyReport.trust_score >= 70 ? 'bg-green-500/5 border-green-500/20' :
                              companyReport.trust_score >= 40 ? 'bg-amber-500/5 border-amber-500/20' :
                              'bg-red-500/5 border-red-500/20'
                            }`}>
                              <p className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant dark:text-slate-500 mb-1">Verdict</p>
                              <p className="text-sm font-bold text-on-surface dark:text-white">{companyReport.verdict}</p>
                            </div>
                          )}

                          {/* Source badge */}
                          <p className="text-[9px] text-on-surface-variant dark:text-slate-600 uppercase tracking-widest text-right">
                            {companyReport.source === 'gemini' ? '≡ƒñû AI-Powered Analysis' : companyReport.source === 'none' ? 'ΓÜá∩╕Å No Company Detected' : companyReport.source === 'error' ? 'Γ¥î Verification Failed' : 'Γä╣∩╕Å Basic Info'}
                          </p>
                        </div>
                      )}

                      {/* Fallback when no result yet */}
                      {!companyLoading && !companyReport && (
                        <p className="text-sm italic text-on-surface-variant dark:text-slate-500">Company verification will appear after analysis.</p>
                      )}
                    </motion.div>

                    {/* Job Details Card ΓÇö CENTER/MAIN */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-2xl border border-slate-200/20 dark:border-slate-800">
                      <h3 className="text-xl font-black text-on-surface dark:text-white mb-6 flex items-center gap-2">
                        <span className="material-symbols-outlined text-primary text-2xl">work</span>Extracted Job Details
                      </h3>
                      {result.job_details && Object.keys(result.job_details).length > 0 ? (
                        <div className="divide-y divide-slate-100 dark:divide-slate-800">
                          {result.job_details.title && <JobDetailRow icon="badge" label="Job Title" value={result.job_details.title} />}
                          {result.job_details.location && <JobDetailRow icon="location_on" label="Location" value={result.job_details.location} />}
                          {result.job_details.salary && <JobDetailRow icon="payments" label="Salary / CTC" value={result.job_details.salary} />}
                          {result.job_details.employment_type && <JobDetailRow icon="schedule" label="Type" value={result.job_details.employment_type} />}
                          {result.job_details.experience && <JobDetailRow icon="trending_up" label="Experience" value={result.job_details.experience} />}
                          {result.job_details.contact && <JobDetailRow icon="call" label="Contact" value={result.job_details.contact} />}
                          {result.job_details.skills && (
                            <div className="py-3">
                              <p className="text-[10px] font-black uppercase tracking-widest text-on-surface-variant dark:text-slate-500 mb-2 flex items-center gap-2">
                                <span className="material-symbols-outlined text-primary text-lg">checklist</span>Skills Required
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {result.job_details.skills.map((skill, i) => (
                                  <span key={i} className="px-3 py-1 text-[11px] font-bold bg-primary/10 text-primary rounded-lg border border-primary/20">{skill}</span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-sm italic text-on-surface-variant dark:text-slate-500">No structured job details could be extracted from the text.</p>
                      )}
                    </motion.div>
                  </div>

                  {/* ===== DETAILED SUMMARY & RED FLAGS ===== */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-xl border border-slate-200/20 dark:border-slate-800 relative overflow-hidden">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-black text-on-surface dark:text-white flex items-center gap-2">
                          <span className="material-symbols-outlined text-primary">article</span>Detailed Summary
                        </h3>
                        {result.analysis?.source === 'gemini' && (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-full">
                            <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none">
                              <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="url(#gemini-grad)" />
                              <defs><linearGradient id="gemini-grad" x1="2" y1="2" x2="22" y2="22"><stop stopColor="#4285F4"/><stop offset="0.5" stopColor="#9B72CB"/><stop offset="1" stopColor="#D96570"/></linearGradient></defs>
                            </svg>
                            <span className="text-[9px] font-black uppercase tracking-widest bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Gemini AI</span>
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-on-surface-variant dark:text-slate-300 leading-relaxed">{result.analysis?.summary}</p>
                      {result.analysis?.company_insight && (
                        <div className="mt-4 p-4 bg-primary/5 dark:bg-primary/10 rounded-2xl border border-primary/10">
                          <h4 className="font-black text-[10px] uppercase tracking-widest text-primary mb-2">Company Insight</h4>
                          <p className="text-sm dark:text-slate-400 font-medium">{result.analysis.company_insight}</p>
                        </div>
                      )}
                      <div className="mt-4 p-4 bg-white/50 dark:bg-slate-800/50 rounded-2xl">
                        <h4 className="font-black text-[10px] uppercase tracking-widest text-primary mb-2">Recommendation</h4>
                        <p className="text-sm dark:text-slate-400 font-bold">{result.analysis?.recommendation}</p>
                      </div>
                    </motion.div>

                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-xl border border-slate-200/20 dark:border-slate-800">
                      <h3 className="text-lg font-black text-on-surface dark:text-white mb-4 uppercase tracking-widest">{result.prediction === 'Fake' ? 'Red Flags' : 'Positive Indicators'}</h3>
                      <div className="flex flex-col gap-3 mb-6">
                        {result.analysis?.red_flags && result.analysis.red_flags.length > 0 ? (
                          result.analysis.red_flags.map((flag, i) => (
                            <div key={i} className="flex gap-3 items-start bg-red-500/5 p-3 rounded-xl border border-red-500/10">
                              <span className="material-symbols-outlined text-red-500 text-sm mt-0.5">warning</span>
                              <p className="text-xs font-bold text-red-700 dark:text-red-400">{flag}</p>
                            </div>
                          ))
                        ) : (
                           <p className="text-sm dark:text-slate-400 italic">No suspicious metadata patterns detected.</p>
                        )}
                      </div>
                      <div>
                        <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">Linguistic Hotspots</h4>
                        <div className="flex flex-wrap gap-2">
                          {result.highlights && result.highlights.map((word, i) => (
                            <span key={i} className="px-3 py-1.5 text-[10px] font-black uppercase tracking-tighter bg-primary/10 text-primary rounded-lg border border-primary/20">{word}</span>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  </div>

                  {/* ===== EXTRACTED TEXT ===== */}
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-xl border border-slate-200/20 dark:border-slate-800">
                    <p className="text-[10px] font-black text-on-surface-variant dark:text-slate-500 uppercase tracking-[0.3em] mb-3">Raw Snippet Extraction</p>
                    <p className="text-sm italic text-on-surface-variant dark:text-slate-400 bg-white/30 dark:bg-slate-950/30 p-5 rounded-xl border border-dashed border-slate-300 dark:border-slate-800 leading-relaxed">{result.extracted_text}</p>
                  </motion.div>

                  {/* ===== ANALYSIS PIPELINE ΓÇö AT THE BOTTOM ===== */}
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="bg-surface-container-lowest dark:bg-slate-900 rounded-[2.5rem] p-8 shadow-xl border border-slate-200/20 dark:border-slate-800">
                    <h3 className="text-lg font-black text-on-surface dark:text-white mb-6 flex items-center gap-2">
                      <span className="material-symbols-outlined text-primary">account_tree</span>How We Analyzed This
                    </h3>
                    <p className="text-sm text-on-surface-variant dark:text-slate-400 mb-8 leading-relaxed">Every input goes through a 5-step neural analysis pipeline. Here's exactly what happened with your submission:</p>
                    <div className="space-y-0">
                      {result.pipeline && result.pipeline.map((step) => (<PipelineCard key={step.step} step={step} />))}
                    </div>
                  </motion.div>
                </motion.section>
              )}
            </AnimatePresence>
          </>
        )}

        {activeView === 'about' && (
          <section className="max-w-4xl mx-auto px-6 mb-24">
             <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-12">
                <div className="text-center">
                  <h2 className="text-4xl font-black tracking-tighter dark:text-white mb-6">About the project</h2>
                  <p className="text-xl text-on-surface-variant dark:text-slate-400">Protecting digital career identity in an age of deception.</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-16">
                   <div className="bg-surface-container-low dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800">
                      <h3 className="text-2xl font-black dark:text-white mb-6 flex items-center gap-3"><span className="material-symbols-outlined text-primary">psychology</span>The AI Engine</h3>
                      <p className="text-on-surface-variant dark:text-slate-400 leading-relaxed">FakeJobDetector uses a sophisticated Random Forest ensemble model trained on over 18,000 real and fraudulent job postings. It doesn't just look for words; it analyzes structural patterns, urgency levels, and data request triggers.</p>
                   </div>
                   <div className="bg-surface-container-low dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800">
                      <h3 className="text-2xl font-black dark:text-white mb-6 flex items-center gap-3"><span className="material-symbols-outlined text-primary">verified</span>Our Mission</h3>
                      <p className="text-on-surface-variant dark:text-slate-400 leading-relaxed">With remote work on the rise, phishing via job boards has increased by 300%. Our goal is to provide job seekers with a free, high-speed neural verification tool to validate opportunities before sharing sensitive personal data.</p>
                   </div>
                   <div className="bg-surface-container-low dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800">
                      <h3 className="text-2xl font-black dark:text-white mb-6 flex items-center gap-3"><span className="material-symbols-outlined text-primary">smart_toy</span>AI Career Assistant</h3>
                      <p className="text-on-surface-variant dark:text-slate-400 leading-relaxed">Powered by Google's Gemini AI, our integrated chatbot analyzes job requirements against your skills, creates personalized learning roadmaps, and guides you through interview preparation ΓÇö all within the same platform.</p>
                   </div>
                   <div className="bg-surface-container-low dark:bg-slate-900 p-10 rounded-[2.5rem] border border-slate-200 dark:border-slate-800">
                      <h3 className="text-2xl font-black dark:text-white mb-6 flex items-center gap-3"><span className="material-symbols-outlined text-primary">school</span>Roadmap Generator</h3>
                      <p className="text-on-surface-variant dark:text-slate-400 leading-relaxed">Missing skills for a role? Our AI creates week-by-week learning plans with curated resources from Coursera, YouTube, and official docs ΓÇö turning skill gaps into actionable study schedules.</p>
                   </div>
                </div>
             </motion.div>
          </section>
        )}


      </main>

      {/* ===== FLOATING CHATBOT ===== */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed bottom-6 right-6 w-[420px] h-[620px] bg-white dark:bg-slate-900 rounded-[2rem] shadow-2xl shadow-black/20 dark:shadow-black/50 border border-slate-200 dark:border-slate-800 flex flex-col overflow-hidden z-[100]"
          >
            {/* Chat Header */}
            <div className="bg-gradient-to-r from-primary to-indigo-700 p-5 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                  <span className="material-symbols-outlined text-white">smart_toy</span>
                </div>
                <div>
                  <h3 className="text-white font-black text-sm tracking-tight">Career Assistant</h3>
                  <p className="text-white/60 text-[10px] font-bold uppercase tracking-widest">Powered by Gemini AI</p>
                </div>
              </div>
              <button onClick={() => setChatOpen(false)} className="w-8 h-8 bg-white/10 hover:bg-white/20 rounded-lg flex items-center justify-center transition-colors">
                <span className="material-symbols-outlined text-white text-lg">close</span>
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 chat-scrollbar">
              {chatMessages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[85%] ${
                    msg.role === 'user' 
                      ? 'bg-primary text-white rounded-2xl rounded-br-md px-4 py-3' 
                      : 'bg-slate-100 dark:bg-slate-800 text-on-surface dark:text-slate-200 rounded-2xl rounded-bl-md px-4 py-3'
                  }`}>
                    {msg.role === 'user' ? (
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                    ) : (
                      <>
                        <div className="chat-markdown">{renderMarkdown(msg.content)}</div>
                        {msg.suggestions && msg.suggestions.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-700">
                            <p className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-2">Have more questions?</p>
                            <div className="flex flex-col gap-2">
                              {msg.suggestions.map((s, idx) => (
                                <button key={idx} onClick={() => handleSuggestionClick(s)} className="text-left px-3 py-2 text-xs font-bold bg-white dark:bg-slate-700 text-primary dark:text-primary-fixed-dim rounded-lg border border-primary/20 hover:bg-primary/5 transition-colors shadow-sm">{s}</button>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </motion.div>
              ))}

              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-2xl rounded-bl-md px-5 py-4">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Quick Suggestions */}
            {chatMessages.length <= 1 && (
              <div className="px-4 pb-2 flex flex-wrap gap-2 shrink-0">
                {chatSuggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestionClick(s)}
                    className="px-3 py-1.5 text-[11px] font-bold bg-primary/10 text-primary rounded-lg border border-primary/20 hover:bg-primary/20 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Chat Input */}
            <div className="p-4 border-t border-slate-200 dark:border-slate-800 shrink-0">
              <div className="flex gap-2">
                <textarea
                  ref={chatInputRef}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={handleChatKeyDown}
                  placeholder="Ask about eligibility, roadmaps, career advice..."
                  rows={1}
                  className="flex-1 px-4 py-3 bg-slate-100 dark:bg-slate-800 rounded-xl border-none focus:ring-2 focus:ring-primary/30 text-sm text-on-surface dark:text-white placeholder:text-slate-400 resize-none"
                />
                <button
                  onClick={handleChatSend}
                  disabled={chatLoading || !chatInput.trim()}
                  className="w-11 h-11 bg-primary hover:bg-primary/90 disabled:opacity-40 rounded-xl flex items-center justify-center transition-all shrink-0"
                >
                  <span className="material-symbols-outlined text-white text-lg">send</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Chat Button (when chat is closed) */}
      {!chatOpen && chatAvailable && (
        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-br from-violet-500 to-purple-700 rounded-2xl shadow-2xl shadow-purple-500/30 flex items-center justify-center hover:scale-110 active:scale-95 transition-all z-[100]"
        >
          <span className="material-symbols-outlined text-white text-2xl">smart_toy</span>
          {result && chatMessages.length === 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full animate-pulse"></span>
          )}
        </motion.button>
      )}

      <footer className="w-full py-16 bg-surface-container-lowest dark:bg-slate-950 border-t border-slate-200/50 dark:border-slate-800/80">
        <div className="max-w-7xl mx-auto px-6 lg:px-10">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
            <div className="col-span-1 md:col-span-2">
              <span className="text-2xl font-black tracking-tighter text-on-surface dark:text-white">FakeJob<span className="text-primary">Detector</span></span>
              <p className="mt-4 text-sm text-on-surface-variant dark:text-slate-400 max-w-sm leading-relaxed font-medium">Advanced open-source neural engine dedicated to identifying fraudulent job postings. Safeguard your career with real-time AI analysis.</p>
            </div>
            <div>
              <h5 className="font-bold text-[10px] uppercase tracking-[0.2em] text-on-surface dark:text-white mb-6">Explore</h5>
              <div className="flex flex-col gap-3">
                <button onClick={() => setActiveView('home')} className="text-sm text-on-surface-variant dark:text-slate-400 hover:text-primary transition-colors text-left font-medium">Analyzer Tool</button>
                <button onClick={() => setActiveView('about')} className="text-sm text-on-surface-variant dark:text-slate-400 hover:text-primary transition-colors text-left font-medium">About the Engine</button>
              </div>
            </div>
            <div>
              <h5 className="font-bold text-[10px] uppercase tracking-[0.2em] text-on-surface dark:text-white mb-6">Open Source</h5>
              <div className="flex flex-col gap-3">
                <a className="text-sm text-on-surface-variant dark:text-slate-400 hover:text-primary transition-colors font-medium" href="#">GitHub Repository</a>
                <a className="text-sm text-on-surface-variant dark:text-slate-400 hover:text-primary transition-colors font-medium" href="#">Kaggle Dataset</a>
                <a className="text-sm text-on-surface-variant dark:text-slate-400 hover:text-primary transition-colors font-medium" href="#">API Access</a>
              </div>
            </div>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-center gap-4 pt-10 border-t border-slate-200/50 dark:border-slate-800/50">
            <p className="text-[10px] font-bold text-on-surface-variant dark:text-slate-500 uppercase tracking-widest">&copy; {new Date().getFullYear()} FakeJobDetector All rights reserved.</p>
            <div className="flex gap-6">
              <a className="text-[10px] font-bold text-on-surface-variant dark:text-slate-500 hover:text-primary transition-colors uppercase tracking-widest" href="#">Privacy Policy</a>
              <a className="text-[10px] font-bold text-on-surface-variant dark:text-slate-500 hover:text-primary transition-colors uppercase tracking-widest" href="#">Terms of Service</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
