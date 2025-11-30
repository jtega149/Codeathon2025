import { Sparkles, Lightbulb, CheckCircle2, AlertCircle, MessageSquare, Send, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import { useDashboard } from "../context/DashboardContext";

interface Suggestion {
  title: string;
  description: string;
  impact: "high" | "medium" | "low";
  category: string;
  estimatedReduction: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export function AISuggestions() {
  const { currentEntityId, currentRecord } = useDashboard();
  const [activeTab, setActiveTab] = useState<"recommendations" | "chat">("recommendations");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I'm your AI sustainability advisor. I can help you understand your emissions data, suggest improvement strategies, and answer questions about your sustainability goals. How can I assist you today?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);

  // Fetch recommendations when entity changes
  useEffect(() => {
    if (!currentEntityId) return;
    
    setLoadingRecommendations(true);
    fetch(`http://localhost:8000/ai/recommendations/${currentEntityId}`)
      .then((res) => res.json())
      .then((data) => {
        setSuggestions(data.recommendations || []);
        setLoadingRecommendations(false);
      })
      .catch((err) => {
        console.error("Error fetching recommendations:", err);
        setLoadingRecommendations(false);
      });
  }, [currentEntityId]);

  // Reset chat when entity changes
  useEffect(() => {
    setChatMessages([
      {
        role: "assistant",
        content: "Hello! I'm your AI sustainability advisor. I can help you understand your emissions data, suggest improvement strategies, and answer questions about your sustainability goals. How can I assist you today?",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  }, [currentEntityId]);

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case "high":
        return "bg-red-100 text-red-700 border-red-200";
      case "medium":
        return "bg-amber-100 text-amber-700 border-amber-200";
      case "low":
        return "bg-blue-100 text-blue-700 border-blue-200";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || !currentEntityId || loadingChat) return;

    const userMessage = chatInput.trim();
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message to chat
    const newUserMessage: ChatMessage = {
      role: "user",
      content: userMessage,
      timestamp
    };
    setChatMessages((prev) => [...prev, newUserMessage]);
    setChatInput("");
    setLoadingChat(true);

    // Prepare conversation history for API
    const conversationHistory = chatMessages.map((msg) => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      const response = await fetch("http://localhost:8000/ai/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          entity_id: currentEntityId,
          message: userMessage,
          conversation_history: conversationHistory,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setChatMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "I'm sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setChatMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoadingChat(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-slate-900">AI-Powered Recommendations</h2>
              <p className="text-slate-500 text-sm">Personalized suggestions to improve your sustainability score</p>
              <p className="text-slate-500 text-sm">(Using results from our custom model; Ai Chat using another ai model)</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-violet-50 border border-violet-200 rounded-lg">
            <AlertCircle className="w-4 h-4 text-violet-600" />
            <span className="text-violet-700 text-sm">Action Required</span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mt-6">
          <button
            onClick={() => setActiveTab("recommendations")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeTab === "recommendations"
                ? "bg-violet-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            <Lightbulb className="w-4 h-4" />
            <span>Recommendations</span>
          </button>
          <button
            onClick={() => setActiveTab("chat")}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeTab === "chat"
                ? "bg-violet-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>AI Assistant</span>
          </button>
        </div>
      </div>

      {/* Recommendations Tab Content */}
      {activeTab === "recommendations" && (
        <div className="p-6">
          {loadingRecommendations ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-violet-600 animate-spin" />
              <span className="ml-3 text-slate-600">Loading AI recommendations...</span>
            </div>
          ) : suggestions.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              No recommendations available at this time.
            </div>
          ) : (
            <div className="space-y-4">
              {suggestions.map((suggestion, index) => (
              <div
                key={index}
                className="group border border-slate-200 rounded-lg p-5 hover:border-violet-300 hover:shadow-md transition-all duration-200"
              >
                <div className="flex items-start gap-4">
                  <div className="mt-1 w-10 h-10 rounded-lg bg-gradient-to-br from-violet-100 to-purple-100 flex items-center justify-center flex-shrink-0 group-hover:from-violet-200 group-hover:to-purple-200 transition-all">
                    <Lightbulb className="w-5 h-5 text-violet-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <h3 className="text-slate-900">{suggestion.title}</h3>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={`px-2.5 py-1 border rounded text-xs ${getImpactColor(suggestion.impact)}`}>
                          {suggestion.impact.toUpperCase()} IMPACT
                        </span>
                        <span className="px-2.5 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                          {suggestion.category}
                        </span>
                      </div>
                    </div>
                    <p className="text-slate-600 mb-3">{suggestion.description}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 border border-emerald-200 rounded">
                          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                          <span className="text-emerald-700 text-sm">
                            Estimated Impact: <span>{suggestion.estimatedReduction}</span>
                          </span>
                        </div>
                      </div>
                      <button className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm">
                        View Details
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Chat Tab Content */}
      {activeTab === "chat" && (
        <div className="flex flex-col h-[600px]">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {chatMessages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[70%] ${
                    message.role === "user"
                      ? "bg-violet-600 text-white rounded-lg rounded-tr-sm"
                      : "bg-slate-100 text-slate-900 rounded-lg rounded-tl-sm"
                  } px-4 py-3`}
                >
                  <p className="text-sm whitespace-pre-line">{message.content}</p>
                  <p
                    className={`text-xs mt-2 ${
                      message.role === "user" ? "text-violet-200" : "text-slate-500"
                    }`}
                  >
                    {message.timestamp}
                  </p>
                </div>
              </div>
            ))}
            {loadingChat && (
              <div className="flex justify-start">
                <div className="bg-slate-100 text-slate-900 rounded-lg rounded-tl-sm px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-slate-500" />
                    <span className="text-sm text-slate-500">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Chat Input */}
          <div className="border-t border-slate-100 p-6">
            <div className="flex gap-3">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                placeholder="Ask about your emissions, request recommendations, or get sustainability advice..."
                className="flex-1 px-4 py-3 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent text-slate-900 placeholder:text-slate-400"
              />
              <button
                onClick={handleSendMessage}
                disabled={loadingChat || !chatInput.trim() || !currentEntityId}
                className="px-6 py-3 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loadingChat ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                <span>Send</span>
              </button>
            </div>
            {!currentEntityId && (
              <p className="text-slate-400 text-xs mt-2">
                Please select an entity to start chatting.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}