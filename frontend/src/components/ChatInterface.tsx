import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, Loader2, AlertCircle } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { apiService, ChatResponse } from "@/services/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card } from "@/components/ui/card";

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
  sources?: ChatResponse['sources'];
  timestamp?: string;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm कानून. I can help you solve your legal queries and provide guidance on legal matters. How can I assist you today?",
      sender: "bot",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  const handleSend = async () => {
    if (inputValue.trim() && !isLoading) {
      const userMessage: Message = {
        id: messages.length + 1,
        text: inputValue,
        sender: "user",
        timestamp: new Date().toISOString(),
      };

      setMessages([...messages, userMessage]);
      setInputValue("");
      setIsLoading(true);
      setError(null);

      try {
        // Call the backend API
        const response = await apiService.sendChatMessage({
          query: inputValue,
          conversation_id: conversationId || undefined,
          top_k: 5,
        });

        // Update conversation ID if this is the first message
        if (!conversationId && response.conversation_id) {
          setConversationId(response.conversation_id);
        }

        const botMessage: Message = {
          id: messages.length + 2,
          text: response.response,
          sender: "bot",
          sources: response.sources,
          timestamp: response.timestamp,
        };
        
        setMessages((prev) => [...prev, botMessage]);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An error occurred";
        setError(errorMessage);
        
        const errorBotMessage: Message = {
          id: messages.length + 2,
          text: "I apologize, but I encountered an error processing your request. Please make sure the backend server is running and try again.",
          sender: "bot",
          timestamp: new Date().toISOString(),
        };
        
        setMessages((prev) => [...prev, errorBotMessage]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-card rounded-xl shadow-elegant border border-border overflow-hidden">
      {/* Chat Header */}
      <div className="bg-gradient-hero p-4 flex items-center gap-3">
        <Bot className="w-10 h-10 text-secondary" />
        <div>
          <h3 className="font-semibold text-primary-foreground">कानून</h3>
          <p className="text-xs text-primary-foreground/80">AI Legal Assistant</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive" className="m-4 mb-0">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Messages Area */}
      <ScrollArea ref={scrollAreaRef} className="h-[500px] p-6 bg-gradient-subtle">
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id}>
              <div
                className={`flex gap-3 ${
                  message.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.sender === "bot" && (
                  <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-secondary-foreground" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] p-4 rounded-lg ${
                    message.sender === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-card text-card-foreground shadow-card border border-border"
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                </div>
                {message.sender === "user" && (
                  <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-accent-foreground" />
                  </div>
                )}
              </div>

              {/* Sources Section */}
              {message.sources && message.sources.length > 0 && (
                <div className="ml-11 mt-2 space-y-2">
                  <p className="text-xs font-semibold text-muted-foreground">Sources:</p>
                  {message.sources.map((source, idx) => (
                    <Card key={idx} className="p-3 text-xs space-y-1">
                      <div className="flex items-start justify-between">
                        <p className="font-semibold">{source.title}</p>
                        <span className="text-muted-foreground text-xs">{source.category}</span>
                      </div>
                      <p className="text-muted-foreground line-clamp-2">{source.preview}</p>
                      {source.url && (
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          View Source
                        </a>
                      )}
                    </Card>
                  ))}
                </div>
              )}
            </div>
          ))}
          
          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-secondary-foreground" />
              </div>
              <div className="bg-card text-card-foreground shadow-card border border-border p-4 rounded-lg">
                <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 bg-card border-t border-border">
        <div className="flex gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask your legal queries here..."
            className="flex-1 bg-background"
            disabled={isLoading}
          />
          <Button 
            onClick={handleSend} 
            variant="secondary" 
            size="icon"
            disabled={isLoading || !inputValue.trim()}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          This chatbot provides general legal information only and is not a substitute for professional legal advice.
        </p>
      </div>
    </div>
  );
};

export default ChatInterface;
