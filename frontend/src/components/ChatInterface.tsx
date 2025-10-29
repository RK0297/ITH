import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User } from "lucide-react";
import { useState } from "react";

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
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

  const handleSend = () => {
    if (inputValue.trim()) {
      const userMessage: Message = {
        id: messages.length + 1,
        text: inputValue,
        sender: "user",
      };

      setMessages([...messages, userMessage]);
      setInputValue("");

      // Simulate bot response
      setTimeout(() => {
        const botMessage: Message = {
          id: messages.length + 2,
          text: "I'm here to help with your legal queries. Please note that this is for informational purposes only and does not constitute legal advice.",
          sender: "bot",
        };
        setMessages((prev) => [...prev, botMessage]);
      }, 1000);
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
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea className="h-[500px] p-6 bg-gradient-subtle">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
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
                <p className="text-sm leading-relaxed">{message.text}</p>
              </div>
              {message.sender === "user" && (
                <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-accent-foreground" />
                </div>
              )}
            </div>
          ))}
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
          />
          <Button onClick={handleSend} variant="secondary" size="icon">
            <Send className="w-4 h-4" />
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
