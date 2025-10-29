import Navigation from "@/components/Navigation";
import ChatInterface from "@/components/ChatInterface";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Scale, BookOpen, Shield } from "lucide-react";
import lawScalesPattern from "@/assets/law-scales-pattern.png";

const Index = () => {
  const scrollToChat = () => {
    const element = document.getElementById("chat");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen">
      <Navigation />

      {/* Hero Section */}
      <section
        id="home"
        className="pt-32 pb-20 px-4 bg-gradient-hero relative overflow-hidden"
        style={{
          backgroundImage: `url(${lawScalesPattern})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
        }}
      >
        <div className="absolute inset-0 bg-primary/90"></div>
        <div className="container mx-auto relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex justify-center mb-6">
              <Scale className="w-20 h-20 text-secondary" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold text-primary-foreground mb-6">
              कानून
            </h1>
            <p className="text-xl text-primary-foreground/90 mb-4">
              AI Legal Advising Chatbot 
            </p>
            <p className="text-lg text-primary-foreground/80 mb-8 max-w-2xl mx-auto">
              Get advanced legal advise that are accessible to everyone and aims 
              to enhance the legal literacy of the people and make it accessible to everyone.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button onClick={scrollToChat} variant="gold" size="lg">
                Start Chat
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Chat Section */}
      <section id="chat" className="py-20 px-4 bg-gradient-subtle">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-foreground mb-4">Ask Your Legal Questions</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              
            </p>
          </div>
          <ChatInterface />
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-4 bg-background">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-bold text-foreground mb-8 text-center">
              About कानून
            </h2>
            <p className="text-lg text-muted-foreground mb-12 text-center max-w-3xl mx-auto">
              कानून is an innovative AI-powered legal advising chatbot designed to democratize access to legal knowledge and enhance legal literacy. Our platform leverages artificial intelligence to provide accurate, comprehensive, and accessible legal guidance for your legal queries.
            </p>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center p-6 rounded-xl bg-card shadow-card border border-border">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                  <BookOpen className="w-8 h-8 text-primary" />
                </div>
                <h3 className="font-semibold text-lg mb-2 text-foreground">
                  Comprehensive Legal Knowledge
                </h3>
                <p className="text-muted-foreground text-sm">
                  Get expert guidance on a wide range of legal topics, fundamental rights, constitutional amendments, and contemporary legal frameworks.
                </p>
              </div>

              <div className="text-center p-6 rounded-xl bg-card shadow-card border border-border">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-secondary/10 flex items-center justify-center">
                  <Scale className="w-8 h-8 text-secondary" />
                </div>
                <h3 className="font-semibold text-lg mb-2 text-foreground">AI-Powered Analysis</h3>
                <p className="text-muted-foreground text-sm">
                  Get instant, intelligent responses to your legal queries. Our AI analyzes legal provisions to provide contextual guidance for your questions.
                </p>
              </div>

              <div className="text-center p-6 rounded-xl bg-card shadow-card border border-border">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-accent/10 flex items-center justify-center">
                  <Shield className="w-8 h-8 text-accent" />
                </div>
                <h3 className="font-semibold text-lg mb-2 text-foreground">Educational Resource</h3>
                <p className="text-muted-foreground text-sm">
                  Enhance your legal literacy through interactive conversations with our knowledgeable AI assistant.
                </p>
              </div>
            </div>
            <div className="mt-12 p-8 rounded-xl bg-accent/5 border-2 border-accent/20">
              <p className="text-center text-muted-foreground">
                <strong className="text-foreground">Note:</strong> कानून provides general legal information for educational purposes and does not constitute formal legal advice. For specific legal matters, please consult with a qualified legal professional.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
