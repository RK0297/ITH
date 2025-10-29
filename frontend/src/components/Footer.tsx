import { Scale, Github, Linkedin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-gradient-hero text-primary-foreground py-12 mt-20">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Scale className="w-6 h-6 text-secondary" />
              <h3 className="text-lg font-bold">कानून</h3>
            </div>
            <p className="text-sm text-primary-foreground/80">
              Your trusted companion for answering your legal queries.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm text-primary-foreground/80">
              <li>
                <a href="#about" className="hover:text-secondary transition-smooth">
                  About Us
                </a>
              </li>
            </ul>
          </div>

          {/* Contact & Social */}
          <div>
            <h4 className="font-semibold mb-4">Connect With Us</h4>
            <div className="flex gap-4 mb-4">
              <a
                href="https://github.com/RK0297"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-full bg-primary-foreground/10 hover:bg-secondary flex items-center justify-center transition-smooth"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://www.linkedin.com/in/radhakrishna-bharuka?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-full bg-primary-foreground/10 hover:bg-secondary flex items-center justify-center transition-smooth"
              >
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        {/* Legal Disclaimer */}
        <div className="mt-12 pt-8 border-t border-primary-foreground/20">
          <p className="text-xs text-primary-foreground/60 text-center mb-4">
            <strong>Legal Disclaimer:</strong> This AI chatbot provides general legal information
            for educational purposes only. It does not constitute legal advice and should not be relied upon as such. For specific legal
            matters, please consult a qualified legal professional.
          </p>
          <p className="text-xs text-primary-foreground/60 text-center">
            © {new Date().getFullYear()} कानून. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
