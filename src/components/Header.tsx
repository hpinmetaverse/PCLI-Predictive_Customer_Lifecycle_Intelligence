import { Link, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import {  LogOut, LayoutDashboard, Activity } from "lucide-react";

export const Header = () => {
  const { user, isAdmin, signOut } = useAuth();
  const loc = useLocation();

  return (
    <header className="glass sticky top-0 z-50 border-b border-border">
      <div className="container flex items-center justify-between h-16">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-xl font-bold gradient-text">ChurnIQ</span>
        </Link>

        <nav className="flex items-center gap-2">
          {user && (
            <>
             <Link to="/dash">
                  <Button variant={loc.pathname === "/dash" ? "secondary" : "ghost"} size="sm">
                    <LayoutDashboard className="w-4 h-4 mr-2" />Dashboard
                  </Button>
                </Link>
              <Button variant="ghost" size="sm" onClick={signOut}>
                <LogOut className="w-4 h-4 mr-2" />Sign out
              </Button>
            </>
          )}
          {!user && (
            <Link to="/auth"><Button size="sm">Sign in</Button></Link>
          )}
        </nav>
      </div>
    </header>
  );
};
