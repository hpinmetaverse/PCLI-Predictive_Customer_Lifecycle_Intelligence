import { useState } from "react";
import { Navigate, useNavigate, Link } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { lovable } from "@/integrations/lovable";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Eye, EyeOff } from "lucide-react";
import { Loader2, TrendingUp } from "lucide-react";

const Auth = () => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [showPassword, setShowPassword] = useState(false);


  if (loading) return null;
  if (user) return <Navigate to="/" replace />;

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (error) toast.error(error.message);
    else {
      toast.success("Welcome back");
      setEmail("");
      setPassword("");
      navigate("/");
    }
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    const { error } = await supabase.auth.signUp({
      email, password,
      options: {
        emailRedirectTo: `${window.location.origin}/`,
        data: { full_name: fullName },
      },
    });
    setBusy(false);
    if (error) toast.error(error.message);
    else {
      toast.success("Account created - check your email to verify.");
      setEmail("");
      setPassword("");
      setFullName("");
    }
  };



  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        <Link to="/" className="flex items-center justify-center gap-2 mb-2">
     
          <span className="text-2xl font-bold gradient-text">ChurnIQ</span>
        </Link>

        <Card className="card-elevated p-6">
          <Tabs defaultValue="signin">
            <TabsList className="grid grid-cols-2 w-full mb-6">
              <TabsTrigger value="signin">Sign in</TabsTrigger>
              <TabsTrigger value="signup">Sign up</TabsTrigger>
            </TabsList>

            <TabsContent value="signin">
              <form onSubmit={handleSignIn} className="space-y-4">
                <div><Label>Email</Label><Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></div>

                <div className="relative">
  <Label>Password</Label>
  <Input
    type={showPassword ? "text" : "password"}
    value={password}
    onChange={(e) => setPassword(e.target.value)}
    required
  />
  <button
    type="button"
    onClick={() => setShowPassword((prev) => !prev)}
    className="absolute right-3 top-9 text-sm text-muted-foreground"
  >
    {showPassword ? (     <EyeOff className="w-4 h-4" />) : (     <Eye className="w-4 h-4" />)}
  </button>
</div>
                <Button type="submit" disabled={busy} className="w-full">{busy && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}Sign in</Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <form onSubmit={handleSignUp} className="space-y-4">
                <div><Label>Full name</Label><Input value={fullName} onChange={(e) => setFullName(e.target.value)} required /></div>
                <div><Label>Email</Label><Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></div>
                <div className="relative">
  <Label>Password</Label>
  <Input
    type={showPassword ? "text" : "password"}
    value={password}
    onChange={(e) => setPassword(e.target.value)}
    required
  />
  <button
    type="button"
    onClick={() => setShowPassword((prev) => !prev)}
    className="absolute right-3 top-9 text-sm text-muted-foreground"
  >
    {showPassword ? (     <EyeOff className="w-4 h-4" />) : (     <Eye className="w-4 h-4" />)}
  </button>
</div>
                <Button type="submit" disabled={busy} className="w-full">{busy && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}Create account</Button>
              </form>
            </TabsContent>
          </Tabs>

         
        </Card>
      </div>
    </div>
  );
};

export default Auth;
