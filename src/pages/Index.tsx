import { useState } from "react";
import { Navigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Header } from "@/components/Header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

type FormState = {
  gender: string;
  SeniorCitizen: boolean;
  Partner: boolean;
  Dependents: boolean;
  PaperlessBilling: boolean;
  PhoneService: boolean;
  MultipleLines: boolean;
  OnlineSecurity: boolean;
  OnlineBackup: boolean;
  DeviceProtection: boolean;
  TechSupport: boolean;
  StreamingTV: boolean;
  StreamingMovies: boolean;
  MonthlyCharges: number;
  Tenure: number;
  InternetService: string;
  Contract: string;
  PaymentMethod: string;
};

const defaultState: FormState = {
  gender: "1", SeniorCitizen: false, Partner: false, Dependents: false,
  PaperlessBilling: false, PhoneService: true, MultipleLines: false,
  OnlineSecurity: false, OnlineBackup: false, DeviceProtection: false, TechSupport: false,
  StreamingTV: false, StreamingMovies: false,
  MonthlyCharges: 70, Tenure: 12, InternetService: "1", Contract: "0", PaymentMethod: "0",
};

const checkboxes: Array<[keyof FormState, string]> = [
  ["SeniorCitizen", "Senior Citizen"],
  ["Partner", "Has a partner"],
  ["Dependents", "Has dependents"],
  ["PaperlessBilling", "Paperless Billing"],
  ["PhoneService", "Phone Service"],
  ["MultipleLines", "Multiple Lines"],
  ["OnlineSecurity", "Online Security"],
  ["OnlineBackup", "Online Backup"],
  ["DeviceProtection", "Device Protection"],
  ["TechSupport", "Tech Support"],
  ["StreamingTV", "Streaming TV"],
  ["StreamingMovies", "Streaming Movies"],
];

const Index = () => {
  const { user, loading } = useAuth();
  const [form, setForm] = useState<FormState>(defaultState);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  if (loading) return null;
  if (!user) return <Navigate to="/auth" replace />;

  const update = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm((p) => ({ ...p, [k]: v }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setResult(null);
    const { data, error } = await supabase.functions.invoke("predict-churn", { body: form });
    setBusy(false);
    if (error || data?.error) {
      toast.error(data?.error || error?.message || "Prediction failed");
      return;
    }
    setResult(data);
    toast.success("Prediction complete");
  };

  const churnPct = result ? Math.round(Number(result.churn_probability) * 100) : 0;
  const churnTier = churnPct < 25 ? { label: "LOW", color: "text-success" }
    : churnPct < 50 ? { label: "MEDIUM", color: "text-primary" }
    : churnPct < 75 ? { label: "HIGH", color: "text-warning" }
    : { label: "EXTREME", color: "text-danger" };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="container py-10 space-y-8">
        <section className="text-center space-y-3 max-w-3xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold gradient-text leading-tight">Customer Churn Prediction</h1>
          <p className="text-muted-foreground text-lg">Explainable AI for churn risk, lifetime value, and retention strategy.</p>
        </section>

        <Card className="card-elevated p-6 md:p-8">
          <form onSubmit={submit} className="space-y-8">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">Service flags</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {checkboxes.map(([k, label]) => (
                  <label key={k} className="flex items-center gap-2 p-3 rounded-lg bg-secondary/50 hover:bg-secondary transition cursor-pointer">
                    <Checkbox checked={form[k] as boolean} onCheckedChange={(v) => update(k, !!v as any)} />
                    <span className="text-sm">{label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="space-y-2">
                <Label>Gender</Label>
                <Select value={form.gender} onValueChange={(v) => update("gender", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="1">Male</SelectItem><SelectItem value="0">Female</SelectItem></SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Internet</Label>
                <Select value={form.InternetService} onValueChange={(v) => update("InternetService", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">DSL</SelectItem>
                    <SelectItem value="2">Fiber optic</SelectItem>
                    <SelectItem value="0">No internet</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Contract</Label>
                <Select value={form.Contract} onValueChange={(v) => update("Contract", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">Month-to-Month</SelectItem>
                    <SelectItem value="1">One Year</SelectItem>
                    <SelectItem value="2">Two Year</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Payment</Label>
                <Select value={form.PaymentMethod} onValueChange={(v) => update("PaymentMethod", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">Electronic Check</SelectItem>
                    <SelectItem value="1">Mailed Check</SelectItem>
                    <SelectItem value="2">Bank Transfer</SelectItem>
                    <SelectItem value="3">Credit Card</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Monthly Charges ($)</Label>
                <Input type="number" step="0.01" value={form.MonthlyCharges} onChange={(e) => update("MonthlyCharges", parseFloat(e.target.value) || 0)} />
              </div>
              <div className="space-y-2">
                <Label>Tenure (months)</Label>
                <Input type="number" value={form.Tenure} onChange={(e) => update("Tenure", parseInt(e.target.value) || 0)} />
              </div>
            </div>

            <Button type="submit" disabled={busy} size="lg" className="w-full md:w-auto">
              {busy ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Predicting…</> : <>Run Prediction</>}
            </Button>
          </form>
        </Card>

        {result && (
          <section className="space-y-6">
            <Card className="card-elevated p-6 text-center">
              <p className="text-2xl md:text-3xl font-semibold">
                Churn probability is{" "}
                <span className={churnTier.color}>{Number(result.churn_probability).toFixed(2)}</span>{" "}
                and Expected Life Time Value is{" "}
                <span className="gradient-text">${Number(result.lifetime_value).toLocaleString()}</span>
              </p>
              <p className={`mt-2 text-sm font-semibold ${churnTier.color}`}>{churnTier.label} RISK</p>
            </Card>

            <div className="grid md:grid-cols-3 gap-4">
              {result.hazard_image && (
                <Card className="card-elevated p-3">
                  <img src={`data:image/png;base64,${result.hazard_image}`} alt="Cumulative Hazard Over Time" className="w-full rounded bg-white" />
                </Card>
              )}
              {result.gauge_image && (
                <Card className="card-elevated p-3">
                  <img src={`data:image/png;base64,${result.gauge_image}`} alt="Churn Probability Gauge" className="w-full rounded bg-white" />
                </Card>
              )}
              {result.survival_image && (
                <Card className="card-elevated p-3">
                  <img src={`data:image/png;base64,${result.survival_image}`} alt="Survival Probability Over Time" className="w-full rounded bg-white" />
                </Card>
              )}
            </div>

            {result.shap_image && (
              <Card className="card-elevated p-3">
                <img src={`data:image/png;base64,${result.shap_image}`} alt="SHAP Explanation" className="w-full rounded bg-white" />
              </Card>
            )}
          </section>
        )}
      </main>
    </div>
  );
};

export default Index;
