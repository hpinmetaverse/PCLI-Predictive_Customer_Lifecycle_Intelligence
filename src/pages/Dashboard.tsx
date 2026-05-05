import hazard from "@/assets/hazard.png";
import model1 from "@/assets/model_1.png";
import featImp from "@/assets/model_feat_imp.png";
import monthly from "@/assets/monthlycharges.png";
import permImp from "@/assets/perm_imp.png";
import shap from "@/assets/shap.png";
import shap1 from "@/assets/shap1.png";
import survival from "@/assets/survival.png";
import tenure from "@/assets/tenure-churn.png"
import { Header } from "@/components/Header";
import { useAuth } from "@/contexts/AuthContext";
import { Navigate } from "react-router-dom";

type Item = { src: string; title: string; desc: string };

const items: Item[] = [
{ src: model1, title: "Confusion Matrix & ROC", desc: "Model performance with AUC of 0.848." },
  { src: featImp, title: "Feature Importances", desc: "Top drivers ranked by coefficient magnitude." },
  { src: hazard, title: "Cumulative Hazard", desc: "Hazard accumulation over tenure." },
  { src: survival, title: "Survival Probability", desc: "Survival function over tenure." },
  { src: shap, title: "SHAP Explanation", desc: "Feature contributions pushing prediction higher / lower." },
  { src: shap1, title: "SHAP Single Prediction", desc: "Per-customer breakdown with values." },
  { src: tenure, title: "Tenure vs Churn", desc: "Churn distribution across tenure months." },
  { src: monthly, title: "Monthly Charges Density", desc: "Churn vs Not Churn by monthly charges." },
  { src: permImp, title: "Permutation Importance", desc: "Weight ± std for each feature." },
];

const ModelAnalytics = () => {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (!user) return <Navigate to="/auth" replace />;
  return (
    <div className="min-h-screen"> 
<Header/>
    <section className="container py-16">
      <header className="mb-10">
        <h1
          className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent"
          style={{ backgroundImage: "var(--gradient-primary)" }}
        >
          Model Analytics
        </h1>
        <p className="mt-3 text-white/60 max-w-2xl">
          Detailed model evaluation, explainability, and survival analysis for the churn prediction model.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        {items.map((it) => (
          <article
            key={it.title}
            className="rounded-2xl border border-white/10 p-5 shadow-xl"
            style={{ background: "var(--gradient-card)" }}
          >
            <h3 className="text-lg font-semibold text-white">{it.title}</h3>
            <p className="text-sm text-white/60 mb-4">{it.desc}</p>
            <div className="rounded-lg overflow-hidden bg-white/5 border border-white/10">
              <img src={it.src} alt={it.title} loading="lazy" className="w-full h-auto" />
            </div>
          </article>
        ))}
      </div>
    </section>
    </div>
  );
};

export default ModelAnalytics;