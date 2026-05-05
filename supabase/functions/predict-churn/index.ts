import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }

    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_ANON_KEY")!,
      { global: { headers: { Authorization: authHeader } } }
    );
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }

    const inputs = await req.json();
    const apiUrl = Deno.env.get("PREDICTION_API_URL");
    if (!apiUrl) {
      return new Response(JSON.stringify({ error: "PREDICTION_API_URL not configured" }), { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }

    let result: any;
    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict_json`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(inputs),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Model API returned ${response.status}: ${text.slice(0, 200)}`);
      }
      result = await response.json();
    } catch (e) {
      return new Response(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }), { status: 502, headers: { ...corsHeaders, "Content-Type": "application/json" } });
    }

    // Persist
    const { data: saved, error: insErr } = await supabase.from("predictions").insert({
      user_id: user.id,
      inputs,
      churn_probability: Number(result.churn_probability ?? 0),
      lifetime_value: Number(result.lifetime_value ?? 0),
      shap_image: result.shap_image ?? null,
      gauge_image: result.gauge_image ?? null,
      hazard_image: result.hazard_image ?? null,
      survival_image: result.survival_image ?? null,
    }).select().single();

    if (insErr) console.error("Insert error:", insErr);

    return new Response(JSON.stringify({ ...result, id: saved?.id }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("predict-churn error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } });
  }
});
