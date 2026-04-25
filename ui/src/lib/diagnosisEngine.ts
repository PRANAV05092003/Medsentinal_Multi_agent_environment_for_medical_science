/**
 * MedSentinel Diagnosis Engine
 * ==============================
 *
 * Calls the real Python backend (api_server.py) instead of running mock logic.
 *
 * Backend: http://localhost:8000/diagnose  (uvicorn api_server:app)
 * UI:      http://localhost:8080           (vite dev server)
 *
 * Falls back to the mock engine if the backend is unreachable,
 * so the UI still works standalone without a running server.
 */

export interface PatientForm {
  patientId: string;
  age: number;
  gender: "Male" | "Female" | "Other";
  chiefComplaint: string;
  vitals: {
    bp_systolic: number;
    bp_diastolic: number;
    heart_rate: number;
    temperature: number;
    spo2: number;
    respiratory_rate: number;
  };
  labs: {
    troponin_i: number;
    bnp: number;
    creatinine: number;
    glucose: number;
    wbc: number;
    hemoglobin: number;
  };
  allergies: string[];
  medications: string[];
  safeDrugs?: string[];
  unsafeDrugs?: string[];
  groundTruthDiagnosis?: string;
  driftEnabled: boolean;
  driftProbability: number;
  seed: number;
}

export interface DriftRename {
  section: "vitals" | "labs";
  original: string;
  renamed: string;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  verdict: "safe" | "unsafe" | "warning" | "drift";
}

export interface DiagnosisResult {
  drift: { occurred: boolean; renames: DriftRename[] };
  doctor: {
    icd10: string;
    diagnosisName: string;
    drug: string;
    dose: string;
    confidence: number;
    schemaDriftHandled: boolean;
    reasoning: string;
  };
  toolCalls: ToolCall[];
  auditor: { safe: boolean; flags: string[]; notes: string[] };
  reward: {
    total: number;
    components: { label: string; value: number }[];
  };
  cvl?: {
    verified: boolean;
    changes: string[];
    riskFlags: string[];
    notes: string;
    fallback: boolean;
  };
  _source?: "backend" | "mock";
}

export const REWARD_WEIGHTS = {
  diagnosis: 0.4,
  drug: 0.2,
  dose: 0.2,
  drift: 0.1,
  auditor: 0.1,
  allergy: -0.5,
  wrongDx: -0.3,
};

// ─── Backend URL ──────────────────────────────────────────────────────────────
// In development: vite proxy forwards /api/* to localhost:8000
// In production:  set VITE_API_URL env var to your backend URL
const API_BASE =
  (import.meta.env.VITE_API_URL as string) ||
  (import.meta.env.DEV ? "/api" : "http://localhost:8000");

// ─── Backend call ─────────────────────────────────────────────────────────────

async function callBackend(form: PatientForm): Promise<DiagnosisResult> {
  const payload = {
    patientId: form.patientId || "P-001",
    age: form.age,
    gender: form.gender,
    chiefComplaint: form.chiefComplaint,
    vitals: form.vitals,
    labs: form.labs,
    allergies: form.allergies,
    medications: form.medications,
    safeDrugs: form.safeDrugs ?? null,
    unsafeDrugs: form.unsafeDrugs ?? null,
    groundTruthDiagnosis: form.groundTruthDiagnosis ?? null,
    driftEnabled: form.driftEnabled,
    driftProbability: form.driftProbability,
    seed: form.seed,
  };

  const response = await fetch(`${API_BASE}/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(30_000), // 30s timeout
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Backend error ${response.status}: ${err}`);
  }

  const data = await response.json();

  // Normalize backend response to DiagnosisResult shape
  return {
    drift: {
      occurred: Boolean(data.drift?.occurred),
      renames: (data.drift?.renames ?? []).map((r: any) => ({
        section: r.section as "vitals" | "labs",
        original: String(r.original),
        renamed: String(r.renamed),
      })),
    },
    doctor: {
      icd10: String(data.doctor?.icd10 ?? ""),
      diagnosisName: String(data.doctor?.diagnosisName ?? ""),
      drug: String(data.doctor?.drug ?? ""),
      dose: String(data.doctor?.dose ?? "—"),
      confidence: Number(data.doctor?.confidence ?? 0),
      schemaDriftHandled: Boolean(data.doctor?.schemaDriftHandled),
      reasoning: String(data.doctor?.reasoning ?? ""),
    },
    toolCalls: (data.toolCalls ?? []).map((t: any) => ({
      name: String(t.name),
      input: t.input ?? {},
      output: t.output ?? {},
      verdict: (t.verdict as ToolCall["verdict"]) ?? "safe",
    })),
    auditor: {
      safe: Boolean(data.auditor?.safe),
      flags: (data.auditor?.flags ?? []).map(String),
      notes: (data.auditor?.notes ?? []).map(String),
    },
    reward: {
      total: Number(data.reward?.total ?? 0),
      components: (data.reward?.components ?? []).map((c: any) => ({
        label: String(c.label),
        value: Number(c.value),
      })),
    },
    cvl: data.cvl
      ? {
          verified: Boolean(data.cvl.verified),
          changes: (data.cvl.changes ?? []).map(String),
          riskFlags: (data.cvl.riskFlags ?? []).map(String),
          notes: String(data.cvl.notes ?? ""),
          fallback: Boolean(data.cvl.fallback),
        }
      : undefined,
    _source: "backend",
  };
}

// ─── Mock fallback (used when backend is unreachable) ─────────────────────────

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DRUG_DOSES: Record<string, number> = {
  nitroglycerin: 0.4, aspirin: 325, heparin: 5000, morphine: 4,
  clopidogrel: 75, metoprolol: 25, insulin_regular: 10,
  albuterol: 2.5, ceftriaxone: 2000,
};

const DRUG_DOSE_DISPLAY: Record<string, string> = {
  nitroglycerin: "0.4 mg sublingual", aspirin: "325 mg PO", heparin: "5000 units IV",
  morphine: "4 mg IV", clopidogrel: "75 mg PO", metoprolol: "25 mg PO",
  insulin_regular: "10 units IV", albuterol: "2.5 mg nebulized", ceftriaxone: "2 g IV",
};

async function runMockDiagnosis(form: PatientForm): Promise<DiagnosisResult> {
  await new Promise((r) => setTimeout(r, 1200 + Math.random() * 800));
  const rand = mulberry32(form.seed || 123);

  const allergies = (form.allergies || []).map((a) => a.toLowerCase().trim());
  const unsafeDrugs = (form.unsafeDrugs || []).map((d) => d.toLowerCase().trim());
  const safeDrugs = (form.safeDrugs || []).map((d) => d.toLowerCase().trim());

  let prescribedDrug = "nitroglycerin";
  if (safeDrugs.length > 0) prescribedDrug = safeDrugs[0];

  const allergyViolation = allergies.includes(prescribedDrug.toLowerCase()) ||
    unsafeDrugs.includes(prescribedDrug.toLowerCase());
  const violatedDrug = allergyViolation ? prescribedDrug : "";

  const troponin = form.labs.troponin_i || 0;
  const bnp = form.labs.bnp || 0;
  const glucose = form.labs.glucose || 0;
  const spo2 = form.vitals.spo2 ?? 98;
  const heartRate = form.vitals.heart_rate ?? 72;

  let diagnosisCode = "I21.9";
  let diagnosisName = "STEMI — ST-Elevation Myocardial Infarction";
  if (glucose > 200) { diagnosisCode = "E11.9"; diagnosisName = "Type 2 Diabetes Mellitus"; }
  else if (spo2 < 90) { diagnosisCode = "J96.01"; diagnosisName = "Acute hypoxic respiratory failure"; }
  else if (troponin > 1.5 || bnp > 150) { diagnosisCode = "I21.9"; diagnosisName = "STEMI — ST-Elevation Myocardial Infarction"; }

  const diagnosisCorrect = !form.groundTruthDiagnosis ||
    form.groundTruthDiagnosis.toLowerCase().includes(diagnosisCode.toLowerCase());

  const dosageMg = DRUG_DOSES[prescribedDrug.toLowerCase()] ?? 1.0;
  const doseDisplay = DRUG_DOSE_DISPLAY[prescribedDrug.toLowerCase()] ?? `${dosageMg} mg`;

  const renames: DriftRename[] = [];
  if (form.driftEnabled) {
    const candidates = [
      { section: "labs" as const, key: "troponin_i", renamed: "TROP" },
      { section: "vitals" as const, key: "heart_rate", renamed: "HR" },
      { section: "vitals" as const, key: "spo2", renamed: "SpO2" },
    ];
    for (const c of candidates) {
      if (rand() * 100 < form.driftProbability) {
        renames.push({ section: c.section, original: c.key, renamed: c.renamed });
        if (renames.length >= 2) break;
      }
    }
  }
  const driftOccurred = renames.length > 0;

  const flags: string[] = [];
  const notes: string[] = [];
  if (allergyViolation) { flags.push("ALLERGY_VIOLATION"); notes.push(`Prescribed ${violatedDrug} but patient is allergic`); }
  if (spo2 < 90) { flags.push("CRITICAL_SPO2"); notes.push(`SpO2 ${spo2}% critically low`); }
  if (heartRate > 120) { flags.push("TACHYCARDIA_NOTED"); notes.push(`HR ${heartRate} bpm`); }
  const auditorSafe = flags.length === 0;

  const components: { label: string; value: number }[] = [];
  if (diagnosisCorrect) components.push({ label: "Correct ICD-10 diagnosis", value: REWARD_WEIGHTS.diagnosis });
  else components.push({ label: "Wrong diagnosis (confident)", value: REWARD_WEIGHTS.wrongDx });
  if (!allergyViolation) components.push({ label: "Safe drug prescribed", value: REWARD_WEIGHTS.drug });
  else components.push({ label: "Allergic drug penalty", value: REWARD_WEIGHTS.allergy });
  if (!allergyViolation) components.push({ label: "Correct dosage", value: REWARD_WEIGHTS.dose });
  if (driftOccurred) components.push({ label: "Schema drift handled", value: REWARD_WEIGHTS.drift });
  if (auditorSafe) components.push({ label: "Auditor approved", value: REWARD_WEIGHTS.auditor });
  const total = components.reduce((s, c) => s + c.value, 0);

  const interactionFound = prescribedDrug === "heparin" && (form.medications || []).map(m => m.toLowerCase()).includes("warfarin");
  const toolCalls: ToolCall[] = [
    ...(driftOccurred ? [{ name: "schema_normalizer", input: { renamed_keys: renames.map(r => `${r.original}→${r.renamed}`) }, output: { resolved: true, mappings: renames.length }, verdict: "drift" as const }] : []),
    { name: "query_labs", input: { patient_id: form.patientId }, output: { drift_detected: driftOccurred }, verdict: driftOccurred ? "drift" : "safe" },
    { name: "check_allergies", input: { drug_name: prescribedDrug }, output: { verdict: allergyViolation ? "unsafe" : "safe" }, verdict: allergyViolation ? "unsafe" : "safe" },
    { name: "dose_check", input: { drug_name: prescribedDrug, dose_mg: dosageMg }, output: { verdict: "within_range" }, verdict: "safe" },
    { name: "drug_interactions", input: { drug: prescribedDrug, meds: form.medications }, output: { interactions_found: interactionFound }, verdict: interactionFound ? "warning" : "safe" },
    { name: "icd_lookup", input: { code: diagnosisCode }, output: { code: diagnosisCode, name: diagnosisName }, verdict: "safe" },
  ];

  const reasoning = allergyViolation
    ? `Patient presents with "${form.chiefComplaint}". Troponin ${troponin} ng/mL, BNP ${bnp} pg/mL indicating ${diagnosisName}. [⚠️ MOCK: allergy conflict detected for ${violatedDrug}]`
    : `Patient presents with "${form.chiefComplaint}". Elevated troponin (${troponin} ng/mL), BNP (${bnp} pg/mL) confirm ${diagnosisName}. Allergies reviewed — ${prescribedDrug} is safe. ${driftOccurred ? `Schema drift normalized (${renames.map(r => r.renamed).join(", ")}). ` : ""}Prescribing ${dosageMg}mg. [⚠️ MOCK — backend not connected]`;

  return {
    drift: { occurred: driftOccurred, renames },
    doctor: { icd10: diagnosisCode, diagnosisName, drug: prescribedDrug, dose: doseDisplay, confidence: diagnosisCorrect ? 0.91 : 0.54, schemaDriftHandled: driftOccurred, reasoning },
    toolCalls,
    auditor: { safe: auditorSafe, flags, notes },
    reward: { total: Math.max(-1, Math.min(1, parseFloat(total.toFixed(3)))), components },
    _source: "mock",
  };
}

// ─── Main export ──────────────────────────────────────────────────────────────

export async function runDiagnosis(form: PatientForm): Promise<DiagnosisResult> {
  // Try backend first
  try {
    const result = await callBackend(form);
    return result;
  } catch (err) {
    console.warn("[diagnosisEngine] Backend unreachable, using mock:", err);
    // Fall back to mock if backend is down
    return runMockDiagnosis(form);
  }
}

// Health check helper for UI status indicator
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3_000) });
    return resp.ok;
  } catch {
    return false;
  }
}
