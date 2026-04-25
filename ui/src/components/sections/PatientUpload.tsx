import { useRef, useState, DragEvent, ChangeEvent } from "react";
import { motion } from "framer-motion";
import { UploadCloud, Download, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { PatientForm } from "@/lib/diagnosisEngine";

type UploadedMeta = { filename: string; patientId: string };

interface Props {
  onLoaded: (form: Partial<PatientForm>, meta: UploadedMeta) => void;
}

const SAFE_EXAMPLE = {
  patient_id: "P-SAFE-001",
  age: 62,
  gender: "Female",
  chief_complaint: "Mild shortness of breath on exertion for 2 weeks. No chest pain.",
  vitals: { bp_systolic: 128, bp_diastolic: 78, heart_rate: 82, temperature: 36.8, spo2: 97, respiratory_rate: 16 },
  lab_results: { troponin_i: 0.02, bnp: 95, creatinine: 0.9, glucose: 104, wbc: 7.1, hemoglobin: 13.4 },
  known_allergies: ["penicillin"],
  current_medications: ["metformin", "atorvastatin"],
  ground_truth_diagnosis: "Stable angina",
  safe_drugs: ["nitroglycerin", "metoprolol"],
  unsafe_drugs: ["amoxicillin"],
};

const UNSAFE_EXAMPLE = {
  patient_id: "P-UNSAFE-007",
  age: 71,
  gender: "Male",
  chief_complaint: "Crushing substernal chest pain radiating to jaw, diaphoresis, nausea for 30 minutes.",
  vitals: { bp_systolic: 168, bp_diastolic: 98, heart_rate: 118, temperature: 37.2, spo2: 91, respiratory_rate: 26 },
  lab_results: { troponin_i: 4.7, bnp: 320, creatinine: 1.4, glucose: 188, wbc: 12.8, hemoglobin: 12.1 },
  known_allergies: ["aspirin", "sulfa"],
  current_medications: ["warfarin", "lisinopril"],
  ground_truth_diagnosis: "STEMI - Acute MI",
  safe_drugs: ["clopidogrel", "heparin", "metoprolol"],
  unsafe_drugs: ["aspirin", "ibuprofen"],
};

const mapToForm = (j: any): Partial<PatientForm> => {
  const out: Partial<PatientForm> = {};
  if (j.patient_id) out.patientId = String(j.patient_id);
  if (typeof j.age === "number") out.age = j.age;
  if (j.gender) out.gender = j.gender;
  if (j.chief_complaint) out.chiefComplaint = j.chief_complaint;
  if (j.vitals && typeof j.vitals === "object") {
    out.vitals = {
      bp_systolic: +j.vitals.bp_systolic || 120,
      bp_diastolic: +j.vitals.bp_diastolic || 80,
      heart_rate: +j.vitals.heart_rate || 75,
      temperature: +j.vitals.temperature || 37,
      spo2: +j.vitals.spo2 || 98,
      respiratory_rate: +j.vitals.respiratory_rate || 16,
    };
  }
  if (j.lab_results && typeof j.lab_results === "object") {
    out.labs = {
      troponin_i: +j.lab_results.troponin_i || 0,
      bnp: +j.lab_results.bnp || 0,
      creatinine: +j.lab_results.creatinine || 1,
      glucose: +j.lab_results.glucose || 100,
      wbc: +j.lab_results.wbc || 7,
      hemoglobin: +j.lab_results.hemoglobin || 14,
    };
  }
  if (Array.isArray(j.known_allergies)) out.allergies = j.known_allergies.map(String);
  if (Array.isArray(j.current_medications)) out.medications = j.current_medications.map(String);
  if (Array.isArray(j.safe_drugs)) out.safeDrugs = j.safe_drugs.map(String);
  if (Array.isArray(j.unsafe_drugs)) out.unsafeDrugs = j.unsafe_drugs.map(String);
  if (typeof j.ground_truth_diagnosis === "string") out.groundTruthDiagnosis = j.ground_truth_diagnosis;
  return out;
};

const REQUIRED = ["age", "gender", "chief_complaint", "vitals", "lab_results"];

export const PatientUpload = ({ onLoaded }: Props) => {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleParsed = (json: any, filename: string) => {
    if (!json || typeof json !== "object") {
      toast.error("Could not parse file — invalid JSON format");
      return;
    }
    if (!json.patient_id) {
      toast.error("File loaded but some fields are missing — defaults applied");
    } else {
      const missing = REQUIRED.filter((k) => !(k in json));
      if (missing.length) toast.warning("File loaded but some fields are missing — defaults applied");
    }
    const mapped = mapToForm(json);
    const pid = String(json.patient_id || "P-UPLOAD");
    toast.success(`Patient record loaded — ${pid}`);
    onLoaded(mapped, { filename, patientId: pid });
  };

  const readFile = (file: File) => {
    if (!file.name.toLowerCase().endsWith(".json")) {
      toast.error("Only .json files are supported");
      return;
    }
    if (file.size > 1024 * 1024) {
      toast.error("File too large — patient records should be under 1MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(String(e.target?.result || ""));
        handleParsed(json, file.name);
      } catch {
        toast.error("Could not parse file — invalid JSON format");
      }
    };
    reader.onerror = () => toast.error("Could not read file");
    reader.readAsText(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setHover(false);
    const file = e.dataTransfer.files?.[0];
    if (file) readFile(file);
  };

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFile(file);
    e.target.value = "";
  };

  const loadExample = (data: any, label: string) => {
    handleParsed(data, `${label}.json`);
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setHover(true);
        }}
        onDragLeave={() => setHover(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed transition-all px-6 py-12 text-center ${
          hover ? "border-solid border-primary bg-primary/10" : "border-primary/50 bg-primary/[0.03] hover:bg-primary/5"
        }`}
      >
        <input ref={inputRef} type="file" accept=".json,application/json" className="hidden" onChange={onPick} />
        <motion.div
          animate={{ y: [0, -6, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary/15 text-primary mb-3"
        >
          <UploadCloud size={28} />
        </motion.div>
        <div className="font-semibold text-base">Drop patient JSON file here</div>
        <div className="text-xs text-muted-foreground mt-1">or click to browse · Supports .json files only</div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <button
          type="button"
          onClick={() => loadExample(SAFE_EXAMPLE, "safe_patient_example")}
          className="flex items-center justify-center gap-2 py-3 rounded-lg border border-success/50 text-success hover:bg-success/10 transition font-medium text-sm"
        >
          <Download size={16} />
          Load Safe Patient Example
        </button>
        <button
          type="button"
          onClick={() => loadExample(UNSAFE_EXAMPLE, "unsafe_patient_example")}
          className="flex items-center justify-center gap-2 py-3 rounded-lg border border-danger/50 text-danger hover:bg-danger/10 transition font-medium text-sm"
        >
          <AlertTriangle size={16} />
          Load Unsafe Patient Example
        </button>
      </div>

      <div className="text-xs text-muted-foreground bg-muted/30 rounded-lg p-3 border border-border/50">
        <div className="font-semibold mb-1 text-foreground/80">Expected schema</div>
        <code className="block opacity-80 leading-relaxed">
          {`{ patient_id, age, gender, chief_complaint, vitals{...}, lab_results{...}, known_allergies[], current_medications[] }`}
        </code>
      </div>
    </div>
  );
};
