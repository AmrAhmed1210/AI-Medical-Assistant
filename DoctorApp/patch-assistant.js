const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'app', '(patient)', 'ai-profile-assistant.tsx');
let content = fs.readFileSync(filePath, 'utf8');

// 1. Types
content = content.replace(
  'type SubIntent = "disease" | "med" | "allergy" | null;',
  'type SubIntent = "disease" | "med" | "allergy" | "surgery" | "vital" | "general" | null;'
);
content = content.replace(
  'onboardingSection: "disease" | "med" | "allergy" | "done";',
  'onboardingSection: "disease" | "med" | "allergy" | "surgery" | "vital" | "general" | "done";'
);

// 2. Imports
content = content.replace(
  /import \{\n  getChronicDiseases, getMedications, getAllergies,\n  createChronicDisease, createMedication, createAllergy,\n  updateChronicDisease, updateMedication, updateAllergy,\n  deleteChronicDisease, deleteMedication, deleteAllergy,\n  ChronicDisease, Medication, AllergyRecord\n\} from "\.\.\/\.\.\/services\/medicalRecordService";/,
  `import {
  getChronicDiseases, getMedications, getAllergies, getSurgeries, getVitals, getMedicalProfile,
  createChronicDisease, createMedication, createAllergy, createSurgery, createVital,
  updateChronicDisease, updateMedication, updateAllergy, updateSurgery, updateVital, updateMedicalProfile,
  deleteChronicDisease, deleteMedication, deleteAllergy, deleteSurgery, deleteVital,
  ChronicDisease, Medication, AllergyRecord, SurgeryRecord, VitalReading
} from "../../services/medicalRecordService";`
);

// 3. States
content = content.replace(
  /  const \[diseases, setDiseases\] = useState<ChronicDisease\[\]>\(\[\]\);\n  const \[meds, setMeds\] = useState<Medication\[\]>\(\[\]\);\n  const \[allergies, setAllergies\] = useState<AllergyRecord\[\]>\(\[\]\);/,
  `  const [diseases, setDiseases] = useState<ChronicDisease[]>([]);
  const [meds, setMeds] = useState<Medication[]>([]);
  const [allergies, setAllergies] = useState<AllergyRecord[]>([]);
  const [surgeries, setSurgeries] = useState<SurgeryRecord[]>([]);
  const [vitals, setVitals] = useState<VitalReading[]>([]);
  const [profile, setProfile] = useState<any>(null);`
);

// 4. loadData
content = content.replace(
  /        const \[d, m, a\] = await Promise\.all\(\[\n          getChronicDiseases\(pid\), getMedications\(pid\), getAllergies\(pid\),\n        \]\);\n        const activeD = d\.filter\(\(x: any\) => x\.isActive !== false\);\n        const activeM = m\.filter\(\(x: any\) => x\.isActive !== false\);\n        const activeA = a\.filter\(\(x: any\) => x\.isActive !== false\);\n        setDiseases\(activeD\); setMeds\(activeM\); setAllergies\(activeA\);\n        return \{ count: activeD\.length \+ activeM\.length \+ activeA\.length, pid \};/,
  `        const [d, m, a, s, v, p] = await Promise.all([
          getChronicDiseases(pid), getMedications(pid), getAllergies(pid),
          getSurgeries(pid), getVitals(pid), getMedicalProfile(pid).catch(() => null)
        ]);
        const activeD = d.filter((x: any) => x.isActive !== false);
        const activeM = m.filter((x: any) => x.isActive !== false);
        const activeA = a.filter((x: any) => x.isActive !== false);
        setDiseases(activeD); setMeds(activeM); setAllergies(activeA);
        setSurgeries(s); setVitals(v); setProfile(p);
        return { count: activeD.length + activeM.length + activeA.length + s.length + v.length, pid };`
);

// 5. catChips
content = content.replace(
  /  const catChips = \(\) => \[\n    \{ label: t\("أمراض", "Diseases"\), value: "DISEASE" \},\n    \{ label: t\("أدوية", "Medications"\), value: "MED" \},\n    \{ label: t\("حساسية", "Allergies"\), value: "ALLERGY" \},\n  \];/,
  `  const catChips = () => [
    { label: t("أمراض", "Diseases"), value: "DISEASE" },
    { label: t("أدوية", "Medications"), value: "MED" },
    { label: t("حساسية", "Allergies"), value: "ALLERGY" },
    { label: t("جراحات", "Surgeries"), value: "SURGERY" },
    { label: t("قياسات", "Vitals"), value: "VITAL" },
    { label: t("بيانات عامة", "General"), value: "GENERAL" },
  ];`
);

// 6. Summary Card
content = content.replace(
  /    const items = \[\n      \{ icon: "heart", color: COLORS\.primary, bg: "#ECFDF5", count: diseases\.length, label: t\("أمراض", "Diseases"\) \},\n      \{ icon: "medkit", color: "#0D9488", bg: "#F0FDFA", count: meds\.length, label: t\("أدوية", "Meds"\) \},\n      \{ icon: "warning", color: "#64748B", bg: "#F8FAFC", count: allergies\.length, label: t\("حساسية", "Allergies"\) \},\n    \];/,
  `    const items = [
      { icon: "heart", color: COLORS.primary, bg: "#ECFDF5", count: diseases.length, label: t("أمراض", "Diseases") },
      { icon: "medkit", color: "#0D9488", bg: "#F0FDFA", count: meds.length, label: t("أدوية", "Meds") },
      { icon: "warning", color: "#64748B", bg: "#F8FAFC", count: allergies.length, label: t("حساسية", "Allergies") },
      { icon: "bandage", color: "#3B82F6", bg: "#EFF6FF", count: surgeries.length, label: t("جراحات", "Surgeries") },
      { icon: "pulse", color: "#F59E0B", bg: "#FEF3C7", count: vitals.length, label: t("قياسات", "Vitals") },
    ];`
);

// Known Chips string array update
content = content.replace(
  /      "DISEASE","MED","ALLERGY","YES","NO",\n      "DRUG","FOOD","OTHER","CONFIRM_DEL","CANCEL_DEL",/,
  `      "DISEASE","MED","ALLERGY","SURGERY","VITAL","GENERAL","YES","NO",
      "DRUG","FOOD","OTHER","CONFIRM_DEL","CANCEL_DEL","MALE","FEMALE",`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Patch complete.');
