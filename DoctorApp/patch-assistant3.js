const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'app', '(patient)', 'ai-profile-assistant.tsx');
let content = fs.readFileSync(filePath, 'utf8');

// Insert after the allergy flow
const searchStr = `          pushState({ intent: "after_action", subIntent: null, step: 0, onboardingSection: "done", draft: {}, messages: next });\n        }\n      }`;

const newFlows = `          pushState({ intent: "after_action", subIntent: null, step: 0, onboardingSection: "done", draft: {}, messages: next });
        }
      }

      // Surgery flow
      else if (subIntent === "surgery") {
        if (step === 1) {
          draft.surgeryName = isSkip ? "Unknown" : label;
          next.push(addMsg(t("متى أجريت هذه العملية؟", "When was this surgery?"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.surgeryDate = isSkip ? new Date().toISOString().split("T")[0] : label;
          const payload = { surgeryName: draft.surgeryName, surgeryDate: draft.surgeryDate, isActive: true };
          if (intent === "edit") await updateSurgery(draft.id, payload);
          else await createSurgery(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }

      // Vital flow
      else if (subIntent === "vital") {
        if (step === 1) {
          draft.readingType = isSkip ? "Blood Pressure" : label;
          next.push(addMsg(t("ما هي القيمة؟ (مثال: 120)", "What is the value? (e.g. 120)"), "assistant", skipChips()));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.value = parseFloat(label) || 120;
          next.push(addMsg(t("ما هي القيمة الثانية (إذا وجدت، للضغط)؟", "Second value (if any, for BP)?"), "assistant", skipNoneChips()));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.value2 = isSkip ? undefined : parseFloat(label) || 80;
          const payload = { readingType: draft.readingType, value: draft.value, value2: draft.value2, unit: "standard", isNormal: true };
          if (intent === "edit") await updateVital(draft.id, payload);
          else await createVital(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }

      // General profile flow
      else if (subIntent === "general") {
        if (step === 1) {
          draft.isSmoker = val === "YES";
          next.push(addMsg(t("فصيلة الدم؟", "Blood Type?"), "assistant", [
            { label: "A+", value: "A+" }, { label: "O+", value: "O+" }, { label: "B+", value: "B+" }, { label: "AB+", value: "AB+" }
          ]));
          pushState({ step: 2, draft, messages: next });
        } else if (step === 2) {
          draft.bloodType = isSkip ? "" : label;
          next.push(addMsg(t("الوزن (كجم)؟", "Weight (kg)?"), "assistant", skipChips()));
          pushState({ step: 3, draft, messages: next });
        } else if (step === 3) {
          draft.weightKg = parseFloat(label) || 70;
          next.push(addMsg(t("الطول (سم)؟", "Height (cm)?"), "assistant", skipChips()));
          pushState({ step: 4, draft, messages: next });
        } else if (step === 4) {
          draft.heightCm = parseFloat(label) || 170;
          const payload = { 
            isSmoker: draft.isSmoker, 
            bloodType: draft.bloodType, 
            weightKg: draft.weightKg, 
            heightCm: draft.heightCm 
          };
          await updateMedicalProfile(pid, payload);
          await loadData();
          next = askAfterAction(next);
          pushState({ intent: "after_action", subIntent: null, step: 0, draft: {}, messages: next });
        }
      }`;

content = content.replace(searchStr, newFlows);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Patch3 complete.');
