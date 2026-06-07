const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'app', '(patient)', 'ai-profile-assistant.tsx');
let content = fs.readFileSync(filePath, 'utf8');

// 1. ADD menu:
content = content.replace(
  /        if \(val === "DISEASE"\) \{\n          next\.push\(addMsg\(t\("ما اسم المرض؟", "Disease name\?"\), "assistant", skipChips\(\)\)\);\n          pushState\(\{ subIntent: "disease", step: 1, draft: \{\}, messages: next \}\);\n        \} else if \(val === "MED"\) \{/,
  `        if (val === "DISEASE") {
          next.push(addMsg(t("ما اسم المرض؟", "Disease name?"), "assistant", skipChips()));
          pushState({ subIntent: "disease", step: 1, draft: {}, messages: next });
        } else if (val === "SURGERY") {
          next.push(addMsg(t("ما اسم العملية الجراحية؟", "Surgery name?"), "assistant", skipChips()));
          pushState({ subIntent: "surgery", step: 1, draft: {}, messages: next });
        } else if (val === "VITAL") {
          next.push(addMsg(t("ما نوع القياس؟", "Vital type?"), "assistant", [
            { label: t("الضغط", "Blood Pressure"), value: "Blood Pressure" },
            { label: t("السكر", "Blood Sugar"), value: "Blood Sugar" },
          ]));
          pushState({ subIntent: "vital", step: 1, draft: {}, messages: next });
        } else if (val === "GENERAL") {
          next.push(addMsg(t("هل أنت مدخن؟", "Are you a smoker?"), "assistant", [
            { label: t("نعم", "Yes"), value: "YES" }, { label: t("لا", "No"), value: "NO" },
          ]));
          pushState({ subIntent: "general", step: 1, draft: profile || {}, messages: next });
        } else if (val === "MED") {`
);

// 2. EDIT menu (Step 0)
content = content.replace(
  /        \} else if \(val === "ALLERGY"\) \{\n          chips = allergies\.map\(\(a\) => \(\{ label: a\.allergenName, value: \`ITEM_\$\{a\.id\}\` \}\)\);\n          if \(\!chips\.length\) chips = \[\{ label: t\("لا يوجد بيانات", "No data"\), value: "HOME" \}\];\n          next\.push\(addMsg\(t\("اختر الحساسية:", "Select allergy:"\), "assistant", chips\)\);\n          pushState\(\{ subIntent: "allergy", step: 0\.5, messages: next \}\);\n        \}/,
  `} else if (val === "ALLERGY") {
          chips = allergies.map((a) => ({ label: a.allergenName, value: \`ITEM_\${a.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الحساسية:", "Select allergy:"), "assistant", chips));
          pushState({ subIntent: "allergy", step: 0.5, messages: next });
        } else if (val === "SURGERY") {
          chips = surgeries.map((s) => ({ label: s.surgeryName, value: \`ITEM_\${s.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر العملية لتعديلها:", "Select surgery:"), "assistant", chips));
          pushState({ subIntent: "surgery", step: 0.5, messages: next });
        } else if (val === "VITAL") {
          chips = vitals.map((v) => ({ label: v.readingType, value: \`ITEM_\${v.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر القياس لتعديله:", "Select vital:"), "assistant", chips));
          pushState({ subIntent: "vital", step: 0.5, messages: next });
        } else if (val === "GENERAL") {
          next.push(addMsg(t("هل أنت مدخن؟", "Are you a smoker?"), "assistant", [
            { label: t("نعم", "Yes"), value: "YES" }, { label: t("لا", "No"), value: "NO" },
          ]));
          pushState({ subIntent: "general", step: 1, draft: profile || {}, messages: next });
        }`
);

// 3. EDIT menu (Step 0.5 - Handling ITEM_)
content = content.replace(
  /        \} else if \(subIntent === "allergy"\) \{\n          const item = allergies\.find\(\(a\) => a\.id === id\);\n          if \(item\) \{\n            next\.push\(addMsg\(t\("نوع الحساسية:", "Allergy type:"\), "assistant", \[\n              \{ label: t\("دواء", "Drug"\), value: "DRUG" \},\n              \{ label: t\("طعام", "Food"\), value: "FOOD" \},\n              \{ label: t\("أخرى", "Other"\), value: "OTHER" \},\n            \]\)\);\n            pushState\(\{ step: 1, draft: \{ \.\.\.item \}, messages: next \}\);\n          \}\n        \}/,
  `} else if (subIntent === "allergy") {
          const item = allergies.find((a) => a.id === id);
          if (item) {
            next.push(addMsg(t("نوع الحساسية:", "Allergy type:"), "assistant", [
              { label: t("دواء", "Drug"), value: "DRUG" },
              { label: t("طعام", "Food"), value: "FOOD" },
              { label: t("أخرى", "Other"), value: "OTHER" },
            ]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        } else if (subIntent === "surgery") {
          const item = surgeries.find((s) => s.id === id);
          if (item) {
            next.push(addMsg(t(\`العملية الحالية: \${item.surgeryName}\\nأدخل الاسم الجديد:\`, \`Current: \${item.surgeryName}\\nNew name:\`), "assistant", [{ label: item.surgeryName, value: item.surgeryName }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        } else if (subIntent === "vital") {
          const item = vitals.find((v) => v.id === id);
          if (item) {
            next.push(addMsg(t(\`القياس الحالي: \${item.readingType}\\nأدخل النوع الجديد:\`, \`Current: \${item.readingType}\\nNew type:\`), "assistant", [{ label: item.readingType, value: item.readingType }]));
            pushState({ step: 1, draft: { ...item }, messages: next });
          }
        }`
);

// 4. DELETE menu (Step 0)
content = content.replace(
  /        \} else if \(val === "ALLERGY"\) \{\n          chips = allergies\.map\(\(a\) => \(\{ label: a\.allergenName, value: \`DEL_\$\{a\.id\}\` \}\)\);\n          if \(\!chips\.length\) chips = \[\{ label: t\("لا يوجد بيانات", "No data"\), value: "HOME" \}\];\n          next\.push\(addMsg\(t\("اختر الحساسية لحذفها:", "Select allergy to delete:"\), "assistant", chips\)\);\n          pushState\(\{ subIntent: "allergy", step: 1, messages: next \}\);\n        \}/,
  `} else if (val === "ALLERGY") {
          chips = allergies.map((a) => ({ label: a.allergenName, value: \`DEL_\${a.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر الحساسية لحذفها:", "Select allergy to delete:"), "assistant", chips));
          pushState({ subIntent: "allergy", step: 1, messages: next });
        } else if (val === "SURGERY") {
          chips = surgeries.map((s) => ({ label: s.surgeryName, value: \`DEL_\${s.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر العملية لحذفها:", "Select surgery to delete:"), "assistant", chips));
          pushState({ subIntent: "surgery", step: 1, messages: next });
        } else if (val === "VITAL") {
          chips = vitals.map((v) => ({ label: v.readingType, value: \`DEL_\${v.id}\` }));
          if (!chips.length) chips = [{ label: t("لا يوجد بيانات", "No data"), value: "HOME" }];
          next.push(addMsg(t("اختر القياس لحذفه:", "Select vital to delete:"), "assistant", chips));
          pushState({ subIntent: "vital", step: 1, messages: next });
        }`
);

// 5. DELETE menu (Step 2 - Handling Confirmation)
content = content.replace(
  /          if \(subIntent === "allergy"\) await deleteAllergy\(draft\.id\);/,
  `          if (subIntent === "allergy") await deleteAllergy(draft.id);
          if (subIntent === "surgery") await deleteSurgery(draft.id);
          if (subIntent === "vital") await deleteVital(draft.id);`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Patch2 complete.');
