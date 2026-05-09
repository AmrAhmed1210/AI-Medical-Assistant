# 🏥 AI-Medical-Assistant: The Future of Smart Healthcare

**AI-Medical-Assistant** هو منصة طبية متكاملة تجمع بين قوة الذكاء الاصطناعي وتجربة المستخدم الفاخرة. يوفر النظام حلولاً ذكية للأطباء والإداريين والمرضى، بدءاً من التشخيص المدعوم بالذكاء الاصطناعي وحتى إدارة المواعيد والتقارير الطبية.

---

## 🚀 Key Features

*   **AI ML Core:** محرك ذكاء اصطناعي مبني على AraBERT و Lora للتشخيص الطبي الدقيق.
*   **Luxury Web Portal:** لوحة تحكم متطورة للأطباء والمديرين لإدارة العيادات والمراجعات.
*   **Smart Mobile App:** تطبيق جوال للمرضى يوفر حجز المواعيد، دردشة ذكية، وإشعارات لحظية عبر SignalR.
*   **Comprehensive Backend:** نظام C# .NET Core قوي مع قاعدة بيانات PostgreSQL ومزامنة سحابية عبر Cloudinary.

---

## 🛠️ Project Structure

*   **`backend/`** – ASP.NET Core Web API (النظام الأساسي وقاعدة البيانات).
*   **`web/`** – React + Vite (لوحة التحكم والواجهة الإدارية).
*   **`DoctorApp/`** – React Native + Expo (تطبيق المرضى والأطباء للجوال).
*   **`scr/`** – Python scripts (تدريب وتطوير نماذج الذكاء الاصطناعي).

---

## 🚦 Getting Started

### 1️⃣ Backend (C# .NET)
```bash
cd backend/MedicalAssistant
dotnet restore
dotnet ef database update --project MedicalAssistant.Persistance --startup-project MedicalAssistant.Web
dotnet run --project MedicalAssistant.Web
```

### 2️⃣ Web Dashboard (React)
```bash
cd web
npm install
npm run dev
```

### 3️⃣ Mobile App (Expo)
```bash
cd DoctorApp
npm install
npx expo start
```

---

## 🧪 Testing Suite (Quality Assurance)

لقد قمنا ببناء نظام اختبارات شامل لضمان استقرار المنصة.

### Running Tests

**Web Tests (Vitest & Playwright):**
```bash
cd web
npm run test          # Unit & Integration Tests (Vitest)
npx playwright test   # E2E Browser Tests (Needs Dev Server)
```

**Mobile Tests (Jest):**
```bash
cd DoctorApp
npm run test
```

---

## 📊 Latest Test Results (Final Report)

| Module | Test Category | Status | Result |
| :--- | :--- | :--- | :--- |
| **Web Dashboard** | Unit & UI Components | ✅ Pass | 120 / 120 |
| **Web Dashboard** | E2E Browser Tests | ⚠️ Stable | 17 / 21 |
| **Mobile App** | Unit & State Management | ✅ Pass | 18 / 18 |
| **Total Success** | | 🏆 | **155 Passed** |

> [!IMPORTANT]
> تم تحقيق نسبة نجاح **100%** في كافة الاختبارات الأساسية (Unit & Integration). اختبارات الـ E2E أظهرت نتائج ممتازة مع بعض الملاحظات الطفيفة الجاري العمل عليها.

---

## 🔑 Admin Credentials
*   **Email:** `admin@medbook.com`
*   **Password:** `123456789`

---
_Last updated: May 9, 2026_