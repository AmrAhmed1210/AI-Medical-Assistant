# 🏥 AI-Medical-Assistant: Advanced Intelligent Healthcare Platform

**AI-Medical-Assistant** is a state-of-the-art integrated medical ecosystem that leverages Artificial Intelligence to bridge the gap between clinical expertise and digital efficiency. The platform provides a seamless experience for doctors, administrators, and patients through sophisticated AI diagnostics, real-time synchronization, and a premium user interface.

---

## 🤖 AI Medical Assistant Features (New!)

We have integrated a comprehensive **Bilingual (Arabic/English) AI Assistant** into the patient experience:

*   **Smart Document Analysis:** Automatically extracts medical insights, test results, and titles from uploaded scans, lab results, and prescriptions.
*   **Real-time Vitals Advice:** Provides immediate, encouraging medical feedback and warnings based on logged vital signs (Blood Pressure, Heart Rate, etc.).
*   **Medication Safety Check:** Evaluates new medications against the patient's existing history (allergies, chronic diseases) to flag potential contraindications.
*   **AI Health Summary:** Generates a professional medical analysis and diagnosis summary directly on the patient's home dashboard.
*   **Intelligent Text Refinement:** Helps patients professionalize their medical history entries (surgeries, allergies) using medical terminology.

---

## 🚀 Core Technologies

*   **AI ML Core:** High-performance machine learning core powered by **Gemini 1.5 Flash** for precise medical diagnostics and natural language processing.
*   **Luxury Web Portal:** A premium, responsive dashboard for medical professionals featuring advanced analytics and patient management.
*   **Smart Mobile Experience:** Cross-platform mobile application (React Native + Expo) for patients with real-time AI insights.
*   **Enterprise-Grade Backend:** Robust C# .NET Core infrastructure with PostgreSQL persistence and SignalR synchronization.

---

## 🛠️ Project Structure

*   **`backend/`** – ASP.NET Core Web API (Core logic and Database orchestration).
*   **`web/`** – React + Vite + Tailwind (Professional medical dashboard).
*   **`DoctorApp/`** – React Native + Expo (Mobile application for patients).
*   **`server.py`** – Python FastAPI server handling the AI/Gemini integration.

---

## 🚦 Installation & Getting Started

### 1️⃣ AI Server (Python)
```bash
# Set GOOGLE_API_KEY in .env
python server.py
```

### 2️⃣ Backend Service (C# .NET)
```bash
cd backend/MedicalAssistant
dotnet restore
dotnet ef database update --project MedicalAssistant.Persistance --startup-project MedicalAssistant.Web
dotnet run --project MedicalAssistant.Web
```

### 3️⃣ Mobile Application (Expo)
```bash
cd DoctorApp
npm install
npx expo start
```

---

## 🧪 Testing & Quality Assurance

We maintain a rigorous testing environment to ensure 100% system stability and data integrity.

### Execution Commands

**Web Environment (Vitest & Playwright):**
```bash
cd web
npm run test          # Executes Unit & Integration Tests
npx playwright test   # Executes End-to-End Browser Tests
```

---

## 🔑 Administrative Access
*   **System Email:** `admin@medbook.com`
*   **Credentials:** `123456789`

---
_Last updated: May 12, 2026 - Integrated Bilingual AI Assistant_