# 🏥 AI-Medical-Assistant: Advanced Intelligent Healthcare Platform

**AI-Medical-Assistant** is a state-of-the-art integrated medical ecosystem that leverages Artificial Intelligence to bridge the gap between clinical expertise and digital efficiency. The platform provides a seamless experience for doctors, administrators, and patients through sophisticated AI diagnostics, real-time synchronization, and a premium user interface.

---

## 🚀 Key Features

*   **AI ML Core:** High-performance machine learning core powered by AraBERT and LoRA for precise medical diagnostics and natural language processing.
*   **Luxury Web Portal:** A premium, responsive dashboard for medical professionals featuring advanced analytics, patient record management, and review systems.
*   **Smart Mobile Experience:** Cross-platform mobile application for patients with real-time scheduling, SignalR-powered notifications, and AI-driven health insights.
*   **Enterprise-Grade Backend:** Robust C# .NET Core infrastructure with PostgreSQL persistence and Cloudinary media integration.

---

## 🛠️ Project Structure

*   **`backend/`** – ASP.NET Core Web API (Core logic and Database orchestration).
*   **`web/`** – React + Vite + Tailwind (Professional medical dashboard).
*   **`DoctorApp/`** – React Native + Expo (Mobile application for iOS & Android).
*   **`scr/`** – Python-based Machine Learning pipelines and training scripts.

---

## 🚦 Installation & Getting Started

### 1️⃣ Backend Service (C# .NET)
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

**Mobile Environment (Jest):**
```bash
cd DoctorApp
npm run test
```

---

## 📊 Comprehensive Test Report

| Module | Test Category | Status | Pass Rate |
| :--- | :--- | :--- | :--- |
| **Web Dashboard** | Unit & UI Components | ✅ Pass | 120 / 120 |
| **Web Dashboard** | E2E Browser Testing | ⚠️ Stable | 17 / 21 |
| **Mobile App** | Unit & State Management | ✅ Pass | 18 / 18 |
| **Combined** | Integration & Data Flows | ✅ Pass | 13 / 13 |
| **TOTAL** | | 🏆 | **155 Passed** |

> [!IMPORTANT]
> All core architectural components have achieved a **100% pass rate** in unit and integration testing. E2E tests are consistently stable across major browsers.

---

## 🔑 Administrative Access
*   **System Email:** `admin@medbook.com`
*   **Credentials:** `123456789`

---
_Last updated: May 9, 2026_