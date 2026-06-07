# MedBook: Advanced Intelligent Healthcare Platform

## Overview
MedBook is an enterprise-grade integrated medical ecosystem designed to bridge the gap between clinical expertise and digital efficiency. By leveraging Artificial Intelligence, the platform delivers a seamless, high-performance experience for healthcare providers, administrators, and patients. It features sophisticated AI diagnostics, real-time data synchronization, and a highly optimized user interface.

## Key Features

*   **Intelligent Document Analysis**: Automatically processes and extracts clinical insights, test results, and document classifications from uploaded medical imagery, laboratory reports, and prescriptions.
*   **Real-time Vitals Monitoring & Analysis**: Delivers immediate clinical feedback and alerts based on patient-logged vital signs, enhancing preventive care.
*   **Medication Safety & Contraindication Checks**: Evaluates new prescriptions against the patient's comprehensive medical history, including documented allergies and chronic conditions, to identify potential adverse drug interactions.
*   **AI-Generated Health Summaries**: Compiles professional clinical analyses and diagnostic summaries, directly accessible via the patient dashboard.
*   **Bilingual Natural Language Processing**: Features full bilingual support (English/Arabic) for AI interactions, ensuring accessibility and precise medical terminology usage.

## Technology Stack

*   **AI & Machine Learning**: Python FastAPI server integrated with Gemini 1.5 Flash for high-throughput natural language processing and diagnostic analysis.
*   **Web Dashboard**: React, Vite, and TailwindCSS providing a highly responsive, data-rich administrative and clinical portal.
*   **Mobile Application**: Cross-platform application built with React Native and Expo, featuring real-time state synchronization and a comprehensive dynamic theming system.
*   **Backend Infrastructure**: ASP.NET Core Web API with Entity Framework Core, PostgreSQL persistence, and SignalR for robust, low-latency communication.

## Architecture & Project Structure

*   `backend/` - ASP.NET Core Web API handling core business logic, authentication, and database orchestration.
*   `web/` - React-based clinical dashboard for medical professionals and system administrators.
*   `DoctorApp/` - React Native application tailored for patient engagement and telemetry.
*   `server.py` - Microservice handling AI inference and large language model integrations.

## Getting Started

### 1. AI Inference Service
Configure the `GOOGLE_API_KEY` in your environment variables, then initialize the server:
```bash
python server.py
```

### 2. Backend API Service
Ensure a local or remote PostgreSQL instance is running, then execute the database migrations and start the server:
```bash
cd backend/MedicalAssistant
dotnet restore
dotnet ef database update --project MedicalAssistant.Persistance --startup-project MedicalAssistant.Web
dotnet run --project MedicalAssistant.Web
```

### 3. Mobile Client Application
Install the required Node dependencies and start the Expo development server:
```bash
cd DoctorApp
npm install
npx expo start
npx expo start --web --port 8082
```

## Testing & Quality Assurance

The repository maintains a comprehensive testing suite to guarantee system stability and prevent regressions.

**Web Client Testing:**
```bash
cd web
npm run test          # Execute Unit and Integration Tests via Vitest
npx playwright test   # Execute End-to-End browser validation via Playwright
```

## Administrative Credentials

Default system administrator access for development environments:
*   **Email:** admin@medbook.com
*   **Password:** 123456789

---
*Maintained by the MedBook Engineering Team*