# MedBook (AI Medical Assistant) 🏥

MedBook is a comprehensive healthcare platform featuring a Doctor/Patient mobile application, a Web Dashboard, and a robust ASP.NET Core backend. It incorporates an AI Medical Assistant to analyze medical records, prescriptions, and lab results.

## 🚀 Live Links & Deployments

| Component | Technology | Live URL |
|-----------|------------|----------|
| **Web Dashboard** | React + Vite | [https://poetic-pithivier-a60d22.netlify.app/](https://poetic-pithivier-a60d22.netlify.app/) (Netlify) <br> *Also available on Vercel* |
| **Backend API** | ASP.NET Core 8 + EF Core | [http://amrahmed1210-001-site1.itempurl.com/](http://amrahmed1210-001-site1.itempurl.com/) |
| **Database** | SQL Server | Hosted on SmarterASP.net (`sql1003.site4now.net`) |
| **Mobile App** | React Native (Expo) | [Download Android APK](https://expo.dev/artifacts/eas/UZexGDLgLq0U2R0-htN0cOIRawn371V-zZmOmLPE_GU.apk) |

---

## 🛠️ Project Structure

The repository is divided into 3 main components:
1. **`/backend/MedicalAssistant/`**: The core API providing authentication, data management, and SignalR for real-time notifications.
2. **`/web/`**: The web application for managing the system, built with React and Vite.
3. **`/DoctorApp/`**: The mobile application for Doctors and Patients, built with Expo React Native.

## 📝 What Was Accomplished

### 1. Backend Deployment (SmarterASP.net)
- Successfully deployed the ASP.NET Core application to SmarterASP.
- Configured connection strings to point to the live SQL Server instance.
- Verified live API endpoints and SignalR Hub connections.

### 2. Web Deployment (Vercel & Netlify)
- Configured environment variables (`.env`) to communicate with the production backend (`VITE_API_BASE_URL` & `VITE_SIGNALR_HUB_URL`).
- Resolved build warnings in SignalR configurations.
- Successfully built and deployed the production `dist` folder to Vercel/Netlify.

### 3. Mobile App Deployment (Expo EAS)
- Updated `constants/api.ts` to route all mobile traffic to the live backend.
- Configured `eas.json` for Android production builds.
- Fixed `expo-image-picker` permissions and integration in patient records (`[category].tsx`).
- Created dynamic localization foundations using `LanguageContext` (Arabic/English toggle).
- Ran `eas build -p android --profile production` to successfully generate the final Play Store-ready APK!

## 🌍 Localization (i18n) - 100% Coverage 🎉
Both the Web Dashboard and the Mobile App have achieved **100% dual-language support (Arabic/English)**!
- Extracted and translated over 500+ medical and technical terms across the entire ecosystem.
- Integrated robust React Contexts (`useLanguage`) to seamlessly switch between LTR and RTL layouts without reloading.
- Every single screen (Doctor Workspace, Patient Records, Chat, AI Analysis, etc.) is fully localized down to the last letter.
