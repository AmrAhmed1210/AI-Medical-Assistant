# Run Project (DoctorApp + Backend + Web)

## 1) Backend (.NET API + SignalR)
1. Open terminal in:
   - `E:\AI\backend\MedicalAssistant\MedicalAssistant.Web`
2. Restore/build (first time):
   - `dotnet restore`
   - `dotnet build`
3. Run API:
   - `dotnet run`
4. API runs on:
   - `http://localhost:5194`
   - LAN access is enabled by program startup (`0.0.0.0:5194`).

Important:
- If build fails with locked DLL files, stop old running API process first, then run again.

## 2) Web Dashboard (React + Vite)
1. Open terminal in:
   - `E:\AI\web`
2. Install deps (first time):
   - `npm install`
3. Run dev server:
   - `npm run dev`
4. Open shown URL (usually `http://localhost:5173`).

## 3) Mobile App (Expo)
1. Open terminal in:
   - `E:\AI\DoctorApp`
2. Install deps (first time):
   - `npm install`
3. Run Expo:
   - `npx expo start`
4. Open with Expo Go (Android/iOS) or web.

Notes for mobile API URL:
- Mobile app reads host from Expo runtime and points to `http://<expo-host>:5194` automatically.
- Make sure phone and PC are on the same network.

## 4) Recommended startup order
1. Backend
2. Web
3. Mobile

## 5) Quick checks if something is not updating
- Restart backend after controller/service changes.
- Hard refresh web page.
- In Expo press `r` to reload app.
- Verify token belongs to correct role (Doctor/Patient) before testing role-specific pages.
