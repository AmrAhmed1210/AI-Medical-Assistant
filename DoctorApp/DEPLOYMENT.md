# Backend Deployment Guide

## Current Status
- **Frontend API URL**: `https://ai-medical-assistant-production-38a3.up.railway.app`
- **Hosting**: Railway (backend app)
- **Database**: Neon PostgreSQL

## Prerequisites
1. Railway CLI installed and logged in
2. Neon account with a PostgreSQL project
3. Backend repository (ASP.NET Core or Node.js) pushed to GitHub/GitLab

## 1. Neon PostgreSQL Setup
1. Go to [neon.tech](https://neon.tech) and create a new project.
2. Create a database (e.g., `ai_medical_db`).
3. Copy the connection string (use the **Pooled** connection string for serverless apps).
   - Format: `postgres://user:password@host/database?sslmode=require`

## 2. Railway Project Setup
1. Create a new project in Railway dashboard.
2. Link your backend repo (GitHub/GitLab).
3. Add a **PostgreSQL** service OR use your external Neon DB.

### If using external Neon DB:
- Go to **Variables** in your Railway service.
- Add:
  ```
  DATABASE_URL=postgres://user:password@neon-host/database?sslmode=require
  ```

## 3. Environment Variables
Add these variables in Railway Dashboard → Variables:

```env
DATABASE_URL=postgres://user:pass@neon-host/db?sslmode=require
JWT_SECRET=your-super-secret-key
ASPNETCORE_ENVIRONMENT=Production
# or for Node.js:
# NODE_ENV=production
# PORT=5194
```

## 4. Auto Database Updates (Migrations)
### ASP.NET Core (Entity Framework)
1. In your backend project, ensure migrations exist:
   ```bash
   dotnet ef migrations add InitialCreate
   ```
2. In Railway, go to your service → **Settings** → **Deploy** → add a **Start Command**:
   ```bash
   dotnet ef database update && dotnet YourApp.dll
   ```
   OR, add it in your `Dockerfile` / `Procfile`.
3. **Alternative (Recommended)**: Run migrations in the app startup code:
   ```csharp
   using (var scope = app.Services.CreateScope())
   {
       var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
       db.Database.Migrate();
   }
   ```

### Node.js (Prisma)
1. Add to `package.json` scripts:
   ```json
   "migrate": "prisma migrate deploy",
   "start": "npm run migrate && node server.js"
   ```
2. Set Railway start command to: `npm run start`

## 5. Deploy
1. Push code to your connected branch.
2. Railway auto-deploys.
3. Check **Deploy Logs** in Railway dashboard.
4. Verify health endpoint: `GET https://your-railway-url/health` or `/api/doctors`

## 6. Connecting Frontend to Backend
The frontend `BASE_URL` in `constants/api.ts` is already set to your Railway app:
```ts
export const BASE_URL = "https://ai-medical-assistant-production-38a3.up.railway.app";
```

## Troubleshooting
- **Connection refused**: Ensure `DATABASE_URL` is correct and Neon allows your IP.
- **Migrations fail**: Check Neon connection limits; use pooled connection string.
- **CORS errors**: Add your Expo/mobile URL to backend CORS policy.
- **Build fails**: Check that your backend runs locally with `dotnet run` or `npm start` before pushing.

## Summary Flow
```
Push to GitHub → Railway Auto-Deploy → Run Migrations → App Live → Neon DB Updated
```
