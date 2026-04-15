# MedBook Backend — Setup & Run Guide

## ⚙️ Requirements

| Tool | Version |
|------|---------|
| .NET SDK | 10.0+ |
| SQL Server Express | Any |
| Postman | Any |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/AmrAhmed1210/AI-Medical-Assistant.git
cd AI-Medical-Assistant/backend/MedicalAssistant
```

### 2. Configure the database connection
Open `MedicalAssistant.Web/appsettings.Development.json` and set:
```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=.\\SQLEXPRESS;Database=MedicalAssistantDb;Trusted_Connection=True;TrustServerCertificate=True;"
  },
  "Jwt": {
    "Key": "YourSuperSecretKeyHereMustBe32CharsMin!!",
    "Issuer": "MedicalAssistant",
    "Audience": "MedicalAssistantApp",
    "ExpiresInDays": 7
  }
}
```
> ⚠️ Change `Jwt:Key` to any secure string (minimum 32 characters)

### 3. Apply database migrations
```bash
cd MedicalAssistant.Web
dotnet ef database update --project ..\MedicalAssistant.Persistance\MedicalAssistant.Persistance.csproj
```

### 4. Seed the database (first time only)
```bash
sqlcmd -S .\SQLEXPRESS -d MedicalAssistantDb -i "SeedData.sql"
```

### 5. Run the project
```bash
dotnet run
```

API: **http://localhost:5194**  
Swagger: **http://localhost:5194/swagger**

---

## 📡 API Endpoints

| # | Method | Endpoint | Auth |
|---|--------|----------|------|
| 1 | POST | `/auth/register` | ❌ |
| 2 | POST | `/auth/login` | ❌ |
| 3 | GET | `/doctors` | ❌ |
| 4 | GET | `/doctors?specialtyId=1` | ❌ |
| 5 | GET | `/doctors/{id}` | ❌ |
| 6 | POST | `/appointments` | ✅ |
| 7 | GET | `/appointments/my` | ✅ |
| 8 | DELETE | `/appointments/{id}` | ✅ |
| 9 | GET | `/reviews/{doctorId}` | ❌ |
| 10 | POST | `/reviews` | ✅ |
| 11 | GET | `/profile/me` | ✅ |
| 12 | PUT | `/profile/me` | ✅ |

---

## 🔐 Auth

### Register — `POST /auth/register`
```json
{
  "name": "Ahmed Ali",
  "email": "ahmed@example.com",
  "passwordHash": "password123",
  "role": "Patient",
  "phone": "01012345678"
}
```

### Login — `POST /auth/login`
```json
{
  "email": "ahmed@example.com",
  "passwordHash": "password123"
}
```

### Response (both endpoints)
```json
{
  "token": "eyJhbGci...",
  "name": "Ahmed Ali",
  "email": "ahmed@example.com",
  "role": "Patient",
  "phone": "01012345678"
}
```

### JWT Claims included in token
```csharp
new Claim("PatientId", patient.Id.ToString()),
new Claim("name",      patient.FullName),
new Claim(ClaimTypes.Email, patient.Email)
```

---

## 📝 Changes Made

### Auth Module (New)
- `AuthController` — `/auth/register` + `/auth/login`
- `AuthService` — BCrypt password hashing
- `IAuthService` interface
- `RegisterDto`, `LoginDto`, `AuthResponseDto`
- `PasswordHash` field added to `Patient` entity

### Appointment Module
- `AppointmentDate` + `AppointmentTime` → `Date` + `Time` (strings)
- Added `PaymentMethod` (`"visa"` or `"cash"`)
- Added `DoctorName` + `Specialty` to response
- `PatientId` from JWT token (not request body)
- DELETE returns `{ message }` instead of 204

### Routes Fixed
| Controller | Before | After |
|---|---|---|
| Doctors | `/api/Doctors` | `/doctors` |
| Reviews | `/api/Reviews/doctor/{id}` | `/reviews/{id}` |
| Appointments | `/appointments/my?patientId=1` | `/appointments/my` |
| Profile | `/profile/me?patientId=1` | `/profile/me` |

### Program.cs
- JWT Authentication added
- All repositories registered explicitly
- `ReviewMappingProfile` registered in AutoMapper
- `IAuthService` registered
- `UseAuthentication()` before `UseAuthorization()`

### DoctorService
- `GetAllDoctorsAsync()` includes Specialty
- `GetDoctorByIdAsync()` includes Specialty

### AppointmentRepository
- `GetByPatientIdWithDoctorAsync()` — Doctor + Specialty included
- `GetByIdWithDoctorAsync()` — Doctor + Specialty included

---

## ⚠️ Known Issues

### Specialty Filter
- Frontend sends `?specialty=Cardiology` (name)
- Backend expects `?specialtyId=1` (ID)
- Fix: update frontend to send `specialtyId`

### AutoMapper Warning
- `AutoMapper 16.1.0` has a vulnerability warning (not a blocker)

---

## 📦 Packages Added

| Package | Project | Version |
|---------|---------|---------|
| `BCrypt.Net-Next` | Services + Web | 4.0.3 |
| `Microsoft.AspNetCore.Authentication.JwtBearer` | Web | 10.0.0 |
| `System.IdentityModel.Tokens.Jwt` | Services | 8.14.0 |
| `Microsoft.IdentityModel.Tokens` | Services | 8.14.0 |
| `Microsoft.Extensions.Configuration.Abstractions` | Services | 10.0.0 |
| `Microsoft.EntityFrameworkCore.SqlServer` | Persistance | 10.0.0 |