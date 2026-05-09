# MedBook Web - AI-Powered Medical Consultation Platform

## Overview

MedBook is an intelligent medical consultation platform that bridges the gap between patients and healthcare providers through AI-driven symptom analysis, electronic appointment scheduling, OCR-based prescription reading, and real-time doctor-patient communication.

## Account Creation Policy

> **Important:** Only the system administrator (Admin) can create new user accounts. There is no open self-registration in this system. Accounts are created exclusively through the Admin Dashboard &rarr; User Management &rarr; Add User.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| React 18 + Vite | Frontend framework |
| TypeScript | Type safety |
| Zustand | State management |
| Axios | HTTP client with JWT interceptors |
| Tailwind CSS | Styling |
| React Router DOM v6 | Navigation |
| Recharts | Data visualization |
| Framer Motion | Animations |
| @microsoft/signalr | Real-time chat |
| date-fns | Date formatting |
| Font Tajawal | Arabic typography |

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd medbook-web

# 2. Install dependencies
npm install

# 3. Configure environment variables
cp .env.example .env
# Update VITE_API_BASE_URL to point to your backend

# 4. Start the development server
npm run dev

# 5. Build for production
npm run build
```

## Project Structure

```
src/
├── api/              # API calls with Axios
│   ├── axiosInstance.ts   # JWT interceptors
│   ├── authApi.ts
│   ├── doctorApi.ts
│   ├── appointmentApi.ts
│   ├── adminApi.ts
│   └── consultApi.ts
├── store/            # Zustand stores
│   ├── authStore.ts
│   ├── doctorStore.ts
│   ├── appointmentStore.ts
│   └── notificationStore.ts
├── components/
│   ├── layout/       # DashboardLayout, Sidebar, TopBar
│   ├── ui/           # Button, Card, Badge, Modal, Table...
│   ├── doctor/       # DoctorCard, AppointmentTable, AIReportCard...
│   └── admin/        # StatCard, UserTable, ModelVersionTable
├── pages/
│   ├── auth/         # LoginPage
│   ├── admin/        # Dashboard, Users, Statistics, Models
│   └── doctor/       # Dashboard, Profile, Schedule, Appointments, Patients, Reports, Chat
├── hooks/            # useAuth, useDoctor, useNotifications
├── lib/              # types.ts, utils.ts, signalr.ts
└── constants/        # config.ts
```

## Roles & Permissions

### Admin (System Administrator)
- Add / disable / delete users
- View system statistics
- Manage AI models
- Hot-reload model configurations

### Doctor
- View and edit profile
- Manage appointments (confirm / cancel / complete)
- View patients and their medical history
- Read AI-generated reports
- Chat with patients
- Set availability schedules

### Patient
- Accounts are created by the Admin only
- Access the platform via the mobile application

## Environment Variables

```env
VITE_API_BASE_URL=http://localhost:5000
VITE_SIGNALR_HUB_URL=http://localhost:5000/hubs/consult
VITE_APP_NAME=MedBook
```

## Color System

| Color | Code | Usage |
|-------|------|-------|
| Primary | `#2563eb` | Buttons and primary elements |
| Success | `#22c55e` | Success and confirmation states |
| Warning | `#f59e0b` | Warnings |
| Danger | `#ef4444` | Errors and cancellations |
| Emergency | `#7f1d1d` | Emergency alerts with pulse animation |

## Security

- JWT tokens are stored in localStorage
- Axios interceptors automatically append the token
- 401 responses redirect to the login page
- Route guards prevent unauthorized access
- Role-based access control (RBAC) for every page
