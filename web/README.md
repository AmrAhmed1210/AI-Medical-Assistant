# MedBook Web - منصة الاستشارات الطبية الذكية

<div dir="rtl">

## 🏥 نظرة عامة
MedBook هي منصة استشارات طبية مدعومة بالذكاء الاصطناعي تربط المرضى بالأطباء من خلال:
- تحليل الأعراض بالذكاء الاصطناعي
- حجز المواعيد الإلكترونية
- قراءة الوصفات الطبية عبر OCR
- محادثات فورية بين الأطباء والمرضى

## ⚠️ سياسة إنشاء الحسابات
> **مهم:** فقط مدير النظام (Admin) يمكنه إنشاء حسابات جديدة للمستخدمين. لا يوجد تسجيل مفتوح (self-registration) في هذا النظام. يتم إنشاء الحسابات من خلال لوحة تحكم الأدمن → إدارة المستخدمين → إضافة مستخدم.

## 🛠️ التقنيات المستخدمة
| التقنية | الاستخدام |
|---------|-----------|
| React 18 + Vite | إطار العمل |
| TypeScript | الكتابة الآمنة |
| Zustand | إدارة الحالة |
| Axios | HTTP client مع JWT interceptors |
| Tailwind CSS | التصميم |
| React Router DOM v6 | التنقل |
| Recharts | الرسوم البيانية |
| Framer Motion | الأنيميشن |
| @microsoft/signalr | الدردشة الفورية |
| date-fns | تنسيق التواريخ |
| Font Tajawal | خط عربي |

## 🚀 تشغيل المشروع

### المتطلبات
- Node.js 18+
- npm أو yarn

### خطوات التثبيت

```bash
# 1. استنساخ المشروع
git clone <repo-url>
cd medbook-web

# 2. تثبيت الحزم
npm install

# 3. إعداد المتغيرات البيئية
cp .env.example .env
# عدّل VITE_API_BASE_URL ليشير إلى الـ backend

# 4. تشغيل بيئة التطوير
npm run dev

# 5. البناء للإنتاج
npm run build
```

## 📁 هيكل المشروع

```
src/
├── api/              # API calls مع Axios
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

## 👥 الأدوار والصلاحيات

### Admin (مدير النظام)
- ✅ إضافة/تعطيل/حذف المستخدمين
- ✅ عرض إحصائيات النظام
- ✅ إدارة نماذج AI
- ✅ إعادة تحميل النماذج ساخناً

### Doctor (طبيب)
- ✅ عرض وتعديل الملف الشخصي
- ✅ إدارة المواعيد (تأكيد/إلغاء/إكمال)
- ✅ عرض المرضى وتاريخهم
- ✅ قراءة تقارير AI
- ✅ المحادثة مع المرضى
- ✅ ضبط أوقات التوافر

### Patient (مريض)
- يتم إنشاء حسابهم من قِبل الأدمن فقط
- يصلون للمنصة عبر التطبيق المحمول

## 🔌 متغيرات البيئة

```env
VITE_API_BASE_URL=http://localhost:5000
VITE_SIGNALR_HUB_URL=http://localhost:5000/hubs/consult
VITE_APP_NAME=MedBook
```

## 🎨 نظام الألوان

| اللون | الكود | الاستخدام |
|-------|-------|-----------|
| Primary | `#2563eb` | الأزرار والعناصر الرئيسية |
| Success | `#22c55e` | النجاح والتأكيد |
| Warning | `#f59e0b` | التحذيرات |
| Danger | `#ef4444` | الأخطاء والإلغاء |
| Emergency | `#7f1d1d` | الطوارئ مع pulse |

## 🔐 الأمان
- JWT tokens محفوظة في localStorage
- Axios interceptors تضيف التوكن تلقائياً
- 401 responses تعيد توجيه لصفحة الدخول
- Route guards تمنع الوصول غير المصرح
- Role-based access control لكل صفحة

</div>
