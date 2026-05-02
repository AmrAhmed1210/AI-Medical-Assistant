-- =====================================================
-- MediCare Database Schema - Complete PostgreSQL Script
-- For Neon Database
-- =====================================================

-- =====================================================
-- 1. CORE TABLES (Users, Doctors, Patients)
-- =====================================================

-- Users Table
CREATE TABLE IF NOT EXISTS "Users" (
    "Id" SERIAL PRIMARY KEY,
    "FullName" VARCHAR(100) NOT NULL,
    "Email" VARCHAR(150) NOT NULL UNIQUE,
    "PasswordHash" VARCHAR(255) NOT NULL,
    "Role" VARCHAR(20) NOT NULL, -- Patient | Doctor | Admin | Secretary
    "PhoneNumber" VARCHAR(20),
    "BirthDate" TIMESTAMP WITH TIME ZONE,
    "PhotoUrl" TEXT,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE
);

-- Specialties Table
CREATE TABLE IF NOT EXISTS "Specialties" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "NameAr" VARCHAR(100)
);

-- Doctors Table
CREATE TABLE IF NOT EXISTS "Doctors" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "SpecialtyId" INTEGER NOT NULL,
    "Name" VARCHAR(100) NOT NULL,
    "License" VARCHAR(100) NOT NULL,
    "Bio" TEXT,
    "ImageUrl" TEXT,
    "ConsultationFee" DECIMAL(10,2),
    "Experience" INTEGER,
    "Rating" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "ReviewCount" INTEGER NOT NULL DEFAULT 0,
    "Location" VARCHAR(200),
    "IsAvailable" BOOLEAN NOT NULL DEFAULT TRUE,
    "IsScheduleVisible" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_Doctors_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_Doctors_Specialties" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE RESTRICT
);

-- Patients Table
CREATE TABLE IF NOT EXISTS "Patients" (
    "Id" SERIAL PRIMARY KEY,
    "FullName" VARCHAR(100) NOT NULL,
    "Email" VARCHAR(150) NOT NULL UNIQUE,
    "PhoneNumber" VARCHAR(20) NOT NULL,
    "PasswordHash" VARCHAR(255) NOT NULL,
    "DateOfBirth" TIMESTAMP WITH TIME ZONE NOT NULL,
    "Gender" VARCHAR(10) NOT NULL,
    "Address" VARCHAR(300),
    "ImageUrl" TEXT,
    "BloodType" VARCHAR(5),
    "MedicalNotes" TEXT,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "UserId" INTEGER,
    
    CONSTRAINT "FK_Patients_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE SET NULL
);

-- Admins Table (inherits from Users via FK)
CREATE TABLE IF NOT EXISTS "Admins" (
    "Id" SERIAL PRIMARY KEY,
    "LastLoginAt" TIMESTAMP WITH TIME ZONE,
    CONSTRAINT "FK_Admins_Users" FOREIGN KEY ("Id") REFERENCES "Users"("Id") ON DELETE CASCADE
);

-- =====================================================
-- 2. DOCTOR SCHEDULING
-- =====================================================

-- Doctor Availability Table
CREATE TABLE IF NOT EXISTS "DoctorAvailabilities" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "DayOfWeek" SMALLINT NOT NULL, -- 0=Sunday, 6=Saturday
    "StartTime" TIME NOT NULL,
    "EndTime" TIME NOT NULL,
    "IsAvailable" BOOLEAN NOT NULL DEFAULT TRUE,
    "SlotDurationMinutes" INTEGER NOT NULL DEFAULT 30,
    
    CONSTRAINT "FK_DoctorAvailabilities_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);

-- Doctor Applications Table
CREATE TABLE IF NOT EXISTS "DoctorApplications" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "Email" VARCHAR(150) NOT NULL,
    "Phone" VARCHAR(20) NOT NULL,
    "SpecialtyId" INTEGER NOT NULL,
    "Experience" INTEGER NOT NULL,
    "Bio" TEXT NOT NULL,
    "LicenseNumber" VARCHAR(100) NOT NULL,
    "Message" TEXT NOT NULL,
    "DocumentUrl" TEXT NOT NULL,
    "PhotoUrl" TEXT,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'Pending',
    "SubmittedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "ProcessedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_DoctorApplications_Specialties" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE RESTRICT
);

-- =====================================================
-- 3. APPOINTMENTS
-- =====================================================

CREATE TABLE IF NOT EXISTS "Appointments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "Date" VARCHAR(20) NOT NULL,
    "Time" VARCHAR(20) NOT NULL,
    "PaymentMethod" VARCHAR(10) NOT NULL DEFAULT 'cash',
    "Status" VARCHAR(20) NOT NULL DEFAULT 'Pending', -- Pending | Confirmed | Completed | Cancelled | NoShow
    "Notes" VARCHAR(1000),
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_Appointments_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE RESTRICT,
    CONSTRAINT "FK_Appointments_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE RESTRICT
);

CREATE INDEX "IX_Appointments_PatientId" ON "Appointments"("PatientId");
CREATE INDEX "IX_Appointments_DoctorId" ON "Appointments"("DoctorId");
CREATE INDEX "IX_Appointments_Status" ON "Appointments"("Status");

-- =====================================================
-- 4. CONSULTATIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS "Consultations" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "Title" VARCHAR(200) NOT NULL,
    "Description" TEXT NOT NULL,
    "ScheduledAt" TIMESTAMP WITH TIME ZONE NOT NULL,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'Scheduled',
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_Consultations_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_Consultations_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- =====================================================
-- 5. REVIEWS
-- =====================================================

CREATE TABLE IF NOT EXISTS "Reviews" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER,
    "Author" VARCHAR(100) NOT NULL,
    "PatientName" VARCHAR(100),
    "Rating" INTEGER NOT NULL,
    "Comment" TEXT NOT NULL,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_Reviews_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_Reviews_DoctorId" ON "Reviews"("DoctorId");

-- =====================================================
-- 6. AI SESSIONS & MESSAGES
-- =====================================================

CREATE TABLE IF NOT EXISTS "Sessions" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "Title" VARCHAR(200),
    "UrgencyLevel" VARCHAR(20),
    "Type" VARCHAR(20) NOT NULL DEFAULT 'AI',
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_Sessions_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "Messages" (
    "Id" SERIAL PRIMARY KEY,
    "SessionId" INTEGER NOT NULL,
    "Role" VARCHAR(20) NOT NULL DEFAULT 'user',
    "Content" TEXT NOT NULL,
    "MessageType" VARCHAR(20) NOT NULL DEFAULT 'text',
    "AttachmentUrl" TEXT,
    "FileName" VARCHAR(255),
    "SenderName" VARCHAR(100) NOT NULL,
    "Timestamp" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "SenderPhotoUrl" TEXT,
    
    CONSTRAINT "FK_Messages_Sessions" FOREIGN KEY ("SessionId") REFERENCES "Sessions"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_Messages_SessionId" ON "Messages"("SessionId");

-- =====================================================
-- 7. FOLLOWED DOCTORS
-- =====================================================

CREATE TABLE IF NOT EXISTS "FollowedDoctors" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "FollowedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_FollowedDoctors_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_FollowedDoctors_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "UQ_FollowedDoctors_Patient_Doctor" UNIQUE ("PatientId", "DoctorId")
);

CREATE INDEX "IX_FollowedDoctors_PatientId" ON "FollowedDoctors"("PatientId");
CREATE INDEX "IX_FollowedDoctors_DoctorId" ON "FollowedDoctors"("DoctorId");

-- =====================================================
-- 8. ANALYSIS RESULTS (AI)
-- =====================================================

CREATE TABLE IF NOT EXISTS "AnalysisResults" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SessionId" INTEGER NOT NULL,
    "MessageId" INTEGER NOT NULL,
    "SymptomsJson" TEXT,
    "UrgencyLevel" VARCHAR(20), -- HIGH, MEDIUM, LOW
    "UrgencyScore" DECIMAL(5,2),
    "Disclaimer" VARCHAR(100) NOT NULL DEFAULT 'This is not medical advice.',
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_AnalysisResults_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_AnalysisResults_PatientId" ON "AnalysisResults"("PatientId");

-- =====================================================
-- 9. HEALTH TRACKING - MEDICAL PROFILE
-- =====================================================

CREATE TABLE IF NOT EXISTS "MedicalProfiles" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL UNIQUE, -- 1:1 relationship
    "BloodType" VARCHAR(5),
    "WeightKg" DECIMAL(5,2),
    "HeightCm" DECIMAL(5,2),
    "IsSmoker" BOOLEAN NOT NULL DEFAULT FALSE,
    "SmokingDetails" TEXT,
    "DrinksAlcohol" BOOLEAN NOT NULL DEFAULT FALSE,
    "ExerciseHabits" VARCHAR(100),
    "EmergencyContactName" VARCHAR(200),
    "EmergencyContactPhone" VARCHAR(30),
    "EmergencyContactRelation" VARCHAR(100),
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_MedicalProfiles_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE UNIQUE INDEX "IX_MedicalProfiles_PatientId" ON "MedicalProfiles"("PatientId");

-- =====================================================
-- 10. HEALTH TRACKING - SURGERY HISTORY
-- =====================================================

CREATE TABLE IF NOT EXISTS "SurgeryHistories" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SurgeryName" VARCHAR(200) NOT NULL,
    "SurgeryDate" DATE NOT NULL,
    "HospitalName" VARCHAR(200),
    "DoctorName" VARCHAR(200),
    "Complications" TEXT,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_SurgeryHistories_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_SurgeryHistories_PatientId" ON "SurgeryHistories"("PatientId");

-- =====================================================
-- 11. HEALTH TRACKING - ALLERGY RECORDS
-- =====================================================

CREATE TABLE IF NOT EXISTS "AllergyRecords" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "AllergyType" VARCHAR(30) NOT NULL, -- medication | food | environmental
    "AllergenName" VARCHAR(200) NOT NULL,
    "Severity" VARCHAR(30) NOT NULL, -- mild | moderate | severe | life_threatening
    "ReactionDescription" TEXT,
    "FirstObservedDate" DATE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_AllergyRecords_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_AllergyRecords_PatientId" ON "AllergyRecords"("PatientId");

-- =====================================================
-- 12. HEALTH TRACKING - CHRONIC DISEASE MONITORS
-- =====================================================

CREATE TABLE IF NOT EXISTS "ChronicDiseaseMonitors" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DiseaseName" VARCHAR(200) NOT NULL, -- "Hypertension", "Type 2 Diabetes"
    "DiseaseType" VARCHAR(50) NOT NULL, -- hypertension | diabetes | cardiac | renal | other
    "DiagnosedDate" DATE,
    "Severity" VARCHAR(20) NOT NULL, -- mild | moderate | severe
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "DoctorNotes" TEXT,
    "TargetValues" TEXT, -- JSON: {"bp": "120/80", "sugar_fasting": "<100"}
    "MonitoringFrequency" VARCHAR(50) NOT NULL, -- daily | weekly | monthly
    "LastCheckDate" DATE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_ChronicDiseaseMonitors_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_ChronicDiseaseMonitors_PatientId" ON "ChronicDiseaseMonitors"("PatientId");

-- =====================================================
-- 13. HEALTH TRACKING - VITAL READINGS (Home Monitoring)
-- =====================================================

CREATE TABLE IF NOT EXISTS "VitalReadings" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "ChronicDiseaseMonitorId" INTEGER, -- nullable
    "ReadingType" VARCHAR(30) NOT NULL, -- blood_pressure | blood_sugar | weight | heart_rate | temperature | spo2
    "Value" DECIMAL(8,2) NOT NULL,
    "Value2" DECIMAL(8,2), -- for BP diastolic
    "Unit" VARCHAR(20) NOT NULL, -- mmHg | mg/dL | kg | bpm | °C | %
    "SugarReadingContext" VARCHAR(20), -- fasting | post_meal | random
    "IsNormal" BOOLEAN NOT NULL DEFAULT TRUE,
    "RecordedBy" VARCHAR(20) NOT NULL DEFAULT 'patient', -- patient | nurse | doctor
    "Notes" TEXT,
    "RecordedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_VitalReadings_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_VitalReadings_ChronicDiseaseMonitors" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL
);

CREATE INDEX "IX_VitalReadings_PatientId" ON "VitalReadings"("PatientId");
CREATE INDEX "IX_VitalReadings_ReadingType" ON "VitalReadings"("ReadingType");
CREATE INDEX "IX_VitalReadings_PatientId_ReadingType_RecordedAt" ON "VitalReadings"("PatientId", "ReadingType", "RecordedAt");

-- =====================================================
-- 14. HEALTH TRACKING - MEDICATION TRACKERS
-- =====================================================

CREATE TABLE IF NOT EXISTS "MedicationTrackers" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "PrescribedByDoctorId" INTEGER, -- nullable
    "ChronicDiseaseMonitorId" INTEGER, -- nullable, link drug to disease
    "MedicationName" VARCHAR(200) NOT NULL,
    "GenericName" VARCHAR(200),
    "Dosage" VARCHAR(100) NOT NULL,
    "Form" VARCHAR(30) NOT NULL, -- tablet | capsule | syrup | injection
    "Frequency" VARCHAR(100) NOT NULL,
    "TimesPerDay" INTEGER NOT NULL,
    "DoseTimes" TEXT NOT NULL, -- JSON: ["08:00", "20:00"]
    "StartDate" DATE NOT NULL,
    "EndDate" DATE,
    "Instructions" TEXT,
    "PillsRemaining" INTEGER,
    "RefillThreshold" INTEGER NOT NULL DEFAULT 7,
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_MedicationTrackers_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_MedicationTrackers_Doctors" FOREIGN KEY ("PrescribedByDoctorId") REFERENCES "Doctors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_MedicationTrackers_ChronicDiseaseMonitors" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL
);

CREATE INDEX "IX_MedicationTrackers_PatientId" ON "MedicationTrackers"("PatientId");

-- =====================================================
-- 15. HEALTH TRACKING - MEDICATION LOGS
-- =====================================================

CREATE TABLE IF NOT EXISTS "MedicationLogs" (
    "Id" SERIAL PRIMARY KEY,
    "MedicationTrackerId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "ScheduledAt" TIMESTAMP WITH TIME ZONE NOT NULL,
    "TakenAt" TIMESTAMP WITH TIME ZONE,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending | taken | missed | skipped
    "NotifiedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_MedicationLogs_MedicationTrackers" FOREIGN KEY ("MedicationTrackerId") REFERENCES "MedicationTrackers"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_MedicationLogs_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_MedicationLogs_MedicationTrackerId" ON "MedicationLogs"("MedicationTrackerId");
CREATE INDEX "IX_MedicationLogs_PatientId" ON "MedicationLogs"("PatientId");
CREATE INDEX "IX_MedicationLogs_ScheduledAt" ON "MedicationLogs"("ScheduledAt");

-- =====================================================
-- 16. HEALTH TRACKING - PATIENT VISITS
-- =====================================================

CREATE TABLE IF NOT EXISTS "PatientVisits" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "AppointmentId" INTEGER,
    "ChiefComplaint" TEXT NOT NULL,
    "PresentIllnessHistory" TEXT,
    "ExaminationFindings" TEXT,
    "Assessment" TEXT,
    "Plan" TEXT,
    "Notes" TEXT,
    "SummarySnapshot" TEXT, -- JSON snapshot
    "VisitDate" DATE NOT NULL DEFAULT CURRENT_DATE,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'active', -- active | completed | cancelled
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    "ClosedAt" TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT "FK_PatientVisits_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_PatientVisits_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE NO ACTION
);

CREATE INDEX "IX_PatientVisits_PatientId" ON "PatientVisits"("PatientId");
CREATE INDEX "IX_PatientVisits_DoctorId" ON "PatientVisits"("DoctorId");
CREATE INDEX "IX_PatientVisits_PatientId_VisitDate" ON "PatientVisits"("PatientId", "VisitDate");
CREATE INDEX "IX_PatientVisits_DoctorId_VisitDate" ON "PatientVisits"("DoctorId", "VisitDate");
CREATE INDEX "IX_PatientVisits_DoctorId_Status" ON "PatientVisits"("DoctorId", "Status");

-- =====================================================
-- 17. HEALTH TRACKING - SYMPTOMS
-- =====================================================

CREATE TABLE IF NOT EXISTS "Symptoms" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "Name" VARCHAR(200) NOT NULL,
    "Severity" VARCHAR(20) NOT NULL, -- mild | moderate | severe | life_threatening
    "Duration" VARCHAR(100),
    "Onset" VARCHAR(20), -- sudden | gradual
    "Progression" VARCHAR(20), -- worsening | stable | improving
    "Location" VARCHAR(100),
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_Symptoms_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_Symptoms_PatientVisitId" ON "Symptoms"("PatientVisitId");

-- =====================================================
-- 18. HEALTH TRACKING - VISIT VITAL SIGNS
-- =====================================================

CREATE TABLE IF NOT EXISTS "VisitVitalSigns" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "PatientVisitId" INTEGER NOT NULL,
    "Type" VARCHAR(30) NOT NULL, -- blood_pressure | blood_sugar | temperature | heart_rate | respiratory_rate | weight | height | spo2
    "Value" DECIMAL(8,2) NOT NULL,
    "Value2" DECIMAL(8,2), -- for BP diastolic
    "Unit" VARCHAR(20) NOT NULL,
    "IsAbnormal" BOOLEAN NOT NULL DEFAULT FALSE,
    "NormalRangeMin" DECIMAL(8,2),
    "NormalRangeMax" DECIMAL(8,2),
    "RecordedBy" VARCHAR(20) NOT NULL DEFAULT 'doctor', -- doctor | nurse
    "Notes" TEXT,
    "RecordedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_VisitVitalSigns_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_VisitVitalSigns_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_VisitVitalSigns_PatientId" ON "VisitVitalSigns"("PatientId");
CREATE INDEX "IX_VisitVitalSigns_PatientVisitId" ON "VisitVitalSigns"("PatientVisitId");

-- =====================================================
-- 19. HEALTH TRACKING - VISIT PRESCRIPTIONS
-- =====================================================

CREATE TABLE IF NOT EXISTS "VisitPrescriptions" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "MedicationName" VARCHAR(200) NOT NULL,
    "GenericName" VARCHAR(200),
    "Dosage" VARCHAR(100) NOT NULL,
    "Form" VARCHAR(30) NOT NULL, -- tablet | syrup | injection
    "Frequency" VARCHAR(100) NOT NULL,
    "TimesPerDay" INTEGER NOT NULL,
    "SpecificTimes" TEXT, -- JSON: ["08:00", "20:00"]
    "Duration" VARCHAR(100), -- "30 days"
    "Quantity" INTEGER,
    "Instructions" TEXT,
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "Refills" INTEGER NOT NULL DEFAULT 0,
    "IsDispensed" BOOLEAN NOT NULL DEFAULT FALSE,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_VisitPrescriptions_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_VisitPrescriptions_PatientVisitId" ON "VisitPrescriptions"("PatientVisitId");

-- =====================================================
-- 20. HEALTH TRACKING - VISIT DOCUMENTS
-- =====================================================

CREATE TABLE IF NOT EXISTS "VisitDocuments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "DocumentType" VARCHAR(30) NOT NULL, -- xray | lab_result | prescription_scan | ecg | other
    "Title" VARCHAR(300) NOT NULL,
    "FileUrl" TEXT NOT NULL,
    "FileType" VARCHAR(50) NOT NULL, -- image/jpeg | application/pdf
    "Description" TEXT,
    "UploadedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT "FK_VisitDocuments_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

CREATE INDEX "IX_VisitDocuments_PatientVisitId" ON "VisitDocuments"("PatientVisitId");

-- =====================================================
-- END OF SCHEMA
-- =====================================================

-- Print summary
SELECT 'Schema created successfully!' AS status;
SELECT COUNT(*) AS total_tables FROM information_schema.tables WHERE table_schema = 'public';

-- List all created tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- Verify foreign keys
SELECT 
    tc.table_name, 
    kcu.column_name, 
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name 
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name;

COMMIT;)
