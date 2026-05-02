-- =====================================================
-- NEON DATABASE FULL RESET
-- This script DROPS all tables and recreates them
-- Run this in Neon SQL Editor
-- =====================================================

-- Step 1: Drop ALL tables (in correct dependency order)
DROP TABLE IF EXISTS "MedicationLogs" CASCADE;
DROP TABLE IF EXISTS "VisitVitalSigns" CASCADE;
DROP TABLE IF EXISTS "VisitPrescriptions" CASCADE;
DROP TABLE IF EXISTS "VisitDocuments" CASCADE;
DROP TABLE IF EXISTS "Symptoms" CASCADE;
DROP TABLE IF EXISTS "VitalReadings" CASCADE;
DROP TABLE IF EXISTS "MedicationTrackers" CASCADE;
DROP TABLE IF EXISTS "Message" CASCADE;
DROP TABLE IF EXISTS "AnalysisResults" CASCADE;
DROP TABLE IF EXISTS "PatientVisits" CASCADE;
DROP TABLE IF EXISTS "SurgeryHistories" CASCADE;
DROP TABLE IF EXISTS "MedicalProfiles" CASCADE;
DROP TABLE IF EXISTS "FollowedDoctors" CASCADE;
DROP TABLE IF EXISTS "Consultations" CASCADE;
DROP TABLE IF EXISTS "ChronicDiseaseMonitors" CASCADE;
DROP TABLE IF EXISTS "Appointments" CASCADE;
DROP TABLE IF EXISTS "AllergyRecords" CASCADE;
DROP TABLE IF EXISTS "Reviews" CASCADE;
DROP TABLE IF EXISTS "DoctorAvailabilities" CASCADE;
DROP TABLE IF EXISTS "DoctorApplications" CASCADE;
DROP TABLE IF EXISTS "Session" CASCADE;
DROP TABLE IF EXISTS "Patients" CASCADE;
DROP TABLE IF EXISTS "Doctors" CASCADE;
DROP TABLE IF EXISTS "Admins" CASCADE;
DROP TABLE IF EXISTS "Users" CASCADE;
DROP TABLE IF EXISTS "Specialties" CASCADE;
DROP TABLE IF EXISTS "__EFMigrationsHistory" CASCADE;

-- =====================================================
-- Step 2: Create ALL tables
-- =====================================================

-- 1. Specialties
CREATE TABLE "Specialties" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "NameAr" TEXT
);

-- 2. Users
CREATE TABLE "Users" (
    "Id" SERIAL PRIMARY KEY,
    "FullName" VARCHAR(120) NOT NULL,
    "Email" VARCHAR(256) NOT NULL,
    "PasswordHash" VARCHAR(512) NOT NULL,
    "Role" VARCHAR(20) NOT NULL,
    "PhoneNumber" VARCHAR(20),
    "BirthDate" TIMESTAMPTZ,
    "PhotoUrl" TEXT,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ
);

-- 3. Admins (TPT from Users)
CREATE TABLE "Admins" (
    "Id" INTEGER NOT NULL PRIMARY KEY REFERENCES "Users"("Id") ON DELETE CASCADE,
    "LastLoginAt" TIMESTAMPTZ
);

-- 4. Doctors
CREATE TABLE "Doctors" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "SpecialtyId" INTEGER NOT NULL,
    "Name" VARCHAR(200) NOT NULL,
    "License" TEXT NOT NULL,
    "Bio" VARCHAR(1000),
    "ImageUrl" VARCHAR(500),
    "ConsultationFee" NUMERIC(10,2),
    "Experience" INTEGER,
    "Rating" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "ReviewCount" INTEGER NOT NULL DEFAULT 0,
    "Location" VARCHAR(200),
    "IsAvailable" BOOLEAN NOT NULL DEFAULT TRUE,
    "IsScheduleVisible" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_Doctors_Specialties" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE RESTRICT,
    CONSTRAINT "FK_Doctors_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE RESTRICT
);

-- 5. DoctorAvailabilities
CREATE TABLE "DoctorAvailabilities" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "DayOfWeek" SMALLINT NOT NULL,
    "StartTime" TIME NOT NULL,
    "EndTime" TIME NOT NULL,
    "IsAvailable" BOOLEAN NOT NULL,
    "SlotDurationMinutes" INTEGER NOT NULL,
    CONSTRAINT "FK_DoctorAvailabilities_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);

-- 6. DoctorApplications
CREATE TABLE "DoctorApplications" (
    "Id" SERIAL PRIMARY KEY,
    "Name" TEXT NOT NULL,
    "Email" TEXT NOT NULL,
    "Phone" TEXT NOT NULL,
    "SpecialtyId" INTEGER NOT NULL,
    "Experience" INTEGER NOT NULL,
    "Bio" TEXT NOT NULL,
    "LicenseNumber" TEXT NOT NULL,
    "Message" TEXT NOT NULL,
    "DocumentUrl" TEXT NOT NULL,
    "PhotoUrl" TEXT,
    "Status" TEXT NOT NULL,
    "SubmittedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "ProcessedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_DoctorApplications_Specialties" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE CASCADE
);

-- 7. Patients
CREATE TABLE "Patients" (
    "Id" SERIAL PRIMARY KEY,
    "FullName" VARCHAR(100) NOT NULL,
    "Email" VARCHAR(150) NOT NULL,
    "PhoneNumber" VARCHAR(20) NOT NULL,
    "PasswordHash" TEXT NOT NULL,
    "DateOfBirth" TIMESTAMPTZ NOT NULL,
    "Gender" VARCHAR(10) NOT NULL,
    "Address" VARCHAR(300),
    "ImageUrl" TEXT,
    "BloodType" VARCHAR(5),
    "MedicalNotes" TEXT,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "UserId" INTEGER,
    CONSTRAINT "FK_Patients_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id")
);

-- 8. Session (singular - matches code)
CREATE TABLE "Session" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "Title" VARCHAR(200),
    "UrgencyLevel" VARCHAR(20),
    "Type" TEXT NOT NULL,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_Session_Users" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE RESTRICT
);

-- 9. Message (singular - matches code)
CREATE TABLE "Message" (
    "Id" SERIAL PRIMARY KEY,
    "SessionId" INTEGER NOT NULL,
    "Role" VARCHAR(20) NOT NULL,
    "Content" TEXT NOT NULL,
    "MessageType" TEXT NOT NULL,
    "AttachmentUrl" TEXT,
    "FileName" TEXT,
    "SenderName" VARCHAR(200) NOT NULL DEFAULT '',
    "Timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "SenderPhotoUrl" TEXT,
    CONSTRAINT "FK_Message_Session" FOREIGN KEY ("SessionId") REFERENCES "Session"("Id") ON DELETE CASCADE
);

-- 10. Appointments
CREATE TABLE "Appointments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "Date" VARCHAR(20) NOT NULL,
    "Time" VARCHAR(20) NOT NULL,
    "PaymentMethod" VARCHAR(10) NOT NULL DEFAULT 'cash',
    "Status" VARCHAR(20) NOT NULL DEFAULT 'Pending',
    "Notes" VARCHAR(1000),
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_Appointments_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE RESTRICT,
    CONSTRAINT "FK_Appointments_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);

-- 11. Reviews
CREATE TABLE "Reviews" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER,
    "Author" VARCHAR(100) NOT NULL,
    "PatientName" TEXT,
    "Rating" INTEGER NOT NULL,
    "Comment" VARCHAR(1000) NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_Reviews_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);

-- 12. Consultations
CREATE TABLE "Consultations" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "Title" TEXT NOT NULL,
    "Description" TEXT NOT NULL,
    "ScheduledAt" TIMESTAMPTZ NOT NULL,
    "Status" TEXT NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_Consultations_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_Consultations_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 13. FollowedDoctors
CREATE TABLE "FollowedDoctors" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "FollowedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_FollowedDoctors_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_FollowedDoctors_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 14. MedicalProfiles
CREATE TABLE "MedicalProfiles" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "BloodType" VARCHAR(5),
    "WeightKg" NUMERIC(5,2),
    "HeightCm" NUMERIC(5,2),
    "IsSmoker" BOOLEAN NOT NULL DEFAULT FALSE,
    "SmokingDetails" TEXT,
    "DrinksAlcohol" BOOLEAN NOT NULL DEFAULT FALSE,
    "ExerciseHabits" VARCHAR(100),
    "EmergencyContactName" VARCHAR(200),
    "EmergencyContactPhone" VARCHAR(30),
    "EmergencyContactRelation" VARCHAR(100),
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_MedicalProfiles_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 15. AllergyRecords
CREATE TABLE "AllergyRecords" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "AllergyType" VARCHAR(30) NOT NULL,
    "AllergenName" VARCHAR(200) NOT NULL,
    "Severity" VARCHAR(30) NOT NULL,
    "ReactionDescription" TEXT,
    "FirstObservedDate" DATE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_AllergyRecords_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 16. SurgeryHistories
CREATE TABLE "SurgeryHistories" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SurgeryName" VARCHAR(300) NOT NULL,
    "SurgeryDate" DATE NOT NULL,
    "HospitalName" VARCHAR(200),
    "DoctorName" VARCHAR(200),
    "Complications" TEXT,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_SurgeryHistories_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 17. ChronicDiseaseMonitors
CREATE TABLE "ChronicDiseaseMonitors" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DiseaseName" VARCHAR(200) NOT NULL,
    "DiseaseType" VARCHAR(50) NOT NULL,
    "DiagnosedDate" DATE,
    "Severity" VARCHAR(20) NOT NULL,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "DoctorNotes" TEXT,
    "TargetValues" TEXT,
    "MonitoringFrequency" VARCHAR(50) NOT NULL,
    "LastCheckDate" DATE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_ChronicDiseaseMonitors_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 18. VitalReadings
CREATE TABLE "VitalReadings" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "ChronicDiseaseMonitorId" INTEGER,
    "ReadingType" VARCHAR(30) NOT NULL,
    "Value" NUMERIC(8,2) NOT NULL,
    "Value2" NUMERIC(8,2),
    "Unit" VARCHAR(20) NOT NULL,
    "SugarReadingContext" VARCHAR(20),
    "IsNormal" BOOLEAN NOT NULL DEFAULT TRUE,
    "RecordedBy" VARCHAR(20) NOT NULL,
    "Notes" TEXT,
    "RecordedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_VitalReadings_ChronicDiseaseMonitors" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_VitalReadings_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);

-- 19. MedicationTrackers
CREATE TABLE "MedicationTrackers" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "PrescribedByDoctorId" INTEGER,
    "ChronicDiseaseMonitorId" INTEGER,
    "MedicationName" VARCHAR(200) NOT NULL,
    "GenericName" VARCHAR(200),
    "Dosage" VARCHAR(100) NOT NULL,
    "Form" VARCHAR(30) NOT NULL,
    "Frequency" VARCHAR(100) NOT NULL,
    "TimesPerDay" INTEGER NOT NULL,
    "DoseTimes" VARCHAR(200) NOT NULL,
    "StartDate" DATE NOT NULL,
    "EndDate" DATE,
    "Instructions" TEXT,
    "PillsRemaining" INTEGER,
    "RefillThreshold" INTEGER NOT NULL DEFAULT 7,
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_MedicationTrackers_ChronicDiseaseMonitors" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_MedicationTrackers_Doctors" FOREIGN KEY ("PrescribedByDoctorId") REFERENCES "Doctors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_MedicationTrackers_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);

-- 20. MedicationLogs
CREATE TABLE "MedicationLogs" (
    "Id" SERIAL PRIMARY KEY,
    "MedicationTrackerId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "ScheduledAt" TIMESTAMPTZ NOT NULL,
    "TakenAt" TIMESTAMPTZ,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'pending',
    "NotifiedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_MedicationLogs_MedicationTrackers" FOREIGN KEY ("MedicationTrackerId") REFERENCES "MedicationTrackers"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_MedicationLogs_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 21. PatientVisits
CREATE TABLE "PatientVisits" (
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
    "SummarySnapshot" TEXT,
    "VisitDate" DATE NOT NULL,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'active',
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "ClosedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_PatientVisits_Doctors" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id"),
    CONSTRAINT "FK_PatientVisits_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);

-- 22. Symptoms
CREATE TABLE "Symptoms" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "Name" VARCHAR(200) NOT NULL,
    "Severity" VARCHAR(20) NOT NULL,
    "Duration" TEXT,
    "Onset" VARCHAR(20),
    "Progression" VARCHAR(20),
    "Location" VARCHAR(100),
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_Symptoms_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

-- 23. VisitDocuments
CREATE TABLE "VisitDocuments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "DocumentType" VARCHAR(30) NOT NULL,
    "Title" VARCHAR(300) NOT NULL,
    "FileUrl" TEXT NOT NULL,
    "FileType" VARCHAR(50) NOT NULL,
    "Description" TEXT,
    "UploadedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_VisitDocuments_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

-- 24. VisitPrescriptions
CREATE TABLE "VisitPrescriptions" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "MedicationName" VARCHAR(200) NOT NULL,
    "GenericName" VARCHAR(200),
    "Dosage" VARCHAR(100) NOT NULL,
    "Form" VARCHAR(30) NOT NULL,
    "Frequency" VARCHAR(100) NOT NULL,
    "TimesPerDay" INTEGER NOT NULL,
    "SpecificTimes" VARCHAR(200),
    "Duration" VARCHAR(100),
    "Quantity" INTEGER,
    "Instructions" TEXT,
    "IsChronic" BOOLEAN NOT NULL DEFAULT FALSE,
    "Refills" INTEGER NOT NULL DEFAULT 0,
    "IsDispensed" BOOLEAN NOT NULL DEFAULT FALSE,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_VisitPrescriptions_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);

-- 25. VisitVitalSigns
CREATE TABLE "VisitVitalSigns" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "PatientVisitId" INTEGER NOT NULL,
    "Type" VARCHAR(30) NOT NULL,
    "Value" NUMERIC(8,2) NOT NULL,
    "Value2" NUMERIC(8,2),
    "Unit" VARCHAR(20) NOT NULL,
    "IsAbnormal" BOOLEAN NOT NULL DEFAULT FALSE,
    "NormalRangeMin" NUMERIC(8,2),
    "NormalRangeMax" NUMERIC(8,2),
    "RecordedBy" VARCHAR(20) NOT NULL,
    "Notes" TEXT,
    "RecordedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_VisitVitalSigns_PatientVisits" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_VisitVitalSigns_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);

-- 26. AnalysisResults
CREATE TABLE "AnalysisResults" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SessionId" INTEGER NOT NULL,
    "MessageId" INTEGER NOT NULL,
    "SymptomsJson" TEXT,
    "UrgencyLevel" TEXT,
    "UrgencyScore" NUMERIC(18,2),
    "Disclaimer" TEXT NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_AnalysisResults_Patients" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);

-- =====================================================
-- Step 3: Create ALL indexes
-- =====================================================

CREATE UNIQUE INDEX "IX_Users_Email" ON "Users" ("Email");
CREATE INDEX "IX_Doctors_SpecialtyId" ON "Doctors" ("SpecialtyId");
CREATE INDEX "IX_Doctors_UserId" ON "Doctors" ("UserId");
CREATE INDEX "IX_DoctorAvailabilities_DoctorId" ON "DoctorAvailabilities" ("DoctorId");
CREATE INDEX "IX_DoctorApplications_SpecialtyId" ON "DoctorApplications" ("SpecialtyId");
CREATE UNIQUE INDEX "IX_Patients_Email" ON "Patients" ("Email");
CREATE INDEX "IX_Patients_UserId" ON "Patients" ("UserId");
CREATE INDEX "IX_Session_UserId" ON "Session" ("UserId");
CREATE INDEX "IX_Messages_SessionId" ON "Message" ("SessionId");
CREATE INDEX "IX_Messages_Timestamp" ON "Message" ("Timestamp");
CREATE INDEX "IX_Appointments_DoctorId" ON "Appointments" ("DoctorId");
CREATE INDEX "IX_Appointments_PatientId" ON "Appointments" ("PatientId");
CREATE INDEX "IX_Reviews_DoctorId" ON "Reviews" ("DoctorId");
CREATE INDEX "IX_Consultations_DoctorId" ON "Consultations" ("DoctorId");
CREATE INDEX "IX_Consultations_PatientId" ON "Consultations" ("PatientId");
CREATE INDEX "IX_FollowedDoctors_DoctorId" ON "FollowedDoctors" ("DoctorId");
CREATE UNIQUE INDEX "IX_FollowedDoctors_PatientId_DoctorId" ON "FollowedDoctors" ("PatientId", "DoctorId");
CREATE UNIQUE INDEX "IX_MedicalProfiles_PatientId" ON "MedicalProfiles" ("PatientId");
CREATE INDEX "IX_AllergyRecords_PatientId" ON "AllergyRecords" ("PatientId");
CREATE INDEX "IX_AllergyRecords_PatientId_IsActive" ON "AllergyRecords" ("PatientId", "IsActive");
CREATE INDEX "IX_SurgeryHistories_PatientId" ON "SurgeryHistories" ("PatientId");
CREATE INDEX "IX_ChronicDiseaseMonitors_PatientId" ON "ChronicDiseaseMonitors" ("PatientId");
CREATE INDEX "IX_ChronicDiseaseMonitors_PatientId_IsActive" ON "ChronicDiseaseMonitors" ("PatientId", "IsActive");
CREATE INDEX "IX_VitalReadings_ChronicDiseaseMonitorId" ON "VitalReadings" ("ChronicDiseaseMonitorId");
CREATE INDEX "IX_VitalReadings_PatientId_ReadingType_RecordedAt" ON "VitalReadings" ("PatientId", "ReadingType", "RecordedAt");
CREATE INDEX "IX_MedicationTrackers_ChronicDiseaseMonitorId" ON "MedicationTrackers" ("ChronicDiseaseMonitorId");
CREATE INDEX "IX_MedicationTrackers_PatientId_IsActive" ON "MedicationTrackers" ("PatientId", "IsActive");
CREATE INDEX "IX_MedicationTrackers_PrescribedByDoctorId" ON "MedicationTrackers" ("PrescribedByDoctorId");
CREATE INDEX "IX_MedicationLogs_MedicationTrackerId_Status" ON "MedicationLogs" ("MedicationTrackerId", "Status");
CREATE INDEX "IX_MedicationLogs_PatientId_ScheduledAt_Status" ON "MedicationLogs" ("PatientId", "ScheduledAt", "Status");
CREATE INDEX "IX_PatientVisits_DoctorId_Status" ON "PatientVisits" ("DoctorId", "Status");
CREATE INDEX "IX_PatientVisits_DoctorId_VisitDate" ON "PatientVisits" ("DoctorId", "VisitDate");
CREATE INDEX "IX_PatientVisits_PatientId_VisitDate" ON "PatientVisits" ("PatientId", "VisitDate");
CREATE INDEX "IX_Symptoms_PatientVisitId" ON "Symptoms" ("PatientVisitId");
CREATE INDEX "IX_VisitDocuments_PatientVisitId" ON "VisitDocuments" ("PatientVisitId");
CREATE INDEX "IX_VisitPrescriptions_PatientVisitId" ON "VisitPrescriptions" ("PatientVisitId");
CREATE INDEX "IX_VisitVitalSigns_PatientId" ON "VisitVitalSigns" ("PatientId");
CREATE INDEX "IX_VisitVitalSigns_PatientVisitId" ON "VisitVitalSigns" ("PatientVisitId");
CREATE INDEX "IX_AnalysisResults_PatientId" ON "AnalysisResults" ("PatientId");

-- =====================================================
-- Step 4: EF Migrations History
-- =====================================================
CREATE TABLE "__EFMigrationsHistory" (
    "MigrationId" VARCHAR(150) NOT NULL PRIMARY KEY,
    "ProductVersion" VARCHAR(32) NOT NULL
);
INSERT INTO "__EFMigrationsHistory" ("MigrationId", "ProductVersion")
VALUES ('20260428193146_InitialCreate', '8.0.0');

-- =====================================================
-- DONE! All 26 tables + indexes + constraints created
-- =====================================================
