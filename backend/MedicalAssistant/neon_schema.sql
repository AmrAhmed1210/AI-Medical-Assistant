-- =====================================================
-- Medical Assistant - PostgreSQL Schema for Neon
-- Generated from EF Core Migration (InitialCreate)
-- =====================================================

-- 1. Specialties
CREATE TABLE IF NOT EXISTS "Specialties" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "NameAr" TEXT
);

-- 2. Users (base table for Admin via TPT)
CREATE TABLE IF NOT EXISTS "Users" (
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
CREATE UNIQUE INDEX IF NOT EXISTS "IX_Users_Email" ON "Users" ("Email");

-- 3. Admins (TPT inheritance from Users)
CREATE TABLE IF NOT EXISTS "Admins" (
    "Id" INTEGER NOT NULL PRIMARY KEY REFERENCES "Users"("Id") ON DELETE CASCADE,
    "LastLoginAt" TIMESTAMPTZ
);

-- 4. Doctors
CREATE TABLE IF NOT EXISTS "Doctors" (
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
    CONSTRAINT "FK_Doctors_Specialties_SpecialtyId" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE RESTRICT,
    CONSTRAINT "FK_Doctors_Users_UserId" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS "IX_Doctors_SpecialtyId" ON "Doctors" ("SpecialtyId");
CREATE INDEX IF NOT EXISTS "IX_Doctors_UserId" ON "Doctors" ("UserId");

-- 5. DoctorAvailabilities
CREATE TABLE IF NOT EXISTS "DoctorAvailabilities" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "DayOfWeek" SMALLINT NOT NULL,
    "StartTime" TIME NOT NULL,
    "EndTime" TIME NOT NULL,
    "IsAvailable" BOOLEAN NOT NULL,
    "SlotDurationMinutes" INTEGER NOT NULL,
    CONSTRAINT "FK_DoctorAvailabilities_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_DoctorAvailabilities_DoctorId" ON "DoctorAvailabilities" ("DoctorId");

-- 6. DoctorApplications
CREATE TABLE IF NOT EXISTS "DoctorApplications" (
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
    CONSTRAINT "FK_DoctorApplications_Specialties_SpecialtyId" FOREIGN KEY ("SpecialtyId") REFERENCES "Specialties"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_DoctorApplications_SpecialtyId" ON "DoctorApplications" ("SpecialtyId");

-- 7. Patients
CREATE TABLE IF NOT EXISTS "Patients" (
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
    CONSTRAINT "FK_Patients_Users_UserId" FOREIGN KEY ("UserId") REFERENCES "Users"("Id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "IX_Patients_Email" ON "Patients" ("Email");
CREATE INDEX IF NOT EXISTS "IX_Patients_UserId" ON "Patients" ("UserId");

-- 8. Sessions
CREATE TABLE IF NOT EXISTS "Sessions" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "Title" VARCHAR(200),
    "UrgencyLevel" VARCHAR(20),
    "Type" TEXT NOT NULL,
    "IsDeleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_Sessions_Users_UserId" FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS "IX_Sessions_UserId" ON "Sessions" ("UserId");

-- 9. Messages
CREATE TABLE IF NOT EXISTS "Message" (
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
    CONSTRAINT "FK_Message_Sessions_SessionId" FOREIGN KEY ("SessionId") REFERENCES "Sessions"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_Messages_SessionId" ON "Message" ("SessionId");
CREATE INDEX IF NOT EXISTS "IX_Messages_Timestamp" ON "Message" ("Timestamp");

-- 10. Appointments
CREATE TABLE IF NOT EXISTS "Appointments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "Date" VARCHAR(20) NOT NULL,
    "Time" VARCHAR(20) NOT NULL,
    "PaymentMethod" VARCHAR(10) NOT NULL DEFAULT 'cash',
    "Status" VARCHAR(20) NOT NULL DEFAULT 'Pending',
    "Notes" VARCHAR(1000),
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_Appointments_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE RESTRICT,
    CONSTRAINT "FK_Appointments_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE NO ACTION
);
CREATE INDEX IF NOT EXISTS "IX_Appointments_DoctorId" ON "Appointments" ("DoctorId");
CREATE INDEX IF NOT EXISTS "IX_Appointments_PatientId" ON "Appointments" ("PatientId");

-- 11. Reviews
CREATE TABLE IF NOT EXISTS "Reviews" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER,
    "Author" VARCHAR(100) NOT NULL,
    "PatientName" TEXT,
    "Rating" INTEGER NOT NULL,
    "Comment" VARCHAR(1000) NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_Reviews_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_Reviews_DoctorId" ON "Reviews" ("DoctorId");

-- 12. Consultations
CREATE TABLE IF NOT EXISTS "Consultations" (
    "Id" SERIAL PRIMARY KEY,
    "DoctorId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "Title" TEXT NOT NULL,
    "Description" TEXT NOT NULL,
    "ScheduledAt" TIMESTAMPTZ NOT NULL,
    "Status" TEXT NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "UpdatedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_Consultations_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_Consultations_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_Consultations_DoctorId" ON "Consultations" ("DoctorId");
CREATE INDEX IF NOT EXISTS "IX_Consultations_PatientId" ON "Consultations" ("PatientId");

-- 13. FollowedDoctors
CREATE TABLE IF NOT EXISTS "FollowedDoctors" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "DoctorId" INTEGER NOT NULL,
    "FollowedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_FollowedDoctors_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_FollowedDoctors_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_FollowedDoctors_DoctorId" ON "FollowedDoctors" ("DoctorId");
CREATE UNIQUE INDEX IF NOT EXISTS "IX_FollowedDoctors_PatientId_DoctorId" ON "FollowedDoctors" ("PatientId", "DoctorId");

-- 14. MedicalProfiles
CREATE TABLE IF NOT EXISTS "MedicalProfiles" (
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
    CONSTRAINT "FK_MedicalProfiles_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "IX_MedicalProfiles_PatientId" ON "MedicalProfiles" ("PatientId");

-- 15. AllergyRecords
CREATE TABLE IF NOT EXISTS "AllergyRecords" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "AllergyType" VARCHAR(30) NOT NULL,
    "AllergenName" VARCHAR(200) NOT NULL,
    "Severity" VARCHAR(30) NOT NULL,
    "ReactionDescription" TEXT,
    "FirstObservedDate" DATE,
    "IsActive" BOOLEAN NOT NULL DEFAULT TRUE,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_AllergyRecords_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_AllergyRecords_PatientId" ON "AllergyRecords" ("PatientId");
CREATE INDEX IF NOT EXISTS "IX_AllergyRecords_PatientId_IsActive" ON "AllergyRecords" ("PatientId", "IsActive");

-- 16. SurgeryHistories
CREATE TABLE IF NOT EXISTS "SurgeryHistories" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SurgeryName" VARCHAR(300) NOT NULL,
    "SurgeryDate" DATE NOT NULL,
    "HospitalName" VARCHAR(200),
    "DoctorName" VARCHAR(200),
    "Complications" TEXT,
    "Notes" TEXT,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_SurgeryHistories_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_SurgeryHistories_PatientId" ON "SurgeryHistories" ("PatientId");

-- 17. ChronicDiseaseMonitors
CREATE TABLE IF NOT EXISTS "ChronicDiseaseMonitors" (
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
    CONSTRAINT "FK_ChronicDiseaseMonitors_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_ChronicDiseaseMonitors_PatientId" ON "ChronicDiseaseMonitors" ("PatientId");
CREATE INDEX IF NOT EXISTS "IX_ChronicDiseaseMonitors_PatientId_IsActive" ON "ChronicDiseaseMonitors" ("PatientId", "IsActive");

-- 18. VitalReadings
CREATE TABLE IF NOT EXISTS "VitalReadings" (
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
    CONSTRAINT "FK_VitalReadings_ChronicDiseaseMonitors_ChronicDiseaseMonitorId" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_VitalReadings_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE NO ACTION
);
CREATE INDEX IF NOT EXISTS "IX_VitalReadings_ChronicDiseaseMonitorId" ON "VitalReadings" ("ChronicDiseaseMonitorId");
CREATE INDEX IF NOT EXISTS "IX_VitalReadings_PatientId_ReadingType_RecordedAt" ON "VitalReadings" ("PatientId", "ReadingType", "RecordedAt");

-- 19. MedicationTrackers
CREATE TABLE IF NOT EXISTS "MedicationTrackers" (
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
    CONSTRAINT "FK_MedicationTrackers_ChronicDiseaseMonitors_ChronicDiseaseMonitorId" FOREIGN KEY ("ChronicDiseaseMonitorId") REFERENCES "ChronicDiseaseMonitors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_MedicationTrackers_Doctors_PrescribedByDoctorId" FOREIGN KEY ("PrescribedByDoctorId") REFERENCES "Doctors"("Id") ON DELETE SET NULL,
    CONSTRAINT "FK_MedicationTrackers_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE NO ACTION
);
CREATE INDEX IF NOT EXISTS "IX_MedicationTrackers_ChronicDiseaseMonitorId" ON "MedicationTrackers" ("ChronicDiseaseMonitorId");
CREATE INDEX IF NOT EXISTS "IX_MedicationTrackers_PatientId_IsActive" ON "MedicationTrackers" ("PatientId", "IsActive");
CREATE INDEX IF NOT EXISTS "IX_MedicationTrackers_PrescribedByDoctorId" ON "MedicationTrackers" ("PrescribedByDoctorId");

-- 20. MedicationLogs
CREATE TABLE IF NOT EXISTS "MedicationLogs" (
    "Id" SERIAL PRIMARY KEY,
    "MedicationTrackerId" INTEGER NOT NULL,
    "PatientId" INTEGER NOT NULL,
    "ScheduledAt" TIMESTAMPTZ NOT NULL,
    "TakenAt" TIMESTAMPTZ,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'pending',
    "NotifiedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_MedicationLogs_MedicationTrackers_MedicationTrackerId" FOREIGN KEY ("MedicationTrackerId") REFERENCES "MedicationTrackers"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_MedicationLogs_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_MedicationLogs_MedicationTrackerId_Status" ON "MedicationLogs" ("MedicationTrackerId", "Status");
CREATE INDEX IF NOT EXISTS "IX_MedicationLogs_PatientId_ScheduledAt_Status" ON "MedicationLogs" ("PatientId", "ScheduledAt", "Status");

-- 21. PatientVisits
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
    "SummarySnapshot" TEXT,
    "VisitDate" DATE NOT NULL,
    "Status" VARCHAR(20) NOT NULL DEFAULT 'active',
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "ClosedAt" TIMESTAMPTZ,
    CONSTRAINT "FK_PatientVisits_Doctors_DoctorId" FOREIGN KEY ("DoctorId") REFERENCES "Doctors"("Id"),
    CONSTRAINT "FK_PatientVisits_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_PatientVisits_DoctorId_Status" ON "PatientVisits" ("DoctorId", "Status");
CREATE INDEX IF NOT EXISTS "IX_PatientVisits_DoctorId_VisitDate" ON "PatientVisits" ("DoctorId", "VisitDate");
CREATE INDEX IF NOT EXISTS "IX_PatientVisits_PatientId_VisitDate" ON "PatientVisits" ("PatientId", "VisitDate");

-- 22. Symptoms
CREATE TABLE IF NOT EXISTS "Symptoms" (
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
    CONSTRAINT "FK_Symptoms_PatientVisits_PatientVisitId" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_Symptoms_PatientVisitId" ON "Symptoms" ("PatientVisitId");

-- 23. VisitDocuments
CREATE TABLE IF NOT EXISTS "VisitDocuments" (
    "Id" SERIAL PRIMARY KEY,
    "PatientVisitId" INTEGER NOT NULL,
    "DocumentType" VARCHAR(30) NOT NULL,
    "Title" VARCHAR(300) NOT NULL,
    "FileUrl" TEXT NOT NULL,
    "FileType" VARCHAR(50) NOT NULL,
    "Description" TEXT,
    "UploadedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_VisitDocuments_PatientVisits_PatientVisitId" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_VisitDocuments_PatientVisitId" ON "VisitDocuments" ("PatientVisitId");

-- 24. VisitPrescriptions
CREATE TABLE IF NOT EXISTS "VisitPrescriptions" (
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
    CONSTRAINT "FK_VisitPrescriptions_PatientVisits_PatientVisitId" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "IX_VisitPrescriptions_PatientVisitId" ON "VisitPrescriptions" ("PatientVisitId");

-- 25. VisitVitalSigns
CREATE TABLE IF NOT EXISTS "VisitVitalSigns" (
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
    CONSTRAINT "FK_VisitVitalSigns_PatientVisits_PatientVisitId" FOREIGN KEY ("PatientVisitId") REFERENCES "PatientVisits"("Id") ON DELETE CASCADE,
    CONSTRAINT "FK_VisitVitalSigns_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id")
);
CREATE INDEX IF NOT EXISTS "IX_VisitVitalSigns_PatientId" ON "VisitVitalSigns" ("PatientId");
CREATE INDEX IF NOT EXISTS "IX_VisitVitalSigns_PatientVisitId" ON "VisitVitalSigns" ("PatientVisitId");

-- 26. AnalysisResults
CREATE TABLE IF NOT EXISTS "AnalysisResults" (
    "Id" SERIAL PRIMARY KEY,
    "PatientId" INTEGER NOT NULL,
    "SessionId" INTEGER NOT NULL,
    "MessageId" INTEGER NOT NULL,
    "SymptomsJson" TEXT,
    "UrgencyLevel" TEXT,
    "UrgencyScore" NUMERIC(18,2),
    "Disclaimer" TEXT NOT NULL,
    "CreatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT "FK_AnalysisResults_Patients_PatientId" FOREIGN KEY ("PatientId") REFERENCES "Patients"("Id") ON DELETE NO ACTION
);
CREATE INDEX IF NOT EXISTS "IX_AnalysisResults_PatientId" ON "AnalysisResults" ("PatientId");

-- 27. EF Core Migrations History (required for EF Core)
CREATE TABLE IF NOT EXISTS "__EFMigrationsHistory" (
    "MigrationId" VARCHAR(150) NOT NULL PRIMARY KEY,
    "ProductVersion" VARCHAR(32) NOT NULL
);

-- Mark migration as applied
INSERT INTO "__EFMigrationsHistory" ("MigrationId", "ProductVersion")
VALUES ('20260428193146_InitialCreate', '8.0.0')
ON CONFLICT ("MigrationId") DO NOTHING;
