IF OBJECT_ID(N'[__EFMigrationsHistory]') IS NULL
BEGIN
    CREATE TABLE [__EFMigrationsHistory] (
        [MigrationId] nvarchar(150) NOT NULL,
        [ProductVersion] nvarchar(32) NOT NULL,
        CONSTRAINT [PK___EFMigrationsHistory] PRIMARY KEY ([MigrationId])
    );
END;
GO

BEGIN TRANSACTION;
GO

CREATE TABLE [Specialties] (
    [Id] int NOT NULL IDENTITY,
    [Name] nvarchar(100) NOT NULL,
    [NameAr] nvarchar(max) NULL,
    CONSTRAINT [PK_Specialties] PRIMARY KEY ([Id])
);
GO

CREATE TABLE [Users] (
    [Id] int NOT NULL IDENTITY,
    [FullName] nvarchar(120) NOT NULL,
    [Email] nvarchar(256) NOT NULL,
    [PasswordHash] nvarchar(512) NOT NULL,
    [Role] nvarchar(20) NOT NULL,
    [PhoneNumber] nvarchar(20) NULL,
    [BirthDate] datetime2 NULL,
    [PhotoUrl] nvarchar(max) NULL,
    [IsActive] bit NOT NULL DEFAULT CAST(1 AS bit),
    [IsDeleted] bit NOT NULL DEFAULT CAST(0 AS bit),
    [CreatedAt] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    [UpdatedAt] datetime2 NULL,
    [Discriminator] nvarchar(5) NOT NULL,
    [LastLoginAt] datetime2 NULL,
    CONSTRAINT [PK_Users] PRIMARY KEY ([Id])
);
GO

CREATE TABLE [DoctorApplications] (
    [Id] int NOT NULL IDENTITY,
    [Name] nvarchar(max) NOT NULL,
    [Email] nvarchar(max) NOT NULL,
    [Phone] nvarchar(max) NOT NULL,
    [SpecialtyId] int NOT NULL,
    [Experience] int NOT NULL,
    [Bio] nvarchar(max) NOT NULL,
    [LicenseNumber] nvarchar(max) NOT NULL,
    [Message] nvarchar(max) NOT NULL,
    [DocumentUrl] nvarchar(max) NOT NULL,
    [PhotoUrl] nvarchar(max) NULL,
    [Status] nvarchar(max) NOT NULL,
    [SubmittedAt] datetime2 NOT NULL,
    [ProcessedAt] datetime2 NULL,
    CONSTRAINT [PK_DoctorApplications] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_DoctorApplications_Specialties_SpecialtyId] FOREIGN KEY ([SpecialtyId]) REFERENCES [Specialties] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Doctors] (
    [Id] int NOT NULL IDENTITY,
    [UserId] int NOT NULL,
    [SpecialtyId] int NOT NULL,
    [Name] nvarchar(200) NOT NULL,
    [License] nvarchar(max) NOT NULL,
    [Bio] nvarchar(1000) NULL,
    [ImageUrl] nvarchar(500) NULL,
    [ConsultationFee] decimal(10,2) NULL,
    [Experience] int NULL,
    [Rating] float NOT NULL DEFAULT 0.0E0,
    [ReviewCount] int NOT NULL DEFAULT 0,
    [Location] nvarchar(200) NULL,
    [IsAvailable] bit NOT NULL DEFAULT CAST(1 AS bit),
    [IsScheduleVisible] bit NOT NULL DEFAULT CAST(1 AS bit),
    [CreatedAt] datetime2 NOT NULL,
    [UpdatedAt] datetime2 NULL,
    CONSTRAINT [PK_Doctors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Doctors_Specialties_SpecialtyId] FOREIGN KEY ([SpecialtyId]) REFERENCES [Specialties] ([Id]) ON DELETE NO ACTION,
    CONSTRAINT [FK_Doctors_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id]) ON DELETE NO ACTION
);
GO

CREATE TABLE [Patients] (
    [Id] int NOT NULL IDENTITY,
    [FullName] nvarchar(100) NOT NULL,
    [Email] nvarchar(150) NOT NULL,
    [PhoneNumber] nvarchar(20) NOT NULL,
    [PasswordHash] nvarchar(max) NOT NULL,
    [DateOfBirth] datetime2 NOT NULL,
    [Gender] nvarchar(10) NOT NULL,
    [Address] nvarchar(300) NULL,
    [ImageUrl] nvarchar(max) NULL,
    [BloodType] nvarchar(5) NULL,
    [MedicalNotes] nvarchar(max) NULL,
    [CreatedAt] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    [IsActive] bit NOT NULL DEFAULT CAST(1 AS bit),
    [UserId] int NULL,
    CONSTRAINT [PK_Patients] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Patients_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id])
);
GO

CREATE TABLE [Session] (
    [Id] int NOT NULL IDENTITY,
    [UserId] int NOT NULL,
    [Title] nvarchar(200) NULL,
    [UrgencyLevel] nvarchar(20) NULL,
    [Type] nvarchar(max) NOT NULL,
    [IsDeleted] bit NOT NULL DEFAULT CAST(0 AS bit),
    [CreatedAt] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    [UpdatedAt] datetime2 NULL,
    CONSTRAINT [PK_Session] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Session_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id]) ON DELETE NO ACTION
);
GO

CREATE TABLE [DoctorAvailabilities] (
    [Id] int NOT NULL IDENTITY,
    [DoctorId] int NOT NULL,
    [DayOfWeek] tinyint NOT NULL,
    [StartTime] time NOT NULL,
    [EndTime] time NOT NULL,
    [IsAvailable] bit NOT NULL,
    [SlotDurationMinutes] int NOT NULL,
    CONSTRAINT [PK_DoctorAvailabilities] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_DoctorAvailabilities_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Reviews] (
    [Id] int NOT NULL IDENTITY,
    [DoctorId] int NOT NULL,
    [PatientId] int NULL,
    [Author] nvarchar(100) NOT NULL,
    [PatientName] nvarchar(max) NULL,
    [Rating] int NOT NULL,
    [Comment] nvarchar(1000) NOT NULL,
    [CreatedAt] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    CONSTRAINT [PK_Reviews] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Reviews_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [AllergyRecords] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [AllergyType] nvarchar(30) NOT NULL,
    [AllergenName] nvarchar(200) NOT NULL,
    [Severity] nvarchar(30) NOT NULL,
    [ReactionDescription] nvarchar(max) NULL,
    [FirstObservedDate] date NULL,
    [IsActive] bit NOT NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_AllergyRecords] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_AllergyRecords_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [AnalysisResults] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [SessionId] int NOT NULL,
    [MessageId] int NOT NULL,
    [SymptomsJson] nvarchar(max) NULL,
    [UrgencyLevel] nvarchar(max) NULL,
    [UrgencyScore] decimal(18,2) NULL,
    [Disclaimer] nvarchar(max) NOT NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_AnalysisResults] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_AnalysisResults_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [Appointments] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [DoctorId] int NOT NULL,
    [Date] nvarchar(20) NOT NULL,
    [Time] nvarchar(20) NOT NULL,
    [PaymentMethod] nvarchar(10) NOT NULL DEFAULT N'cash',
    [Status] nvarchar(20) NOT NULL DEFAULT N'Pending',
    [Notes] nvarchar(1000) NULL,
    [CreatedAt] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    CONSTRAINT [PK_Appointments] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Appointments_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE NO ACTION,
    CONSTRAINT [FK_Appointments_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [ChronicDiseaseMonitors] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [DiseaseName] nvarchar(200) NOT NULL,
    [DiseaseType] nvarchar(50) NOT NULL,
    [DiagnosedDate] date NULL,
    [Severity] nvarchar(20) NOT NULL,
    [IsActive] bit NOT NULL,
    [DoctorNotes] nvarchar(max) NULL,
    [TargetValues] nvarchar(max) NULL,
    [MonitoringFrequency] nvarchar(50) NOT NULL,
    [LastCheckDate] date NULL,
    [CreatedAt] datetime2 NOT NULL,
    [UpdatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_ChronicDiseaseMonitors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_ChronicDiseaseMonitors_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Consultations] (
    [Id] int NOT NULL IDENTITY,
    [DoctorId] int NOT NULL,
    [PatientId] int NOT NULL,
    [Title] nvarchar(max) NOT NULL,
    [Description] nvarchar(max) NOT NULL,
    [ScheduledAt] datetime2 NOT NULL,
    [Status] nvarchar(max) NOT NULL,
    [CreatedAt] datetime2 NOT NULL,
    [UpdatedAt] datetime2 NULL,
    CONSTRAINT [PK_Consultations] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Consultations_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_Consultations_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [FollowedDoctors] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [DoctorId] int NOT NULL,
    [FollowedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_FollowedDoctors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_FollowedDoctors_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_FollowedDoctors_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [MedicalProfiles] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [BloodType] nvarchar(5) NULL,
    [WeightKg] decimal(5,2) NULL,
    [HeightCm] decimal(5,2) NULL,
    [IsSmoker] bit NOT NULL,
    [SmokingDetails] nvarchar(max) NULL,
    [DrinksAlcohol] bit NOT NULL,
    [ExerciseHabits] nvarchar(100) NULL,
    [EmergencyContactName] nvarchar(200) NULL,
    [EmergencyContactPhone] nvarchar(30) NULL,
    [EmergencyContactRelation] nvarchar(100) NULL,
    [CreatedAt] datetime2 NOT NULL,
    [UpdatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_MedicalProfiles] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_MedicalProfiles_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [PatientVisits] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [DoctorId] int NOT NULL,
    [AppointmentId] int NULL,
    [ChiefComplaint] nvarchar(max) NOT NULL,
    [PresentIllnessHistory] nvarchar(max) NULL,
    [ExaminationFindings] nvarchar(max) NULL,
    [Assessment] nvarchar(max) NULL,
    [Plan] nvarchar(max) NULL,
    [Notes] nvarchar(max) NULL,
    [SummarySnapshot] nvarchar(max) NULL,
    [VisitDate] date NOT NULL,
    [Status] nvarchar(20) NOT NULL DEFAULT N'active',
    [CreatedAt] datetime2 NOT NULL,
    [ClosedAt] datetime2 NULL,
    CONSTRAINT [PK_PatientVisits] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_PatientVisits_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]),
    CONSTRAINT [FK_PatientVisits_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [SurgeryHistories] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [SurgeryName] nvarchar(300) NOT NULL,
    [SurgeryDate] date NOT NULL,
    [HospitalName] nvarchar(200) NULL,
    [DoctorName] nvarchar(200) NULL,
    [Complications] nvarchar(max) NULL,
    [Notes] nvarchar(max) NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_SurgeryHistories] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_SurgeryHistories_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Message] (
    [Id] int NOT NULL IDENTITY,
    [SessionId] int NOT NULL,
    [Role] nvarchar(20) NOT NULL,
    [Content] text NOT NULL,
    [MessageType] nvarchar(max) NOT NULL,
    [AttachmentUrl] nvarchar(max) NULL,
    [FileName] nvarchar(max) NULL,
    [SenderName] nvarchar(200) NOT NULL DEFAULT N'',
    [Timestamp] datetime2 NOT NULL DEFAULT (GETUTCDATE()),
    [SenderPhotoUrl] nvarchar(max) NULL,
    CONSTRAINT [PK_Message] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Message_Session_SessionId] FOREIGN KEY ([SessionId]) REFERENCES [Session] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [MedicationTrackers] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [PrescribedByDoctorId] int NULL,
    [ChronicDiseaseMonitorId] int NULL,
    [MedicationName] nvarchar(200) NOT NULL,
    [GenericName] nvarchar(200) NULL,
    [Dosage] nvarchar(100) NOT NULL,
    [Form] nvarchar(30) NOT NULL,
    [Frequency] nvarchar(100) NOT NULL,
    [TimesPerDay] int NOT NULL,
    [DoseTimes] nvarchar(200) NOT NULL,
    [StartDate] date NOT NULL,
    [EndDate] date NULL,
    [Instructions] nvarchar(max) NULL,
    [PillsRemaining] int NULL,
    [RefillThreshold] int NOT NULL DEFAULT 7,
    [IsChronic] bit NOT NULL,
    [IsActive] bit NOT NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_MedicationTrackers] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_MedicationTrackers_ChronicDiseaseMonitors_ChronicDiseaseMonitorId] FOREIGN KEY ([ChronicDiseaseMonitorId]) REFERENCES [ChronicDiseaseMonitors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_MedicationTrackers_Doctors_PrescribedByDoctorId] FOREIGN KEY ([PrescribedByDoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_MedicationTrackers_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [VitalReadings] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [ChronicDiseaseMonitorId] int NULL,
    [ReadingType] nvarchar(30) NOT NULL,
    [Value] decimal(8,2) NOT NULL,
    [Value2] decimal(8,2) NULL,
    [Unit] nvarchar(20) NOT NULL,
    [SugarReadingContext] nvarchar(20) NULL,
    [IsNormal] bit NOT NULL,
    [RecordedBy] nvarchar(20) NOT NULL,
    [Notes] nvarchar(max) NULL,
    [RecordedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_VitalReadings] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VitalReadings_ChronicDiseaseMonitors_ChronicDiseaseMonitorId] FOREIGN KEY ([ChronicDiseaseMonitorId]) REFERENCES [ChronicDiseaseMonitors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_VitalReadings_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [Symptoms] (
    [Id] int NOT NULL IDENTITY,
    [PatientVisitId] int NOT NULL,
    [Name] nvarchar(200) NOT NULL,
    [Severity] nvarchar(20) NOT NULL,
    [Duration] nvarchar(max) NULL,
    [Onset] nvarchar(20) NULL,
    [Progression] nvarchar(20) NULL,
    [Location] nvarchar(100) NULL,
    [IsChronic] bit NOT NULL,
    [Notes] nvarchar(max) NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_Symptoms] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Symptoms_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitDocuments] (
    [Id] int NOT NULL IDENTITY,
    [PatientVisitId] int NOT NULL,
    [DocumentType] nvarchar(30) NOT NULL,
    [Title] nvarchar(300) NOT NULL,
    [FileUrl] nvarchar(max) NOT NULL,
    [FileType] nvarchar(50) NOT NULL,
    [Description] nvarchar(max) NULL,
    [UploadedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_VisitDocuments] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitDocuments_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitPrescriptions] (
    [Id] int NOT NULL IDENTITY,
    [PatientVisitId] int NOT NULL,
    [MedicationName] nvarchar(200) NOT NULL,
    [GenericName] nvarchar(200) NULL,
    [Dosage] nvarchar(100) NOT NULL,
    [Form] nvarchar(30) NOT NULL,
    [Frequency] nvarchar(100) NOT NULL,
    [TimesPerDay] int NOT NULL,
    [SpecificTimes] nvarchar(200) NULL,
    [Duration] nvarchar(100) NULL,
    [Quantity] int NULL,
    [Instructions] nvarchar(max) NULL,
    [IsChronic] bit NOT NULL,
    [Refills] int NOT NULL DEFAULT 0,
    [IsDispensed] bit NOT NULL,
    [Notes] nvarchar(max) NULL,
    [CreatedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_VisitPrescriptions] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitPrescriptions_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitVitalSigns] (
    [Id] int NOT NULL IDENTITY,
    [PatientId] int NOT NULL,
    [PatientVisitId] int NOT NULL,
    [Type] nvarchar(30) NOT NULL,
    [Value] decimal(8,2) NOT NULL,
    [Value2] decimal(8,2) NULL,
    [Unit] nvarchar(20) NOT NULL,
    [IsAbnormal] bit NOT NULL,
    [NormalRangeMin] decimal(8,2) NULL,
    [NormalRangeMax] decimal(8,2) NULL,
    [RecordedBy] nvarchar(20) NOT NULL,
    [Notes] nvarchar(max) NULL,
    [RecordedAt] datetime2 NOT NULL,
    CONSTRAINT [PK_VisitVitalSigns] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitVitalSigns_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_VisitVitalSigns_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [MedicationLogs] (
    [Id] int NOT NULL IDENTITY,
    [MedicationTrackerId] int NOT NULL,
    [PatientId] int NOT NULL,
    [ScheduledAt] datetime2 NOT NULL,
    [TakenAt] datetime2 NULL,
    [Status] nvarchar(20) NOT NULL DEFAULT N'pending',
    [NotifiedAt] datetime2 NULL,
    CONSTRAINT [PK_MedicationLogs] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_MedicationLogs_MedicationTrackers_MedicationTrackerId] FOREIGN KEY ([MedicationTrackerId]) REFERENCES [MedicationTrackers] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_MedicationLogs_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE INDEX [IX_AllergyRecords_PatientId] ON [AllergyRecords] ([PatientId]);
GO

CREATE INDEX [IX_AllergyRecords_PatientId_IsActive] ON [AllergyRecords] ([PatientId], [IsActive]);
GO

CREATE INDEX [IX_AnalysisResults_PatientId] ON [AnalysisResults] ([PatientId]);
GO

CREATE INDEX [IX_Appointments_DoctorId] ON [Appointments] ([DoctorId]);
GO

CREATE INDEX [IX_Appointments_PatientId] ON [Appointments] ([PatientId]);
GO

CREATE INDEX [IX_ChronicDiseaseMonitors_PatientId] ON [ChronicDiseaseMonitors] ([PatientId]);
GO

CREATE INDEX [IX_ChronicDiseaseMonitors_PatientId_IsActive] ON [ChronicDiseaseMonitors] ([PatientId], [IsActive]);
GO

CREATE INDEX [IX_Consultations_DoctorId] ON [Consultations] ([DoctorId]);
GO

CREATE INDEX [IX_Consultations_PatientId] ON [Consultations] ([PatientId]);
GO

CREATE INDEX [IX_DoctorApplications_SpecialtyId] ON [DoctorApplications] ([SpecialtyId]);
GO

CREATE INDEX [IX_DoctorAvailabilities_DoctorId] ON [DoctorAvailabilities] ([DoctorId]);
GO

CREATE INDEX [IX_Doctors_SpecialtyId] ON [Doctors] ([SpecialtyId]);
GO

CREATE INDEX [IX_Doctors_UserId] ON [Doctors] ([UserId]);
GO

CREATE INDEX [IX_FollowedDoctors_DoctorId] ON [FollowedDoctors] ([DoctorId]);
GO

CREATE UNIQUE INDEX [IX_FollowedDoctors_PatientId_DoctorId] ON [FollowedDoctors] ([PatientId], [DoctorId]);
GO

CREATE UNIQUE INDEX [IX_MedicalProfiles_PatientId] ON [MedicalProfiles] ([PatientId]);
GO

CREATE INDEX [IX_MedicationLogs_MedicationTrackerId_Status] ON [MedicationLogs] ([MedicationTrackerId], [Status]);
GO

CREATE INDEX [IX_MedicationLogs_PatientId_ScheduledAt_Status] ON [MedicationLogs] ([PatientId], [ScheduledAt], [Status]);
GO

CREATE INDEX [IX_MedicationTrackers_ChronicDiseaseMonitorId] ON [MedicationTrackers] ([ChronicDiseaseMonitorId]);
GO

CREATE INDEX [IX_MedicationTrackers_PatientId_IsActive] ON [MedicationTrackers] ([PatientId], [IsActive]);
GO

CREATE INDEX [IX_MedicationTrackers_PrescribedByDoctorId] ON [MedicationTrackers] ([PrescribedByDoctorId]);
GO

CREATE INDEX [IX_Messages_SessionId] ON [Message] ([SessionId]);
GO

CREATE INDEX [IX_Messages_Timestamp] ON [Message] ([Timestamp]);
GO

CREATE UNIQUE INDEX [IX_Patients_Email] ON [Patients] ([Email]);
GO

CREATE INDEX [IX_Patients_UserId] ON [Patients] ([UserId]);
GO

CREATE INDEX [IX_PatientVisits_DoctorId_Status] ON [PatientVisits] ([DoctorId], [Status]);
GO

CREATE INDEX [IX_PatientVisits_DoctorId_VisitDate] ON [PatientVisits] ([DoctorId], [VisitDate]);
GO

CREATE INDEX [IX_PatientVisits_PatientId_VisitDate] ON [PatientVisits] ([PatientId], [VisitDate]);
GO

CREATE INDEX [IX_Reviews_DoctorId] ON [Reviews] ([DoctorId]);
GO

CREATE INDEX [IX_Session_UserId] ON [Session] ([UserId]);
GO

CREATE INDEX [IX_SurgeryHistories_PatientId] ON [SurgeryHistories] ([PatientId]);
GO

CREATE INDEX [IX_Symptoms_PatientVisitId] ON [Symptoms] ([PatientVisitId]);
GO

CREATE UNIQUE INDEX [IX_Users_Email] ON [Users] ([Email]);
GO

CREATE INDEX [IX_VisitDocuments_PatientVisitId] ON [VisitDocuments] ([PatientVisitId]);
GO

CREATE INDEX [IX_VisitPrescriptions_PatientVisitId] ON [VisitPrescriptions] ([PatientVisitId]);
GO

CREATE INDEX [IX_VisitVitalSigns_PatientId] ON [VisitVitalSigns] ([PatientId]);
GO

CREATE INDEX [IX_VisitVitalSigns_PatientVisitId] ON [VisitVitalSigns] ([PatientVisitId]);
GO

CREATE INDEX [IX_VitalReadings_ChronicDiseaseMonitorId] ON [VitalReadings] ([ChronicDiseaseMonitorId]);
GO

CREATE INDEX [IX_VitalReadings_PatientId_ReadingType_RecordedAt] ON [VitalReadings] ([PatientId], [ReadingType], [RecordedAt]);
GO

INSERT INTO [__EFMigrationsHistory] ([MigrationId], [ProductVersion])
VALUES (N'20260428193146_InitialCreate', N'8.0.24');
GO

COMMIT;
GO

