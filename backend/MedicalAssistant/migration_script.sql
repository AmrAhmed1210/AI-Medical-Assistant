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
    [Id] integer NOT NULL,
    [Name] character varying(100) NOT NULL,
    [NameAr] text NULL,
    CONSTRAINT [PK_Specialties] PRIMARY KEY ([Id])
);
GO

CREATE TABLE [Users] (
    [Id] integer NOT NULL,
    [FullName] character varying(120) NOT NULL,
    [Email] character varying(256) NOT NULL,
    [PasswordHash] character varying(512) NOT NULL,
    [Role] character varying(20) NOT NULL,
    [PhoneNumber] character varying(20) NULL,
    [BirthDate] timestamp with time zone NULL,
    [PhotoUrl] text NULL,
    [IsActive] boolean NOT NULL DEFAULT CAST(1 AS boolean),
    [IsDeleted] boolean NOT NULL DEFAULT CAST(0 AS boolean),
    [CreatedAt] timestamp with time zone NOT NULL DEFAULT (NOW()),
    [UpdatedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_Users] PRIMARY KEY ([Id])
);
GO

CREATE TABLE [DoctorApplications] (
    [Id] integer NOT NULL,
    [Name] text NOT NULL,
    [Email] text NOT NULL,
    [Phone] text NOT NULL,
    [SpecialtyId] integer NOT NULL,
    [Experience] integer NOT NULL,
    [Bio] text NOT NULL,
    [LicenseNumber] text NOT NULL,
    [Message] text NOT NULL,
    [DocumentUrl] text NOT NULL,
    [PhotoUrl] text NULL,
    [Status] text NOT NULL,
    [SubmittedAt] timestamp with time zone NOT NULL,
    [ProcessedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_DoctorApplications] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_DoctorApplications_Specialties_SpecialtyId] FOREIGN KEY ([SpecialtyId]) REFERENCES [Specialties] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Admins] (
    [Id] integer NOT NULL,
    [LastLoginAt] timestamp with time zone NULL,
    CONSTRAINT [PK_Admins] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Admins_Users_Id] FOREIGN KEY ([Id]) REFERENCES [Users] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Doctors] (
    [Id] integer NOT NULL,
    [UserId] integer NOT NULL,
    [SpecialtyId] integer NOT NULL,
    [Name] character varying(200) NOT NULL,
    [License] text NOT NULL,
    [Bio] character varying(1000) NULL,
    [ImageUrl] character varying(500) NULL,
    [ConsultationFee] numeric(10,2) NULL,
    [Experience] integer NULL,
    [Rating] double precision NOT NULL DEFAULT 0.0E0,
    [ReviewCount] integer NOT NULL DEFAULT 0,
    [Location] character varying(200) NULL,
    [IsAvailable] boolean NOT NULL DEFAULT CAST(1 AS boolean),
    [IsScheduleVisible] boolean NOT NULL DEFAULT CAST(1 AS boolean),
    [CreatedAt] timestamp with time zone NOT NULL,
    [UpdatedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_Doctors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Doctors_Specialties_SpecialtyId] FOREIGN KEY ([SpecialtyId]) REFERENCES [Specialties] ([Id]) ON DELETE NO ACTION,
    CONSTRAINT [FK_Doctors_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id]) ON DELETE NO ACTION
);
GO

CREATE TABLE [Patients] (
    [Id] integer NOT NULL,
    [FullName] character varying(100) NOT NULL,
    [Email] character varying(150) NOT NULL,
    [PhoneNumber] character varying(20) NOT NULL,
    [PasswordHash] text NOT NULL,
    [DateOfBirth] timestamp with time zone NOT NULL,
    [Gender] character varying(10) NOT NULL,
    [Address] character varying(300) NULL,
    [ImageUrl] text NULL,
    [BloodType] character varying(5) NULL,
    [MedicalNotes] text NULL,
    [CreatedAt] timestamp with time zone NOT NULL DEFAULT (NOW()),
    [IsActive] boolean NOT NULL DEFAULT CAST(1 AS boolean),
    [UserId] integer NULL,
    CONSTRAINT [PK_Patients] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Patients_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id])
);
GO

CREATE TABLE [Session] (
    [Id] integer NOT NULL,
    [UserId] integer NOT NULL,
    [Title] character varying(200) NULL,
    [UrgencyLevel] character varying(20) NULL,
    [Type] text NOT NULL,
    [IsDeleted] boolean NOT NULL DEFAULT CAST(0 AS boolean),
    [CreatedAt] timestamp with time zone NOT NULL DEFAULT (NOW()),
    [UpdatedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_Session] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Session_Users_UserId] FOREIGN KEY ([UserId]) REFERENCES [Users] ([Id]) ON DELETE NO ACTION
);
GO

CREATE TABLE [DoctorAvailabilities] (
    [Id] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [DayOfWeek] smallint NOT NULL,
    [StartTime] interval NOT NULL,
    [EndTime] interval NOT NULL,
    [IsAvailable] boolean NOT NULL,
    [SlotDurationMinutes] integer NOT NULL,
    CONSTRAINT [PK_DoctorAvailabilities] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_DoctorAvailabilities_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Reviews] (
    [Id] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [PatientId] integer NULL,
    [Author] character varying(100) NOT NULL,
    [PatientName] text NULL,
    [Rating] integer NOT NULL,
    [Comment] character varying(1000) NOT NULL,
    [CreatedAt] timestamp with time zone NOT NULL DEFAULT (NOW()),
    CONSTRAINT [PK_Reviews] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Reviews_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [AllergyRecords] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [AllergyType] character varying(30) NOT NULL,
    [AllergenName] character varying(200) NOT NULL,
    [Severity] character varying(30) NOT NULL,
    [ReactionDescription] text NULL,
    [FirstObservedDate] date NULL,
    [IsActive] boolean NOT NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_AllergyRecords] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_AllergyRecords_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [AnalysisResults] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [SessionId] integer NOT NULL,
    [MessageId] integer NOT NULL,
    [SymptomsJson] text NULL,
    [UrgencyLevel] text NULL,
    [UrgencyScore] numeric NULL,
    [Disclaimer] text NOT NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_AnalysisResults] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_AnalysisResults_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Appointments] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [Date] character varying(20) NOT NULL,
    [Time] character varying(20) NOT NULL,
    [PaymentMethod] character varying(10) NOT NULL DEFAULT 'cash',
    [Status] character varying(20) NOT NULL DEFAULT 'Pending',
    [Notes] character varying(1000) NULL,
    [CreatedAt] timestamp with time zone NOT NULL DEFAULT (NOW()),
    CONSTRAINT [PK_Appointments] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Appointments_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE NO ACTION,
    CONSTRAINT [FK_Appointments_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE NO ACTION
);
GO

CREATE TABLE [ChronicDiseaseMonitors] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [DiseaseName] character varying(200) NOT NULL,
    [DiseaseType] character varying(50) NOT NULL,
    [DiagnosedDate] date NULL,
    [Severity] character varying(20) NOT NULL,
    [IsActive] boolean NOT NULL,
    [DoctorNotes] text NULL,
    [TargetValues] text NULL,
    [MonitoringFrequency] character varying(50) NOT NULL,
    [LastCheckDate] date NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    [UpdatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_ChronicDiseaseMonitors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_ChronicDiseaseMonitors_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Consultations] (
    [Id] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [Title] text NOT NULL,
    [Description] text NOT NULL,
    [ScheduledAt] timestamp with time zone NOT NULL,
    [Status] text NOT NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    [UpdatedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_Consultations] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Consultations_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_Consultations_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [FollowedDoctors] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [FollowedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_FollowedDoctors] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_FollowedDoctors_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_FollowedDoctors_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [MedicalProfiles] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [BloodType] character varying(5) NULL,
    [WeightKg] numeric(5,2) NULL,
    [HeightCm] numeric(5,2) NULL,
    [IsSmoker] boolean NOT NULL,
    [SmokingDetails] text NULL,
    [DrinksAlcohol] boolean NOT NULL,
    [ExerciseHabits] character varying(100) NULL,
    [EmergencyContactName] character varying(200) NULL,
    [EmergencyContactPhone] character varying(30) NULL,
    [EmergencyContactRelation] character varying(100) NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    [UpdatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_MedicalProfiles] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_MedicalProfiles_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [PatientVisits] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [DoctorId] integer NOT NULL,
    [AppointmentId] integer NULL,
    [ChiefComplaint] text NOT NULL,
    [PresentIllnessHistory] text NULL,
    [ExaminationFindings] text NULL,
    [Assessment] text NULL,
    [Plan] text NULL,
    [Notes] text NULL,
    [SummarySnapshot] text NULL,
    [VisitDate] date NOT NULL,
    [Status] character varying(20) NOT NULL DEFAULT 'active',
    [CreatedAt] timestamp with time zone NOT NULL,
    [ClosedAt] timestamp with time zone NULL,
    CONSTRAINT [PK_PatientVisits] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_PatientVisits_Doctors_DoctorId] FOREIGN KEY ([DoctorId]) REFERENCES [Doctors] ([Id]),
    CONSTRAINT [FK_PatientVisits_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [SurgeryHistories] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [SurgeryName] character varying(300) NOT NULL,
    [SurgeryDate] date NOT NULL,
    [HospitalName] character varying(200) NULL,
    [DoctorName] character varying(200) NULL,
    [Complications] text NULL,
    [Notes] text NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_SurgeryHistories] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_SurgeryHistories_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Message] (
    [Id] integer NOT NULL,
    [SessionId] integer NOT NULL,
    [Role] character varying(20) NOT NULL,
    [Content] text NOT NULL,
    [MessageType] text NOT NULL,
    [AttachmentUrl] text NULL,
    [FileName] text NULL,
    [SenderName] character varying(200) NOT NULL DEFAULT '',
    [Timestamp] timestamp with time zone NOT NULL DEFAULT (NOW()),
    [SenderPhotoUrl] text NULL,
    CONSTRAINT [PK_Message] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Message_Session_SessionId] FOREIGN KEY ([SessionId]) REFERENCES [Session] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [MedicationTrackers] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [PrescribedByDoctorId] integer NULL,
    [ChronicDiseaseMonitorId] integer NULL,
    [MedicationName] character varying(200) NOT NULL,
    [GenericName] character varying(200) NULL,
    [Dosage] character varying(100) NOT NULL,
    [Form] character varying(30) NOT NULL,
    [Frequency] character varying(100) NOT NULL,
    [TimesPerDay] integer NOT NULL,
    [DoseTimes] character varying(200) NOT NULL,
    [StartDate] date NOT NULL,
    [EndDate] date NULL,
    [Instructions] text NULL,
    [PillsRemaining] integer NULL,
    [RefillThreshold] integer NOT NULL DEFAULT 7,
    [IsChronic] boolean NOT NULL,
    [IsActive] boolean NOT NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_MedicationTrackers] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_MedicationTrackers_ChronicDiseaseMonitors_ChronicDiseaseMon~] FOREIGN KEY ([ChronicDiseaseMonitorId]) REFERENCES [ChronicDiseaseMonitors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_MedicationTrackers_Doctors_PrescribedByDoctorId] FOREIGN KEY ([PrescribedByDoctorId]) REFERENCES [Doctors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_MedicationTrackers_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VitalReadings] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [ChronicDiseaseMonitorId] integer NOT NULL,
    [ReadingType] character varying(30) NOT NULL,
    [Value] numeric(8,2) NOT NULL,
    [Value2] numeric(8,2) NULL,
    [Unit] character varying(20) NOT NULL,
    [SugarReadingContext] character varying(20) NULL,
    [IsNormal] boolean NOT NULL,
    [RecordedBy] character varying(20) NOT NULL,
    [Notes] text NULL,
    [RecordedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_VitalReadings] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VitalReadings_ChronicDiseaseMonitors_ChronicDiseaseMonitorId] FOREIGN KEY ([ChronicDiseaseMonitorId]) REFERENCES [ChronicDiseaseMonitors] ([Id]) ON DELETE SET NULL,
    CONSTRAINT [FK_VitalReadings_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [Symptoms] (
    [Id] integer NOT NULL,
    [PatientVisitId] integer NOT NULL,
    [Name] character varying(200) NOT NULL,
    [Severity] character varying(20) NOT NULL,
    [Duration] text NULL,
    [Onset] character varying(20) NULL,
    [Progression] character varying(20) NULL,
    [Location] character varying(100) NULL,
    [IsChronic] boolean NOT NULL,
    [Notes] text NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_Symptoms] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_Symptoms_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitDocuments] (
    [Id] integer NOT NULL,
    [PatientVisitId] integer NOT NULL,
    [DocumentType] character varying(30) NOT NULL,
    [Title] character varying(300) NOT NULL,
    [FileUrl] text NOT NULL,
    [FileType] character varying(50) NOT NULL,
    [Description] text NULL,
    [UploadedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_VisitDocuments] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitDocuments_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitPrescriptions] (
    [Id] integer NOT NULL,
    [PatientVisitId] integer NOT NULL,
    [MedicationName] character varying(200) NOT NULL,
    [GenericName] character varying(200) NULL,
    [Dosage] character varying(100) NOT NULL,
    [Form] character varying(30) NOT NULL,
    [Frequency] character varying(100) NOT NULL,
    [TimesPerDay] integer NOT NULL,
    [SpecificTimes] character varying(200) NULL,
    [Duration] character varying(100) NULL,
    [Quantity] integer NULL,
    [Instructions] text NULL,
    [IsChronic] boolean NOT NULL,
    [Refills] integer NOT NULL DEFAULT 0,
    [IsDispensed] boolean NOT NULL,
    [Notes] text NULL,
    [CreatedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_VisitPrescriptions] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitPrescriptions_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE
);
GO

CREATE TABLE [VisitVitalSigns] (
    [Id] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [PatientVisitId] integer NOT NULL,
    [Type] character varying(30) NOT NULL,
    [Value] numeric(8,2) NOT NULL,
    [Value2] numeric(8,2) NULL,
    [Unit] character varying(20) NOT NULL,
    [IsAbnormal] boolean NOT NULL,
    [NormalRangeMin] numeric(8,2) NULL,
    [NormalRangeMax] numeric(8,2) NULL,
    [RecordedBy] character varying(20) NOT NULL,
    [Notes] text NULL,
    [RecordedAt] timestamp with time zone NOT NULL,
    CONSTRAINT [PK_VisitVitalSigns] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_VisitVitalSigns_PatientVisits_PatientVisitId] FOREIGN KEY ([PatientVisitId]) REFERENCES [PatientVisits] ([Id]) ON DELETE CASCADE,
    CONSTRAINT [FK_VisitVitalSigns_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id])
);
GO

CREATE TABLE [MedicationLogs] (
    [Id] integer NOT NULL,
    [MedicationTrackerId] integer NOT NULL,
    [PatientId] integer NOT NULL,
    [ScheduledAt] timestamp with time zone NOT NULL,
    [TakenAt] timestamp with time zone NULL,
    [Status] character varying(20) NOT NULL DEFAULT 'pending',
    [NotifiedAt] timestamp with time zone NULL,
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
VALUES (N'20260502185055_InitialPostgresCreate', N'8.0.24');
GO

COMMIT;
GO

BEGIN TRANSACTION;
GO

INSERT INTO [__EFMigrationsHistory] ([MigrationId], [ProductVersion])
VALUES (N'20260503122539_AddSecretarySystem', N'8.0.24');
GO

COMMIT;
GO

