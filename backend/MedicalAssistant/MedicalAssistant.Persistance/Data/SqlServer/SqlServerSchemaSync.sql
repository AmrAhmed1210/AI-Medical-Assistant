-- Idempotent SQL Server schema sync (safe to run multiple times)

IF COL_LENGTH('Users', 'RefreshToken') IS NULL
    ALTER TABLE [Users] ADD [RefreshToken] nvarchar(max) NULL;

IF COL_LENGTH('Users', 'RefreshTokenExpiryTime') IS NULL
    ALTER TABLE [Users] ADD [RefreshTokenExpiryTime] datetime2 NULL;

IF COL_LENGTH('MedicalProfiles', 'AiDiagnosisSummary') IS NULL
    ALTER TABLE [MedicalProfiles] ADD [AiDiagnosisSummary] nvarchar(max) NULL;

IF COL_LENGTH('MedicalProfiles', 'LastAiAnalysisAt') IS NULL
    ALTER TABLE [MedicalProfiles] ADD [LastAiAnalysisAt] datetime2 NULL;

IF COL_LENGTH('MedicationTrackers', 'DaysOfWeek') IS NULL
    ALTER TABLE [MedicationTrackers] ADD [DaysOfWeek] nvarchar(max) NOT NULL
        CONSTRAINT [DF_MedicationTrackers_DaysOfWeek] DEFAULT (N'');

IF COL_LENGTH('PatientVisits', 'FollowUpAfterDays') IS NULL
    ALTER TABLE [PatientVisits] ADD [FollowUpAfterDays] int NULL;

IF COL_LENGTH('PatientVisits', 'FollowUpNotes') IS NULL
    ALTER TABLE [PatientVisits] ADD [FollowUpNotes] nvarchar(max) NULL;

IF COL_LENGTH('PatientVisits', 'FollowUpRequired') IS NULL
    ALTER TABLE [PatientVisits] ADD [FollowUpRequired] bit NULL;

IF OBJECT_ID(N'[PatientDocuments]', N'U') IS NULL
BEGIN
    CREATE TABLE [PatientDocuments] (
        [Id] int NOT NULL IDENTITY,
        [PatientId] int NOT NULL,
        [DocumentType] nvarchar(max) NOT NULL,
        [Title] nvarchar(max) NOT NULL,
        [FileUrl] nvarchar(max) NOT NULL,
        [FileType] nvarchar(max) NOT NULL,
        [Description] nvarchar(max) NULL,
        [DocumentDate] datetime2 NOT NULL,
        [UploadedAt] datetime2 NOT NULL,
        CONSTRAINT [PK_PatientDocuments] PRIMARY KEY ([Id]),
        CONSTRAINT [FK_PatientDocuments_Patients_PatientId] FOREIGN KEY ([PatientId]) REFERENCES [Patients] ([Id]) ON DELETE CASCADE
    );

    CREATE INDEX [IX_PatientDocuments_PatientId] ON [PatientDocuments] ([PatientId]);
END;

IF COL_LENGTH('VitalReadings', 'ChronicDiseaseMonitorId') IS NOT NULL
   AND (SELECT is_nullable FROM sys.columns
        WHERE object_id = OBJECT_ID('VitalReadings') AND name = 'ChronicDiseaseMonitorId') = 0
BEGIN
    ALTER TABLE [VitalReadings] ALTER COLUMN [ChronicDiseaseMonitorId] int NULL;
END;
