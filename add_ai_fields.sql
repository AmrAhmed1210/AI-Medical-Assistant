-- SQL script to add AI Diagnosis fields to the MedicalProfiles table
-- Run this on your Railway/Neon database to fix the HTTP 500 error

ALTER TABLE "MedicalProfiles" 
ADD COLUMN IF NOT EXISTS "AiDiagnosisSummary" TEXT,
ADD COLUMN IF NOT EXISTS "LastAiAnalysisAt" TIMESTAMP WITH TIME ZONE;
