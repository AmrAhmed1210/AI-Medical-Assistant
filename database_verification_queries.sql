-- =====================================================
-- MediCare Database Verification & Test Queries
-- Run these after creating the schema to verify everything
-- =====================================================

-- 1. COUNT ALL TABLES
SELECT 'TOTAL TABLES' as check_type, COUNT(*) as count
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE';

-- 2. LIST ALL TABLES
SELECT table_name as table_name
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 3. CHECK FOREIGN KEYS
SELECT 
    'FOREIGN KEYS' as check_type,
    COUNT(*) as count
FROM information_schema.table_constraints 
WHERE constraint_type = 'FOREIGN KEY' 
AND table_schema = 'public';

-- 4. CHECK UNIQUE CONSTRAINTS
SELECT 
    'UNIQUE CONSTRAINTS' as check_type,
    COUNT(*) as count
FROM information_schema.table_constraints 
WHERE constraint_type = 'UNIQUE' 
AND table_schema = 'public';

-- 5. CHECK INDEXES
SELECT 
    'INDEXES' as check_type,
    COUNT(*) as count
FROM pg_indexes 
WHERE schemaname = 'public';

-- 6. DETAILED TABLE LIST WITH COUNTS (will be 0 initially)
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns c WHERE c.table_name = t.table_name) as column_count,
    0 as row_count  -- placeholder, will show actual counts after inserts
FROM information_schema.tables t
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 7. TEST INSERTS (Run these to verify data insertion works)

-- Insert a test specialty
-- INSERT INTO "Specialties" ("Name") VALUES ('Cardiology') RETURNING "Id";

-- Insert a test user
-- INSERT INTO "Users" ("FullName", "Email", "PasswordHash", "Role") 
-- VALUES ('Test User', 'test@test.com', 'hash', 'Patient') RETURNING "Id";

-- 8. RELATIONSHIP VERIFICATION QUERIES

-- Check Patient -> MedicalProfile (1:1)
-- SELECT p."FullName", mp."BloodType" 
-- FROM "Patients" p
-- LEFT JOIN "MedicalProfiles" mp ON p."Id" = mp."PatientId";

-- Check Patient -> SurgeryHistories (1:N)
-- SELECT p."FullName", COUNT(sh."Id") as surgery_count
-- FROM "Patients" p
-- LEFT JOIN "SurgeryHistories" sh ON p."Id" = sh."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check Patient -> AllergyRecords (1:N)
-- SELECT p."FullName", COUNT(ar."Id") as allergy_count
-- FROM "Patients" p
-- LEFT JOIN "AllergyRecords" ar ON p."Id" = ar."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check Patient -> ChronicDiseaseMonitors (1:N)
-- SELECT p."FullName", COUNT(cdm."Id") as disease_count
-- FROM "Patients" p
-- LEFT JOIN "ChronicDiseaseMonitors" cdm ON p."Id" = cdm."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check Patient -> VitalReadings (1:N)
-- SELECT p."FullName", COUNT(vr."Id") as reading_count
-- FROM "Patients" p
-- LEFT JOIN "VitalReadings" vr ON p."Id" = vr."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check Patient -> MedicationTrackers (1:N)
-- SELECT p."FullName", COUNT(mt."Id") as medication_count
-- FROM "Patients" p
-- LEFT JOIN "MedicationTrackers" mt ON p."Id" = mt."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check Patient -> PatientVisits (1:N)
-- SELECT p."FullName", COUNT(pv."Id") as visit_count
-- FROM "Patients" p
-- LEFT JOIN "PatientVisits" pv ON p."Id" = pv."PatientId"
-- GROUP BY p."Id", p."FullName";

-- Check PatientVisit -> Symptoms (1:N)
-- SELECT pv."Id", pv."ChiefComplaint", COUNT(s."Id") as symptom_count
-- FROM "PatientVisits" pv
-- LEFT JOIN "Symptoms" s ON pv."Id" = s."PatientVisitId"
-- GROUP BY pv."Id", pv."ChiefComplaint";

-- Check PatientVisit -> VisitVitalSigns (1:N)
-- SELECT pv."Id", pv."ChiefComplaint", COUNT(vvs."Id") as vital_count
-- FROM "PatientVisits" pv
-- LEFT JOIN "VisitVitalSigns" vvs ON pv."Id" = vvs."PatientVisitId"
-- GROUP BY pv."Id", pv."ChiefComplaint";

-- Check PatientVisit -> VisitPrescriptions (1:N)
-- SELECT pv."Id", pv."ChiefComplaint", COUNT(vp."Id") as prescription_count
-- FROM "PatientVisits" pv
-- LEFT JOIN "VisitPrescriptions" vp ON pv."Id" = vp."PatientVisitId"
-- GROUP BY pv."Id", pv."ChiefComplaint";

-- Check PatientVisit -> VisitDocuments (1:N)
-- SELECT pv."Id", pv."ChiefComplaint", COUNT(vd."Id") as document_count
-- FROM "PatientVisits" pv
-- LEFT JOIN "VisitDocuments" vd ON pv."Id" = vd."PatientVisitId"
-- GROUP BY pv."Id", pv."ChiefComplaint";

-- Check ChronicDiseaseMonitor -> VitalReadings (1:N)
-- SELECT cdm."DiseaseName", COUNT(vr."Id") as reading_count
-- FROM "ChronicDiseaseMonitors" cdm
-- LEFT JOIN "VitalReadings" vr ON cdm."Id" = vr."ChronicDiseaseMonitorId"
-- GROUP BY cdm."Id", cdm."DiseaseName";

-- Check ChronicDiseaseMonitor -> MedicationTrackers (1:N)
-- SELECT cdm."DiseaseName", COUNT(mt."Id") as medication_count
-- FROM "ChronicDiseaseMonitors" cdm
-- LEFT JOIN "MedicationTrackers" mt ON cdm."Id" = mt."ChronicDiseaseMonitorId"
-- GROUP BY cdm."Id", cdm."DiseaseName";

-- Check MedicationTracker -> MedicationLogs (1:N)
-- SELECT mt."MedicationName", COUNT(ml."Id") as log_count
-- FROM "MedicationTrackers" mt
-- LEFT JOIN "MedicationLogs" ml ON mt."Id" = ml."MedicationTrackerId"
-- GROUP BY mt."Id", mt."MedicationName";

-- Check Doctor -> PatientVisits (1:N)
-- SELECT d."Name", COUNT(pv."Id") as visit_count
-- FROM "Doctors" d
-- LEFT JOIN "PatientVisits" pv ON d."Id" = pv."DoctorId"
-- GROUP BY d."Id", d."Name";

-- Check Doctor -> MedicationTrackers (1:N via PrescribedBy)
-- SELECT d."Name", COUNT(mt."Id") as prescriptions_count
-- FROM "Doctors" d
-- LEFT JOIN "MedicationTrackers" mt ON d."Id" = mt."PrescribedByDoctorId"
-- GROUP BY d."Id", d."Name";

-- =====================================================
-- PERFORMANCE CHECK QUERIES
-- =====================================================

-- Check table sizes
SELECT 
    relname as table_name,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    pg_size_pretty(pg_relation_size(relid)) as table_size,
    pg_size_pretty(pg_indexes_size(relid)) as index_size
FROM pg_catalog.pg_statio_user_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(relid) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;

-- =====================================================
-- SUMMARY REPORT
-- =====================================================

SELECT '=== DATABASE SCHEMA SUMMARY ===' as report;

SELECT 
    'Total Tables: ' || COUNT(*)::text as summary
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE';

SELECT 
    'Total Foreign Keys: ' || COUNT(*)::text as summary
FROM information_schema.table_constraints 
WHERE constraint_type = 'FOREIGN KEY' 
AND table_schema = 'public';

SELECT 
    'Total Indexes: ' || COUNT(*)::text as summary
FROM pg_indexes 
WHERE schemaname = 'public';

SELECT 
    'Total Columns: ' || COUNT(*)::text as summary
FROM information_schema.columns 
WHERE table_schema = 'public';

SELECT '=== SCHEMA IS READY FOR USE ===' as report;

COMMIT;
