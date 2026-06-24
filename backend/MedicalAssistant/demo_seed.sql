-- =====================================================
-- 🎬 DEMO SEED DATA - Medical Assistant
-- Run this on SQL Server to clean & seed demo data
-- All passwords use BCrypt (compatible with .NET BCrypt)
-- =====================================================

-- =====================================================
-- STEP 1: DELETE ALL DATA (respect FK order)
-- =====================================================

DELETE FROM "MedicationLogs";
DELETE FROM "VisitVitalSigns";
DELETE FROM "VisitPrescriptions";
DELETE FROM "VisitDocuments";
DELETE FROM "Symptoms";
DELETE FROM "PatientVisits";
DELETE FROM "VitalReadings";
DELETE FROM "MedicationTrackers";
DELETE FROM "AnalysisResults";
DELETE FROM "Message";
DELETE FROM "Session";
DELETE FROM "SurgeryHistories";
DELETE FROM "MedicalProfiles";
DELETE FROM "PatientDocuments";
DELETE FROM "FollowedDoctors";
DELETE FROM "Consultations";
DELETE FROM "ChronicDiseaseMonitors";
DELETE FROM "Reviews";
DELETE FROM "Appointments";
DELETE FROM "AllergyRecords";
DELETE FROM "DoctorAvailabilities";
DELETE FROM "DoctorApplications";
DELETE FROM "Secretaries";
DELETE FROM "Doctors";
DELETE FROM "Patients";
DELETE FROM "Admins";
DELETE FROM "Users";
DELETE FROM "Specialties";

-- Reset identity seeds
DBCC CHECKIDENT ('"Specialties"', RESEED, 0);
DBCC CHECKIDENT ('"Users"', RESEED, 0);
DBCC CHECKIDENT ('"Doctors"', RESEED, 0);
DBCC CHECKIDENT ('"Patients"', RESEED, 0);
DBCC CHECKIDENT ('"Appointments"', RESEED, 0);
DBCC CHECKIDENT ('"Reviews"', RESEED, 0);
DBCC CHECKIDENT ('"DoctorAvailabilities"', RESEED, 0);
DBCC CHECKIDENT ('"Session"', RESEED, 0);
DBCC CHECKIDENT ('"Message"', RESEED, 0);
DBCC CHECKIDENT ('"Consultations"', RESEED, 0);
DBCC CHECKIDENT ('"FollowedDoctors"', RESEED, 0);
DBCC CHECKIDENT ('"MedicalProfiles"', RESEED, 0);
DBCC CHECKIDENT ('"AllergyRecords"', RESEED, 0);
DBCC CHECKIDENT ('"SurgeryHistories"', RESEED, 0);
DBCC CHECKIDENT ('"ChronicDiseaseMonitors"', RESEED, 0);
DBCC CHECKIDENT ('"VitalReadings"', RESEED, 0);
DBCC CHECKIDENT ('"MedicationTrackers"', RESEED, 0);
DBCC CHECKIDENT ('"MedicationLogs"', RESEED, 0);
DBCC CHECKIDENT ('"PatientVisits"', RESEED, 0);
DBCC CHECKIDENT ('"Symptoms"', RESEED, 0);
DBCC CHECKIDENT ('"VisitVitalSigns"', RESEED, 0);
DBCC CHECKIDENT ('"VisitPrescriptions"', RESEED, 0);
DBCC CHECKIDENT ('"VisitDocuments"', RESEED, 0);
DBCC CHECKIDENT ('"AnalysisResults"', RESEED, 0);
DBCC CHECKIDENT ('"DoctorApplications"', RESEED, 0);

PRINT '✅ All tables cleared & identity seeds reset';

-- =====================================================
-- STEP 2: INSERT SPECIALTIES
-- =====================================================

SET IDENTITY_INSERT "Specialties" ON;

INSERT INTO "Specialties" ("Id", "Name", "NameAr") VALUES
(1,  'Cardiology',          N'أمراض القلب'),
(2,  'Dermatology',         N'الأمراض الجلدية'),
(3,  'Orthopedics',         N'جراحة العظام'),
(4,  'Pediatrics',          N'طب الأطفال'),
(5,  'Neurology',           N'الأمراض العصبية'),
(6,  'Ophthalmology',       N'طب العيون'),
(7,  'ENT',                 N'الأنف والأذن والحنجرة'),
(8,  'Internal Medicine',   N'الباطنة'),
(9,  'General Surgery',     N'الجراحة العامة'),
(10, 'Psychiatry',          N'الطب النفسي'),
(11, 'Obstetrics & Gynecology', N'النساء والتوليد'),
(12, 'Urology',             N'المسالك البولية'),
(13, 'Dentistry',           N'طب الأسنان'),
(14, 'Radiology',           N'الأشعة'),
(15, 'Oncology',            N'الأورام');

SET IDENTITY_INSERT "Specialties" OFF;

PRINT '✅ 15 Specialties inserted';

-- =====================================================
-- STEP 3: INSERT USERS (Admin + Doctors + Patients)
-- =====================================================
-- Password for ALL accounts: Demo@2026
-- BCrypt hash: $2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2

SET IDENTITY_INSERT "Users" ON;

-- Admin User (Id = 1)
INSERT INTO "Users" ("Id", "FullName", "Email", "PasswordHash", "Role", "PhoneNumber", "BirthDate", "PhotoUrl", "IsActive", "IsDeleted", "CreatedAt")
VALUES (1, 'Admin', 'admin@medassist.com', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Admin', '01000000001', '1990-01-15', NULL, 1, 0, GETUTCDATE());

-- Doctor Users (Id = 2-9)
INSERT INTO "Users" ("Id", "FullName", "Email", "PasswordHash", "Role", "PhoneNumber", "BirthDate", "PhotoUrl", "IsActive", "IsDeleted", "CreatedAt")
VALUES
(2,  N'Dr. Ahmed Hassan',      'dr.ahmed@medassist.com',    '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345001', '1980-03-15', 'https://randomuser.me/api/portraits/men/32.jpg',   1, 0, GETUTCDATE()),
(3,  N'Dr. Sara Mohamed',      'dr.sara@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345002', '1985-07-22', 'https://randomuser.me/api/portraits/women/44.jpg',  1, 0, GETUTCDATE()),
(4,  N'Dr. Khaled Ibrahim',    'dr.khaled@medassist.com',   '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345003', '1978-11-08', 'https://randomuser.me/api/portraits/men/45.jpg',   1, 0, GETUTCDATE()),
(5,  N'Dr. Nour El-Din',       'dr.nour@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345004', '1982-05-30', 'https://randomuser.me/api/portraits/men/67.jpg',   1, 0, GETUTCDATE()),
(6,  N'Dr. Fatma Ali',         'dr.fatma@medassist.com',    '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345005', '1988-09-12', 'https://randomuser.me/api/portraits/women/65.jpg', 1, 0, GETUTCDATE()),
(7,  N'Dr. Omar Youssef',      'dr.omar@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345006', '1975-02-18', 'https://randomuser.me/api/portraits/men/52.jpg',   1, 0, GETUTCDATE()),
(8,  N'Dr. Mona Adel',         'dr.mona@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345007', '1990-12-25', 'https://randomuser.me/api/portraits/women/33.jpg', 1, 0, GETUTCDATE()),
(9,  N'Dr. Yasser Mahmoud',    'dr.yasser@medassist.com',   '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Doctor', '01012345008', '1983-06-05', 'https://randomuser.me/api/portraits/men/78.jpg',   1, 0, GETUTCDATE());

-- Patient Users (Id = 10-14)
INSERT INTO "Users" ("Id", "FullName", "Email", "PasswordHash", "Role", "PhoneNumber", "BirthDate", "PhotoUrl", "IsActive", "IsDeleted", "CreatedAt")
VALUES
(10, N'Mohamed Tarek',     'mohamed@medassist.com',   '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Patient', '01098765001', '1995-04-10', 'https://randomuser.me/api/portraits/men/11.jpg',   1, 0, GETUTCDATE()),
(11, N'Amira Hassan',      'amira@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Patient', '01098765002', '1998-08-20', 'https://randomuser.me/api/portraits/women/22.jpg', 1, 0, GETUTCDATE()),
(12, N'Youssef Kamal',     'youssef@medassist.com',   '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Patient', '01098765003', '2000-01-15', 'https://randomuser.me/api/portraits/men/25.jpg',   1, 0, GETUTCDATE()),
(13, N'Layla Mostafa',     'layla@medassist.com',     '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Patient', '01098765004', '1992-11-03', 'https://randomuser.me/api/portraits/women/55.jpg', 1, 0, GETUTCDATE()),
(14, N'Hassan Samir',      'hassan@medassist.com',    '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2', 'Patient', '01098765005', '1988-06-28', 'https://randomuser.me/api/portraits/men/36.jpg',   1, 0, GETUTCDATE());

SET IDENTITY_INSERT "Users" OFF;

PRINT '✅ 14 Users inserted (1 Admin + 8 Doctors + 5 Patients)';

-- =====================================================
-- STEP 4: INSERT ADMIN
-- =====================================================

INSERT INTO "Admins" ("Id", "LastLoginAt")
VALUES (1, GETUTCDATE());

PRINT '✅ Admin record created';

-- =====================================================
-- STEP 5: INSERT DOCTORS
-- =====================================================

SET IDENTITY_INSERT "Doctors" ON;

INSERT INTO "Doctors" ("Id", "UserId", "SpecialtyId", "Name", "License", "Bio", "ImageUrl", "ConsultationFee", "Experience", "Rating", "ReviewCount", "Location", "IsAvailable", "IsScheduleVisible", "CreatedAt")
VALUES
(1, 2, 1, N'Dr. Ahmed Hassan',    'LIC-CAR-2010-001',
 N'Consultant Cardiologist with extensive experience in interventional cardiology and heart failure management. Fellow of the Egyptian Society of Cardiology.',
 'https://randomuser.me/api/portraits/men/32.jpg', 350.00, 14, 4.8, 127, N'Cairo, Nasr City', 1, 1, GETUTCDATE()),

(2, 3, 2, N'Dr. Sara Mohamed',    'LIC-DER-2012-002',
 N'Board-certified Dermatologist specializing in cosmetic dermatology, laser treatments, and chronic skin conditions. Member of the Egyptian Dermatology Society.',
 'https://randomuser.me/api/portraits/women/44.jpg', 300.00, 12, 4.9, 215, N'Cairo, Maadi', 1, 1, GETUTCDATE()),

(3, 4, 3, N'Dr. Khaled Ibrahim',  'LIC-ORT-2008-003',
 N'Senior Orthopedic Surgeon with expertise in joint replacement, sports medicine, and spinal surgery. Published researcher with over 20 papers in international journals.',
 'https://randomuser.me/api/portraits/men/45.jpg', 400.00, 16, 4.7, 98, N'Giza, Dokki', 1, 1, GETUTCDATE()),

(4, 5, 4, N'Dr. Nour El-Din',     'LIC-PED-2011-004',
 N'Pediatric Specialist with a focus on neonatal care and childhood infectious diseases. Known for his patient and caring approach with children and families.',
 'https://randomuser.me/api/portraits/men/67.jpg', 250.00, 13, 4.6, 183, N'Cairo, Heliopolis', 1, 1, GETUTCDATE()),

(5, 6, 5, N'Dr. Fatma Ali',       'LIC-NEU-2013-005',
 N'Neurologist specializing in epilepsy management, migraine treatment, and neurodegenerative disorders. Certified in EEG and EMG interpretation.',
 'https://randomuser.me/api/portraits/women/65.jpg', 380.00, 11, 4.8, 156, N'Cairo, Zamalek', 1, 1, GETUTCDATE()),

(6, 7, 8, N'Dr. Omar Youssef',    'LIC-INT-2005-006',
 N'Internal Medicine Consultant with 20+ years of experience in managing diabetes, hypertension, and complex chronic conditions. Head of the Internal Medicine Department.',
 'https://randomuser.me/api/portraits/men/52.jpg', 280.00, 19, 4.5, 312, N'Alexandria, Smouha', 1, 1, GETUTCDATE()),

(7, 8, 11, N'Dr. Mona Adel',      'LIC-OBG-2015-007',
 N'OB/GYN Specialist with expertise in high-risk pregnancies, laparoscopic gynecological surgery, and fertility management. Compassionate and detail-oriented.',
 'https://randomuser.me/api/portraits/women/33.jpg', 320.00, 9, 4.9, 198, N'Cairo, New Cairo', 1, 1, GETUTCDATE()),

(8, 9, 13, N'Dr. Yasser Mahmoud', 'LIC-DEN-2009-008',
 N'Dental Surgeon specializing in cosmetic dentistry, dental implants, and orthodontics. Uses the latest digital dentistry techniques for precise treatments.',
 'https://randomuser.me/api/portraits/men/78.jpg', 200.00, 15, 4.7, 267, N'Cairo, Mohandessin', 1, 1, GETUTCDATE());

SET IDENTITY_INSERT "Doctors" OFF;

PRINT '✅ 8 Doctors inserted';

-- =====================================================
-- STEP 6: INSERT DOCTOR AVAILABILITIES
-- =====================================================
-- DayOfWeek: 0=Sunday, 1=Monday, 2=Tuesday, 3=Wednesday, 4=Thursday, 5=Friday, 6=Saturday

-- Dr. Ahmed Hassan (Id=1) - Sun, Tue, Thu
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(1, 0, '09:00', '14:00', 1, 30),
(1, 2, '09:00', '14:00', 1, 30),
(1, 4, '16:00', '21:00', 1, 30);

-- Dr. Sara Mohamed (Id=2) - Mon, Wed, Sat
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(2, 1, '10:00', '15:00', 1, 20),
(2, 3, '10:00', '15:00', 1, 20),
(2, 6, '10:00', '14:00', 1, 20);

-- Dr. Khaled Ibrahim (Id=3) - Sun, Mon, Wed
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(3, 0, '08:00', '13:00', 1, 30),
(3, 1, '08:00', '13:00', 1, 30),
(3, 3, '15:00', '20:00', 1, 30);

-- Dr. Nour El-Din (Id=4) - Sun, Tue, Thu, Sat
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(4, 0, '09:00', '13:00', 1, 20),
(4, 2, '09:00', '13:00', 1, 20),
(4, 4, '14:00', '18:00', 1, 20),
(4, 6, '10:00', '14:00', 1, 20);

-- Dr. Fatma Ali (Id=5) - Mon, Wed, Thu
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(5, 1, '11:00', '16:00', 1, 30),
(5, 3, '11:00', '16:00', 1, 30),
(5, 4, '17:00', '21:00', 1, 30);

-- Dr. Omar Youssef (Id=6) - Sun, Tue, Wed, Thu
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(6, 0, '08:00', '12:00', 1, 30),
(6, 2, '08:00', '12:00', 1, 30),
(6, 3, '14:00', '18:00', 1, 30),
(6, 4, '14:00', '18:00', 1, 30);

-- Dr. Mona Adel (Id=7) - Mon, Tue, Thu
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(7, 1, '09:00', '14:00', 1, 30),
(7, 2, '09:00', '14:00', 1, 30),
(7, 4, '16:00', '20:00', 1, 30);

-- Dr. Yasser Mahmoud (Id=8) - Sun, Mon, Wed, Sat
INSERT INTO "DoctorAvailabilities" ("DoctorId", "DayOfWeek", "StartTime", "EndTime", "IsAvailable", "SlotDurationMinutes") VALUES
(8, 0, '10:00', '15:00', 1, 20),
(8, 1, '10:00', '15:00', 1, 20),
(8, 3, '16:00', '21:00', 1, 20),
(8, 6, '10:00', '14:00', 1, 20);

PRINT '✅ Doctor availabilities inserted';

-- =====================================================
-- STEP 7: INSERT PATIENTS
-- =====================================================

SET IDENTITY_INSERT "Patients" ON;

INSERT INTO "Patients" ("Id", "FullName", "Email", "PhoneNumber", "PasswordHash", "DateOfBirth", "Gender", "Address", "ImageUrl", "BloodType", "MedicalNotes", "CreatedAt", "IsActive", "UserId")
VALUES
(1, N'Mohamed Tarek',   'mohamed@medassist.com',  '01098765001', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2',
 '1995-04-10', 'Male',   N'15 El-Tahrir St, Dokki, Giza',          'https://randomuser.me/api/portraits/men/11.jpg',   'A+',  NULL, GETUTCDATE(), 1, 10),

(2, N'Amira Hassan',    'amira@medassist.com',    '01098765002', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2',
 '1998-08-20', 'Female', N'22 El-Nasr Road, Nasr City, Cairo',      'https://randomuser.me/api/portraits/women/22.jpg', 'B+',  NULL, GETUTCDATE(), 1, 11),

(3, N'Youssef Kamal',   'youssef@medassist.com',  '01098765003', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2',
 '2000-01-15', 'Male',   N'8 Corniche El-Nile, Maadi, Cairo',       'https://randomuser.me/api/portraits/men/25.jpg',   'O+',  NULL, GETUTCDATE(), 1, 12),

(4, N'Layla Mostafa',   'layla@medassist.com',    '01098765004', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2',
 '1992-11-03', 'Female', N'45 El-Horreya Ave, Alexandria',           'https://randomuser.me/api/portraits/women/55.jpg', 'AB+', NULL, GETUTCDATE(), 1, 13),

(5, N'Hassan Samir',    'hassan@medassist.com',   '01098765005', '$2b$11$kifrESbHessyclnjSeQTaeFzFuA7GY/9hYgP37dR0amy0XKiH48B2',
 '1988-06-28', 'Male',   N'33 El-Gomhoreya St, Tanta, Gharbia',     'https://randomuser.me/api/portraits/men/36.jpg',   'A-',  NULL, GETUTCDATE(), 1, 14);

SET IDENTITY_INSERT "Patients" OFF;

PRINT '✅ 5 Patients inserted';

-- =====================================================
-- STEP 8: INSERT SOME REVIEWS (for demo display)
-- =====================================================

INSERT INTO "Reviews" ("DoctorId", "PatientId", "Author", "PatientName", "Rating", "Comment", "CreatedAt") VALUES
-- Reviews for Dr. Ahmed Hassan (Cardiology)
(1, 1, N'Mohamed Tarek',  N'Mohamed Tarek',  5, N'Excellent cardiologist! Very thorough examination and clear explanation of my condition. Highly recommended.', DATEADD(DAY, -30, GETUTCDATE())),
(1, 2, N'Amira Hassan',   N'Amira Hassan',   4, N'Professional and knowledgeable. The wait time was a bit long but the consultation was worth it.', DATEADD(DAY, -20, GETUTCDATE())),
(1, 4, N'Layla Mostafa',  N'Layla Mostafa',  5, N'Dr. Ahmed is the best heart specialist I have visited. Very caring and attentive.', DATEADD(DAY, -10, GETUTCDATE())),

-- Reviews for Dr. Sara Mohamed (Dermatology)
(2, 2, N'Amira Hassan',   N'Amira Hassan',   5, N'Amazing dermatologist! My skin condition improved significantly after following her treatment plan.', DATEADD(DAY, -25, GETUTCDATE())),
(2, 4, N'Layla Mostafa',  N'Layla Mostafa',  5, N'Dr. Sara is fantastic. She takes her time to explain everything and the results are remarkable.', DATEADD(DAY, -15, GETUTCDATE())),

-- Reviews for Dr. Khaled Ibrahim (Orthopedics)
(3, 5, N'Hassan Samir',   N'Hassan Samir',   5, N'Outstanding surgeon! My knee replacement surgery went perfectly. Recovery was smooth.', DATEADD(DAY, -40, GETUTCDATE())),
(3, 1, N'Mohamed Tarek',  N'Mohamed Tarek',  4, N'Great orthopedic doctor. Helped me recover from my sports injury quickly.', DATEADD(DAY, -22, GETUTCDATE())),

-- Reviews for Dr. Nour El-Din (Pediatrics)
(4, 2, N'Amira Hassan',   N'Amira Hassan',   5, N'My children love Dr. Nour! He is patient, gentle, and very skilled with kids.', DATEADD(DAY, -18, GETUTCDATE())),
(4, 4, N'Layla Mostafa',  N'Layla Mostafa',  4, N'Wonderful pediatrician. Always available for follow-up questions. Very reassuring.', DATEADD(DAY, -8, GETUTCDATE())),

-- Reviews for Dr. Fatma Ali (Neurology)
(5, 3, N'Youssef Kamal',  N'Youssef Kamal',  5, N'Dr. Fatma helped me manage my chronic migraines effectively. Life-changing treatment!', DATEADD(DAY, -35, GETUTCDATE())),
(5, 5, N'Hassan Samir',   N'Hassan Samir',   4, N'Knowledgeable neurologist. The diagnosis was accurate and the treatment plan worked well.', DATEADD(DAY, -12, GETUTCDATE())),

-- Reviews for Dr. Omar Youssef (Internal Medicine)
(6, 1, N'Mohamed Tarek',  N'Mohamed Tarek',  4, N'Very experienced internist. Manages my diabetes effectively with regular follow-ups.', DATEADD(DAY, -28, GETUTCDATE())),
(6, 5, N'Hassan Samir',   N'Hassan Samir',   5, N'Dr. Omar is incredibly thorough. He caught an issue other doctors missed. Very grateful.', DATEADD(DAY, -5, GETUTCDATE())),

-- Reviews for Dr. Mona Adel (OB/GYN)
(7, 2, N'Amira Hassan',   N'Amira Hassan',   5, N'Dr. Mona made my pregnancy journey comfortable and safe. Best OB/GYN in Cairo!', DATEADD(DAY, -45, GETUTCDATE())),
(7, 4, N'Layla Mostafa',  N'Layla Mostafa',  5, N'Exceptional care and professionalism. I felt safe and supported throughout.', DATEADD(DAY, -14, GETUTCDATE())),

-- Reviews for Dr. Yasser Mahmoud (Dentistry)
(8, 3, N'Youssef Kamal',  N'Youssef Kamal',  4, N'Great dentist. My dental implants look and feel natural. Minimal pain during the procedure.', DATEADD(DAY, -33, GETUTCDATE())),
(8, 1, N'Mohamed Tarek',  N'Mohamed Tarek',  5, N'Dr. Yasser is excellent! Best dental experience I have ever had. Very modern clinic.', DATEADD(DAY, -7, GETUTCDATE()));

PRINT '✅ 17 Reviews inserted';

-- =====================================================
-- ✅ DONE! Demo data is ready for video recording
-- =====================================================

PRINT '';
PRINT '========================================';
PRINT '🎬 DEMO DATA SEEDED SUCCESSFULLY!';
PRINT '========================================';
PRINT '📊 Summary:';
PRINT '   • 15 Specialties';
PRINT '   • 1 Admin';
PRINT '   • 8 Doctors (with availabilities)';
PRINT '   • 5 Patients';
PRINT '   • 17 Reviews';
PRINT '========================================';
PRINT '🔑 All accounts password: Demo@2026';
PRINT '========================================';
