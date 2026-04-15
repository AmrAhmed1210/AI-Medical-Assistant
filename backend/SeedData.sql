-- =============================================
-- MedBook Seed Data
-- Run this in SQL Server Management Studio
-- or Azure Data Studio on: MedicalAssistantDb
-- =============================================

USE MedicalAssistantDb;

-- =============================================
-- 1. Specialties
-- =============================================
INSERT INTO Specialties (Name) VALUES
('Cardiology'),
('Dermatology'),
('Neurology'),
('Orthopedics'),
('Pediatrics'),
('Gynecology'),
('Ophthalmology'),
('ENT');

-- =============================================
-- 2. Doctors
-- =============================================
INSERT INTO Doctors (Name, Location, Experience, ConsultationFee, Rating, ReviewCount, IsAvailable, Bio, ImageUrl, SpecialtyId)
VALUES
('Dr. Sarah Ahmed',    'Cairo, Egypt',       10, 150, 4.8, 124, 1, 'Specialist in cardiovascular diseases with 10 years of experience at Cairo University Hospital.', 'default-doctor.png', 1),
('Dr. Mohamed Ali',   'Giza, Egypt',          8, 120, 4.6,  98, 1, 'Expert dermatologist specializing in skin disorders and cosmetic dermatology.', 'default-doctor.png', 2),
('Dr. Nour Hassan',   'Alexandria, Egypt',   12, 200, 4.9,  87, 1, 'Senior neurologist with expertise in stroke management and epilepsy treatment.', 'default-doctor.png', 3),
('Dr. Ahmed Khaled',  'Cairo, Egypt',         7, 130, 4.5,  76, 1, 'Orthopedic surgeon specializing in joint replacement and sports injuries.', 'default-doctor.png', 4),
('Dr. Mona Samir',    'Giza, Egypt',          9, 100, 4.7, 112, 1, 'Pediatrician with special interest in child development and vaccination.', 'default-doctor.png', 5),
('Dr. Hana Youssef',  'Cairo, Egypt',        15, 180, 4.9,  95, 1, 'Gynecologist and obstetrician with extensive experience in high-risk pregnancies.', 'default-doctor.png', 6),
('Dr. Tarek Ibrahim', 'Alexandria, Egypt',    6, 110, 4.4,  63, 0, 'Ophthalmologist specializing in cataract surgery and laser vision correction.', 'default-doctor.png', 7),
('Dr. Rania Mostafa', 'Cairo, Egypt',        11, 140, 4.6,  89, 1, 'ENT specialist with expertise in hearing disorders and sinus surgery.', 'default-doctor.png', 8);

-- =============================================
-- 3. Patients (for testing)
-- =============================================
INSERT INTO Patients (FullName, Email, PhoneNumber, DateOfBirth, Gender, IsActive, CreatedAt)
VALUES
('Ahmed Ali',   'ahmed@example.com',  '01012345678', '1990-05-15', 'Male',   1, GETUTCDATE()),
('Sara Mohamed','sara@example.com',   '01098765432', '1995-08-20', 'Female', 1, GETUTCDATE());

-- =============================================
-- 4. Reviews
-- =============================================
INSERT INTO Reviews (DoctorId, Author, Rating, Comment, CreatedAt)
VALUES
(1, 'Ahmed Ali',    5, 'Excellent doctor, very professional and caring.',        '2026-03-06'),
(1, 'Sara Mohamed', 4, 'Great experience, explained everything clearly.',         '2026-03-10'),
(1, 'Omar Hassan',  5, 'Best cardiologist I have ever visited.',                 '2026-03-15'),
(2, 'Mona Khaled',  4, 'Very knowledgeable and friendly.',                       '2026-03-08'),
(2, 'Ahmed Ali',    5, 'Fixed my skin problem in just 2 visits.',                '2026-03-12'),
(3, 'Sara Mohamed', 5, 'Outstanding neurologist, highly recommended.',            '2026-03-01'),
(4, 'Omar Hassan',  4, 'Good surgeon, recovery was smooth.',                     '2026-03-05'),
(5, 'Nadia Samir',  5, 'Amazing with kids, very patient and gentle.',            '2026-03-18'),
(6, 'Hana Ali',     5, 'Best gynecologist, made me feel very comfortable.',      '2026-03-20'),
(8, 'Ahmed Ali',    4, 'Very thorough examination, good advice.',                '2026-03-14');

-- =============================================
-- Verify data
-- =============================================
SELECT 'Specialties' AS TableName, COUNT(*) AS Count FROM Specialties
UNION ALL
SELECT 'Doctors',  COUNT(*) FROM Doctors
UNION ALL
SELECT 'Patients', COUNT(*) FROM Patients
UNION ALL
SELECT 'Reviews',  COUNT(*) FROM Reviews;
