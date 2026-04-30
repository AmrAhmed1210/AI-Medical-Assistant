using MedicalAssistant.Domain.Entities.AdminModule;
using MedicalAssistant.Domain.Entities.AnalysisModule;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.ConsultationsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Persistance.Data.Configurations;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant.Persistance.Data.DbContexts
{
    public class MedicalAssistantDbContext : DbContext
    {
        public MedicalAssistantDbContext(DbContextOptions<MedicalAssistantDbContext> options) : base(options)
        {
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.ApplyConfigurationsFromAssembly(typeof(DoctorConfiguration).Assembly);
            modelBuilder.Entity<Session>().ToTable("Session");
            modelBuilder.Entity<FollowedDoctor>()
                .HasIndex(f => new { f.PatientId, f.DoctorId })
                .IsUnique();
            
            // Fix: Use TPT (Table Per Type) for Admin inheritance - avoids Discriminator column
            modelBuilder.Entity<Admin>().ToTable("Admins");
            
            base.OnModelCreating(modelBuilder);
        }

        public DbSet<Doctor> Doctors { get; set; }
        public DbSet<DoctorAvailability> DoctorAvailabilities { get; set; }
        public DbSet<Specialty> Specialties { get; set; }
        public DbSet<Patient> Patients { get; set; }
        public DbSet<Appointment> Appointments { get; set; }
        public DbSet<Consultation> Consultations { get; set; }
        public DbSet<Review> Reviews { get; set; }
        public DbSet<User> Users { get; set; }
        public DbSet<Admin> Admins { get; set; }
        public DbSet<Session> Sessions { get; set; }
        public DbSet<Message> Messages { get; set; }
        public DbSet<FollowedDoctor> FollowedDoctors { get; set; }
        public DbSet<DoctorApplication> DoctorApplications { get; set; }

        public DbSet<MedicalProfile> MedicalProfiles { get; set; }
        public DbSet<SurgeryHistory> SurgeryHistories { get; set; }
        public DbSet<AllergyRecord> AllergyRecords { get; set; }
        public DbSet<ChronicDiseaseMonitor> ChronicDiseaseMonitors { get; set; }
        public DbSet<VitalReading> VitalReadings { get; set; }
        public DbSet<MedicationTracker> MedicationTrackers { get; set; }
        public DbSet<MedicationLog> MedicationLogs { get; set; }
        public DbSet<PatientVisit> PatientVisits { get; set; }
        public DbSet<Symptom> Symptoms { get; set; }
        public DbSet<VisitVitalSign> VisitVitalSigns { get; set; }
        public DbSet<VisitPrescription> VisitPrescriptions { get; set; }
        public DbSet<VisitDocument> VisitDocuments { get; set; }
        public DbSet<AnalysisResult> AnalysisResults { get; set; }
    }
}