using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Infrastructure.Data.Configurations.DoctorsModule;
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

            // Unique constraint for follows
            modelBuilder.Entity<FollowedDoctor>()
                .HasIndex(f => new { f.PatientId, f.DoctorId })
                .IsUnique();

            base.OnModelCreating(modelBuilder);
        }

        // Module tables
        public DbSet<Doctor> Doctors { get; set; }
        public DbSet<DoctorAvailability> DoctorAvailabilities { get; set; }
        public DbSet<Specialty> Specialties { get; set; }
        public DbSet<Patient> Patients { get; set; }
        public DbSet<Appointment> Appointments { get; set; }
        public DbSet<Review> Reviews { get; set; }
        public DbSet<User> Users { get; set; }
        public DbSet<Session> Sessions { get; set; }
        public DbSet<FollowedDoctor> FollowedDoctors { get; set; }
    }
}