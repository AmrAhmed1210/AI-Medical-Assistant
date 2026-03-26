using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.AdminModule;
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
            // تطبيق جميع الإعدادات (Configurations) الموجودة في نفس الـ Assembly الذي يحتوي على DoctorConfiguration
            // هذا يضمن تشغيل قيود البيانات التي حددناها للأطباء والتخصصات
            modelBuilder.ApplyConfigurationsFromAssembly(typeof(DoctorConfiguration).Assembly);

            base.OnModelCreating(modelBuilder);
        }

        // الجداول الأساسية لمديول الأطباء

        // جدول الأطباء الذي يحتوي على البيانات المعروضة في الكروت
        public DbSet<Doctor> Doctors { get; set; }

        // جدول التخصصات لملء قائمة الاختيارات والبحث
        public DbSet<Specialty> Specialties { get; set; }
        public DbSet<Patient> Patients { get; set; }
        public DbSet<Appointment> Appointments { get; set; }
        public DbSet<Review> Reviews { get; set; }
        public DbSet<User> Users { get; set; }
        public DbSet<Admin> Admins { get; set; }
    }
}