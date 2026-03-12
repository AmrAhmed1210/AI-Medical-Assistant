using MedicalAssistant.Domain.Entities.AppointmentsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class AppointmentConfiguration : IEntityTypeConfiguration<Appointment>
    {
        public void Configure(EntityTypeBuilder<Appointment> builder)
        {
            // Primary key
            builder.HasKey(a => a.Id);

            // Required relationships
            builder.HasOne(a => a.Patient)
                   .WithMany()
                   .HasForeignKey(a => a.PatientId)
                   .OnDelete(DeleteBehavior.Restrict);

            builder.HasOne(a => a.Doctor)
                   .WithMany()
                   .HasForeignKey(a => a.DoctorId)
                   .OnDelete(DeleteBehavior.Restrict);

            // Date and time required
            builder.Property(a => a.AppointmentDate)
                   .IsRequired();
            builder.Property(a => a.AppointmentTime)
                   .IsRequired();

            // Status required, max length 20
            builder.Property(a => a.Status)
                   .IsRequired()
                   .HasMaxLength(20)
                   .HasDefaultValue("Pending");

            // Notes optional, can be long
            builder.Property(a => a.Notes)
                   .HasMaxLength(1000);

            // CreatedAt required, default now
            builder.Property(a => a.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("GETUTCDATE()");
        }
    }
}
