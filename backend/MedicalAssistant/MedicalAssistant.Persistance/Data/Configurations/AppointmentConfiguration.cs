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

            // SessionId optional (no navigation configured)
            builder.Property(a => a.SessionId)
                   .IsRequired(false);

            // ScheduledAt required
            builder.Property(a => a.ScheduledAt)
                   .IsRequired();

            // Status required, max length 20
            builder.Property(a => a.Status)
                   .IsRequired()
                   .HasMaxLength(20)
                   .HasDefaultValue("Pending");

            // Reason optional
            builder.Property(a => a.Reason)
                   .HasMaxLength(500)
                   .IsRequired(false);

            // Notes optional, can be long
            builder.Property(a => a.Notes)
                   .HasMaxLength(1000)
                   .IsRequired(false);

            // Soft delete flag
            builder.Property(a => a.IsDeleted)
                   .IsRequired()
                   .HasDefaultValue(false);

            // CreatedAt required, default now (UTC)
            builder.Property(a => a.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("GETUTCDATE()");

            // UpdatedAt nullable
            builder.Property(a => a.UpdatedAt)
                   .IsRequired(false);
        }
    }
}
