using MedicalAssistant.Domain.Entities.AppointmentsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class AppointmentConfiguration : IEntityTypeConfiguration<Appointment>
    {
        public void Configure(EntityTypeBuilder<Appointment> builder)
        {
            builder.HasKey(a => a.Id);

            builder.HasOne(a => a.Patient)
                   .WithMany()
                   .HasForeignKey(a => a.PatientId)
                   .OnDelete(DeleteBehavior.Restrict);

            builder.HasOne(a => a.Doctor)
                   .WithMany()
                   .HasForeignKey(a => a.DoctorId)
                   .OnDelete(DeleteBehavior.Restrict);

            builder.Property(a => a.Date)
                   .IsRequired()
                   .HasMaxLength(20);

            builder.Property(a => a.Time)
                   .IsRequired()
                   .HasMaxLength(20);

            builder.Property(a => a.PaymentMethod)
                   .IsRequired()
                   .HasMaxLength(10)
                   .HasDefaultValue("cash");

            builder.Property(a => a.Status)
                   .IsRequired()
                   .HasMaxLength(20)
                   .HasDefaultValue("Pending");

            builder.Property(a => a.Notes)
                   .HasMaxLength(1000);

            builder.Property(a => a.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("NOW()");
        }
    }
}
