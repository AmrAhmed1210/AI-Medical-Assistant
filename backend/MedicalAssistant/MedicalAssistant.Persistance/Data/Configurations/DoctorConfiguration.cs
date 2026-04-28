using MedicalAssistant.Domain.Entities.DoctorsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations;

public class DoctorConfiguration : IEntityTypeConfiguration<Doctor>
{
    public void Configure(EntityTypeBuilder<Doctor> builder)
    {
        builder.ToTable("Doctors");

        builder.HasKey(d => d.Id);

        builder.Property(d => d.Name)
               .HasMaxLength(200);

        builder.Property(d => d.Bio)
               .HasMaxLength(1000);

        builder.Property(d => d.ConsultationFee)
               .HasColumnType("decimal(10,2)");

        builder.Property(d => d.ImageUrl)
               .HasMaxLength(500);

        builder.Property(d => d.Location)
               .HasMaxLength(200);

        builder.Property(d => d.Rating)
               .HasDefaultValue(0.0);

        builder.Property(d => d.ReviewCount)
               .HasDefaultValue(0);

        builder.Property(d => d.IsAvailable)
               .HasDefaultValue(true);

        builder.Property(d => d.IsScheduleVisible)
               .HasDefaultValue(true);

        builder.HasOne(d => d.Specialty)
               .WithMany(s => s.Doctors)
               .HasForeignKey(d => d.SpecialtyId)
               .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(d => d.User)
               .WithMany()
               .HasForeignKey(d => d.UserId)
               .OnDelete(DeleteBehavior.Restrict);
    }
}
