using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.UserModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Infrastructure.Data.Configurations.DoctorsModule;

public class DoctorConfiguration : IEntityTypeConfiguration<Doctor>
{
    public void Configure(EntityTypeBuilder<Doctor> builder)
    {
        builder.ToTable("Doctors");

        builder.HasKey(d => d.Id);

        builder.Property(d => d.Name)
               .IsRequired()
               .HasMaxLength(150);

        builder.Property(d => d.Location)
               .HasMaxLength(200);

        builder.Property(d => d.Bio)
               .HasMaxLength(1000);

        builder.Property(d => d.ConsultationFee)
               .HasColumnType("decimal(10,2)");

        builder.Property(d => d.ImageUrl)
               .HasMaxLength(500);

        builder.HasOne(d => d.Specialty)
               .WithMany(s => s.Doctors)
               .HasForeignKey(d => d.SpecialtyId)
               .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(d => d.User)
               .WithMany()
               .HasForeignKey(d => d.UserId)
               .OnDelete(DeleteBehavior.SetNull);
    }
}