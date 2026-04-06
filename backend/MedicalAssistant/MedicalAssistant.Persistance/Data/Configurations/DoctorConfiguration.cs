using MedicalAssistant.Domain.Entities.DoctorsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Infrastructure.Data.Configurations.DoctorsModule;

public class DoctorConfiguration : IEntityTypeConfiguration<Doctor>
{
    public void Configure(EntityTypeBuilder<Doctor> builder)
    {
        // 1. Table Name matching SQL Schema
        builder.ToTable("Doctor Profiles");

        builder.HasKey(d => d.Id);

        // 2. Link to User Account (FK -> Users.Id)
        builder.Property(d => d.UserId)
               .IsRequired();

        builder.HasIndex(d => d.UserId)
               .IsUnique();

        // 3. Specialty Relationship (Normalization)
        // حذفنا Property(d => d.Specialty) اللي كانت String 
        // واستبدلناها بالعلاقة دي:
        builder.HasOne(d => d.Specialty)
               .WithMany(s => s.Doctors)
               .HasForeignKey(d => d.SpecialtyId)
               .OnDelete(DeleteBehavior.Restrict);

        // 4. License & Professional Info
        builder.Property(d => d.License)
               .IsRequired()
               .HasMaxLength(60);

        builder.HasIndex(d => d.License)
               .IsUnique();

        builder.Property(d => d.YearsExperience)
               .IsRequired(false);

        // 5. Content Properties
        builder.Property(d => d.Bio)
               .HasMaxLength(1000);

        builder.Property(d => d.PhotoUrl)
               .HasMaxLength(500);

        builder.Property(d => d.ConsultFee)
               .HasColumnType("decimal(10,2)");

        // 6. Timestamps
        builder.Property(d => d.CreatedAt)
               .HasDefaultValueSql("GETUTCDATE()");

        builder.Property(d => d.UpdatedAt)
               .IsRequired(false);

        // في DoctorConfiguration.cs
        builder.Property(d => d.ConsultFee).HasPrecision(18, 2);
    }
}