using MedicalAssistant.Domain.Entities.PatientModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class PatientConfiguration : IEntityTypeConfiguration<Patient>
    {
        public void Configure(EntityTypeBuilder<Patient> builder)
        {
            // Properties configuration

            // Full name is required with max length 100
            builder.Property(p => p.FullName)
                   .IsRequired()
                   .HasMaxLength(100);

            // Email is required and unique
            builder.Property(p => p.Email)
                   .IsRequired()
                   .HasMaxLength(150);

            builder.HasIndex(p => p.Email)
                   .IsUnique();

            // Phone number is required
            builder.Property(p => p.PhoneNumber)
                   .IsRequired()
                   .HasMaxLength(20);

            // Date of birth is required
            builder.Property(p => p.DateOfBirth)
                   .IsRequired();

            // Gender is required
            builder.Property(p => p.Gender)
                   .IsRequired()
                   .HasMaxLength(10);

            // Address is optional
            builder.Property(p => p.Address)
                   .HasMaxLength(300);

            // Profile image url is optional
            builder.Property(p => p.ImageUrl);

            // Blood type is optional
            builder.Property(p => p.BloodType)
                   .HasMaxLength(5);

            // Medical notes can be long
            builder.Property(p => p.MedicalNotes);

            // CreatedAt is required with default value
            builder.Property(p => p.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("NOW()");

            // IsActive is true by default
            builder.Property(p => p.IsActive)
                   .HasDefaultValue(true);
        }
    }
}
