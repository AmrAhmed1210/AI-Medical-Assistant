using MedicalAssistant.Domain.Entities.PatientModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class MedicalProfileConfiguration : IEntityTypeConfiguration<MedicalProfile>
    {
        public void Configure(EntityTypeBuilder<MedicalProfile> builder)
        {
            builder.HasKey(x => x.Id);

            // 1:1 مع Patient
            builder.HasOne(x => x.Patient)
                   .WithOne(p => p.MedicalProfile)
                   .HasForeignKey<MedicalProfile>(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.BloodType).HasMaxLength(5);
            builder.Property(x => x.WeightKg).HasColumnType("decimal(5,2)");
            builder.Property(x => x.HeightCm).HasColumnType("decimal(5,2)");
            builder.Property(x => x.ExerciseHabits).HasMaxLength(100);
            builder.Property(x => x.EmergencyContactName).HasMaxLength(200);
            builder.Property(x => x.EmergencyContactPhone).HasMaxLength(30);
            builder.Property(x => x.EmergencyContactRelation).HasMaxLength(100);

            builder.HasIndex(x => x.PatientId).IsUnique(); // UNIQUE عشان 1:1
        }
    }
}
