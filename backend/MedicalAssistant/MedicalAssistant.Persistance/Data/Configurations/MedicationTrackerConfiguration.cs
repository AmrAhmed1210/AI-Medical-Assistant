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
    public class MedicationTrackerConfiguration : IEntityTypeConfiguration<MedicationTracker>
    {
        public void Configure(EntityTypeBuilder<MedicationTracker> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.MedicationTrackers)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasOne(x => x.PrescribedByDoctor)
                   .WithMany()
                   .HasForeignKey(x => x.PrescribedByDoctorId)
                   .OnDelete(DeleteBehavior.SetNull)
                   .IsRequired(false);

            builder.HasOne(x => x.ChronicDiseaseMonitor)
                   .WithMany(c => c.Medications)
                   .HasForeignKey(x => x.ChronicDiseaseMonitorId)
                   .OnDelete(DeleteBehavior.SetNull)
                   .IsRequired(false);

            builder.Property(x => x.MedicationName).HasMaxLength(200).IsRequired();
            builder.Property(x => x.GenericName).HasMaxLength(200);
            builder.Property(x => x.Dosage).HasMaxLength(100).IsRequired();
            builder.Property(x => x.Form).HasMaxLength(30).IsRequired();
            builder.Property(x => x.Frequency).HasMaxLength(100).IsRequired();
            builder.Property(x => x.DoseTimes).HasColumnType("nvarchar(200)"); // JSON
            builder.Property(x => x.RefillThreshold).HasDefaultValue(7);

            builder.HasIndex(x => new { x.PatientId, x.IsActive });
        }
    }
}
