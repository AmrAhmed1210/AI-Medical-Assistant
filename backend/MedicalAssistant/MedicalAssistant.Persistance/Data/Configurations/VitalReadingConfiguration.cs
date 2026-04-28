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
    public class VitalReadingConfiguration : IEntityTypeConfiguration<VitalReading>
    {
        public void Configure(EntityTypeBuilder<VitalReading> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.VitalReadings)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasOne(x => x.ChronicDiseaseMonitor)
                   .WithMany(c => c.VitalReadings)
                   .HasForeignKey(x => x.ChronicDiseaseMonitorId)
                   .OnDelete(DeleteBehavior.SetNull)  // لو المرض اتحذف، القراءات تفضل
                   .IsRequired(false);

            builder.Property(x => x.ReadingType).HasMaxLength(30).IsRequired();
            builder.Property(x => x.Value).HasColumnType("decimal(8,2)");
            builder.Property(x => x.Value2).HasColumnType("decimal(8,2)");
            builder.Property(x => x.Unit).HasMaxLength(20).IsRequired();
            builder.Property(x => x.SugarReadingContext).HasMaxLength(20);
            builder.Property(x => x.RecordedBy).HasMaxLength(20).IsRequired();

            // مهم جداً للـ trend queries
            builder.HasIndex(x => new { x.PatientId, x.ReadingType, x.RecordedAt });
        }
    }
}
