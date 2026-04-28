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
    public class ChronicDiseaseMonitorConfiguration : IEntityTypeConfiguration<ChronicDiseaseMonitor>
    {
        public void Configure(EntityTypeBuilder<ChronicDiseaseMonitor> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.ChronicDiseaseMonitors)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.DiseaseName).HasMaxLength(200).IsRequired();
            builder.Property(x => x.DiseaseType).HasMaxLength(50).IsRequired();
            builder.Property(x => x.Severity).HasMaxLength(20).IsRequired();
            builder.Property(x => x.MonitoringFrequency).HasMaxLength(50).IsRequired();
            // TargetValues بيتخزن كـ JSON string
            builder.Property(x => x.TargetValues).HasColumnType("nvarchar(max)");

            builder.HasIndex(x => x.PatientId);
            builder.HasIndex(x => new { x.PatientId, x.IsActive });
        }
    }
}
