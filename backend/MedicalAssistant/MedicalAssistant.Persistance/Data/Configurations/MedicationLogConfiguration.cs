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
    public class MedicationLogConfiguration : IEntityTypeConfiguration<MedicationLog>
    {
        public void Configure(EntityTypeBuilder<MedicationLog> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.MedicationTracker)
                   .WithMany(m => m.Logs)
                   .HasForeignKey(x => x.MedicationTrackerId)
                   .OnDelete(DeleteBehavior.Cascade);

            // NoAction عشان Patient فيه بالفعل Cascade من MedicationTracker
            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.MedicationLogs)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.Status).HasMaxLength(20).HasDefaultValue("pending");

            // الـ Cron بيـ query على ScheduledAt كتير
            builder.HasIndex(x => new { x.PatientId, x.ScheduledAt, x.Status });
            builder.HasIndex(x => new { x.MedicationTrackerId, x.Status });
        }
    }
}
