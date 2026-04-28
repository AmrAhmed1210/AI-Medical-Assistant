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
    public class VisitVitalSignConfiguration : IEntityTypeConfiguration<VisitVitalSign>
    {
        public void Configure(EntityTypeBuilder<VisitVitalSign> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.PatientVisit)
                   .WithMany(v => v.VitalSigns)
                   .HasForeignKey(x => x.PatientVisitId)
                   .OnDelete(DeleteBehavior.Cascade)
                   .IsRequired(false);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.VisitVitalSigns)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.NoAction);

            builder.Property(x => x.Type).HasMaxLength(30).IsRequired();
            builder.Property(x => x.Value).HasColumnType("decimal(8,2)");
            builder.Property(x => x.Value2).HasColumnType("decimal(8,2)");
            builder.Property(x => x.Unit).HasMaxLength(20).IsRequired();
            builder.Property(x => x.NormalRangeMin).HasColumnType("decimal(8,2)");
            builder.Property(x => x.NormalRangeMax).HasColumnType("decimal(8,2)");
            builder.Property(x => x.RecordedBy).HasMaxLength(20).IsRequired();

            builder.HasIndex(x => x.PatientVisitId);
        }
    }
}
