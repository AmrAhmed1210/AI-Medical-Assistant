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
    public class SymptomConfiguration : IEntityTypeConfiguration<Symptom>
    {
        public void Configure(EntityTypeBuilder<Symptom> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.PatientVisit)
                   .WithMany(v => v.Symptoms)
                   .HasForeignKey(x => x.PatientVisitId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.Name).HasMaxLength(200).IsRequired();
            builder.Property(x => x.Severity).HasMaxLength(20).IsRequired();
            builder.Property(x => x.Onset).HasMaxLength(20);
            builder.Property(x => x.Progression).HasMaxLength(20);
            builder.Property(x => x.Location).HasMaxLength(100);

            builder.HasIndex(x => x.PatientVisitId);
        }
    }
}
