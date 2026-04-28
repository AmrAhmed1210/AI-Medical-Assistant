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
    public class AllergyRecordConfiguration : IEntityTypeConfiguration<AllergyRecord>
    {
        public void Configure(EntityTypeBuilder<AllergyRecord> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.AllergyRecords)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.AllergyType).HasMaxLength(30).IsRequired();
            builder.Property(x => x.AllergenName).HasMaxLength(200).IsRequired();
            builder.Property(x => x.Severity).HasMaxLength(30).IsRequired();

            builder.HasIndex(x => x.PatientId);
            // index على IsActive عشان query الحساسيات الفعالة بتيجي كتير
            builder.HasIndex(x => new { x.PatientId, x.IsActive });
        }
    }
}
