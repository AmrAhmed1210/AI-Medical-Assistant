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
    public class VisitPrescriptionConfiguration : IEntityTypeConfiguration<VisitPrescription>
    {
        public void Configure(EntityTypeBuilder<VisitPrescription> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.PatientVisit)
                   .WithMany(v => v.Prescriptions)
                   .HasForeignKey(x => x.PatientVisitId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.Id).ValueGeneratedOnAdd();

            builder.Property(x => x.MedicationName).HasMaxLength(200).IsRequired();
            builder.Property(x => x.GenericName).HasMaxLength(200);
            builder.Property(x => x.Dosage).HasMaxLength(100).IsRequired();
            builder.Property(x => x.Form).HasMaxLength(30).IsRequired();
            builder.Property(x => x.Frequency).HasMaxLength(100).IsRequired();
            builder.Property(x => x.SpecificTimes).HasColumnType("nvarchar(200)"); // JSON
            builder.Property(x => x.Duration).HasMaxLength(100);
            builder.Property(x => x.Refills).HasDefaultValue(0);

            builder.HasIndex(x => x.PatientVisitId);
        }
    }
}
