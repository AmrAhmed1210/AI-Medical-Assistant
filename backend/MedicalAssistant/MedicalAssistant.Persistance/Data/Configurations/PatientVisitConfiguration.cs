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
    public class PatientVisitConfiguration : IEntityTypeConfiguration<PatientVisit>
    {
        public void Configure(EntityTypeBuilder<PatientVisit> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.PatientVisits)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.HasOne(x => x.Doctor)
                   .WithMany(d => d.PatientVisits)
                   .HasForeignKey(x => x.DoctorId)
                   .OnDelete(DeleteBehavior.NoAction); // مش هنحذف الزيارة لو الدكتور اتحذف

            builder.Property(x => x.ChiefComplaint).IsRequired();
            builder.Property(x => x.Status).HasMaxLength(20).HasDefaultValue("active");
            builder.Property(x => x.SummarySnapshot).HasColumnType("nvarchar(max)");

            builder.HasIndex(x => new { x.PatientId, x.VisitDate });
            builder.HasIndex(x => new { x.DoctorId, x.VisitDate });
            builder.HasIndex(x => new { x.DoctorId, x.Status }); // لـ today's active visits
        }
    }
}
