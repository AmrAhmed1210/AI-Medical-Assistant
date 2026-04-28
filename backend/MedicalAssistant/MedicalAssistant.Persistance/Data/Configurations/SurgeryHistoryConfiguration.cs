using MedicalAssistant.Domain.Entities.PatientModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class SurgeryHistoryConfiguration : IEntityTypeConfiguration<SurgeryHistory>
    {
        public void Configure(EntityTypeBuilder<SurgeryHistory> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.Patient)
                   .WithMany(p => p.SurgeryHistories)
                   .HasForeignKey(x => x.PatientId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.SurgeryName).HasMaxLength(300).IsRequired();
            builder.Property(x => x.HospitalName).HasMaxLength(200);
            builder.Property(x => x.DoctorName).HasMaxLength(200);

            builder.HasIndex(x => x.PatientId);
        }
    }
}
