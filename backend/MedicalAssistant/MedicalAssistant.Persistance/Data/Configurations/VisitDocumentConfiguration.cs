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
    public class VisitDocumentConfiguration : IEntityTypeConfiguration<VisitDocument>
    {
        public void Configure(EntityTypeBuilder<VisitDocument> builder)
        {
            builder.HasKey(x => x.Id);

            builder.HasOne(x => x.PatientVisit)
                   .WithMany(v => v.Documents)
                   .HasForeignKey(x => x.PatientVisitId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(x => x.DocumentType).HasMaxLength(30).IsRequired();
            builder.Property(x => x.Title).HasMaxLength(300).IsRequired();
            builder.Property(x => x.FileUrl).IsRequired();
            builder.Property(x => x.FileType).HasMaxLength(50).IsRequired();

            builder.HasIndex(x => x.PatientVisitId);
        }
    }
}
