using MedicalAssistant.Domain.Entities.ReviewsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace MedicalAssistant.Persistence.Configurations
{
    public class ReviewConfiguration : IEntityTypeConfiguration<Review>
    {
        public void Configure(EntityTypeBuilder<Review> builder)
        {
            builder.ToTable("Reviews");

            builder.HasKey(r => r.Id);

            builder.Property(r => r.Author)
                   .IsRequired()
                   .HasMaxLength(100);

            builder.Property(r => r.Comment)
                   .IsRequired()
                   .HasMaxLength(1000);

            builder.Property(r => r.Rating)
                   .IsRequired();

            builder.Property(r => r.CreatedAt)
                   .HasDefaultValueSql("NOW()");

            builder.HasOne(r => r.Doctor)
                   .WithMany(d => d.Reviews)
                   .HasForeignKey(r => r.DoctorId)
                   .OnDelete(DeleteBehavior.Cascade);
        }
    }
}