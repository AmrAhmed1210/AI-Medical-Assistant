using MedicalAssistant.Domain.Entities.SessionsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class SessionConfiguration : IEntityTypeConfiguration<Session>
    {
        public void Configure(EntityTypeBuilder<Session> builder)
        {
            builder.HasKey(s => s.Id);
            builder.HasOne(a => a.User)
                              .WithMany()
                              .HasForeignKey(a => a.UserId)
                              .OnDelete(DeleteBehavior.Restrict);

            builder.Property(s => s.Title)
                   .HasMaxLength(200)
                   .IsRequired(false);

            builder.Property(s => s.UrgencyLevel)
                   .HasMaxLength(20)
                   .IsRequired(false);

            builder.Property(s => s.IsDeleted)
                   .IsRequired()
                   .HasDefaultValue(false);

            builder.Property(s => s.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("GETUTCDATE()");

            builder.Property(s => s.UpdatedAt)
                   .IsRequired(false);
        }
    }
}
