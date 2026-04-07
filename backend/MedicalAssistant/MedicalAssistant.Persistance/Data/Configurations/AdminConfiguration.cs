using MedicalAssistant.Domain.Entities.UserModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class UserConfiguration : IEntityTypeConfiguration<User>
    {
        public void Configure(EntityTypeBuilder<User> builder)
        {
            builder.ToTable("Users");

            builder.HasKey(u => u.Id);

            builder.Property(u => u.FullName)
                   .IsRequired()
                   .HasMaxLength(120);

            builder.Property(u => u.Email)
                   .IsRequired()
                   .HasMaxLength(256);

            builder.HasIndex(u => u.Email)
                   .IsUnique();

            builder.Property(u => u.PasswordHash)
                   .IsRequired()
                   .HasMaxLength(512);

            builder.Property(u => u.Role)
                   .IsRequired()
                   .HasMaxLength(20);

            builder.Property(u => u.PhoneNumber)
                   .HasMaxLength(20);

            builder.Property(u => u.IsActive)
                   .IsRequired()
                   .HasDefaultValue(true);

            builder.Property(u => u.IsDeleted)
                   .IsRequired()
                   .HasDefaultValue(false);

            builder.Property(u => u.CreatedAt)
                   .IsRequired()
                   .HasDefaultValueSql("GETUTCDATE()");

            builder.Property(u => u.UpdatedAt)
                   .IsRequired(false);

            builder.HasQueryFilter(u => !u.IsDeleted);
        }
    }
}