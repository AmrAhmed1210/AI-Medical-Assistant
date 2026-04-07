using MedicalAssistant.Domain.Entities.SessionsModule;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MedicalAssistant.Persistance.Data.Configurations
{
    public class MessageConfiguration : IEntityTypeConfiguration<Message>
    {
        public void Configure(EntityTypeBuilder<Message> builder)
        {
            builder.HasKey(m => m.Id);

            builder.Property(m => m.SessionId).IsRequired();

            builder.HasOne(m => m.Session)
                   .WithMany()
                   .HasForeignKey(m => m.SessionId)
                   .OnDelete(DeleteBehavior.Cascade);

            builder.Property(m => m.Role)
                   .IsRequired()
                   .HasMaxLength(20);

            builder.Property(m => m.Content)
                   .IsRequired()
                   .HasColumnType("nvarchar(max)");

            builder.Property(m => m.Timestamp)
                   .IsRequired()
                   .HasDefaultValueSql("GETUTCDATE()");

            builder.HasIndex(m => m.SessionId).HasDatabaseName("IX_Messages_SessionId");
            builder.HasIndex(m => m.Timestamp).HasDatabaseName("IX_Messages_Timestamp");
        }
    }
}
