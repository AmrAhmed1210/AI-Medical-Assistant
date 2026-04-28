using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.SessionsModule
{
    public class Session : BaseEntity
    {
        public int UserId { get; set; }
        public virtual User User { get; set; } = null!;

        public string? Title { get; set; }

        public string? UrgencyLevel { get; set; }

        public string Type { get; set; } = "AI";

        public bool IsDeleted { get; set; } = false;

        public DateTime CreatedAt { get; set; }

        public DateTime? UpdatedAt { get; set; }
    }
}
