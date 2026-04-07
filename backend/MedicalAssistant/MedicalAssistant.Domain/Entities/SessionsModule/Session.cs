using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.SessionsModule
{
    public class Session : BaseEntity
    {
        // Owning patient (user)
        public int UserId { get; set; }
        public virtual User User { get; set; } = default!;

        // Auto-generated title
        public string? Title { get; set; }

        // Urgency: LOW, MEDIUM, HIGH, EMERGENCY
        public string? UrgencyLevel { get; set; }

        // Soft delete
        public bool IsDeleted { get; set; } = false;

        // Created at (session start)
        public DateTime CreatedAt { get; set; }

        // Updated at (last message time)
        public DateTime? UpdatedAt { get; set; }
    }
}
