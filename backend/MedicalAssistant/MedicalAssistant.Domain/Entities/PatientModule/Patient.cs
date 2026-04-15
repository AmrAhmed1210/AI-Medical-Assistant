namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class Patient : BaseEntity
    {
        public string FullName { get; set; } = default!;
        public string Email { get; set; } = default!;
        public string PhoneNumber { get; set; } = default!;
        public string PasswordHash { get; set; } = string.Empty;
        public DateTime DateOfBirth { get; set; }
        public string Gender { get; set; } = default!;
        public string? Address { get; set; }
        public string? ImageUrl { get; set; }
        public string? BloodType { get; set; }
        public string? MedicalNotes { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool IsActive { get; set; }
    }
}