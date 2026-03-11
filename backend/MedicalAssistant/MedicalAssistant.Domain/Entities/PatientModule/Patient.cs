namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class Patient : BaseEntity
    {
        // Patient full name
        public string FullName { get; set; } = default!;

        // Patient email
        public string Email { get; set; } = default!;

        // Phone number
        public string PhoneNumber { get; set; } = default!;

        // Date of birth
        public DateTime DateOfBirth { get; set; }

        // Gender (Male/Female)
        public string Gender { get; set; } = default!;

        // Address
        public string? Address { get; set; }

        // Profile image url (optional)
        public string? ImageUrl { get; set; }

        // Blood type (optional)
        public string? BloodType { get; set; }

        // Medical notes / chronic conditions
        public string? MedicalNotes { get; set; }

        // Registration date
        public DateTime CreatedAt { get; set; }

        // Account activation status
        public bool IsActive { get; set; }
    }
}
