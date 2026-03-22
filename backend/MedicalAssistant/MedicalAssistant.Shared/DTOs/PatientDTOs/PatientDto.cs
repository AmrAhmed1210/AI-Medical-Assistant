namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    /// <summary>
    /// Patient DTO used for returning patient data to clients.
    /// </summary>
    public class PatientDto
    {
        public int Id { get; set; }
        public string FullName { get; set; } = default!;
        public string Email { get; set; } = default!;
        public string PhoneNumber { get; set; } = default!;
        public DateTime DateOfBirth { get; set; }
        public string Gender { get; set; } = default!;
        public string? Address { get; set; }
        public string? ImageUrl { get; set; }
        public string? BloodType { get; set; }
        public string? MedicalNotes { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool IsActive { get; set; }

        /// <summary>
        /// Patient age calculated from DateOfBirth.
        /// </summary>
        public int Age => CalculateAge();

        private int CalculateAge()
        {
            var today = DateTime.Today;
            var age = today.Year - DateOfBirth.Year;
            if (DateOfBirth.Date > today.AddYears(-age)) age--;
            return age;
        }
    }
}
