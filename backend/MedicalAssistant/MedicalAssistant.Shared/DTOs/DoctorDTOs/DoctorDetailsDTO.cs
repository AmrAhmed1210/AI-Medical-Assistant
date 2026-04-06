namespace MedicalAssistant.Shared.DTOs.DoctorDTOs
{
    public class DoctorDetailsDTO : DoctorDTO
    {
        public int? YearsExperience { get; set; }

        public string? Bio { get; set; }

        public string License { get; set; } = string.Empty;

        public string Email { get; set; } = string.Empty;
    }
}
